# Paper Generation Service - services.py
import os
import json
import logging
import asyncio
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Any, Dict, List, Optional
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from filelock import FileLock
from fastapi import BackgroundTasks

# Third-party AI clients
import google.generativeai as genai
from google import genai as google_genai_client

# --- New modular imports ---
import config
import models
import utils
from cache import (
    create_cache_key,
    add_cache_version,
    get_latest_cache_version,
    get_cache_version_by_id,
)

# Import specific functions from the original codebase structure
from full_paper.run_full_pipeline import (
    derive_plan_from_filedata,
    build_retrieval_objective,
    retrieve_from_pinecone,
    mmr_and_stratified_sample,
    build_generator_prompt_questions_only,
    call_gemini,
    parse_generator_response,
    load_schema_row,
)

# Import Google embedding service (with dimension reduction to match Pinecone)
from google_embedding_service_fixed import embed_texts_google, embed_query_google, load_google_embeddings

# Import structured paper generation (simpler alternative to DSPy)
from structured_paper_generator import generate_paper_with_structured_output
from full_paper.retrieval_and_summarization import (
    generate_dense_evidence_summary_with_llm,
)
from logger import log

from pydantic import ValidationError

# -------- Module-level State --------
LOG = logging.getLogger("uvicorn.error")
_executor: ThreadPoolExecutor = None
df_content = pd.DataFrame()
df_prompt = pd.DataFrame()

# For LLM Researcher
llm_chat_model = None
embedding_client = None

# -------- Custom Exceptions --------
class SchemaNotFoundError(Exception):
    """Custom exception for when a schema is not found."""
    pass

class GenerationError(Exception):
    """Custom exception for failures during the generation process."""
    def __init__(self, message, raw_output=None):
        super().__init__(message)
        self.raw_output = raw_output

# -------- Initialization and State Management --------
def set_executor(executor: ThreadPoolExecutor):
    """Receives the executor instance from main.py during startup."""
    global _executor
    _executor = executor

def load_heavy_models_and_data():
    """
    Loads all heavy resources: CSVs, embedding models, and configures API clients.
    This is a blocking function intended to be run in the executor during startup.
    """
    global df_content, df_prompt, llm_chat_model, embedding_client
    LOG.info("Initiating loading of heavy models and data...")

    # 1. Load CSV data
    try:
        df_content = pd.read_csv(config.CONTENT_CSV_PATH)
        LOG.info(
            f"Loaded content CSV from {config.CONTENT_CSV_PATH} ({len(df_content)} rows)"
        )
    except Exception as e:
        LOG.error(f"Could not load content CSV at {config.CONTENT_CSV_PATH}: {e}")

    try:
        df_prompt = pd.read_csv(config.PROMPT_CSV_PATH)
        LOG.info(
            f"Loaded prompt CSV from {config.PROMPT_CSV_PATH} ({len(df_prompt)} rows)"
        )
    except Exception as e:
        LOG.error(f"Could not load prompt CSV at {config.PROMPT_CSV_PATH}: {e}")

    # 2. Configure Google Generative AI clients
    if config.GOOGLE_API_KEY:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        LOG.info("Configured google.generativeai client.")
        try:
            # For LLM Researcher part
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-lite",
                generation_config=generation_config,
            )
            llm_chat_model = model.start_chat(history=[])
            embedding_client = google_genai_client.Client(api_key=config.GOOGLE_API_KEY)
            LOG.info("LLM research model and client initialized.")
        except Exception as e:
            LOG.exception("Failed to initialize research LLM or client: %s", e)
    else:
        LOG.warning("GOOGLE_API_KEY not found. Generative calls will fail.")

    # 3. Initialize Google embedding service (with dimension reduction for Pinecone compatibility)
    LOG.info("Initializing Google embedding service...")
    load_google_embeddings()
    LOG.info("Google embedding service initialized successfully.")

def get_data_status():
    """Returns the row counts of the loaded dataframes for the health check."""
    return len(df_content), len(df_prompt)

# -------- Core Service for /generate_full --------
async def handle_generate_full(req: models.GenerateRequest, background_tasks: BackgroundTasks, version_id: Optional[str] = None):
    """
    Service logic with a self-healing AND self-patching cache migration strategy.
    """
    loop = asyncio.get_event_loop()
    cache_key = create_cache_key(req.board, req.class_label, req.subject)

    # --- CACHE HIT LOGIC ---
    latest_version = None
    if not version_id:
        latest_version = get_latest_cache_version(cache_key)
    else:
        latest_version = get_cache_version_by_id(cache_key, version_id)

    if latest_version:
        try:
            # Attempt to validate the cached data directly
            validated_response = models.PaperResponse(**latest_version)
            log.info("Serving latest VALID cached version %s for %s", latest_version.get("version_id"), cache_key)
            if not version_id:
                background_tasks.add_task(_generate_and_cache_background, req)
            return validated_response
        except ValidationError as e:
            # --- SELF-PATCHING MIGRATION LOGIC ---
            log.warning(f"Cache data for key {cache_key} is invalid. Attempting in-place migration.")
            
            migrated_version = deepcopy(latest_version) # Work on a copy
            is_migrated = False

            # Check if the specific error we expect is present
            if 'value' in migrated_version and isinstance(migrated_version['value'], dict):
                paper_data = migrated_version['value']
                # THE CORE MIGRATION: Check for 'class' and rename it to 'class_label'
                if 'class' in paper_data and 'class_label' not in paper_data:
                    paper_data['class_label'] = paper_data.pop('class')
                    is_migrated = True
                    log.info(f"Successfully migrated 'class' to 'class_label' for key {cache_key}.")

            if is_migrated:
                try:
                    # Retry validation with the migrated data
                    validated_response = models.PaperResponse(**migrated_version)
                    log.info(f"Migration successful. Serving patched version for key {cache_key}.")
                    
                    # IMPORTANT: Save the fixed version back to the cache.
                    add_cache_version(cache_key, migrated_version['value'])
                    log.info(f"Saved newly patched version back to cache for key {cache_key}.")

                    if not version_id:
                        background_tasks.add_task(_generate_and_cache_background, req)
                    
                    return validated_response
                except ValidationError as final_e:
                    log.error(f"Migration attempt failed validation for key {cache_key}. Falling back. Error: {final_e}")
            else:
                 log.warning(f"Could not automatically migrate cache for key {cache_key}. Falling back.")
            
            # If we reach here, migration failed, so we fall through to the cache miss logic.

    # --- CACHE MISS LOGIC ---
    log.info("Cache miss. Running synchronous generation for %s", cache_key)
    try:
        generated_paper_object = await _run_full_generation_pipeline(req)
        json_safe_paper = utils.sanitize_for_json(generated_paper_object)
        new_version_id = add_cache_version(cache_key, json_safe_paper)
        log.info("Synchronous generation saved as version %s for %s", new_version_id, cache_key)

        response_object = {
            "version_id": new_version_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "value": generated_paper_object
        }
        
        return response_object

    except Exception as e:
        log.error(f"Failed during synchronous generation: {e}", exc_info=True)
        raw_output = getattr(e, 'raw_output', '[Could not get raw text]')
        raise GenerationError(f"Failed to generate paper: {e}", raw_output=raw_output) from e

def _generate_and_cache_background(req: models.GenerateRequest):
    """
    Self-contained background task to run the generation pipeline and update the cache.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    log.info(f"BACKGROUND TASK: Starting generation for {req.board} {req.class_label} {req.subject}...")
    try:
        # 1. Generate the raw, Pydantic-compatible paper object
        final_paper_object = loop.run_until_complete(_run_full_generation_pipeline(req))
        
        # 2. Sanitize it for storage in Redis
        json_safe_paper = utils.sanitize_for_json(final_paper_object)
        cache_key = create_cache_key(req.board, req.class_label, req.subject)
        
        version_id = add_cache_version(cache_key, json_safe_paper)
        log.info("BACKGROUND TASK: Successfully updated cache key %s with new version %s", cache_key, version_id)
    except Exception as e:
        log.error(f"BACKGROUND TASK FAILED for {req.subject}. Error: {e}", exc_info=True)
    finally:
        loop.close()

async def _run_full_generation_pipeline(req: models.GenerateRequest) -> Dict[str, Any]:
    """
    The core, reusable logic for generating a paper from scratch.
    """
    loop = asyncio.get_event_loop()

    # 1. Load Schema
    row = await loop.run_in_executor(
        _executor,
        lambda: load_schema_row(
            config.INPUT_CSV_PATH, req.board, req.class_label, req.subject
        ),
    )
    if not row:
        raise SchemaNotFoundError(
            f"Schema not found for {req.board}, {req.class_label}, {req.subject}"
        )

    # 2. Derive Plan
    file_data = row.get("File_Data", "")
    plan = derive_plan_from_filedata(file_data, req.subject)

    # 3. Process Sections Concurrently
    section_tasks = [
        loop.run_in_executor(
            _executor,
            _process_section_sync,
            sec,
            file_data,
            req.class_label,
            req.subject,
        )
        for sec in plan["sections"]
    ]
    sections_results = await asyncio.gather(*section_tasks)

    # 4. Build Prompt
    slot_summaries = [
        {
            "slot_id": r["section_id"],
            "slot_meta": r.get("slot_meta", ""),
            "summaries": r.get("summaries", []),
        }
        for r in sections_results
    ]
    planner_text = plan.get("planner_text", "Generate a standard exam paper.")
    prompt = build_generator_prompt_questions_only(planner_text, slot_summaries, plan)

    # 5. Generate Paper using Structured Output or Legacy JSON Parsing
    if config.PAPER_GENERATION_METHOD == "structured":
        try:
            log.info("Using structured output for paper generation")
            # Use structured generation with Pydantic validation
            structured_paper = await loop.run_in_executor(
                _executor,
                lambda: generate_paper_with_structured_output(
                    planner_text, slot_summaries, req.board, req.class_label, req.subject, config.GOOGLE_API_KEY
                )
            )
            
            # Structured output returns a dict compatible with our final_paper format
            final_paper = {
                "paper_id": structured_paper.get("paper_id", "unknown-id"),
                "board": req.board,
                "class_label": req.class_label,
                "subject": req.subject,
                "total_marks": plan.get("total_marks"),
                "time_allowed_minutes": plan.get("time_minutes"),
                "general_instructions": plan.get("general_instructions"),
                "questions": structured_paper.get("questions", []),
                "retrieval_metadata": sections_results,
            }
            log.info(f"Structured output successfully generated {len(final_paper['questions'])} questions")
            
        except Exception as structured_error:
            log.error(f"Structured generation failed: {structured_error}")
            if not config.ENABLE_STRUCTURED_FALLBACK:
                raise GenerationError(f"Structured generation failed: {structured_error}") from structured_error
            
            log.info("Falling back to legacy JSON parsing method")
            # Fall back to legacy JSON parsing
            gen_resp = None
            try:
                gen_resp = await loop.run_in_executor(
                    _executor,
                    lambda: call_gemini(prompt, model_name="models/gemini-2.5-flash-lite"),
                )
                raw_text = gen_resp.get("text", "")
                parsed_llm_json = parse_generator_response(raw_text, call_gemini)
                
                # 6. Clean and Finalize (Legacy)
                cleaned_questions = utils._post_process_and_clean_questions(
                    parsed_llm_json.get("questions", [])
                )
                final_paper = {
                    "paper_id": parsed_llm_json.get("paper_id", "unknown-id"),
                    "board": req.board,
                    "class_label": req.class_label,
                    "subject": req.subject,
                    "total_marks": plan.get("total_marks"),
                    "time_allowed_minutes": plan.get("time_minutes"),
                    "general_instructions": plan.get("general_instructions"),
                    "questions": cleaned_questions,
                    "retrieval_metadata": sections_results,
                }
                log.info("Fallback to JSON parsing succeeded")
            except Exception as fallback_error:
                raw_output = gen_resp.get("text", "") if gen_resp else "[LLM call failed]"
                raise GenerationError(
                    f"Both structured and JSON parsing failed. Structured error: {structured_error}. Fallback error: {fallback_error}", 
                    raw_output=raw_output
                ) from fallback_error
    else:
        # Legacy JSON parsing method
        log.info("Using legacy JSON parsing for paper generation")
        gen_resp = None
        try:
            gen_resp = await loop.run_in_executor(
                _executor,
                lambda: call_gemini(prompt, model_name="models/gemini-2.5-flash-lite"),
            )
            raw_text = gen_resp.get("text", "")
            parsed_llm_json = parse_generator_response(raw_text, call_gemini)
        except Exception as e:
            raw_output = gen_resp.get("text", "") if gen_resp else "[LLM call failed]"
            raise GenerationError(
                f"LLM call or parsing failed: {e}", raw_output=raw_output
            ) from e

        # 6. Clean and Finalize (Legacy)
        cleaned_questions = utils._post_process_and_clean_questions(
            parsed_llm_json.get("questions", [])
        )
        final_paper = {
            "paper_id": parsed_llm_json.get("paper_id", "unknown-id"),
            "board": req.board,
            "class_label": req.class_label,
            "subject": req.subject,
            "total_marks": plan.get("total_marks"),
            "time_allowed_minutes": plan.get("time_minutes"),
            "general_instructions": plan.get("general_instructions"),
            "questions": cleaned_questions,
            "retrieval_metadata": sections_results,
        }
    return final_paper

def _process_section_sync(
    sec: Dict[str, Any], file_data: str, class_label: str, subject: str
) -> Dict[str, Any]:
    """
    Synchronous function to process a single section of the paper plan. Intended to be run in the executor.
    """
    try:
        objective = build_retrieval_objective(
            sec, subject_guidelines=file_data, user_mode="balanced"
        )
        specific_objective = (
            f"For {class_label} {subject}, find content for: {objective}"
        )
        LOG.info(f"Running retrieval for Section {sec.get('section_id')}...")

        class_val = "".join(re.findall(r"\d+", class_label))
        subject_val = utils.standardize_subject_name(subject)
        filters = {"class": {"$eq": class_val}, "subject": {"$eq": subject_val}}

        candidates = retrieve_from_pinecone(specific_objective, filters, top_k=20)
        if not candidates:
            LOG.warning(f"No candidates found for Section {sec.get('section_id')}")
            return {
                "section_id": sec.get("section_id"),
                "summaries": [{"id": "N/A", "summary": "No relevant evidence found."}],
                "slot_meta": sec.get("title", ""),
            }

        valid_candidates = [c for c in candidates if c.get("embedding") is not None]
        if not valid_candidates:
            LOG.warning(
                f"Candidates for Section {sec.get('section_id')} had no embeddings."
            )
            return {
                "section_id": sec.get("section_id"),
                "summaries": [{"id": "N/A", "summary": "No valid evidence processed."}],
                "slot_meta": sec.get("title", ""),
            }

        emb_matrix = np.vstack([c["embedding"] for c in valid_candidates])
        query_emb = embed_query_google(specific_objective)
        desired_snippets = min(8, max(5, int(sec.get("num_questions", 5)) + 2))

        picks_indices = mmr_and_stratified_sample(
            query_emb,
            emb_matrix,
            [{"id": c.get("snippet_id")} for c in valid_candidates],
            metadata=[c.get("metadata", {}) for c in valid_candidates],
            n_samples=desired_snippets,
        )
        selected_snips = [valid_candidates[i] for i in picks_indices]

        dense_summary = generate_dense_evidence_summary_with_llm(
            selected_snips, specific_objective, call_gemini
        )
        final_summaries = [
            {
                "id": ",".join([str(s.get("snippet_id", "")) for s in selected_snips]),
                "summary": dense_summary,
            }
        ]

        return {
            "section_id": sec.get("section_id"),
            "summaries": final_summaries,
            "slot_meta": sec.get("title", ""),
        }
    except Exception as e:
        LOG.error(
            f"FATAL ERROR in _process_section_sync for section {sec.get('section_id')}: {e}",
            exc_info=True,
        )
        return {
            "section_id": sec.get("section_id"),
            "summaries": [{"id": "ERROR", "summary": "An error occurred."}],
            "slot_meta": sec.get("title", ""),
        }
