"""
Structured Paper Generation Module

This module provides a simpler alternative to DSPy that still guarantees structured outputs
by using Pydantic schemas with guided generation and validation.
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import google.generativeai as genai

LOG = logging.getLogger("uvicorn.error")

# =============================================================================
# 1. Pydantic Output Schemas (Same as DSPy version)
# =============================================================================

class QuestionChoice(BaseModel):
    """Represents a single choice in a multiple choice or internal choice question."""
    q_text: str = Field(..., description="The question text for this choice")
    sources: Optional[List[str]] = Field(default=[], description="Source IDs used for this choice")

class Question(BaseModel):
    """Represents a single question in the paper."""
    section_id: str = Field(..., description="Section identifier (e.g., 'A', 'B', 'C')")
    q_id: str = Field(..., description="Unique question identifier (e.g., 'A.1', 'B.2')")
    q_text: Union[str, List[QuestionChoice], Dict[str, Any]] = Field(
        ..., 
        description="Question text - can be string, list of choices, or case study object"
    )
    type: str = Field(..., description="Question type (SA, LA, MCQ, Case-Study, etc.)")
    marks: float = Field(..., description="Marks allocated to this question")
    difficulty: Optional[str] = Field(default="Medium", description="Difficulty level")
    is_choice: Optional[bool] = Field(default=False, description="Whether this question has internal choices")
    sources: Optional[List[str]] = Field(default=[], description="Source IDs used for this question")

    @validator('q_text', pre=True)
    def validate_q_text_structure(cls, v, values):
        """Ensure q_text structure matches is_choice flag."""
        is_choice = values.get('is_choice', False)
        if is_choice and isinstance(v, str):
            # Convert string to choice format if is_choice is True
            return [QuestionChoice(q_text=v, sources=[])]
        return v

class PaperStructure(BaseModel):
    """The complete paper structure that will be generated."""
    paper_id: str = Field(..., description="Unique identifier for the paper")
    board: str = Field(..., description="Educational board (e.g., CBSE, ICSE)")
    class_label: str = Field(..., description="Class/grade level")
    subject: str = Field(..., description="Subject name")
    questions: List[Question] = Field(..., description="List of all questions in the paper")

    @validator('questions')
    def validate_questions_not_empty(cls, v):
        """Ensure at least one question is generated."""
        if not v:
            raise ValueError("Paper must contain at least one question")
        return v

# =============================================================================
# 2. Structured Generation Service
# =============================================================================

class StructuredPaperService:
    """Service class that generates structured papers using guided prompting and validation."""
    
    def __init__(self, google_api_key: str):
        """
        Initialize the structured paper service.
        
        Args:
            google_api_key: Google API key for Gemini access
        """
        self.google_api_key = google_api_key
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Configure Gemini for structured generation."""
        try:
            genai.configure(api_key=self.google_api_key)
            # Use the more powerful Gemini Pro model for better structured output
            self.model = genai.GenerativeModel("gemini-2.5-pro")
            LOG.info("Structured paper service configured with Gemini Pro")
        except Exception as e:
            LOG.error(f"Failed to setup Gemini: {e}")
            raise
    
    def generate_paper(self, planner_summary: str, slot_summaries: List[Dict[str, Any]], 
                      board: str, class_label: str, subject: str) -> Dict[str, Any]:
        """
        Generate a paper using structured prompting and Pydantic validation.
        
        Args:
            planner_summary: Summary of the paper plan
            slot_summaries: List of evidence summaries per section
            board: Educational board
            class_label: Class/grade level
            subject: Subject name
            
        Returns:
            Dict: Paper data compatible with existing models.PaperModel
        """
        try:
            # Create structured prompt
            prompt = self._build_structured_prompt(
                planner_summary, slot_summaries, board, class_label, subject
            )
            
            # Generate with retries and validation
            paper_structure = self._generate_with_validation(prompt, board, class_label, subject)
            
            # Convert to legacy format
            paper_dict = self._convert_to_legacy_format(paper_structure, slot_summaries)
            
            LOG.info(f"Successfully generated structured paper with {len(paper_dict['questions'])} questions")
            return paper_dict
            
        except Exception as e:
            LOG.error(f"Structured paper generation failed: {e}")
            raise
    
    def _build_structured_prompt(self, planner_summary: str, slot_summaries: List[Dict[str, Any]], 
                                board: str, class_label: str, subject: str) -> str:
        """Build a structured prompt that guides the LLM to generate valid JSON."""
        
        # Format evidence summaries
        evidence_text = self._format_evidence_summaries(slot_summaries)
        
        # Create the JSON schema as a string for the prompt with LaTeX examples
        schema_example = {
            "paper_id": f"{subject.upper()}_{class_label.replace(' ', '_').upper()}_EXAM_2024",
            "board": board,
            "class_label": class_label,
            "subject": subject,
            "questions": [
                {
                    "section_id": "A",
                    "q_id": "A.1",
                    "q_text": "What is photosynthesis?",
                    "type": "SA",
                    "marks": 2.0,
                    "difficulty": "Easy",
                    "is_choice": False,
                    "sources": ["source_001"]
                },
                {
                    "section_id": "A",
                    "q_id": "A.2",
                    "q_text": "Find the derivative of $\\\\sin(x) + \\\\cos(x)$.",
                    "type": "SA",
                    "marks": 3.0,
                    "difficulty": "Medium",
                    "is_choice": False,
                    "sources": ["math_001"]
                }
            ]
        }
        
        prompt = f"""You are an expert exam paper generator. Generate a structured exam paper in VALID JSON format.

PLANNER SUMMARY:
{planner_summary}

EVIDENCE SUMMARIES:
{evidence_text}

CRITICAL JSON ESCAPING RULES:
1. ALL backslashes in LaTeX formulas MUST be doubled: \\ becomes \\\\
2. Examples of CORRECT escaping:
   - "\\\\sin(x)" not "\\sin(x)"
   - "\\\\frac{{a}}{{b}}" not "\\frac{{a}}{{b}}"
   - "\\\\int_0^1" not "\\int_0^1"
   - "\\\\begin{{bmatrix}}" not "\\begin{{bmatrix}}"
3. Every single backslash in mathematical expressions needs to be escaped

REQUIREMENTS:
1. Generate questions based on the planner summary and evidence provided
2. Use the evidence summaries to create relevant, curriculum-aligned questions
3. Follow the exact JSON schema structure shown below
4. Each question must have all required fields: section_id, q_id, q_text, type, marks
5. Optional fields: difficulty, is_choice, sources

JSON SCHEMA EXAMPLE:
{json.dumps(schema_example, indent=2)}

MATHEMATICAL FORMULA EXAMPLES (CORRECT JSON ESCAPING):
- "The derivative of $\\\\sin(x)$ is $\\\\cos(x)$"
- "Evaluate $\\\\int_0^{{\\\\pi}} \\\\sin(x) dx$"
- "If $A = \\\\begin{{bmatrix}} 1 & 2 \\\\\\\\ 3 & 4 \\\\end{{bmatrix}}$"
- "The limit $\\\\lim_{{x \\\\to 0}} \\\\frac{{\\\\sin(x)}}{{x}} = 1$"

VALIDATION CHECKLIST:
✓ Every backslash in LaTeX is doubled (\\\\)
✓ All string values are properly quoted
✓ No trailing commas in JSON objects or arrays
✓ All braces and brackets are properly matched
✓ No comments or explanations outside JSON

CRITICAL: Before generating, remember that in JSON strings, every single backslash character must be escaped as double backslash (\\\\).

Generate the exam paper JSON:"""

        return prompt
    
    def _generate_with_validation(self, prompt: str, board: str, class_label: str, 
                                 subject: str, max_retries: int = 3) -> PaperStructure:
        """Generate paper with validation and retries."""
        
        for attempt in range(max_retries):
            try:
                LOG.info(f"Generating structured paper (attempt {attempt + 1}/{max_retries})")
                
                # Call Gemini
                response = self.model.generate_content(prompt)
                raw_text = response.text.strip()
                
                # Extract JSON from response
                json_text = self._extract_json(raw_text)
                
                # Parse JSON
                paper_data = json.loads(json_text)
                
                # Validate with Pydantic
                paper_structure = PaperStructure(**paper_data)
                
                LOG.info("Successfully generated and validated structured paper")
                return paper_structure
                
            except json.JSONDecodeError as e:
                LOG.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Add specific guidance based on error type
                    error_msg = str(e)
                    if "Invalid \\escape" in error_msg:
                        feedback = f"\n\nPREVIOUS ATTEMPT FAILED: LaTeX escaping error - {e}. CRITICAL: You MUST double ALL backslashes in mathematical formulas. Every \\ must become \\\\. Examples: \\\\sin, \\\\frac, \\\\int, \\\\begin, etc."
                    else:
                        feedback = f"\n\nPREVIOUS ATTEMPT FAILED: JSON syntax error - {e}. Check for missing commas, unmatched brackets, or invalid characters."
                    
                    prompt += feedback
                    continue
                else:
                    raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts: {e}")
            
            except Exception as e:
                LOG.warning(f"Generation failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError(f"Failed to generate valid paper after {max_retries} attempts: {e}")
        
        raise ValueError("Exhausted all retry attempts")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response text."""
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        
        return text[start:end]
    
    def _format_evidence_summaries(self, slot_summaries: List[Dict[str, Any]]) -> str:
        """Format slot summaries into a text format for prompt input."""
        evidence_parts = []
        
        for slot in slot_summaries:
            slot_id = slot.get("slot_id", "UNKNOWN")
            slot_meta = slot.get("slot_meta", "")
            
            evidence_parts.append(f"Section {slot_id}: {slot_meta}")
            
            for summary in slot.get("summaries", []):
                summary_text = summary.get("summary", "")[:600]  # Limit length
                source_id = summary.get("id", "")
                evidence_parts.append(f"- [{source_id}] {summary_text}")
        
        return "\n".join(evidence_parts)
    
    def _convert_to_legacy_format(self, paper_structure: PaperStructure, 
                                 slot_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert PaperStructure to format compatible with existing models.PaperModel.
        
        Args:
            paper_structure: Generated paper structure
            slot_summaries: Original slot summaries for metadata
            
        Returns:
            Dict compatible with models.PaperModel
        """
        # Convert questions to legacy format
        legacy_questions = []
        for question in paper_structure.questions:
            legacy_q = {
                "q_id": question.q_id,
                "q_text": question.q_text,
                "marks": question.marks,
                "type": question.type,
            }
            
            # Add optional fields if present
            if question.is_choice:
                legacy_q["is_choice"] = question.is_choice
            if question.difficulty:
                legacy_q["difficulty"] = question.difficulty
            if question.sources:
                legacy_q["sources"] = question.sources
            
            legacy_questions.append(legacy_q)
        
        # Convert slot summaries to retrieval metadata format
        retrieval_metadata = []
        for slot in slot_summaries:
            metadata = {
                "section_id": slot.get("slot_id", ""),
                "summaries": slot.get("summaries", []),
                "slot_meta": slot.get("slot_meta", "")
            }
            retrieval_metadata.append(metadata)
        
        return {
            "paper_id": paper_structure.paper_id,
            "board": paper_structure.board,
            "class_label": paper_structure.class_label,
            "subject": paper_structure.subject,
            "total_marks": None,  # Will be set by calling code
            "time_allowed_minutes": None,  # Will be set by calling code
            "general_instructions": None,  # Will be set by calling code
            "questions": legacy_questions,
            "retrieval_metadata": retrieval_metadata,
        }

# =============================================================================
# 3. Global Service Instance
# =============================================================================

_structured_service: Optional[StructuredPaperService] = None

def get_structured_service(google_api_key: str) -> StructuredPaperService:
    """Get or create the global structured service instance."""
    global _structured_service
    if _structured_service is None:
        _structured_service = StructuredPaperService(google_api_key)
    return _structured_service

def generate_paper_with_structured_output(planner_summary: str, slot_summaries: List[Dict[str, Any]], 
                                        board: str, class_label: str, subject: str,
                                        google_api_key: str) -> Dict[str, Any]:
    """
    Convenience function to generate a paper using structured output.
    
    This function can be used as a drop-in replacement for the existing
    JSON parsing approach in the paper generation pipeline.
    """
    service = get_structured_service(google_api_key)
    return service.generate_paper(planner_summary, slot_summaries, board, class_label, subject)
