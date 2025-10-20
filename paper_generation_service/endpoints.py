# Paper Generation Service - endpoints.py
import json
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    status,
)
from pydantic import ValidationError

import models
import services
from logger import log

# -------- API Router Setup --------
router = APIRouter()

# -------- Paper Generation Endpoints --------
@router.post(
    "/generate_full", response_model=models.PaperResponse, tags=["Paper Generation"]
)
async def generate_full_paper(
    req: models.GenerateRequest,
    background_tasks: BackgroundTasks,
    version_id: Optional[str] = None,
):
    """
    Orchestrates the full RAG pipeline with a Stale-While-Revalidate cache.
    - Immediately returns a cached version if available.
    - Triggers a background task to refresh the cache with a new version.
    - If no cache exists, performs a synchronous generation, caches the result, and returns it.
    """
    try:
        # Delegate the entire complex logic to the service layer
        paper_response = await services.handle_generate_full(
            req=req, background_tasks=background_tasks, version_id=version_id
        )
        return paper_response
    except services.SchemaNotFoundError as e:
        log.error(f"Schema not found for request: {req.dict()}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        log.exception(
            f"Unhandled error in /generate_full endpoint for request: {req.dict()}"
        )
        # The original code provided a raw LLM output on failure, which is good for debugging.
        detail = f"An internal error occurred during paper generation: {e}"
        if hasattr(e, "raw_output"):
            detail += f". Raw LLM Output: {e.raw_output}"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
        )
