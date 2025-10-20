# Paper Generation Service - main.py
import os





import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from endpoints import router as api_router
import services
import config
from logger import log

# -------- Logging --------
LOG = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

# -------- App Initialization --------
app = FastAPI(
    title="Paper Generation Service",
    description="Independent FastAPI service for paper generation functionality.",
    version="1.0.0"
)

# -------- Middleware --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Global Executor --------
executor: ThreadPoolExecutor = None

# -------- Application Lifecycle Events --------
@app.on_event("startup")
async def startup_event():
    """
    Handles application startup logic.
    - Initializes the global ThreadPoolExecutor.
    - Triggers the loading of heavy resources (like ML models and CSVs) in the background.
    """
    global executor
    executor = ThreadPoolExecutor(max_workers=config.EXECUTOR_WORKERS)
    LOG.info(f"ThreadPoolExecutor initialized with {config.EXECUTOR_WORKERS} workers.")

    # Delegate the loading of heavy resources to the services module.
    loop = asyncio.get_event_loop()
    
    # Trigger the embedding model to be loaded in the executor to avoid blocking.
    services.set_executor(executor)
    await loop.run_in_executor(executor, services.load_heavy_models_and_data)
    
    LOG.info("Startup complete: Heavy models and data loading has been initiated.")

@app.on_event("shutdown")
def shutdown_event():
    """
    Handles application shutdown logic.
    - Gracefully shuts down the ThreadPoolExecutor.
    """
    global executor
    if executor:
        executor.shutdown(wait=False)
        LOG.info("ThreadPoolExecutor shutdown initiated.")

# -------- API Router Inclusion --------
app.include_router(api_router)

# -------- Health Check Endpoint --------
@app.get("/health", tags=["Monitoring"])
def health() -> Dict[str, Any]:
    """
    A simple health check endpoint to confirm the API is running
    and to get a basic status of loaded data.
    """
    content_rows, prompt_rows = services.get_data_status()
    return {
        "status": "ok",
        "service": "paper_generation",
        "content_rows": content_rows,
        "prompt_rows": prompt_rows,
    }
