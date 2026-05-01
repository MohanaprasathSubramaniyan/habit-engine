"""
Habit Formation Prediction & Intervention Engine
FastAPI Backend — Production-Grade REST API
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import os
from contextlib import asynccontextmanager

from backend.app.api.routes import router
from backend.app.core.config import settings

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("habit-engine")


# ── Lifespan (startup/shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Habit Formation Engine starting up...")
    # Warm up models on startup
    try:
        from backend.app.services.ml_service import MLService
        ml = MLService()
        ml.load_models()
        app.state.ml_service = ml
        logger.info("✅ ML models loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  ML models not found, run training first: {e}")
        app.state.ml_service = None
    yield
    logger.info("👋 Shutting down...")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Habit Formation Prediction Engine",
    description="""
    ## Predictive engine to identify habit-forming milestones in users' first 30 days.
    
    ### Features
    - **Clustering**: Segments users by fitness routine consistency (KMeans)
    - **Churn Prediction**: Ensemble model (RF + GBM + LR) with calibrated probabilities
    - **Habit Formation Score**: Composite metric from engagement + consistency signals
    - **LLM Recommendations**: Personalized intervention suggestions
    - **Real-time Inference**: Sub-50ms predictions for live users
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Request timing middleware ──────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration)
    return response


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": app.state.ml_service is not None,
    }


# ── Mount routers ──────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )