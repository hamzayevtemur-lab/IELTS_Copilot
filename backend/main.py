import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.routers.autocomplete import router as autocomplete_router
from backend.services.model_service import ModelService



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager.
    Initializes the model once at server startup and cleans up on shutdown.
    """

    print("\n" + "=" * 60)
    print("🚀 Initializing IELTS Writing Copilot API...")
    print("=" * 60)

    # Load model into RAM on server startup
    service = ModelService.get_instance()
    service.initialize()

    print("\n Backend is ready to accept requests on http://0.0.0.0:8000")
    print(" Interactive Swagger Docs available at http://localhost:8000/docs\n")

    yield

    print("\n🛑 Shutting down IELTS Writing Copilot API...")



# Create FastAPI application
app=FastAPI(
    title="IELTS Writing Copilot API",
    description="Real-time context-aware autocomplete backend powered by fine-tuned Qwen2.5-1.5B.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Mount Routers
app.include_router(autocomplete_router)



@app.get("/", tags=["Health"])
async def root():
    """Root endpoint welcoming users and providing documentation link."""
    return {
        "service": "IELTS Writing Copilot API",
        "status": "online",
        "docs": "/docs",
        "model": ModelService.get_instance().model_name,
    }



@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Docker / orchestration."""
    service = ModelService.get_instance()
    return {
        "status": "healthy",
        "model_loaded": service.engine is not None,
        "model_name": service.model_name,
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)