import os
import threading

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from models import models as model_classes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS: read allowed origins from env, default to ["*"] for dev
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded model cache
loaded_models = {}
_model_locks = {}
_global_lock = threading.Lock()


def get_model(model_id: str):
    """Load a model on first request, then cache it."""
    if model_id in loaded_models:
        return loaded_models[model_id]

    if model_id not in model_classes:
        return None

    # Per-model lock to avoid duplicate loading
    with _global_lock:
        if model_id not in _model_locks:
            _model_locks[model_id] = threading.Lock()

    with _model_locks[model_id]:
        if model_id in loaded_models:
            return loaded_models[model_id]
        print(f"Loading model: {model_id}")
        loaded_models[model_id] = model_classes[model_id]()
        return loaded_models[model_id]


class PredictRequest(BaseModel):
    pixels: list
    model_id: str


@app.post('/predict')
async def predict(req: PredictRequest):
    try:
        model = get_model(req.model_id)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model_id}' not found"
            )
        if not req.pixels or len(req.pixels) == 0:
            raise HTTPException(status_code=400, detail="Pixels array is empty")

        X = np.array(req.pixels, dtype=np.float32)

        if req.model_id.startswith('torch_CNN'):
            if len(req.pixels) == 784:
                X = X.reshape(28, 28)
            if X.max() > 1.0:
                X = X / 255.0
        else:
            X = X.reshape(1, -1)
            if X.max() > 1.0:
                X = X / 255.0

        y_pred = model.predict(X)
        return {
            "prediction": int(y_pred[0]),
            "model_used": req.model_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get('/models')
async def list_models():
    return {
        "available_models": list(model_classes.keys()),
        "total": len(model_classes)
    }


@app.get('/health')
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "models_available": len(model_classes)
    }


# --- Static file serving for the React SPA ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(STATIC_DIR):
    # Serve static assets (js, css, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for any non-API route (SPA routing)."""
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))