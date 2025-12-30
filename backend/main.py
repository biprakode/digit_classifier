from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import joblib, torch
import numpy as np
from models import models as model_classes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

loaded_models = {}

class PredictRequest(BaseModel):
    pixels: list
    model_id: str

@app.on_event('startup')
async def load_all():
    global loaded_models
    try:
        for model_name , model_class in model_classes.items():
            try:
                loaded_models[model_name] = model_class()
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    except Exception as e:
        print(f"Error during model loading: {e}")

@app.post('/predict')
async def predict(req: PredictRequest):
    try:
        if req.model_id not in loaded_models:
            raise HTTPException(
                status_code = 404,
                detail=f"Model '{req.model_id}' not found"
            )
        if not req.pixels or len(req.pixels) == 0:
            raise HTTPException(status_code=400, detail="Pixels array is empty")
        X = np.array(req.pixels , dtype=np.float32)
        if req.model_id.startswith('torch_CNN'):
            if len(req.pixels) == 784:  # Flattened 28x28
                X = X.reshape(28, 28)
            
            if X.max() <= 1.0: # already normalized
                pass 
            else:
                # Scale down if needed
                X = X / 255.0
        else:
            X = X.reshape(1, -1)
            if X.max() > 1.0:
                X = X / 255.0
        
        model = loaded_models[req.model_id]
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
        "available_models": list(loaded_models.keys()),
        "total": len(loaded_models)
    }

@app.get('/health')
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models)
    }