from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import joblib, torch
import numpy as np
from models import models

app = FastAPI()
models = {}

class PredictRequest(BaseModel):
    pixels: list
    model_id: str

@app.post('/predict')
async def predict(req: PredictRequest):
    X = np.array(req.pixels).reshape(1, -1) / 255.0
    model = models[req.model_id]

    if(callable): # if a func -> (custom)
        y_pred = model(X)
    else:
        y_pred = model.predict(X)

    return y_pred