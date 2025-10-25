from fastapi import FastAPI
from pydantic import BaseModel
import joblib, torch
import numpy as np

app = FastAPI()
models = {}


def load_models():
