from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import json
from typing import List 
app = FastAPI()

model = tf.keras.models.load_model("model.h5")

with open("labels.json") as f:
    labels = json.load(f)

idx_to_label = {v: k for k, v in labels.items()}

class InputData(BaseModel):
    sequence: List[List[float]]  # List of FRAMES_COUNT frames, each with 126 values
                                  # (2 hand slots x 21 landmarks x [x,y,z]).
                                  # Hand slot with no detected hand = all zeros.

@app.post("/predict")
def predict(data: InputData):   
    x= np.array(data.sequence)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    pred = model.predict(x)
    idx = int(np.argmax(pred))
    return {"predicted_label": idx_to_label[idx], "confidence": float(pred[0][idx])}

@app.get("/test")
def test():
    x= np.load("dataset/please/001.npy")
    pred = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    idx = int(np.argmax(pred))
    return {"predicted_label": idx_to_label[idx], "confidence": float(pred[idx])}