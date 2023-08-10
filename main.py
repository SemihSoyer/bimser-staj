import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import json
from sklearn.linear_model import LinearRegression


class LinReg(BaseModel):
    val1: int
    val2: int


app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/health_check")
async def root():
    return {"status": "ok"}

@app.get("/retrain")
async def retrain_model():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3

    # model
    reg = LinearRegression().fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(reg, f)
        print("dosya yazıldı!")
    return {"status": "ok"}

@app.post("/predict")
async def lin_reg_predictor(inputs: LinReg):
    values = json.loads(inputs.json())
    val1 = values["val1"]
    val2 = values["val2"]
    result = model.predict(np.array([[val1, val2]]))
    return {"result": result[0]}
