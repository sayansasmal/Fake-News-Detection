from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load('model.joblib')

class NewsInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "News Classification API Running"}

@app.post("/predict")
def predict(data: NewsInput):
    prediction = model.predict([data.text])[0]
    prediction = int(prediction)

    confidence = model.predict_proba([data.text]).max()

    label_map = {0: "Fake", 1: "Real"}

    return {
        "prediction": label_map.get(prediction, prediction),
        "confidence": round(float(confidence) * 100, 2)
    }