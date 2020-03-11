from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictData(BaseModel):
    text: str

@app.post("/api/predict")
def predict(data: PredictData):
    return 'You submitted: {}'.format(data.text)
