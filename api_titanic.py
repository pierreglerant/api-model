import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd

app = FastAPI()


@app.post("/titanic")
def prediction_api(pclass: int, sex: int, age: int) -> bool:
    # Load model
    # predict
    return True


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
