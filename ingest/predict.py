#!/usr/bin/env python3
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

model = joblib.load(Path("models/baseline.joblib"))
ALL_COLS = list(model.feature_names_in_)  # full column set

class Features(BaseModel):
    lag_1h: float = Field(..., example=0.42)
    lag_2h: float = Field(..., example=0.38)
    lag_3h: float = Field(..., example=0.35)
    lag_6h: float = Field(..., example=0.30)

app = FastAPI()

# @app.post("/predict")
# async def predict(f: Features):
#     data = {c: None for c in ALL_COLS}   # start with full schema
#     data.update(f.model_dump())          # overwrite supplied lags
#     df = pd.DataFrame([data], columns=ALL_COLS)
#     y_hat = model.predict(df)[0]
#     return {"power_forecast": float(y_hat)}

@app.post("/predict", include_in_schema=False)
def predict(payload: dict):
    """
    If champion is tree/GBM: send a feature dict as before.
    If champion is LSTM: send {"window": [[...],[...],...]} (timesteps x features).
    """
    if MODEL_TYPE in ("sklearn","xgboost"):
        return {"power_forecast": PREDICTOR.predict(payload)}
    elif MODEL_TYPE == "lstm":
        if "window" not in payload:
            raise HTTPException(status_code=400, detail="LSTM champion requires 'window' field: [[timestep features] ...].")
        return {"power_forecast": PREDICTOR.predict(payload["window"])}
