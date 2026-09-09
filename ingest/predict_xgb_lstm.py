from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json, joblib, pathlib, torch
"""
To test a local curl request against the legacy compatibility endpoint:
uvicorn predict_xgb_lstm:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:3000/<compatibility-prediction-route> \
  -H 'Content-Type: application/json' \
  -d '{
        "features": {
          "cpu_usage_percent": 12.3,
          "memory_used_bytes": 8.2e9,
          "network_bw_rx_b/s": 155000,
          "lag_1h": 320.5,
          "lag_2h": 315.1,
          "lag_3h": 318.9,
          "lag_6h": 310.2
        }
      }'
"""

ROOT = pathlib.Path(__file__).resolve().parents[1]

# --- Champion loader ---
class SklearnPredictor:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.features = bundle.get("feature_names", [])
    def predict(self, row: Dict[str, Any]) -> float:
        import pandas as pd
        X = pd.DataFrame([row])
        # pad known features
        for c in self.features:
            if c not in X: X[c] = 0.0
        X = X[self.features] if self.features else X
        return float(self.model.predict(X)[0])

class LSTMPredictor:
    def __init__(self, model_path: pathlib.Path):
        ckpt = torch.load(model_path, map_location="cpu")
        from scripts.train_lstm import LSTMReg
        self.model = LSTMReg(ckpt["in_features"], 128, 2, 0.1)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
    def predict(self, window_2d: List[List[float]]) -> float:
        x = torch.tensor([window_2d], dtype=torch.float32)
        with torch.no_grad():
            y = self.model(x).cpu().numpy()[0]
        return float(y)

def load_champion():
    meta = json.loads((ROOT/"models"/"champion.json").read_text())
    mtype, mpath = meta["model_type"], ROOT / meta["model_path"]
    if mtype in ("xgboost", "sklearn"):
        return mtype, SklearnPredictor(joblib.load(mpath))
    elif mtype == "lstm":
        return mtype, LSTMPredictor(mpath)
    raise RuntimeError(f"Unsupported model_type: {mtype}")

MODEL_TYPE, PREDICTOR = load_champion()

# --- API schema ---
class TabularRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Flat feature dict")

class SequenceRequest(BaseModel):
    window: List[List[float]] = Field(..., description="[timesteps][features] window")

app = FastAPI(title="ECoMEP Inference API", version="1.0")

@app.get("/health")
def health(): return {"status": "ok", "model_type": MODEL_TYPE}

@app.get("/model")
def model_info():
    meta = json.loads((ROOT/"models"/"champion.json").read_text())
    return meta

@app.post("/predict", include_in_schema=False)
def predict(payload: Dict[str, Any]):
    if MODEL_TYPE in ("xgboost","sklearn"):
        if "features" not in payload:
            raise HTTPException(400, "Expected {'features': {...}} for tabular model.")
        return {"power_forecast": PREDICTOR.predict(payload["features"])}
    elif MODEL_TYPE == "lstm":
        if "window" not in payload:
            raise HTTPException(400, "Expected {'window': [[...],[...],...]} for LSTM model.")
        return {"power_forecast": PREDICTOR.predict(payload["window"])}
