from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predictor_v2 import predict_games
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predictions")
def get_predictions(date: str = None):
    # Using a fixed date for demo if no data for today
    preds = predict_games(date)
    if not preds and not date:
        preds = predict_games("2024-05-11")
    return preds

@app.get("/model-info")
def get_model_info():
    return {
        "features": [
            {"name": "Rolling Win Pct (162/30)", "description": "Measures team momentum and historical baseline."},
            {"name": "OBS (On-Base + Slugging)", "description": "Comprehensive offensive capability metric."},
            {"name": "Modified ERA/WHIP (35/10)", "description": "Pitcher performance smoothed for sample size."}
        ],
        "algorithm": "LightGBM Classifier",
        "training_data": "Retrosheet Game Logs (2015-2025)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
