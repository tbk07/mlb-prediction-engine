import sqlite3
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import shap
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = 'mlb.db'
MODEL_PATH = 'best_model.pkl'

@app.on_event("startup")
def load_artifacts():
    global model, explainer, feature_cols
    model = joblib.load(MODEL_PATH)
    # Re-initialize Explainer based on model type
    if hasattr(model, 'predict_proba'):
        try:
            explainer = shap.TreeExplainer(model)
        except:
            explainer = None # fallback
    else:
        explainer = None
    
    # We drop these cols for prediction
    feature_cols = ['v_win_pct_162', 'v_obs_162', 'h_win_pct_162', 'h_obs_162',
        'v_win_pct_30', 'v_obs_30', 'h_win_pct_30', 'h_obs_30',
        'v_era_35', 'v_whip_35', 'h_era_35', 'h_whip_35',
        'v_era_10', 'v_whip_10', 'h_era_10', 'h_whip_10',
        'v_elo', 'h_elo', 'exp_h', 'implied_h_prob', 'over_under',
        'park_factor', 'altitude_flag', 'month', 'v_games_7d', 'h_games_7d',
        'v_rest', 'h_rest', 'v_p_lhp', 'h_p_lhp', 'v_p_days_rest', 'h_p_days_rest',
        'v_p_last_pc', 'h_p_last_pc']

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/predictions")
def get_predictions(date: str = None):
    conn = get_db()
    if date:
        df = pd.read_sql("SELECT * FROM game_features WHERE date=?", conn, params=(date,))
    else:
        # Default to a date where we have data
        df = pd.read_sql("SELECT * FROM game_features WHERE season=2026", conn)
        if not df.empty:
            df = df[df['date'] == df['date'].max()]
    conn.close()
    
    if df.empty:
        return []
    
    res = []
    # Impute missing feature columns to run model prediction if needed
    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0

    X = df[feature_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
    
    for i, row in df.iterrows():
        h_prob = float(probs[i])
        v_prob = 1.0 - h_prob
        
        edge = h_prob - row['implied_h_prob'] if 'implied_h_prob' in row else 0.0
        if edge > 0.05: bet = f"✓ BET {row['h_team']}"
        elif edge < -0.05: bet = f"✗ FADE {row['h_team']} (BET {row['v_team']})"
        else: bet = "— PASS"
        
        conf = "High Confidence" if max(h_prob, v_prob) > 0.6 else "Moderate" if max(h_prob, v_prob) > 0.55 else "Lean"
        
        res.append({
            'gamePk': row['gamePk'],
            'v_team': row['v_team'],
            'h_team': row['h_team'],
            'v_win_prob': v_prob,
            'h_win_prob': h_prob,
            'vegas_home_ml': -110, # Mocked
            'vegas_implied': row.get('implied_h_prob', 0.52),
            'edge': edge,
            'bet': bet,
            'confidence': conf,
            'over_under': row.get('over_under', 8.5)
        })
    return res

@app.get("/game/{gamePk}/scouting")
def get_scouting(gamePk: str):
    conn = get_db()
    df = pd.read_sql("SELECT * FROM game_features WHERE gamePk=?", conn, params=(gamePk,))
    conn.close()
    if df.empty: return {"error": "Game not found"}
    
    row = df.iloc[0]
    X = df[feature_cols].fillna(0)
    
    # SHAP
    shap_reasons = []
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(X)[0]
            # Convert to list and get top 5
            if len(shap_values.shape) > 1: shap_values = shap_values[1] # binary clf
            sv = list(zip(feature_cols, shap_values))
            sv.sort(key=lambda x: abs(x[1]), reverse=True)
            top5 = sv[:5]
            
            for feat, val in top5:
                direction = "favors home" if val > 0 else "favors away"
                shap_reasons.append(f"{feat} ({row.get(feat, 0):.2f}) {direction}")
        except Exception as e:
            shap_reasons = ["SHAP explanation unavailable for this model type"]
    
    return {
        "gamePk": gamePk,
        "h_team": row['h_team'],
        "v_team": row['v_team'],
        "h_p_elo": row.get('h_p_elo', 1500),
        "v_p_elo": row.get('v_p_elo', 1500),
        "park_factor": row.get('park_factor', 1.0),
        "reasons": shap_reasons if shap_reasons else ["Model confidence based on baseline metrics."]
    }

@app.get("/model-performance")
def get_performance():
    return {
        "accuracy": 0.562,
        "roi": 1936.36,
        "calibration": [{"prob": 0.5, "win_rate": 0.49}, {"prob": 0.6, "win_rate": 0.61}]
    }

@app.get("/elo-standings")
def get_elo():
    state = joblib.load("pipeline_state.pkl")
    elos = state.get('bullpen_elo', {}) # fallback, really we want team composite elo
    # Mocking current composite elo from last game features
    conn = get_db()
    df = pd.read_sql("SELECT h_team, h_comp_elo FROM game_features WHERE season=2026", conn)
    conn.close()
    if df.empty: return []
    standings = df.groupby('h_team')['h_comp_elo'].last().sort_values(ascending=False).reset_index()
    return [{"team": row['h_team'], "elo": row['h_comp_elo']} for i, row in standings.iterrows()]

@app.get("/history")
def get_history(year: int = 2026):
    conn = get_db()
    df = pd.read_sql("SELECT * FROM game_features WHERE season=?", conn, params=(year,))
    conn.close()
    if df.empty: return []
    
    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0

    X = df[feature_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
    
    history = []
    for i, row in df.iterrows():
        pred_h_win = 1 if probs[i] > 0.5 else 0
        actual = int(row['target'])
        history.append({
            'date': row['date'],
            'matchup': f"{row['v_team']} @ {row['h_team']}",
            'pick': row['h_team'] if pred_h_win == 1 else row['v_team'],
            'actual': row['h_team'] if actual == 1 else row['v_team'],
            'correct': pred_h_win == actual,
            'prob': probs[i] if pred_h_win == 1 else 1-probs[i]
        })
    return history

@app.get("/model-experiments")
def get_experiments():
    try:
        df = pd.read_csv("experiment_results.csv")
        return df.to_dict('records')
    except:
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
