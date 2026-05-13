import sqlite3
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import shap
import json
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = 'mlb.db'
MODEL_PATH = 'best_model.pkl'

# Serve static files from the frontend/dist directory if it exists
if os.path.exists("frontend/dist"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/favicon.svg")
    async def favicon():
        return FileResponse("frontend/dist/favicon.svg")

    @app.get("/icons.svg")
    async def icons():
        return FileResponse("frontend/dist/icons.svg")

@app.get("/")
async def read_index():
    if os.path.exists("frontend/dist/index.html"):
        return FileResponse("frontend/dist/index.html")
    return {
        "status": "online",
        "message": "MLB Prediction API is running.",
        "info": "Frontend build not found. If this is a deployment, ensure 'npm run build' was executed in the frontend directory."
    }

# The exact 86 features used during training
FEATURE_COLS = [
    'v_comp_elo', 'h_comp_elo', 'v_off_elo', 'h_off_elo', 'v_def_elo', 'h_def_elo', 
    'v_p_elo', 'h_p_elo', 'exp_h', 'implied_h_prob', 'over_under', 'park_factor', 
    'altitude_flag', 'month', 'v_games_7d', 'h_games_7d', 'v_rest', 'h_rest', 
    'v_p_lhp', 'h_p_lhp', 'v_p_days_rest', 'h_p_days_rest', 'v_p_last_pc', 'h_p_last_pc', 
    'v_10_win_pct', 'v_10_ops', 'v_10_r_scored', 'v_10_r_allowed', 'v_10_r_diff', 'v_10_err', 
    'v_10_sb_pct', 'v_30_win_pct', 'v_30_ops', 'v_30_r_scored', 'v_30_r_allowed', 'v_30_r_diff', 
    'v_30_err', 'v_30_sb_pct', 'v_162_win_pct', 'v_162_ops', 'v_162_r_scored', 'v_162_r_allowed', 
    'v_162_r_diff', 'v_162_err', 'v_162_sb_pct', 'h_10_win_pct', 'h_10_ops', 'h_10_r_scored', 
    'h_10_r_allowed', 'h_10_r_diff', 'h_10_err', 'h_10_sb_pct', 'h_30_win_pct', 'h_30_ops', 
    'h_30_r_scored', 'h_30_r_allowed', 'h_30_r_diff', 'h_30_err', 'h_30_sb_pct', 'h_162_win_pct', 
    'h_162_ops', 'h_162_r_scored', 'h_162_r_allowed', 'h_162_r_diff', 'h_162_err', 'h_162_sb_pct', 
    'vp_10_era', 'vp_10_whip', 'vp_10_k9', 'vp_10_bb9', 'vp_35_era', 'vp_35_whip', 'vp_35_k9', 
    'vp_35_bb9', 'vp_days_rest', 'vp_last_pitch_count', 'hp_10_era', 'hp_10_whip', 'hp_10_k9', 
    'hp_10_bb9', 'hp_35_era', 'hp_35_whip', 'hp_35_k9', 'hp_35_bb9', 'hp_days_rest', 'hp_last_pitch_count'
]

@app.on_event("startup")
def load_artifacts():
    global model, explainer
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        try:
            # TreeExplainer is fast for LGBM
            explainer = shap.TreeExplainer(model)
        except:
            explainer = None
    else:
        model = None
        explainer = None

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/predictions")
def get_predictions(date: str = None):
    if model is None:
        return {"error": "Model not loaded"}
        
    conn = get_db()
    if date:
        df = pd.read_sql("SELECT * FROM game_features WHERE date=?", conn, params=(date,))
    else:
        # Default to the most recent date with valid team names
        df = pd.read_sql("SELECT * FROM game_features WHERE h_team != 'UNK' AND v_team != 'UNK' ORDER BY date DESC LIMIT 30", conn)
        if not df.empty:
            latest_date = df['date'].max()
            df = df[df['date'] == latest_date]
    conn.close()
    
    if df.empty:
        return []
    
    # Ensure all required features are present (impute if missing)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLS].fillna(0)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    
    res = []
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        h_prob = float(probs[i])
        v_prob = 1.0 - h_prob
        
        implied = row.get('implied_h_prob', 0.52)
        edge = h_prob - implied
        
        if edge > 0.05: bet = f"✓ BET {row['h_team']}"
        elif edge < -0.05: bet = f"✗ FADE {row['h_team']} (BET {row['v_team']})"
        else: bet = "— PASS"
        
        conf = "High Confidence" if max(h_prob, v_prob) > 0.6 else "Moderate" if max(h_prob, v_prob) > 0.55 else "Lean"
        
        res.append({
            'gamePk': str(row['gamePk']),
            'v_team': row['v_team'],
            'h_team': row['h_team'],
            'v_win_prob': v_prob,
            'h_win_prob': h_prob,
            'vegas_home_ml': -110, 
            'vegas_implied': implied,
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
    
    # Ensure all required features are present
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[FEATURE_COLS].fillna(0)
    
    # SHAP explanations
    shap_reasons = []
    if explainer is not None:
        try:
            # Get SHAP values for the first row
            sv_all = explainer.shap_values(X)
            # For binary classification, sv_all might be a list [neg_probs, pos_probs]
            if isinstance(sv_all, list):
                sv = sv_all[1][0] 
            else:
                sv = sv_all[0]
                
            features_with_weights = list(zip(FEATURE_COLS, sv))
            features_with_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            top5 = features_with_weights[:5]
            
            for feat, val in top5:
                direction = "favors Home" if val > 0 else "favors Away"
                val_str = f"{row.get(feat, 0):.2f}"
                shap_reasons.append(f"{feat} ({val_str}) {direction}")
        except Exception as e:
            shap_reasons = [f"Scouting analytics engine error: {str(e)}"]
    
    return {
        "gamePk": gamePk,
        "h_team": row['h_team'],
        "v_team": row['v_team'],
        "h_p_elo": float(row.get('h_p_elo', 1500)),
        "v_p_elo": float(row.get('v_p_elo', 1500)),
        "park_factor": float(row.get('park_factor', 1.0)),
        "reasons": shap_reasons if shap_reasons else ["High confidence based on composite seasonal trend."]
    }

@app.get("/elo-standings")
def get_elo():
    conn = get_db()
    # Get the latest Elo for each team from the most recent valid season
    df = pd.read_sql("SELECT h_team, h_comp_elo, date, season FROM game_features WHERE h_team != 'UNK' ORDER BY date DESC", conn)
    conn.close()
    if df.empty: return []
    
    latest_season = df['season'].iloc[0]
    df = df[df['season'] == latest_season]
    standings = df.sort_values('date').groupby('h_team').tail(1)
    standings = standings.sort_values('h_comp_elo', ascending=False)
    
    return [{"team": row['h_team'], "elo": float(row['h_comp_elo'])} for i, row in standings.iterrows()]

@app.get("/history")
def get_history(year: int = None):
    if model is None: return []
    
    conn = get_db()
    if year:
        df = pd.read_sql("SELECT * FROM game_features WHERE season=? AND h_team != 'UNK'", conn, params=(year,))
    else:
        # Latest valid season
        df_latest = pd.read_sql("SELECT season FROM game_features WHERE h_team != 'UNK' ORDER BY date DESC LIMIT 1", conn)
        if df_latest.empty: return []
        latest_year = int(df_latest['season'].iloc[0])
        df = pd.read_sql("SELECT * FROM game_features WHERE season=? AND h_team != 'UNK'", conn, params=(latest_year,))
    conn.close()
    
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLS].fillna(0)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    
    history = []
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        pred_h_win = 1 if probs[i] > 0.5 else 0
        actual = int(row['target'])
        history.append({
            'date': row['date'],
            'matchup': f"{row['v_team']} @ {row['h_team']}",
            'pick': row['h_team'] if pred_h_win == 1 else row['v_team'],
            'actual': row['h_team'] if actual == 1 else row['v_team'],
            'correct': bool(pred_h_win == actual),
            'prob': float(probs[i] if pred_h_win == 1 else 1-probs[i])
        })
    return history

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
