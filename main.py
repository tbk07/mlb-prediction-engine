import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import requests
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "mlb.db"
MODEL_PATH = "best_model.pkl"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

TEAM_ALIASES = {
    "LAD": "LAN", "SF": "SFN", "NYM": "NYN", "NYY": "NYA", "CHC": "CHN",
    "CWS": "CHA", "KC": "KCA", "TB": "TBA", "SD": "SDN", "WSH": "WAS",
    "AZ": "ARI", "ATH": "OAK",
}

MLB_TEAM_ID_TO_ABBR = {
    108: "ANA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHN", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KCA", 119: "LAN",
    120: "WAS", 121: "NYN", 133: "OAK", 134: "PIT", 135: "SDN", 136: "SEA",
    137: "SFN", 138: "SLN", 139: "TBA", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CHA", 146: "MIA", 147: "NYA", 158: "MIL",
}

TEAM_FULL_NAMES = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHN": "Chicago Cubs", "CHA": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCA": "Kansas City Royals",
    "ANA": "Los Angeles Angels", "LAN": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYA": "New York Yankees",
    "NYN": "New York Mets", "OAK": "Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDN": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFN": "San Francisco Giants", "SLN": "St. Louis Cardinals", "TBA": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WAS": "Washington Nationals",
}

ESPN_LOGO_CODES = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BOS": "bos", "CHN": "chc",
    "CHA": "chw", "CIN": "cin", "CLE": "cle", "COL": "col", "DET": "det",
    "HOU": "hou", "KCA": "kc", "ANA": "laa", "LAN": "lad", "MIA": "mia",
    "MIL": "mil", "MIN": "min", "NYA": "nyy", "NYN": "nym", "OAK": "ath",
    "PHI": "phi", "PIT": "pit", "SDN": "sd", "SEA": "sea", "SFN": "sf",
    "SLN": "stl", "TBA": "tb", "TEX": "tex", "TOR": "tor", "WAS": "wsh",
}

FEATURE_COLS = [
    "v_comp_elo", "h_comp_elo", "v_off_elo", "h_off_elo", "v_def_elo", "h_def_elo",
    "v_p_elo", "h_p_elo", "v_bp_elo", "h_bp_elo", "exp_h", "over_under", "park_factor",
    "altitude_flag", "temp", "month", "v_games_7d", "h_games_7d", "v_rest", "h_rest",
    "v_dist", "h_dist", "h_back_home",
    "v_p_lhp", "h_p_lhp", "v_p_days_rest", "h_p_days_rest", "v_p_last_pc", "h_p_last_pc",
    "v_10_win_pct", "v_10_ops", "v_10_r_scored", "v_10_r_allowed", "v_10_r_diff", "v_10_err",
    "v_10_sb_pct", "v_30_win_pct", "v_30_ops", "v_30_r_scored", "v_30_r_allowed", "v_30_r_diff",
    "v_30_err", "v_30_sb_pct", "v_162_win_pct", "v_162_ops", "v_162_r_scored", "v_162_r_allowed",
    "v_162_r_diff", "v_162_err", "v_162_sb_pct", "h_10_win_pct", "h_10_ops", "h_10_r_scored",
    "h_10_r_allowed", "h_10_r_diff", "h_10_err", "h_10_sb_pct", "h_30_win_pct", "h_30_ops",
    "h_30_r_scored", "h_30_r_allowed", "h_30_r_diff", "h_30_err", "h_30_sb_pct", "h_162_win_pct",
    "h_162_ops", "h_162_r_scored", "h_162_r_allowed", "h_162_r_diff", "h_162_err", "h_162_sb_pct",
    "vp_10_era", "vp_10_whip", "vp_10_k9", "vp_10_bb9", "vp_35_era", "vp_35_whip", "vp_35_k9",
    "vp_35_bb9", "vp_days_rest", "vp_last_pitch_count", "hp_10_era", "hp_10_whip", "hp_10_k9",
    "hp_10_bb9", "hp_35_era", "hp_35_whip", "hp_35_k9", "hp_35_bb9", "hp_days_rest", "hp_last_pitch_count",
]

model = None
explainer = None

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
        "info": "Frontend build not found. If this is a deployment, ensure 'npm run build' was executed in the frontend directory.",
    }


@app.on_event("startup")
def load_artifacts():
    global model, explainer
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None
    
    # Run data updates in a background thread so the server can bind to the port immediately
    import threading
    def run_updates():
        try:
            from pipeline.fetch_data import init_db, fetch_mlb_api_2026, fetch_odds
            from pipeline.features import compute_features
            
            print("Background Startup: Initializing database...")
            init_db()
            
            print("Background Startup: Updating MLB schedule and scores...")
            fetch_mlb_api_2026()
            
            print("Background Startup: Fetching latest market odds...")
            fetch_odds()
            
            print("Background Startup: Re-calculating model features...")
            compute_features()
            
            print("Background Startup update complete.")
        except Exception as e:
            print(f"Background Startup update failed: {e}")

    threading.Thread(target=run_updates, daemon=True).start()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_team_abbr(abbr, team_id=None):
    if team_id in MLB_TEAM_ID_TO_ABBR:
        return MLB_TEAM_ID_TO_ABBR[team_id]
    if not abbr:
        return "UNK"
    return TEAM_ALIASES.get(abbr, abbr)


def team_payload(abbr, full_name=None, team_id=None):
    abbr = normalize_team_abbr(abbr)
    espn_code = ESPN_LOGO_CODES.get(abbr, abbr.lower())
    return {
        "abbr": abbr,
        "full_name": full_name or TEAM_FULL_NAMES.get(abbr, abbr),
        "mlb_logo": f"https://www.mlbstatic.com/team-logos/{team_id}.svg" if team_id else None,
        "espn_logo": f"https://a.espncdn.com/i/teamlogos/mlb/500/{espn_code}.png",
    }


def moneyline_to_prob(ml):
    if ml is None:
        return None
    ml = int(ml)
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def is_default_odds(home_ml, away_ml, over_under):
    return int(home_ml or -110) == -110 and int(away_ml or -110) == -110 and float(over_under or 8.5) == 8.5


def fetch_schedule(start_date, end_date):
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "gameTypes": "R",
            "hydrate": "probablePitcher",
        },
        timeout=20,
    )
    response.raise_for_status()

    schedule = []
    for date_group in response.json().get("dates", []):
        for game in date_group.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]
            away_team = away["team"]
            home_team = home["team"]
            v_abbr = normalize_team_abbr(away_team.get("abbreviation"), away_team.get("id"))
            h_abbr = normalize_team_abbr(home_team.get("abbreviation"), home_team.get("id"))
            schedule.append({
                "gamePk": str(game["gamePk"]),
                "date": date_group.get("date"),
                "game_datetime": game.get("gameDate"),
                "status": game.get("status", {}).get("detailedState") or game.get("status", {}).get("abstractGameState"),
                "venue": game.get("venue", {}).get("name"),
                "v_team": v_abbr,
                "h_team": h_abbr,
                "v_team_info": team_payload(v_abbr, away_team.get("name"), away_team.get("id")),
                "h_team_info": team_payload(h_abbr, home_team.get("name"), home_team.get("id")),
                "v_score": away.get("score"),
                "h_score": home.get("score"),
            })
    return schedule


def build_current_season_elo(year):
    schedule = fetch_schedule(f"{year}-01-01", datetime.now().strftime("%Y-%m-%d"))
    finals = [
        game for game in schedule
        if game.get("status") == "Final" and game.get("v_score") is not None and game.get("h_score") is not None
    ]
    if not finals:
        return []

    conn = get_db()
    prior = pd.read_sql(
        """
        SELECT h_team AS team, h_comp_elo AS elo, date
        FROM game_features
        WHERE h_team != 'UNK' AND season < ?
        UNION ALL
        SELECT v_team AS team, v_comp_elo AS elo, date
        FROM game_features
        WHERE v_team != 'UNK' AND season < ?
        """,
        conn,
        params=(year, year),
    )
    conn.close()

    elo = {abbr: 1500.0 for abbr in TEAM_FULL_NAMES}
    if not prior.empty:
        latest_prior = prior.sort_values("date").groupby("team").tail(1)
        for _, row in latest_prior.iterrows():
            elo[row["team"]] = float(row["elo"])

    k_factor = 20.0
    for game in sorted(finals, key=lambda item: item["game_datetime"] or item["date"]):
        away = game["v_team"]
        home = game["h_team"]
        if away == "UNK" or home == "UNK":
            continue
        away_elo = elo.get(away, 1500.0)
        home_elo = elo.get(home, 1500.0)
        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + 35)) / 400))
        actual_home = 1.0 if int(game["h_score"]) > int(game["v_score"]) else 0.0
        delta = k_factor * (actual_home - expected_home)
        elo[home] = home_elo + delta
        elo[away] = away_elo - delta

    latest_date_by_team = {}
    for game in finals:
        for abbr in (game["v_team"], game["h_team"]):
            if abbr != "UNK":
                latest_date_by_team[abbr] = game["game_datetime"] or game["date"]

    standings = [
        {
            "team": abbr,
            "team_info": team_payload(abbr),
            "elo": value,
            "date": latest_date_by_team.get(abbr),
            "season": year,
        }
        for abbr, value in elo.items()
        if abbr in latest_date_by_team
    ]
    standings.sort(key=lambda item: item["elo"], reverse=True)
    return standings


def ensure_features(df):
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    
    # Ensure all numeric columns are actually numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    return df


def predict_home_probs(df):
    if model is None or df.empty:
        return []
    df = ensure_features(df.copy())
    x = df[FEATURE_COLS].fillna(0)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def get_odds_map(conn, game_pks):
    if not game_pks:
        return {}
    placeholders = ",".join("?" for _ in game_pks)
    rows = conn.execute(f"SELECT * FROM odds WHERE gamePk IN ({placeholders})", game_pks).fetchall()
    return {str(row["gamePk"]): row for row in rows}


def format_bet(h_team, v_team, h_prob, implied):
    if h_prob is None or implied is None:
        return "PASS"
    edge = h_prob - implied
    if edge > 0.05:
        return f"BET {h_team}"
    if edge < -0.05:
        return f"FADE {h_team} (BET {v_team})"
    return "PASS"


def prediction_payload(row, h_prob, odds_row=None, schedule_item=None):
    h_team = row["h_team"]
    v_team = row["v_team"]
    v_prob = None if h_prob is None else 1.0 - h_prob
    home_ml = odds_row["home_ml"] if odds_row else None
    away_ml = odds_row["away_ml"] if odds_row else None
    over_under = odds_row["over_under"] if odds_row else row.get("over_under")
    odds_available = bool(odds_row) and not is_default_odds(home_ml, away_ml, over_under)
    implied = moneyline_to_prob(home_ml) if odds_available else None
    edge = None if h_prob is None or implied is None else h_prob - implied

    return {
        "gamePk": str(row["gamePk"]),
        "date": schedule_item.get("date") if schedule_item else row.get("date"),
        "game_datetime": schedule_item.get("game_datetime") if schedule_item else row.get("date"),
        "status": schedule_item.get("status") if schedule_item else "Final",
        "venue": schedule_item.get("venue") if schedule_item else None,
        "v_team": v_team,
        "h_team": h_team,
        "v_team_info": schedule_item.get("v_team_info") if schedule_item else team_payload(v_team),
        "h_team_info": schedule_item.get("h_team_info") if schedule_item else team_payload(h_team),
        "v_score": schedule_item.get("v_score") if schedule_item else None,
        "h_score": schedule_item.get("h_score") if schedule_item else None,
        "v_win_prob": v_prob,
        "h_win_prob": h_prob,
        "vegas_home_ml": home_ml if odds_available else None,
        "vegas_away_ml": away_ml if odds_available else None,
        "vegas_implied": implied,
        "odds_available": odds_available,
        "edge": edge,
        "bet": format_bet(h_team, v_team, h_prob, implied),
        "confidence": "Pending" if h_prob is None else "High Confidence" if max(h_prob, v_prob) > 0.6 else "Moderate" if max(h_prob, v_prob) > 0.55 else "Lean",
        "over_under": over_under if odds_available else None,
    }


@app.get("/predictions")
def get_predictions(date: str = None):
    if model is None:
        return {"error": "Model not loaded"}

    target_date = date or datetime.now().strftime("%Y-%m-%d")
    conn = get_db()

    try:
        schedule = fetch_schedule(target_date, target_date)
    except Exception:
        schedule = []

    if schedule:
        game_pks = [item["gamePk"] for item in schedule]
        odds_map = get_odds_map(conn, game_pks)
        placeholders = ",".join("?" for _ in game_pks)
        df = pd.read_sql(f"SELECT * FROM game_features WHERE gamePk IN ({placeholders})", conn, params=game_pks)
        probs = predict_home_probs(df)
        feature_rows = {str(row["gamePk"]): (row, float(probs[i])) for i, row in df.reset_index(drop=True).iterrows()}
        results = []
        for item in schedule:
            match = feature_rows.get(item["gamePk"])
            if match:
                row, h_prob = match
                results.append(prediction_payload(row, h_prob, odds_map.get(item["gamePk"]), item))
            else:
                fallback = {"gamePk": item["gamePk"], "date": item["date"], "h_team": item["h_team"], "v_team": item["v_team"]}
                results.append(prediction_payload(fallback, None, odds_map.get(item["gamePk"]), item))
        conn.close()
        return results

    if date:
        df = pd.read_sql("SELECT * FROM game_features WHERE date=?", conn, params=(date,))
    else:
        df = pd.read_sql(
            "SELECT * FROM game_features WHERE h_team != 'UNK' AND v_team != 'UNK' ORDER BY date DESC LIMIT 30",
            conn,
        )
        if not df.empty:
            df = df[df["date"] == df["date"].max()]

    if df.empty:
        conn.close()
        return []

    odds_map = get_odds_map(conn, [str(x) for x in df["gamePk"].tolist()])
    conn.close()
    probs = predict_home_probs(df)
    return [
        prediction_payload(row, float(probs[i]), odds_map.get(str(row["gamePk"])))
        for i, row in df.reset_index(drop=True).iterrows()
    ]


@app.get("/game/{gamePk}/scouting")
def get_scouting(gamePk: str):
    conn = get_db()
    df = pd.read_sql("SELECT * FROM game_features WHERE gamePk=?", conn, params=(gamePk,))
    conn.close()
    if df.empty:
        return {"error": "Game not found"}

    row = df.iloc[0]
    df = ensure_features(df)
    x = df[FEATURE_COLS].fillna(0)

    shap_reasons = []
    if explainer is not None:
        try:
            sv_all = explainer.shap_values(x)
            sv = sv_all[1][0] if isinstance(sv_all, list) else sv_all[0]
            features_with_weights = list(zip(FEATURE_COLS, sv))
            features_with_weights.sort(key=lambda item: abs(item[1]), reverse=True)
            for feat, val in features_with_weights[:5]:
                direction = "favors Home" if val > 0 else "favors Away"
                shap_reasons.append(f"{feat} ({row.get(feat, 0):.2f}) {direction}")
        except Exception as exc:
            shap_reasons = [f"Scouting analytics engine error: {exc}"]

    return {
        "gamePk": gamePk,
        "h_team": row["h_team"],
        "v_team": row["v_team"],
        "h_p_elo": float(row.get("h_p_elo", 1500)),
        "v_p_elo": float(row.get("v_p_elo", 1500)),
        "h_bp_elo": float(row.get("h_bp_elo", 1500)),
        "v_bp_elo": float(row.get("v_bp_elo", 1500)),
        "h_off_elo": float(row.get("h_off_elo", 1500)),
        "v_off_elo": float(row.get("v_off_elo", 1500)),
        "park_factor": float(row.get("park_factor", 1.0)),
        "reasons": shap_reasons if shap_reasons else ["High confidence based on composite seasonal trend."],
    }


@app.get("/elo-standings")
def get_elo(year: int = None):
    conn = get_db()
    selected_year = year or datetime.now().year
    df = pd.read_sql(
        """
        SELECT h_team AS team, h_comp_elo AS elo, h_off_elo AS off_elo, h_def_elo AS def_elo, h_p_elo AS p_elo, h_bp_elo AS bp_elo, date, season
        FROM game_features
        WHERE h_team != 'UNK' AND season = ?
        UNION ALL
        SELECT v_team AS team, v_comp_elo AS elo, v_off_elo AS off_elo, v_def_elo AS def_elo, v_p_elo AS p_elo, v_bp_elo AS bp_elo, date, season
        FROM game_features
        WHERE v_team != 'UNK' AND season = ?
        """,
        conn,
        params=(selected_year, selected_year),
    )
    if df.empty:
        # Fallback to latest available season
        df_latest = pd.read_sql("SELECT MAX(season) AS season FROM game_features WHERE h_team != 'UNK'", conn)
        if df_latest.empty or not df_latest["season"].iloc[0]:
            conn.close()
            return []
        latest_season = int(df_latest["season"].iloc[0])
        df = pd.read_sql(
            """
            SELECT h_team AS team, h_comp_elo AS elo, h_off_elo AS off_elo, h_def_elo AS def_elo, h_p_elo AS p_elo, h_bp_elo AS bp_elo, date, season
            FROM game_features
            WHERE h_team != 'UNK' AND season = ?
            UNION ALL
            SELECT v_team AS team, v_comp_elo AS elo, v_off_elo AS off_elo, v_def_elo AS def_elo, v_p_elo AS p_elo, v_bp_elo AS bp_elo, date, season
            FROM game_features
            WHERE v_team != 'UNK' AND season = ?
            """,
            conn,
            params=(latest_season, latest_season),
        )
    conn.close()
    if df.empty:
        return []

    standings = df.sort_values("date").groupby("team").tail(1)
    standings = standings.sort_values("elo", ascending=False)
    return [
        {
            "team": row["team"],
            "team_info": team_payload(row["team"]),
            "elo": float(row["elo"]),
            "off_elo": float(row.get("off_elo", 1500)),
            "def_elo": float(row.get("def_elo", 1500)),
            "p_elo": float(row.get("p_elo", 1500)),
            "bp_elo": float(row.get("bp_elo", 1500)),
            "date": row["date"],
            "season": int(row["season"]),
        }
        for _, row in standings.iterrows()
    ]


@app.get("/history")
def get_history(year: int = None):
    if model is None:
        return {"history": [], "summary": {}}

    selected_year = year
    schedule = []
    if selected_year:
        try:
            schedule = fetch_schedule(f"{selected_year}-01-01", f"{selected_year}-12-31")
        except Exception:
            schedule = []

    conn = get_db()
    if year:
        df = pd.read_sql("SELECT * FROM game_features WHERE season=? AND h_team != 'UNK'", conn, params=(year,))
    else:
        df_latest = pd.read_sql("SELECT season FROM game_features WHERE h_team != 'UNK' ORDER BY date DESC LIMIT 1", conn)
        if df_latest.empty:
            conn.close()
            return {"history": [], "summary": {}}
        latest_year = int(df_latest["season"].iloc[0])
        df = pd.read_sql("SELECT * FROM game_features WHERE season=? AND h_team != 'UNK'", conn, params=(latest_year,))

    odds_map = get_odds_map(conn, [str(x) for x in df["gamePk"].tolist()]) if not df.empty else {}
    conn.close()

    probs = predict_home_probs(df)
    history = []
    schedule_by_pk = {item["gamePk"]: item for item in schedule}

    total_correct = 0
    total_games = 0
    weekly_stats = {}

    for i, row in df.reset_index(drop=True).iterrows():
        odds_row = odds_map.get(str(row["gamePk"]))
        odds_available = bool(odds_row) and not is_default_odds(odds_row["home_ml"], odds_row["away_ml"], odds_row["over_under"])
        
        pred_h_win = 1 if probs[i] > 0.5 else 0
        target = row.get("target")
        
        is_correct = None
        actual_team = None
        if target is not None and not pd.isna(target):
            actual = int(target)
            is_correct = bool(pred_h_win == actual)
            actual_team = row["h_team"] if actual == 1 else row["v_team"]
            
        scheduled = schedule_by_pk.pop(str(row["gamePk"]), None)
        
        history.append({
            "gamePk": str(row["gamePk"]),
            "date": row["date"],
            "game_datetime": scheduled.get("game_datetime") if scheduled else row["date"],
            "status": scheduled.get("status") if scheduled else ("Final" if is_correct is not None else "Preview"),
            "v_team": row["v_team"],
            "h_team": row["h_team"],
            "v_team_info": scheduled.get("v_team_info") if scheduled else team_payload(row["v_team"]),
            "h_team_info": scheduled.get("h_team_info") if scheduled else team_payload(row["h_team"]),
            "matchup": f"{TEAM_FULL_NAMES.get(row['v_team'], row['v_team'])} @ {TEAM_FULL_NAMES.get(row['h_team'], row['h_team'])}",
            "pick": row["h_team"] if pred_h_win == 1 else row["v_team"],
            "actual": actual_team,
            "correct": is_correct,
            "prob": float(probs[i] if pred_h_win == 1 else 1 - probs[i]),
            "vegas_home_ml": odds_row["home_ml"] if odds_available else "N/A",
            "vegas_away_ml": odds_row["away_ml"] if odds_available else "N/A",
        })

    for item in schedule_by_pk.values():
        history.append({
            "gamePk": item["gamePk"],
            "date": item["date"],
            "game_datetime": item["game_datetime"],
            "status": item["status"],
            "v_team": item["v_team"],
            "h_team": item["h_team"],
            "v_team_info": item["v_team_info"],
            "h_team_info": item["h_team_info"],
            "matchup": f"{item['v_team_info']['full_name']} @ {item['h_team_info']['full_name']}",
            "pick": None,
            "actual": None,
            "correct": None,
            "prob": None,
            "vegas_home_ml": None,
            "vegas_away_ml": None,
        })

    # Only show completed games in the audit history
    history = [h for h in history if h["correct"] is not None]
    history.sort(key=lambda item: (item.get("game_datetime") or item["date"], item["gamePk"]))
    
    total_correct = sum(1 for h in history if h["correct"])
    total_games = len(history)
    
    weekly_stats = {}
    for h in history:
        date_dt = pd.to_datetime(h["date"])
        week_str = f"Week {date_dt.isocalendar()[1]} ({date_dt.year})"
        if week_str not in weekly_stats:
            weekly_stats[week_str] = {"correct": 0, "total": 0}
        weekly_stats[week_str]["correct"] += 1 if h["correct"] else 0
        weekly_stats[week_str]["total"] += 1

    summary = {
        "total_accuracy": (total_correct / total_games) if total_games > 0 else 0,
        "total_games": total_games,
        "weekly": sorted(
            [{"week": k, "accuracy": v["correct"] / v["total"], "games": v["total"]} for k, v in weekly_stats.items()],
            key=lambda x: x["week"]
        )
    }
    
    return {"history": history, "summary": summary}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
