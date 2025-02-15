import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import re

MODEL = joblib.load("mlb_model.pkl")
STATE = joblib.load("latest_stats.pkl")

def get_avg(d, default=0.0):
    return np.mean(d) if len(d) > 0 else default

def name_to_id(name):
    # Rough approximation of Retrosheet ID: lastn(5) + firstn(1) + 01
    parts = name.lower().replace(".", "").replace("'", "").split()
    if len(parts) < 2: return name[:8]
    last, first = parts[-1], parts[0]
    rid = (last[:5] + first[:1]).ljust(6, '0') + "01"
    return rid

def get_vegas_odds():
    url = "https://www.vegasinsider.com/mlb/odds/las-vegas/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'lxml')
        # This is a brittle scraper, vegas odds sites change often
        # Just returning a placeholder if it fails
        return {}
    except:
        return {}

def predict_games(date_str=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=lineups,probablePitcher"
    resp = requests.get(url).json()
    
    predictions = []
    for date in resp.get('dates', []):
        for game in date.get('games', []):
            away = game['teams']['away']
            home = game['teams']['home']
            v_team = away['team']['abbreviation']
            h_team = home['team']['abbreviation']
            
            mapping = {"LAD": "LAN", "SF": "SFN", "NYM": "NYN", "NYY": "NYA", "CHC": "CHN", "CWS": "CHA", "KC": "KCA", "TB": "TBA", "SD": "SDN", "WSH": "WAS"}
            v_t = mapping.get(v_team, v_team)
            h_t = mapping.get(h_team, h_team)

            v_p_name = away.get('probablePitcher', {}).get('fullName', '')
            h_p_name = home.get('probablePitcher', {}).get('fullName', '')
            
            v_p_id = name_to_id(v_p_name) if v_p_name else "unknown"
            h_p_id = name_to_id(h_p_name) if h_p_name else "unknown"

            v_p_stat = get_avg(STATE['player_stats'].get(v_p_id, []), -4.0)
            h_p_stat = get_avg(STATE['player_stats'].get(h_p_id, []), -4.0)
            
            # Lineup extraction (simplified)
            v_l_stat = 0.5
            h_l_stat = 0.5
            # In a real scenario, iterate through lineups and call name_to_id
            
            features = [[
                get_avg(STATE['team_wins'].get(v_t, []), 0.5),
                get_avg(STATE['team_runs'].get(v_t, []), 4.0),
                get_avg(STATE['team_wins'].get(h_t, []), 0.5),
                get_avg(STATE['team_runs'].get(h_t, []), 4.0),
                v_p_stat, h_p_stat, v_l_stat, h_l_stat
            ]]
            
            prob = MODEL.predict_proba(features)[0][1]
            predictions.append({
                'match': f"{v_team} @ {h_team}",
                'v_win_prob': prob,
                'h_win_prob': 1 - prob,
                'v_pitcher': v_p_name,
                'h_pitcher': h_p_name
            })
            
    return predictions

if __name__ == "__main__":
    results = predict_games() # Tries today's games
    if not results:
        print("No games found for today, checking 2024-05-11...")
        results = predict_games("2024-05-11")
        
    for r in results:
        print(f"{r['match']} ({r['v_pitcher']} vs {r['h_pitcher']})")
        print(f"  Model: Away {r['v_win_prob']:.1%} | Home {r['h_win_prob']:.1%}")
