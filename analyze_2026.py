import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import copy

MODEL = joblib.load("mlb_model_v5.pkl")
STATE = copy.deepcopy(joblib.load("latest_stats_v5.pkl"))

def get_2026_games():
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2026-03-20&endDate=2026-05-10&hydrate=probablePitcher"
    resp = requests.get(url).json()
    games_list = []
    for date in resp.get('dates', []):
        for game in date.get('games', []):
            if game['status']['abstractGameState'] != 'Final': continue
            away, home = game['teams']['away'], game['teams']['home']
            games_list.append({
                'id': game['gamePk'],
                'v_team': away['team'].get('abbreviation', 'UNK'),
                'h_team': home['team'].get('abbreviation', 'UNK'),
                'v_p': away.get('probablePitcher', {}).get('fullName', 'Unknown'),
                'h_p': home.get('probablePitcher', {}).get('fullName', 'Unknown'),
                'target': 1 if home.get('score', 0) > away.get('score', 0) else 0
            })
    return games_list

def get_team_f(team):
    hist = STATE['team_hist'].get(team, [])
    res = {}
    for w in [162, 30]:
        recent = hist[-w:] if hist else []
        if not recent: res[f'{w}_win_pct'], res[f'{w}_obs'] = 0.5, 0.75
        else:
            res[f'{w}_win_pct'] = np.mean([x['win'] for x in recent])
            h_hit, ab, bb = sum(x['h'] for x in recent), sum(x['ab'] for x in recent), sum(x['bb'] for x in recent)
            h2, h3, hr = sum(x['2b'] for x in recent), sum(x['3b'] for x in recent), sum(x['hr'] for x in recent)
            res[f'{w}_obs'] = ((h_hit + bb) / (ab + bb) if (ab + bb) > 0 else 0.3) + ((h_hit + h2 + 2*h3 + 3*hr) / ab if ab > 0 else 0.4)
    return res

def analyze_2026():
    games = get_2026_games()
    mapping = {"LAD": "LAN", "SF": "SFN", "NYM": "NYN", "NYY": "NYA", "CHC": "CHN", "CWS": "CHA", "KC": "KCA", "TB": "TBA", "SD": "SDN", "WSH": "WAS"}
    
    audit_data = []
    for g in games:
        v_t, h_t = mapping.get(g['v_team'], g['v_team']), mapping.get(g['h_team'], g['h_team'])
        vf, hf = get_team_f(v_t), get_team_f(h_t)
        v_elo, h_elo = STATE['elo_ratings'].get(v_t, 1500), STATE['elo_ratings'].get(h_t, 1500)
        exp_h = 1 / (10**(-(h_elo + 30 - v_elo) / 400) + 1)
        
        audit_data.append({
            'exp_h': exp_h,
            'h_win_pct_162': hf['162_win_pct'],
            'v_win_pct_162': vf['162_win_pct'],
            'target': g['target']
        })
        
    df_2026 = pd.DataFrame(audit_data)
    print("2026 Correlations with Target:")
    print(df_2026.corr()['target'])
    
    elo_acc = ( (df_2026['exp_h'] > 0.5) == df_2026['target'] ).mean()
    print(f"\n2026 Elo Accuracy: {elo_acc:.4f}")

if __name__ == "__main__":
    analyze_2026()
