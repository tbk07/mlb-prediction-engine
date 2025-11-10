import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
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
                'date': date['date'],
                'v_team': away['team'].get('abbreviation', 'UNK'),
                'h_team': home['team'].get('abbreviation', 'UNK'),
                'v_p': away.get('probablePitcher', {}).get('fullName', 'Unknown'),
                'h_p': home.get('probablePitcher', {}).get('fullName', 'Unknown'),
                'v_score': away.get('score', 0),
                'h_score': home.get('score', 0),
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

def get_p_f(name):
    p_id = (name.split()[-1][:5] + name.split()[0][:1]).lower() if ' ' in name else 'unk'
    hist = STATE['pitcher_hist'].get(p_id, [])
    res = {}
    for w in [35, 10]:
        recent = hist[-w:] if hist else []
        if not recent: res[f'{w}_era'], res[f'{w}_whip'] = 4.5, 1.3
        else:
            er, h_hit, bb, ip = sum(x['er'] for x in recent), sum(x['h'] for x in recent), sum(x['bb'] for x in recent), sum(x['ip'] for x in recent)
            ip_mod = max(ip, w * 3.0)
            res[f'{w}_era'], res[f'{w}_whip'] = ((er + (5/9)*(ip_mod-ip))/ip_mod)*9, (h_hit+bb)/ip_mod
    return res

def audit_2026():
    games = get_2026_games()
    print(f"Auditing {len(games)} games from 2026...")
    
    mapping = {"LAD": "LAN", "SF": "SFN", "NYM": "NYN", "NYY": "NYA", "CHC": "CHN", "CWS": "CHA", "KC": "KCA", "TB": "TBA", "SD": "SDN", "WSH": "WAS"}
    y_true, y_pred = [], []

    for g in games:
        v_t, h_t = mapping.get(g['v_team'], g['v_team']), mapping.get(g['h_team'], g['h_team'])
        
        vf, hf = get_team_f(v_t), get_team_f(h_t)
        vpf, hpf = get_p_f(g['v_p']), get_p_f(g['h_p'])
        v_elo, h_elo = STATE['elo_ratings'].get(v_t, 1500), STATE['elo_ratings'].get(h_t, 1500)
        exp_h = 1 / (10**(-(h_elo + 30 - v_elo) / 400) + 1)

        features = pd.DataFrame([[
            vf['162_win_pct'], vf['162_obs'], hf['162_win_pct'], hf['162_obs'],
            vf['30_win_pct'], vf['30_obs'], hf['30_win_pct'], hf['30_obs'],
            vpf['35_era'], vpf['35_whip'], hpf['35_era'], hpf['35_whip'],
            vpf['10_era'], vpf['10_whip'], hpf['10_era'], hpf['10_whip'],
            v_elo, h_elo, exp_h, 1, 1, 1.0, 0, 0
        ]], columns=[
            'v_win_pct_162', 'v_obs_162', 'h_win_pct_162', 'h_obs_162',
            'v_win_pct_30', 'v_obs_30', 'h_win_pct_30', 'h_obs_30',
            'v_era_35', 'v_whip_35', 'h_era_35', 'h_whip_35',
            'v_era_10', 'v_whip_10', 'h_era_10', 'h_whip_10',
            'v_elo', 'h_elo', 'exp_h', 'v_rest', 'h_rest', 'park_factor', 'v_p_lhp', 'h_p_lhp'
        ])
        
        pred_prob = MODEL.predict_proba(features)[0][1]
        pred_label = 1 if pred_prob > 0.5 else 0
        
        y_pred.append(pred_label)
        y_true.append(g['target'])

        # Walk-forward update
        s_h = g['target']
        STATE['elo_ratings'][h_t] = h_elo + 20 * (s_h - exp_h)
        STATE['elo_ratings'][v_t] = v_elo + 20 * ((1-s_h) - (1-exp_h))
        
    print(f"\nActual Home Win Rate: {np.mean(y_true):.2%}")
    print(f"Predicted Home Win Rate: {np.mean(y_pred):.2%}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    audit_2026()
