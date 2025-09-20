import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

MODEL = joblib.load("mlb_model_v2.pkl")
STATE = joblib.load("latest_stats_v2.pkl")

def name_to_id(name):
    parts = name.lower().replace(".", "").replace("'", "").split()
    if len(parts) < 2: return name[:8]
    last, first = parts[-1], parts[0]
    rid = (last[:5] + first[:1]).ljust(6, '0') + "01"
    return rid

def get_team_f(team):
    hist = STATE['team_hist'].get(team, [])
    res = {}
    for w in [162, 30]:
        recent = hist[-w:] if hist else []
        if not recent:
            res[f'{w}_win_pct'], res[f'{w}_obs'] = 0.5, 0.75
        else:
            res[f'{w}_win_pct'] = np.mean([x['win'] for x in recent])
            h_hit = sum(x['h'] for x in recent)
            ab = sum(x['ab'] for x in recent)
            bb = sum(x['bb'] for x in recent)
            h2, h3, hr = sum(x['2b'] for x in recent), sum(x['3b'] for x in recent), sum(x['hr'] for x in recent)
            obp = (h_hit + bb) / (ab + bb) if (ab + bb) > 0 else 0.3
            slg = (h_hit + h2 + 2*h3 + 3*hr) / ab if ab > 0 else 0.4
            res[f'{w}_obs'] = obp + slg
    return res

def get_pitcher_f(p_id):
    hist = STATE['pitcher_hist'].get(p_id, [])
    res = {}
    for w in [35, 10]:
        recent = hist[-w:] if hist else []
        if not recent:
            res[f'{w}_era'], res[f'{w}_whip'] = 4.5, 1.3
        else:
            er, h_hit, bb, ip = sum(x['er'] for x in recent), sum(x['h'] for x in recent), sum(x['bb'] for x in recent), sum(x['ip'] for x in recent)
            ip_mod = max(ip, w * 3.0)
            er_mod = er + (5/9) * (ip_mod - ip)
            res[f'{w}_era'], res[f'{w}_whip'] = (er_mod / ip_mod) * 9, (h_hit + bb) / ip_mod
    return res

def predict_games(date_str=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher"
    resp = requests.get(url).json()
    
    predictions = []
    for date in resp.get('dates', []):
        for game in date.get('games', []):
            away = game['teams']['away']
            home = game['teams']['home']
            v_team = away['team'].get('abbreviation', away['team'].get('name', 'UNK'))
            h_team = home['team'].get('abbreviation', home['team'].get('name', 'UNK'))
            
            mapping = {"LAD": "LAN", "SF": "SFN", "NYM": "NYN", "NYY": "NYA", "CHC": "CHN", "CWS": "CHA", "KC": "KCA", "TB": "TBA", "SD": "SDN", "WSH": "WAS"}
            v_t, h_t = mapping.get(v_team, v_team), mapping.get(h_team, h_team)

            v_p_name = away.get('probablePitcher', {}).get('fullName', 'Unknown')
            h_p_name = home.get('probablePitcher', {}).get('fullName', 'Unknown')
            v_p_id, h_p_id = name_to_id(v_p_name), name_to_id(h_p_name)

            vf, hf = get_team_f(v_t), get_team_f(h_t)
            vpf, hpf = get_pitcher_f(v_p_id), get_pitcher_f(h_p_id)

            features = pd.DataFrame([[
                vf['162_win_pct'], vf['162_obs'], hf['162_win_pct'], hf['162_obs'],
                vf['30_win_pct'], vf['30_obs'], hf['30_win_pct'], hf['30_obs'],
                vpf['35_era'], vpf['35_whip'], hpf['35_era'], hpf['35_whip'],
                vpf['10_era'], vpf['10_whip'], hpf['10_era'], hpf['10_whip']
            ]], columns=[
                'v_win_pct_162', 'v_obs_162', 'h_win_pct_162', 'h_obs_162',
                'v_win_pct_30', 'v_obs_30', 'h_win_pct_30', 'h_obs_30',
                'v_era_35', 'v_whip_35', 'h_era_35', 'h_whip_35',
                'v_era_10', 'v_whip_10', 'h_era_10', 'h_whip_10'
            ])
            
            prob = MODEL.predict_proba(features)[0][1] # Prob of h_win
            
            # Generate "Why" reasoning
            reasoning = []
            if hpf['10_era'] < vpf['10_era'] - 0.5:
                reasoning.append(f"{h_p_name} has been significantly sharper lately ({hpf['10_era']:.2f} ERA) compared to {v_p_name} ({vpf['10_era']:.2f} ERA).")
            if hf['30_win_pct'] > vf['30_win_pct'] + 0.1:
                reasoning.append(f"The {h_team} are carrying much stronger momentum, winning {hf['30_win_pct']:.0%} of their last 30 games.")
            if hf['162_obs'] > vf['162_obs'] + 0.05:
                reasoning.append(f"The {h_team} lineup has a historical offensive edge with a superior OBS.")
            
            if not reasoning:
                reasoning.append("Model indicates a close matchup based on balanced team and pitcher metrics.")

            predictions.append({
                'id': game['gamePk'],
                'match': f"{v_team} @ {h_team}",
                'h_team': h_team, 'v_team': v_team,
                'h_p': h_p_name, 'v_p': v_p_name,
                'h_win_prob': float(prob),
                'v_win_prob': float(1 - prob),
                'reasoning': reasoning,
                'stats': {
                    'h_era_10': float(hpf['10_era']),
                    'v_era_10': float(vpf['10_era']),
                    'h_obs_30': float(hf['30_obs']),
                    'v_obs_30': float(vf['30_obs'])
                }
            })
    return predictions
