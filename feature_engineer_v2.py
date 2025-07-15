import pandas as pd
import numpy as np
import glob
from collections import deque
import os

def process_gamelogs():
    # Retrosheet column indices (0-indexed)
    # 0: date, 3: v_team, 6: h_team, 9: v_score, 10: h_score, 101: v_pitcher, 103: h_pitcher
    # Hitting stats start at 21 for visitor, 48 for home (approx)
    # Let's use specific column indices for box score stats
    # 21: v_ab, 22: v_h, 23: v_2b, 24: v_3b, 25: v_hr, 27: v_bb, 33: v_sb, 34: v_cs, 36: v_err
    # 48: h_ab, 49: h_h, 50: h_2b, 51: h_3b, 52: h_hr, 54: h_bb, 60: h_sb, 61: h_cs, 63: h_err
    
    cols = {
        0: 'date', 3: 'v_team', 6: 'h_team', 9: 'v_score', 10: 'h_score',
        101: 'v_pitcher', 103: 'h_pitcher',
        21: 'v_ab', 22: 'v_h', 23: 'v_2b', 24: 'v_3b', 25: 'v_hr', 27: 'v_bb', 33: 'v_sb', 34: 'v_cs', 36: 'v_err',
        48: 'h_ab', 49: 'h_h', 50: 'h_2b', 51: 'h_3b', 52: 'h_hr', 54: 'h_bb', 60: 'h_sb', 61: 'h_cs', 63: 'h_err'
    }
    
    files = glob.glob("data/gl*.txt")
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None, low_memory=False, usecols=list(cols.keys()))
        df = df.rename(columns=cols)
        dfs.append(df)
    
    df = pd.concat(dfs).sort_values('date')
    df['h_win'] = (df['h_score'] > df['v_score']).astype(int)
    
    # Team stats tracking (162 and 30 game windows)
    team_hist = {} # team -> list of game results
    
    # Pitcher stats tracking (35 and 10 game windows)
    pitcher_hist = {} # pitcher -> list of game results (R, IP, BB, H, etc.)
    
    # Constants for smoothing
    IP_DEF = 3.0
    BF_DEF = 12.0
    ER_DEF = 5/9
    
    features = []
    
    for _, row in df.iterrows():
        v, h = row['v_team'], row['h_team']
        vp, hp = row['v_pitcher'], row['h_pitcher']
        
        def get_team_features(team):
            hist = team_hist.get(team, [])
            res = {}
            for w in [162, 30]:
                recent = hist[-w:] if hist else []
                if not recent:
                    res[f'{w}_win_pct'] = 0.5
                    res[f'{w}_obs'] = 0.75 # Typical OBS
                else:
                    win_pct = np.mean([x['win'] for x in recent])
                    h = sum(x['h'] for x in recent)
                    ab = sum(x['ab'] for x in recent)
                    bb = sum(x['bb'] for x in recent)
                    h2 = sum(x['2b'] for x in recent)
                    h3 = sum(x['3b'] for x in recent)
                    hr = sum(x['hr'] for x in recent)
                    
                    obp = (h + bb) / (ab + bb) if (ab + bb) > 0 else 0.3
                    slg = (h + h2 + 2*h3 + 3*hr) / ab if ab > 0 else 0.4
                    res[f'{w}_win_pct'] = win_pct
                    res[f'{w}_obs'] = obp + slg
            return res

        def get_pitcher_features(p_id):
            hist = pitcher_hist.get(p_id, [])
            res = {}
            for w in [35, 10]:
                recent = hist[-w:] if hist else []
                if not recent:
                    res[f'{w}_era'] = 4.5
                    res[f'{w}_whip'] = 1.3
                else:
                    er = sum(x['er'] for x in recent)
                    h = sum(x['h'] for x in recent)
                    bb = sum(x['bb'] for x in recent)
                    ip = sum(x['ip'] for x in recent)
                    
                    # Smooth like the original code
                    ip_mod = max(ip, w * IP_DEF)
                    er_mod = er + ER_DEF * (ip_mod - ip)
                    res[f'{w}_era'] = (er_mod / ip_mod) * 9
                    res[f'{w}_whip'] = (h + bb) / ip_mod
            return res

        vf = get_team_features(v)
        hf = get_team_features(h)
        vpf = get_pitcher_features(vp)
        hpf = get_pitcher_features(hp)
        
        features.append([
            vf['162_win_pct'], vf['162_obs'], hf['162_win_pct'], hf['162_obs'],
            vf['30_win_pct'], vf['30_obs'], hf['30_win_pct'], hf['30_obs'],
            vpf['35_era'], vpf['35_whip'], hpf['35_era'], hpf['35_whip'],
            vpf['10_era'], vpf['10_whip'], hpf['10_era'], hpf['10_whip'],
            row['h_win']
        ])
        
        # Update team history
        for team, win, ab, h_hit, h2, h3, hr, bb in [
            (v, 1-row['h_win'], row['v_ab'], row['v_h'], row['v_2b'], row['v_3b'], row['v_hr'], row['v_bb']),
            (h, row['h_win'], row['h_ab'], row['h_h'], row['h_2b'], row['h_3b'], row['h_hr'], row['h_bb'])
        ]:
            if team not in team_hist: team_hist[team] = []
            team_hist[team].append({'win': win, 'ab': ab, 'h': h_hit, '2b': h2, '3b': h3, 'hr': hr, 'bb': bb})
            
        # Update pitcher history (rough IP estimate as 6 for starter)
        # In real retrosheet data we could extract exact IP, but 6 is a fair proxy for starters in most eras
        for p, er, h_allow, bb_allow in [
            (vp, row['h_score'], row['h_h'], row['h_bb']),
            (hp, row['v_score'], row['v_h'], row['v_bb'])
        ]:
            if p not in pitcher_hist: pitcher_hist[p] = []
            pitcher_hist[p].append({'er': er, 'h': h_allow, 'bb': bb_allow, 'ip': 6.0})

    f_cols = [
        'v_win_pct_162', 'v_obs_162', 'h_win_pct_162', 'h_obs_162',
        'v_win_pct_30', 'v_obs_30', 'h_win_pct_30', 'h_obs_30',
        'v_era_35', 'v_whip_35', 'h_era_35', 'h_whip_35',
        'v_era_10', 'v_whip_10', 'h_era_10', 'h_whip_10',
        'target'
    ]
    f_df = pd.DataFrame(features, columns=f_cols)
    f_df.to_csv("features_v2.csv", index=False)
    
    import joblib
    joblib.dump({'team_hist': team_hist, 'pitcher_hist': pitcher_hist}, "latest_stats_v2.pkl")

if __name__ == "__main__":
    process_gamelogs()
