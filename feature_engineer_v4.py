import pandas as pd
import numpy as np
import glob
from collections import deque
import os
import joblib

def process_gamelogs():
    cols = {
        0: 'date', 3: 'v_team', 6: 'h_team', 9: 'v_score', 10: 'h_score',
        101: 'v_pitcher', 103: 'h_pitcher',
        21: 'v_ab', 22: 'v_h', 23: 'v_2b', 24: 'v_3b', 25: 'v_hr', 27: 'v_bb', 33: 'v_sb', 34: 'v_cs', 36: 'v_err',
        48: 'h_ab', 49: 'h_h', 50: 'h_2b', 51: 'h_3b', 52: 'h_hr', 54: 'h_bb', 60: 'h_sb', 61: 'h_cs', 63: 'h_err',
        12: 'v_outs', 13: 'h_outs' # for IP calculation
    }
    
    files = glob.glob("data/gl*.txt")
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None, low_memory=False, usecols=list(cols.keys()))
        df = df.rename(columns=cols)
        dfs.append(df)
    
    df = pd.concat(dfs).sort_values('date')
    df['h_win'] = (df['h_score'] > df['v_score']).astype(int)
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    team_hist = {} 
    pitcher_hist = {} 
    bullpen_hist = {} # team -> list of bullpen stats
    last_game_date = {} # team -> date
    
    elo_ratings = {} 
    K, HFA = 20, 30 
    
    def get_elo(team): return elo_ratings.get(team, 1500)

    IP_DEF, ER_DEF = 3.0, 5/9
    features = []
    
    for _, row in df.iterrows():
        v, h = row['v_team'], row['h_team']
        vp, hp = row['v_pitcher'], row['h_pitcher']
        date = row['date_dt']
        
        v_elo, h_elo = get_elo(v), get_elo(h)
        exp_h = 1 / (10**(-(h_elo + HFA - v_elo) / 400) + 1)
        
        # Rest days
        v_rest = (date - last_game_date.get(v, date)).days
        h_rest = (date - last_game_date.get(h, date)).days
        
        def get_team_features(team):
            hist = team_hist.get(team, [])
            res = {}
            for w in [162, 30]:
                recent = hist[-w:] if hist else []
                if not recent:
                    res[f'{w}_win_pct'], res[f'{w}_obs'] = 0.5, 0.75
                else:
                    res[f'{w}_win_pct'] = np.mean([x['win'] for x in recent])
                    h_hit, ab, bb = sum(x['h'] for x in recent), sum(x['ab'] for x in recent), sum(x['bb'] for x in recent)
                    h2, h3, hr = sum(x['2b'] for x in recent), sum(x['3b'] for x in recent), sum(x['hr'] for x in recent)
                    res[f'{w}_obs'] = ((h_hit + bb) / (ab + bb) if (ab + bb) > 0 else 0.3) + ((h_hit + h2 + 2*h3 + 3*hr) / ab if ab > 0 else 0.4)
            return res

        def get_pitcher_features(p_id):
            hist = pitcher_hist.get(p_id, [])
            res = {}
            for w in [35, 10]:
                recent = hist[-w:] if hist else []
                if not recent:
                    res[f'{w}_era'], res[f'{w}_whip'] = 4.5, 1.3
                else:
                    er, h_hit, bb, ip = sum(x['er'] for x in recent), sum(x['h'] for x in recent), sum(x['bb'] for x in recent), sum(x['ip'] for x in recent)
                    ip_mod = max(ip, w * IP_DEF)
                    res[f'{w}_era'], res[f'{w}_whip'] = ((er + ER_DEF * (ip_mod - ip)) / ip_mod) * 9, (h_hit + bb) / ip_mod
            return res

        def get_bullpen_features(team):
            hist = bullpen_hist.get(team, [])
            res = {}
            for w in [35]: # Just 35 for bullpen stability
                recent = hist[-w:] if hist else []
                if not recent:
                    res[f'{w}_b_whip'] = 1.4
                else:
                    h_hit, bb, ip = sum(x['h'] for x in recent), sum(x['bb'] for x in recent), sum(x['ip'] for x in recent)
                    res[f'{w}_b_whip'] = (h_hit + bb) / max(ip, 1.0)
            return res

        vf, hf = get_team_features(v), get_team_features(h)
        vpf, hpf = get_pitcher_features(vp), get_pitcher_features(hp)
        vbf, hbf = get_bullpen_features(v), get_bullpen_features(h)
        
        features.append([
            vf['162_win_pct'], vf['162_obs'], hf['162_win_pct'], hf['162_obs'],
            vf['30_win_pct'], vf['30_obs'], hf['30_win_pct'], hf['30_obs'],
            vpf['35_era'], vpf['35_whip'], hpf['35_era'], hpf['35_whip'],
            vpf['10_era'], vpf['10_whip'], hpf['10_era'], hpf['10_whip'],
            vbf['35_b_whip'], hbf['35_b_whip'],
            v_elo, h_elo, exp_h, v_rest, h_rest,
            row['h_win']
        ])
        
        # Updates
        s_h = row['h_win']
        elo_ratings[h] = h_elo + K * (s_h - exp_h)
        elo_ratings[v] = v_elo + K * ((1-s_h) - (1-exp_h))
        last_game_date[v], last_game_date[h] = date, date
        
        for team, win, ab, h_hit, h2, h3, hr, bb in [
            (v, 1-row['h_win'], row['v_ab'], row['v_h'], row['v_2b'], row['v_3b'], row['v_hr'], row['v_bb']),
            (h, row['h_win'], row['h_ab'], row['h_h'], row['h_2b'], row['h_3b'], row['h_hr'], row['h_bb'])
        ]:
            if team not in team_hist: team_hist[team] = []
            team_hist[team].append({'win': win, 'ab': ab, 'h': h_hit, '2b': h2, '3b': h3, 'hr': hr, 'bb': bb})
            
        # Update Pitcher/Bullpen (IP proxy)
        # Assuming starter goes 6, rest is bullpen
        try:
            v_outs = float(row['v_outs'])
            h_outs = float(row['h_outs'])
        except (ValueError, TypeError):
            v_outs, h_outs = 27, 27 # Default to 9 innings
            
        total_ip_v = h_outs / 3 # v_pitcher innings is home team outs
        total_ip_h = v_outs / 3 # h_pitcher innings is visitor team outs
        
        # v_pitcher stats (vp)
        if vp not in pitcher_hist: pitcher_hist[vp] = []
        pitcher_hist[vp].append({'er': row['h_score'] * (6/max(total_ip_v, 6)), 'h': row['h_h'] * (6/max(total_ip_v, 6)), 'bb': row['h_bb'] * (6/max(total_ip_v, 6)), 'ip': 6.0})
        
        # h_pitcher stats (hp)
        if hp not in pitcher_hist: pitcher_hist[hp] = []
        pitcher_hist[hp].append({'er': row['v_score'] * (6/max(total_ip_h, 6)), 'h': row['v_h'] * (6/max(total_ip_h, 6)), 'bb': row['v_bb'] * (6/max(total_ip_h, 6)), 'ip': 6.0})

        # Bullpen stats (remaining)
        for team, b_h, b_bb, b_ip in [
            (v, row['h_h'] * max(0, total_ip_v-6)/max(total_ip_v, 1), row['h_bb'] * max(0, total_ip_v-6)/max(total_ip_v, 1), max(0, total_ip_v-6)),
            (h, row['v_h'] * max(0, total_ip_h-6)/max(total_ip_h, 1), row['v_bb'] * max(0, total_ip_h-6)/max(total_ip_h, 1), max(0, total_ip_h-6))
        ]:
            if team not in bullpen_hist: bullpen_hist[team] = []
            bullpen_hist[team].append({'h': b_h, 'bb': b_bb, 'ip': b_ip})

    f_cols = [
        'v_win_pct_162', 'v_obs_162', 'h_win_pct_162', 'h_obs_162',
        'v_win_pct_30', 'v_obs_30', 'h_win_pct_30', 'h_obs_30',
        'v_era_35', 'v_whip_35', 'h_era_35', 'h_whip_35',
        'v_era_10', 'v_whip_10', 'h_era_10', 'h_whip_10',
        'v_b_whip_35', 'h_b_whip_35',
        'v_elo', 'h_elo', 'exp_h', 'v_rest', 'h_rest', 'target'
    ]
    pd.DataFrame(features, columns=f_cols).to_csv("features_v4.csv", index=False)
    joblib.dump({'team_hist': team_hist, 'pitcher_hist': pitcher_hist, 'bullpen_hist': bullpen_hist, 'elo_ratings': elo_ratings, 'last_game_date': last_game_date}, "latest_stats_v4.pkl")

if __name__ == "__main__":
    process_gamelogs()
