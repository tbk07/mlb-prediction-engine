import pandas as pd
import numpy as np
import glob
from collections import deque
import os
import joblib

def process_gamelogs():
    # Adding 16: park_id, 102: v_p_hand, 104: h_p_hand
    cols = {
        0: 'date', 3: 'v_team', 6: 'h_team', 9: 'v_score', 10: 'h_score',
        16: 'park_id', 101: 'v_pitcher', 102: 'v_p_hand', 103: 'h_pitcher', 104: 'h_p_hand',
        21: 'v_ab', 22: 'v_h', 23: 'v_2b', 24: 'v_3b', 25: 'v_hr', 27: 'v_bb', 33: 'v_sb', 34: 'v_cs', 36: 'v_err',
        48: 'h_ab', 49: 'h_h', 50: 'h_2b', 51: 'h_3b', 52: 'h_hr', 54: 'h_bb', 60: 'h_sb', 61: 'h_cs', 63: 'h_err',
        12: 'v_outs', 13: 'h_outs'
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
    df['total_runs'] = df['v_score'] + df['h_score']
    
    team_hist, pitcher_hist, bullpen_hist = {}, {}, {}
    last_game_date, elo_ratings = {}, {}
    
    # Park Factors (rolling runs per game in park)
    park_runs = {} # park -> list of total runs
    league_runs = deque(maxlen=1000) # league-wide rolling runs
    
    K, HFA = 20, 30
    def get_elo(team): return elo_ratings.get(team, 1500)
    IP_DEF, ER_DEF = 3.0, 5/9
    
    features = []
    for _, row in df.iterrows():
        v, h, p_id = row['v_team'], row['h_team'], row['park_id']
        vp, hp = row['v_pitcher'], row['h_pitcher']
        vph, hph = row['v_p_hand'], row['h_p_hand'] # 'L' or 'R'
        date = row['date_dt']
        
        v_elo, h_elo = get_elo(v), get_elo(h)
        exp_h = 1 / (10**(-(h_elo + HFA - v_elo) / 400) + 1)
        
        v_rest = min(10, (date - last_game_date.get(v, date)).days)
        h_rest = min(10, (date - last_game_date.get(h, date)).days)
        
        # Park Factor (current runs avg / league avg)
        p_avg = np.mean(park_runs.get(p_id, [8.5])) if park_runs.get(p_id) else 8.5
        l_avg = np.mean(league_runs) if league_runs else 8.5
        park_factor = p_avg / l_avg

        def get_team_f(team):
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

        def get_pitcher_f(p_id):
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

        vf, hf = get_team_f(v), get_team_f(h)
        vpf, hpf_stats = get_pitcher_f(vp), get_pitcher_f(hp)
        
        features.append([
            vf['162_win_pct'], vf['162_obs'], hf['162_win_pct'], hf['162_obs'],
            vf['30_win_pct'], vf['30_obs'], hf['30_win_pct'], hf['30_obs'],
            vpf['35_era'], vpf['35_whip'], hpf_stats['35_era'], hpf_stats['35_whip'],
            vpf['10_era'], vpf['10_whip'], hpf_stats['10_era'], hpf_stats['10_whip'],
            v_elo, h_elo, exp_h, v_rest, h_rest, park_factor,
            1 if vph == 'L' else 0, 1 if hph == 'L' else 0, # Handedness
            row['h_win']
        ])
        
        # Updates
        s_h = row['h_win']
        elo_ratings[h] = h_elo + K * (s_h - exp_h)
        elo_ratings[v] = v_elo + K * ((1-s_h) - (1-exp_h))
        last_game_date[v], last_game_date[h] = date, date
        
        if p_id not in park_runs: park_runs[p_id] = deque(maxlen=100)
        park_runs[p_id].append(row['total_runs'])
        league_runs.append(row['total_runs'])

        for team, win, ab, h_hit, h2, h3, hr, bb in [
            (v, 1-row['h_win'], row['v_ab'], row['v_h'], row['v_2b'], row['v_3b'], row['v_hr'], row['v_bb']),
            (h, row['h_win'], row['h_ab'], row['h_h'], row['h_2b'], row['h_3b'], row['h_hr'], row['h_bb'])
        ]:
            if team not in team_hist: team_hist[team] = []
            team_hist[team].append({'win': win, 'ab': ab, 'h': h_hit, '2b': h2, '3b': h3, 'hr': hr, 'bb': bb})
            
        try:
            v_outs, h_outs = float(row['v_outs']), float(row['h_outs'])
        except: v_outs, h_outs = 27, 27
            
        for p, er, h_a, bb_a, total_ip in [(vp, row['h_score'], row['h_h'], row['h_bb'], h_outs/3), (hp, row['v_score'], row['v_h'], row['v_bb'], v_outs/3)]:
            if p not in pitcher_hist: pitcher_hist[p] = []
            pitcher_hist[p].append({'er': er * (6/max(total_ip, 6)), 'h': h_a * (6/max(total_ip, 6)), 'bb': bb_a * (6/max(total_ip, 6)), 'ip': 6.0})

    f_cols = [
        'v_win_pct_162', 'v_obs_162', 'h_win_pct_162', 'h_obs_162',
        'v_win_pct_30', 'v_obs_30', 'h_win_pct_30', 'h_obs_30',
        'v_era_35', 'v_whip_35', 'h_era_35', 'h_whip_35',
        'v_era_10', 'v_whip_10', 'h_era_10', 'h_whip_10',
        'v_elo', 'h_elo', 'exp_h', 'v_rest', 'h_rest', 'park_factor',
        'v_p_lhp', 'h_p_lhp', 'target'
    ]
    pd.DataFrame(features, columns=f_cols).to_csv("features_v5.csv", index=False)
    joblib.dump({'team_hist': team_hist, 'pitcher_hist': pitcher_hist, 'elo_ratings': elo_ratings, 'last_game_date': last_game_date, 'park_runs': park_runs, 'league_runs': league_runs}, "latest_stats_v5.pkl")

if __name__ == "__main__":
    process_gamelogs()
