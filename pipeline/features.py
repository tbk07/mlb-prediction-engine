import sqlite3
import pandas as pd
import numpy as np
from collections import deque
import json
import os
import joblib

DB_PATH = 'mlb.db'

def compute_features():
    print("Connecting to DB and loading games...")
    conn = sqlite3.connect(DB_PATH)
    
    # Load games
    games_df = pd.read_sql("SELECT * FROM games ORDER BY date", conn)
    pitcher_df = pd.read_sql("SELECT * FROM pitcher_game_stats", conn)
    batter_df = pd.read_sql("SELECT * FROM batter_game_stats", conn)
    odds_df = pd.read_sql("SELECT * FROM odds", conn)
    
    # Build indexes for quick lookup
    print("Building indexes...")
    pitcher_games = pitcher_df.groupby('gamePk')
    batter_games = batter_df.groupby('gamePk')
    odds_dict = odds_df.set_index('gamePk').to_dict('index')
    
    # Elo System State
    pitcher_elo = {}
    batter_elo = {}
    bullpen_elo = {} # team -> elo
    
    # Rolling State
    team_stats = {} # team -> deque
    team_home_stats = {}
    team_away_stats = {}
    pitcher_stats = {} # pitcher -> deque
    park_runs = {} # park_id -> deque
    league_runs = deque(maxlen=1000)
    
    K_PITCHER = 32
    K_BATTER = 20
    K_TEAM = 16
    HFA = 30
    DEFAULT_ELO = 1500
    
    def get_p_elo(p): return pitcher_elo.get(p, DEFAULT_ELO)
    def get_b_elo(b): return batter_elo.get(b, DEFAULT_ELO)
    def get_bp_elo(t): return bullpen_elo.get(t, DEFAULT_ELO)
    
    features_list = []
    
    print(f"Processing {len(games_df)} games...")
    
    for i, row in games_df.iterrows():
        gamePk = row['gamePk']
        v = row['v_team']
        h = row['h_team']
        vp = row['v_p_starter']
        hp = row['h_p_starter']
        date_str = str(row['date'])
        if len(date_str) < 10:
            date_dt = pd.to_datetime(date_str, format="%Y%m%d")
        else:
            date_dt = pd.to_datetime(date_str)
        
        month = date_dt.month
        
        # Elos
        v_p_elo = get_p_elo(vp)
        h_p_elo = get_p_elo(hp)
        v_bp_elo = get_bp_elo(v)
        h_bp_elo = get_bp_elo(h)
        
        # Get batters for this game if available
        v_batters, h_batters = [], []
        if gamePk in batter_games.groups:
            game_batters = batter_games.get_group(gamePk)
            v_batters_df = game_batters[(game_batters['team'] == v) & (game_batters['is_starter'] == 1)]
            h_batters_df = game_batters[(game_batters['team'] == h) & (game_batters['is_starter'] == 1)]
            v_batters = v_batters_df['player_id'].tolist()[:9]
            h_batters = h_batters_df['player_id'].tolist()[:9]
            
        v_off_elo = np.mean([get_b_elo(b) for b in v_batters]) if v_batters else DEFAULT_ELO
        h_off_elo = np.mean([get_b_elo(b) for b in h_batters]) if h_batters else DEFAULT_ELO
        
        v_def_elo = 0.6 * v_p_elo + 0.4 * v_bp_elo
        h_def_elo = 0.6 * h_p_elo + 0.4 * h_bp_elo
        
        v_comp_elo = 0.5 * v_off_elo + 0.5 * v_def_elo
        h_comp_elo = 0.5 * h_off_elo + 0.5 * h_def_elo
        
        exp_h = 1 / (10**(-(h_comp_elo + HFA - v_comp_elo) / 400) + 1)
        
        # Features calculation functions
        def get_t_feats(team, stats_dict):
            hist = stats_dict.get(team, [])
            res = {}
            for w in [10, 30, 162]:
                recent = list(hist)[-w:] if hist else []
                if not recent:
                    res[f'{w}_win_pct'] = 0.5
                    res[f'{w}_ops'] = 0.75
                    res[f'{w}_r_scored'] = 4.5
                    res[f'{w}_r_allowed'] = 4.5
                    res[f'{w}_r_diff'] = 0.0
                    res[f'{w}_err'] = 0.5
                    res[f'{w}_sb_pct'] = 0.7
                else:
                    res[f'{w}_win_pct'] = np.mean([x['win'] for x in recent])
                    ab, h_hit, h2, h3, hr, bb = sum(x['ab'] for x in recent), sum(x['h'] for x in recent), sum(x['2b'] for x in recent), sum(x['3b'] for x in recent), sum(x['hr'] for x in recent), sum(x['bb'] for x in recent)
                    sf = 0 # approx
                    obp = (h_hit + bb) / max(ab + bb, 1)
                    slg = (h_hit + h2 + 2*h3 + 3*hr) / max(ab, 1)
                    res[f'{w}_ops'] = obp + slg
                    res[f'{w}_r_scored'] = np.mean([x['rs'] for x in recent])
                    res[f'{w}_r_allowed'] = np.mean([x['ra'] for x in recent])
                    res[f'{w}_r_diff'] = res[f'{w}_r_scored'] - res[f'{w}_r_allowed']
                    res[f'{w}_err'] = np.mean([x['err'] for x in recent])
                    sb, cs = sum(x['sb'] for x in recent), sum(x['cs'] for x in recent)
                    res[f'{w}_sb_pct'] = sb / max(sb + cs, 1)
            return res
            
        def get_p_feats(p):
            hist = pitcher_stats.get(p, [])
            res = {}
            for w in [10, 35]:
                recent = list(hist)[-w:] if hist else []
                if not recent:
                    res[f'{w}_era'] = 4.5
                    res[f'{w}_whip'] = 1.3
                    res[f'{w}_k9'] = 8.0
                    res[f'{w}_bb9'] = 3.0
                else:
                    er, h_allow, bb, k, ip = sum(x['er'] for x in recent), sum(x['h'] for x in recent), sum(x['bb'] for x in recent), sum(x['k'] for x in recent), sum(x['ip'] for x in recent)
                    ip_mod = max(ip, w * 3.0)
                    res[f'{w}_era'] = ((er + (5/9)*(ip_mod-ip))/ip_mod)*9
                    res[f'{w}_whip'] = (h_allow+bb)/ip_mod
                    res[f'{w}_k9'] = (k/ip_mod)*9
                    res[f'{w}_bb9'] = (bb/ip_mod)*9
            
            # Days rest & pitch count
            if hist:
                last = hist[-1]
                res['days_rest'] = min(30, (date_dt - last['date']).days)
                res['last_pitch_count'] = last.get('pc', 90)
            else:
                res['days_rest'] = 5
                res['last_pitch_count'] = 90
            return res
            
        vf, hf = get_t_feats(v, team_stats), get_t_feats(h, team_stats)
        vf_away, hf_home = get_t_feats(v, team_away_stats), get_t_feats(h, team_home_stats)
        vpf, hpf = get_p_feats(vp), get_p_feats(hp)
        
        # Park factor
        p_id = str(row['park_id'])
        p_avg = np.mean(park_runs.get(p_id, [8.5])) if park_runs.get(p_id) else 8.5
        l_avg = np.mean(league_runs) if league_runs else 8.5
        park_factor = p_avg / l_avg
        
        # Contextual
        v_games_7d = sum(1 for x in team_stats.get(v, []) if (date_dt - x['date']).days <= 7)
        h_games_7d = sum(1 for x in team_stats.get(h, []) if (date_dt - x['date']).days <= 7)
        v_rest = (date_dt - team_stats[v][-1]['date']).days if team_stats.get(v) else 3
        h_rest = (date_dt - team_stats[h][-1]['date']).days if team_stats.get(h) else 3
        
        # Odds
        odds = odds_dict.get(gamePk, {})
        h_ml = odds.get('home_ml', -110)
        v_ml = odds.get('away_ml', -110)
        ou = odds.get('over_under', 8.5)
        
        def ml_to_prob(ml):
            if ml < 0: return -ml / (-ml + 100)
            else: return 100 / (ml + 100)
        implied_h_prob = ml_to_prob(h_ml)
        
        # Platoon approx
        v_p_lhp = 1 if row['v_p_hand'] == 'L' else 0
        h_p_lhp = 1 if row['h_p_hand'] == 'L' else 0
        
        # Save features
        feat_row = {
            'gamePk': gamePk,
            'season': row['season'],
            'date': date_str,
            'target': 1 if row['h_score'] > row['v_score'] else 0,
            'h_team': h,
            'v_team': v,
            'v_comp_elo': v_comp_elo, 'h_comp_elo': h_comp_elo,
            'v_off_elo': v_off_elo, 'h_off_elo': h_off_elo,
            'v_def_elo': v_def_elo, 'h_def_elo': h_def_elo,
            'v_p_elo': v_p_elo, 'h_p_elo': h_p_elo,
            'exp_h': exp_h,
            'implied_h_prob': implied_h_prob,
            'over_under': ou,
            'park_factor': park_factor,
            'altitude_flag': row['altitude_flag'],
            'month': month,
            'v_games_7d': v_games_7d, 'h_games_7d': h_games_7d,
            'v_rest': v_rest, 'h_rest': h_rest,
            'v_p_lhp': v_p_lhp, 'h_p_lhp': h_p_lhp,
            'v_p_days_rest': vpf['days_rest'], 'h_p_days_rest': hpf['days_rest'],
            'v_p_last_pc': vpf['last_pitch_count'], 'h_p_last_pc': hpf['last_pitch_count']
        }
        
        for w in [10, 30, 162]:
            for k, v_val in vf.items(): feat_row[f'v_{k}'] = v_val
            for k, h_val in hf.items(): feat_row[f'h_{k}'] = h_val
        for w in [10, 35]:
            for k, v_val in vpf.items(): feat_row[f'vp_{k}'] = v_val
            for k, h_val in hpf.items(): feat_row[f'hp_{k}'] = h_val
            
        features_list.append(feat_row)
        
        # --- ELO & STATE UPDATES ---
        h_win = 1 if row['h_score'] > row['v_score'] else 0
        
        # Pitcher Elo update (use runs allowed vs expected roughly)
        # Expected runs ~ 4.5.
        v_er = row['h_score'] # proxy
        h_er = row['v_score'] # proxy
        
        v_p_s = max(0, min(1, 0.5 + (4.5 - v_er)/9))
        h_p_s = max(0, min(1, 0.5 + (4.5 - h_er)/9))
        
        pitcher_elo[vp] = v_p_elo + K_PITCHER * (v_p_s - (1 - exp_h))
        pitcher_elo[hp] = h_p_elo + K_PITCHER * (h_p_s - exp_h)
        
        # Batter Elo
        # Simple update based on whether team scored > 4.5 runs
        if v_batters:
            v_off_s = max(0, min(1, 0.5 + (row['v_score'] - 4.5)/9))
            for b in v_batters: batter_elo[b] = get_b_elo(b) + K_BATTER * (v_off_s - (1 - exp_h))
        if h_batters:
            h_off_s = max(0, min(1, 0.5 + (row['h_score'] - 4.5)/9))
            for b in h_batters: batter_elo[b] = get_b_elo(b) + K_BATTER * (h_off_s - exp_h)
            
        # Update rolling stats
        # Visitor
        v_game_stat = {'date': date_dt, 'win': 1-h_win, 'rs': row['v_score'], 'ra': row['h_score'], 'err': 0, 'ab': 33, 'h': 8, '2b': 2, '3b': 0, 'hr': 1, 'bb': 3, 'sb': 0, 'cs': 0}
        h_game_stat = {'date': date_dt, 'win': h_win, 'rs': row['h_score'], 'ra': row['v_score'], 'err': 0, 'ab': 33, 'h': 8, '2b': 2, '3b': 0, 'hr': 1, 'bb': 3, 'sb': 0, 'cs': 0}
        
        if v not in team_stats: team_stats[v] = deque(maxlen=162)
        if h not in team_stats: team_stats[h] = deque(maxlen=162)
        team_stats[v].append(v_game_stat)
        team_stats[h].append(h_game_stat)
        
        if v not in team_away_stats: team_away_stats[v] = deque(maxlen=162)
        if h not in team_home_stats: team_home_stats[h] = deque(maxlen=162)
        team_away_stats[v].append(v_game_stat)
        team_home_stats[h].append(h_game_stat)
        
        if vp not in pitcher_stats: pitcher_stats[vp] = deque(maxlen=35)
        if hp not in pitcher_stats: pitcher_stats[hp] = deque(maxlen=35)
        pitcher_stats[vp].append({'date': date_dt, 'er': v_er, 'h': 6, 'bb': 2, 'k': 5, 'ip': 6.0, 'pc': 90})
        pitcher_stats[hp].append({'date': date_dt, 'er': h_er, 'h': 6, 'bb': 2, 'k': 5, 'ip': 6.0, 'pc': 90})
        
        if p_id not in park_runs: park_runs[p_id] = deque(maxlen=100)
        park_runs[p_id].append(row['v_score'] + row['h_score'])
        league_runs.append(row['v_score'] + row['h_score'])

    print("Saving features to database...")
    features_df = pd.DataFrame(features_list)
    features_df.to_sql('game_features', conn, if_exists='replace', index=False)
    
    # Save the final state for future inference
    joblib.dump({
        'pitcher_elo': pitcher_elo,
        'batter_elo': batter_elo,
        'bullpen_elo': bullpen_elo,
        'team_stats': team_stats,
        'team_home_stats': team_home_stats,
        'team_away_stats': team_away_stats,
        'pitcher_stats': pitcher_stats,
        'park_runs': park_runs,
        'league_runs': league_runs
    }, "pipeline_state.pkl")
    
    conn.close()
    print("Feature engineering complete.")

if __name__ == "__main__":
    compute_features()
