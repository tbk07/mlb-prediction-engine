import sqlite3
import pandas as pd
import numpy as np
from collections import deque
import json
import os
import joblib

DB_PATH = 'mlb.db'

def get_dist(p1, p2):
    if not p1 or not p2: return 0
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 69 # Approx miles

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

def get_p_feats(p, pitcher_stats, date_dt):
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
            ip_mod = max(ip, 0.1)
            res[f'{w}_era'] = (er / ip_mod) * 9
            res[f'{w}_whip'] = (h_allow + bb) / ip_mod
            res[f'{w}_k9'] = (k / ip_mod) * 9
            res[f'{w}_bb9'] = (bb / ip_mod) * 9
    
    # Days rest & pitch count
    if hist:
        last = hist[-1]
        res['days_rest'] = min(30, (date_dt - last['date']).days)
        res['last_pitch_count'] = last.get('pc', 90)
    else:
        res['days_rest'] = 5
        res['last_pitch_count'] = 90
    return res

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
    batter_elo_lhp = {}
    batter_elo_rhp = {}
    bullpen_elo = {} # team -> elo
    
    # Team-level Elo fallbacks to ensure unique ratings
    team_off_elo = {}
    team_starter_elo = {}
    
    # Rolling State
    team_stats = {} # team -> deque
    team_home_stats = {}
    team_away_stats = {}
    team_last_city = {} # team -> (lat, lon)
    team_last_was_home = {} # team -> bool
    pitcher_stats = {} # pitcher -> deque
    park_runs = {} # park_id -> deque
    league_runs = deque(maxlen=1000)
    
    # Park Coordinates (Approximate for distance calculation)
    PARK_COORDS = {
        "ANA": (33.8003, -117.8827), "ARI": (33.4455, -112.0667), "ATL": (33.8907, -84.4676),
        "BAL": (39.284, -76.6215), "BOS": (42.3467, -71.0972), "CHA": (41.8299, -87.6339),
        "CHN": (41.9484, -87.6553), "CIN": (39.0979, -84.5081), "CLE": (41.4958, -81.6853),
        "COL": (39.7559, -104.9942), "DET": (42.339, -83.0485), "HOU": (29.7573, -95.3555),
        "KCA": (39.0517, -94.4803), "LAN": (34.0739, -118.24), "MIA": (25.7783, -80.2197),
        "MIL": (43.0284, -87.9712), "MIN": (44.9817, -93.2778), "NYA": (40.8296, -73.9262),
        "NYN": (40.7571, -73.8458), "OAK": (37.7516, -122.2005), "PHI": (39.9061, -75.1665),
        "PIT": (40.4473, -80.0057), "SDN": (32.7076, -117.157), "SEA": (47.5914, -122.3323),
        "SFN": (37.7786, -122.3893), "SLN": (38.6226, -90.1928), "TBA": (27.7682, -82.6534),
        "TEX": (32.7512, -97.0825), "TOR": (43.6414, -79.3894), "WAS": (38.873, -77.0074)
    }

    K_PITCHER = 32
    K_BATTER = 20
    K_TEAM = 16
    HFA = 35
    DEFAULT_ELO = 1500
    
    def get_p_elo(p, team): 
        if p is None: return team_starter_elo.get(team, DEFAULT_ELO)
        return pitcher_elo.get(str(p), team_starter_elo.get(team, DEFAULT_ELO))
    def get_b_elo(b, hand): 
        if b is None: return DEFAULT_ELO
        if hand == 'L': return batter_elo_lhp.get(str(b), DEFAULT_ELO)
        return batter_elo_rhp.get(str(b), DEFAULT_ELO)
    def get_bp_elo(t): 
        if t is None or t == 'UNK': return DEFAULT_ELO
        return bullpen_elo.get(t, DEFAULT_ELO)
    
    features_list = []
    
    current_season = None
    print(f"Processing {len(games_df)} games...")
    
    for i, row in games_df.iterrows():
        gamePk = row['gamePk']
        v = row['v_team']
        h = row['h_team']
        vp = row['v_p_starter']
        hp = row['h_p_starter']
        v_hand = row['v_p_hand']
        h_hand = row['h_p_hand']
        season = row['season']
        
        # --- Season Regression ---
        if current_season is not None and season != current_season:
            # Regress 50% towards 1500 between seasons to keep ratings grounded
            for p in pitcher_elo: pitcher_elo[p] = 0.5 * pitcher_elo[p] + 0.5 * 1500
            for b in batter_elo_lhp: batter_elo_lhp[b] = 0.5 * batter_elo_lhp[b] + 0.5 * 1500
            for b in batter_elo_rhp: batter_elo_rhp[b] = 0.5 * batter_elo_rhp[b] + 0.5 * 1500
            for t in bullpen_elo: bullpen_elo[t] = 0.5 * bullpen_elo[t] + 0.5 * 1500
            for t in team_off_elo: team_off_elo[t] = 0.5 * team_off_elo[t] + 0.5 * 1500
            for t in team_starter_elo: team_starter_elo[t] = 0.5 * team_starter_elo[t] + 0.5 * 1500
        current_season = season

        date_str = str(row['date'])
        if len(date_str) < 10:
            date_dt = pd.to_datetime(date_str, format="%Y%m%d")
        else:
            date_dt = pd.to_datetime(date_str)
        
        month = date_dt.month
        
        # Elos
        v_p_elo = get_p_elo(vp, v)
        h_p_elo = get_p_elo(hp, h)
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
            
        v_off_elo = np.mean([get_b_elo(b, h_hand) for b in v_batters]) if v_batters else team_off_elo.get(v, DEFAULT_ELO)
        h_off_elo = np.mean([get_b_elo(b, v_hand) for b in h_batters]) if h_batters else team_off_elo.get(h, DEFAULT_ELO)
        
        v_def_elo = 0.6 * v_p_elo + 0.4 * v_bp_elo
        h_def_elo = 0.6 * h_p_elo + 0.4 * h_bp_elo
        
        v_comp_elo = 0.5 * v_off_elo + 0.5 * v_def_elo
        h_comp_elo = 0.5 * h_off_elo + 0.5 * h_def_elo
        
        exp_h = 1 / (10**(-(h_comp_elo + HFA - v_comp_elo) / 400) + 1)
        
        # Travel & Fatigue
        h_coord = PARK_COORDS.get(h)
        v_dist = get_dist(team_last_city.get(v), h_coord)
        h_dist = get_dist(team_last_city.get(h), h_coord)
        
        # Back home correctly
        h_back_home = 1 if (team_last_was_home.get(h) == False) else 0 # h is home. If h's last was away, h_back_home = 1.
        
        vf, hf = get_t_feats(v, team_stats), get_t_feats(h, team_stats)
        vpf, hpf = get_p_feats(vp, pitcher_stats, date_dt), get_p_feats(hp, pitcher_stats, date_dt)
        
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
        
        # Check if game is completed
        status = row.get('status', 'Final')
        is_completed = (status == 'Final') or (row['v_score'] + row['h_score'] > 0 and status != 'Preview')
        
        # Save features
        feat_row = {
            'gamePk': gamePk,
            'season': row['season'],
            'date': date_str,
            'target': (1 if row['h_score'] > row['v_score'] else 0) if is_completed else None,
            'h_team': h,
            'v_team': v,
            'v_comp_elo': v_comp_elo, 'h_comp_elo': h_comp_elo,
            'v_off_elo': v_off_elo, 'h_off_elo': h_off_elo,
            'v_def_elo': v_def_elo, 'h_def_elo': h_def_elo,
            'v_p_elo': v_p_elo, 'h_p_elo': h_p_elo,
            'v_bp_elo': v_bp_elo, 'h_bp_elo': h_bp_elo,
            'exp_h': exp_h,
            'implied_h_prob': implied_h_prob,
            'over_under': ou,
            'park_factor': park_factor,
            'altitude_flag': row['altitude_flag'],
            'temp': row.get('temperature', 70),
            'month': month,
            'v_games_7d': v_games_7d, 'h_games_7d': h_games_7d,
            'v_rest': v_rest, 'h_rest': h_rest,
            'v_dist': v_dist, 'h_dist': h_dist,
            'h_back_home': h_back_home,
            'v_p_lhp': v_p_lhp, 'h_p_lhp': h_p_lhp,
            'v_p_days_rest': vpf['days_rest'], 'h_p_days_rest': hpf['days_rest'],
            'v_p_last_pc': vpf['last_pitch_count'], 'h_p_last_pc': hpf['last_pitch_count']
        }
        
        # Win Rate windows
        for k, v_val in vf.items(): feat_row[f'v_{k}'] = v_val
        for k, h_val in hf.items(): feat_row[f'h_{k}'] = h_val
        # Pitcher windows
        for k, v_val in vpf.items(): 
            if k not in ['days_rest', 'last_pitch_count']: feat_row[f'vp_{k}'] = v_val
        for k, h_val in hpf.items(): 
            if k not in ['days_rest', 'last_pitch_count']: feat_row[f'hp_{k}'] = h_val
            
        features_list.append(feat_row)
        
        # --- ELO & STATE UPDATES (Only for Completed Games) ---
        if is_completed:
            h_win = 1 if row['h_score'] > row['v_score'] else 0
            
            # Get actual stats from DB
            v_p_stats = pitcher_games.get_group(gamePk)[pitcher_games.get_group(gamePk)['player_id'] == vp] if gamePk in pitcher_games.groups else pd.DataFrame()
            h_p_stats = pitcher_games.get_group(gamePk)[pitcher_games.get_group(gamePk)['player_id'] == hp] if gamePk in pitcher_games.groups else pd.DataFrame()
            
            v_er = v_p_stats['ER'].iloc[0] if not v_p_stats.empty else row['h_score']
            h_er = h_p_stats['ER'].iloc[0] if not h_p_stats.empty else row['v_score']
            
            v_p_s = max(0, min(1, 0.5 + (4.5 - v_er)/9))
            h_p_s = max(0, min(1, 0.5 + (4.5 - h_er)/9))
            
            # Individual Pitcher Elo
            if vp is not None:
                pitcher_elo[str(vp)] = v_p_elo + K_PITCHER * (v_p_s - (1 - exp_h))
            if hp is not None:
                pitcher_elo[str(hp)] = h_p_elo + K_PITCHER * (h_p_s - exp_h)
            
            # Team Starter Elo (Rotation average)
            if v != 'UNK':
                team_starter_elo[v] = team_starter_elo.get(v, DEFAULT_ELO) + (K_PITCHER/5) * (v_p_s - (1 - exp_h))
            if h != 'UNK':
                team_starter_elo[h] = team_starter_elo.get(h, DEFAULT_ELO) + (K_PITCHER/5) * (h_p_s - exp_h)

            # Bullpen Elo Update
            v_bp_s = max(0, min(1, 0.5 + (4.5 - (row['h_score'] - v_er))/9))
            h_bp_s = max(0, min(1, 0.5 + (4.5 - (row['v_score'] - h_er))/9))
            if v != 'UNK':
                bullpen_elo[v] = v_bp_elo + K_TEAM * (v_bp_s - (1 - exp_h))
            if h != 'UNK':
                bullpen_elo[h] = h_bp_elo + K_TEAM * (h_bp_s - exp_h)
            
            # Batter Elo
            if v_batters:
                v_off_s = max(0, min(1, 0.5 + (row['v_score'] - 4.5)/9))
                for b in v_batters:
                    cur_elo = get_b_elo(b, h_hand)
                    new_elo = cur_elo + K_BATTER * (v_off_s - (1 - exp_h))
                    if h_hand == 'L': batter_elo_lhp[str(b)] = new_elo
                    else: batter_elo_rhp[str(b)] = new_elo
            
            if h_batters:
                h_off_s = max(0, min(1, 0.5 + (row['h_score'] - 4.5)/9))
                for b in h_batters:
                    cur_elo = get_b_elo(b, v_hand)
                    new_elo = cur_elo + K_BATTER * (h_off_s - exp_h)
                    if v_hand == 'L': batter_elo_lhp[str(b)] = new_elo
                    else: batter_elo_rhp[str(b)] = new_elo
            
            # Team Offense Elo (Fallback)
            if v != 'UNK':
                v_off_s = max(0, min(1, 0.5 + (row['v_score'] - 4.5)/9))
                team_off_elo[v] = team_off_elo.get(v, DEFAULT_ELO) + K_TEAM * (v_off_s - (1 - exp_h))
            if h != 'UNK':
                h_off_s = max(0, min(1, 0.5 + (row['h_score'] - 4.5)/9))
                team_off_elo[h] = team_off_elo.get(h, DEFAULT_ELO) + K_TEAM * (h_off_s - exp_h)
                
            # Update rolling stats
            def get_team_stat_inner(team, rs, ra, pk):
                bat = batter_games.get_group(pk)[batter_games.get_group(pk)['team'] == team] if pk in batter_games.groups else pd.DataFrame()
                if bat.empty:
                    return {'date': date_dt, 'win': 1 if rs > ra else 0, 'rs': rs, 'ra': ra, 'err': 0, 'ab': 33, 'h': 8, '2b': 2, '3b': 0, 'hr': 1, 'bb': 3, 'sb': 0, 'cs': 0}
                return {
                    'date': date_dt, 'win': 1 if rs > ra else 0, 'rs': rs, 'ra': ra, 'err': 0,
                    'ab': bat['AB'].sum(), 'h': bat['H'].sum(), '2b': 0, '3b': 0, 'hr': bat['HR'].sum(), 'bb': bat['BB'].sum(), 'sb': 0, 'cs': 0
                }
            
            v_game_stat = get_team_stat_inner(v, row['v_score'], row['h_score'], gamePk)
            h_game_stat = get_team_stat_inner(h, row['h_score'], row['v_score'], gamePk)
            
            if v not in team_stats: team_stats[v] = deque(maxlen=162)
            if h not in team_stats: team_stats[h] = deque(maxlen=162)
            team_stats[v].append(v_game_stat)
            team_stats[h].append(h_game_stat)
            
            if v not in team_away_stats: team_away_stats[v] = deque(maxlen=162)
            if h not in team_home_stats: team_home_stats[h] = deque(maxlen=162)
            team_away_stats[v].append(v_game_stat)
            team_home_stats[h].append(h_game_stat)
            
            # Update last city & home/away status
            team_last_city[v] = h_coord
            team_last_city[h] = h_coord
            team_last_was_home[v] = False
            team_last_was_home[h] = True
            
            if not pd.isna(vp) and vp is not None:
                if str(vp) not in pitcher_stats: pitcher_stats[str(vp)] = deque(maxlen=35)
                v_p_stat = {'date': date_dt, 'er': v_er, 'h': v_p_stats['H'].iloc[0] if not v_p_stats.empty else 6, 'bb': v_p_stats['BB'].iloc[0] if not v_p_stats.empty else 2, 'k': v_p_stats['K'].iloc[0] if not v_p_stats.empty else 5, 'ip': v_p_stats['IP'].iloc[0] if not v_p_stats.empty else 6.0, 'pc': v_p_stats['pitch_count'].iloc[0] if not v_p_stats.empty else 90}
                pitcher_stats[str(vp)].append(v_p_stat)
            
            if not pd.isna(hp) and hp is not None:
                if str(hp) not in pitcher_stats: pitcher_stats[str(hp)] = deque(maxlen=35)
                h_p_stat = {'date': date_dt, 'er': h_er, 'h': h_p_stats['H'].iloc[0] if not h_p_stats.empty else 6, 'bb': h_p_stats['BB'].iloc[0] if not h_p_stats.empty else 2, 'k': h_p_stats['K'].iloc[0] if not h_p_stats.empty else 5, 'ip': h_p_stats['IP'].iloc[0] if not h_p_stats.empty else 6.0, 'pc': h_p_stats['pitch_count'].iloc[0] if not h_p_stats.empty else 90}
                pitcher_stats[str(hp)].append(h_p_stat)
            
            if p_id not in park_runs: park_runs[p_id] = deque(maxlen=100)
            park_runs[p_id].append(row['v_score'] + row['h_score'])
            league_runs.append(row['v_score'] + row['h_score'])

    print("Saving features to database...")
    features_df = pd.DataFrame(features_list)
    features_df.to_sql('game_features', conn, if_exists='replace', index=False)
    
    # Save the final state for future inference
    joblib.dump({
        'pitcher_elo': pitcher_elo,
        'batter_elo_lhp': batter_elo_lhp,
        'batter_elo_rhp': batter_elo_rhp,
        'bullpen_elo': bullpen_elo,
        'team_off_elo': team_off_elo,
        'team_starter_elo': team_starter_elo,
        'team_stats': team_stats,
        'team_home_stats': team_home_stats,
        'team_away_stats': team_away_stats,
        'team_last_city': team_last_city,
        'team_last_was_home': team_last_was_home,
        'pitcher_stats': pitcher_stats,
        'park_runs': park_runs,
        'league_runs': league_runs
    }, "pipeline_state.pkl")
    
    conn.close()
    print("Feature engineering complete.")
    print(f"Sample team starter Elos: {list(team_starter_elo.items())[:5]}")

if __name__ == "__main__":
    compute_features()
