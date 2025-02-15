import pandas as pd
import numpy as np
import glob
from collections import deque

def process_gamelogs():
    files = glob.glob("data/gl*.txt")
    cols = {
        0: 'date', 3: 'v_team', 6: 'h_team', 9: 'v_score', 10: 'h_score',
        101: 'v_pitcher', 103: 'h_pitcher'
    }
    for i in range(9):
        cols[105 + i*3] = f'v_hitter_{i+1}'
        cols[132 + i*3] = f'h_hitter_{i+1}'
    
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None, low_memory=False, usecols=list(cols.keys()))
        df = df.rename(columns=cols)
        dfs.append(df)
    
    df = pd.concat(dfs).sort_values('date')
    df['v_win'] = (df['v_score'] > df['h_score']).astype(int)
    
    team_wins = {} # team -> deque of (0/1)
    team_runs = {} # team -> deque of runs
    player_stats = {} # player -> deque of values
    
    T_WINDOW = 20
    P_WINDOW = 10

    def get_avg(d, default=0.0):
        return np.mean(d) if len(d) > 0 else default

    features = []
    for _, row in df.iterrows():
        v, h = row['v_team'], row['h_team']
        vp, hp = row['v_pitcher'], row['h_pitcher']
        
        v_win_pct = get_avg(team_wins.get(v, []), 0.5)
        v_avg_runs = get_avg(team_runs.get(v, []), 4.0)
        h_win_pct = get_avg(team_wins.get(h, []), 0.5)
        h_avg_runs = get_avg(team_runs.get(h, []), 4.0)
        
        v_p_stat = get_avg(player_stats.get(vp, []), -4.0)
        h_p_stat = get_avg(player_stats.get(hp, []), -4.0)
        
        v_l_stat = np.mean([get_avg(player_stats.get(row[f'v_hitter_{i+1}'], []), 0.5) for i in range(9)])
        h_l_stat = np.mean([get_avg(player_stats.get(row[f'h_hitter_{i+1}'], []), 0.5) for i in range(9)])

        features.append([
            v_win_pct, v_avg_runs, h_win_pct, h_avg_runs,
            v_p_stat, h_p_stat, v_l_stat, h_l_stat, row['v_win']
        ])
        
        # Update team stats
        for t, win, runs in [(v, row['v_win'], row['v_score']), (h, 1-row['v_win'], row['h_score'])]:
            if t not in team_wins: team_wins[t] = deque(maxlen=T_WINDOW)
            if t not in team_runs: team_runs[t] = deque(maxlen=T_WINDOW)
            team_wins[t].append(win)
            team_runs[t].append(runs)
            
        # Update pitcher stats (negative runs allowed)
        for p, val in [(vp, -row['h_score']), (hp, -row['v_score'])]:
            if p not in player_stats: player_stats[p] = deque(maxlen=P_WINDOW)
            player_stats[p].append(val)
            
        # Update hitter stats (runs / 9 as proxy)
        for i in range(9):
            v_h, h_h = row[f'v_hitter_{i+1}'], row[f'h_hitter_{i+1}']
            if v_h not in player_stats: player_stats[v_h] = deque(maxlen=P_WINDOW)
            player_stats[v_h].append(row['v_score']/9)
            if h_h not in player_stats: player_stats[h_h] = deque(maxlen=P_WINDOW)
            player_stats[h_h].append(row['h_score']/9)

    f_df = pd.DataFrame(features, columns=[
        'v_win_pct', 'v_runs', 'h_win_pct', 'h_runs', 
        'v_p_stat', 'h_p_stat', 'v_l_stat', 'h_l_stat', 'target'
    ])
    f_df.to_csv("features.csv", index=False)
    
    import joblib
    state = {
        'team_wins': team_wins,
        'team_runs': team_runs,
        'player_stats': player_stats
    }
    joblib.dump(state, "latest_stats.pkl")

if __name__ == "__main__":
    process_gamelogs()
