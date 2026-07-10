import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('mlb.db')
games_df = pd.read_sql("SELECT * FROM games ORDER BY date", conn)

team_starter_elo = {}
pitcher_elo = {}
DEFAULT_ELO = 1500

current_season = None

for i, row in games_df.iterrows():
    v = row['v_team']
    h = row['h_team']
    season = row['season']
    
    if current_season is not None and season != current_season:
        for t in team_starter_elo: team_starter_elo[t] = 0.5 * team_starter_elo[t] + 0.5 * 1500
    current_season = season

    if season == 2026 and row['date'] == '2026-09-27' and h == 'BOS':
        print(f"2026-09-27 BOS team_starter_elo: {team_starter_elo.get('BOS', DEFAULT_ELO)}")

    # simplified update
    status = row.get('status', 'Final')
    is_completed = (status == 'Final') or (row['v_score'] + row['h_score'] > 0 and status != 'Preview')
    
    if is_completed:
        # just assume some arbitrary update for testing
        if v != 'UNK': team_starter_elo[v] = team_starter_elo.get(v, DEFAULT_ELO) + 1
        if h != 'UNK': team_starter_elo[h] = team_starter_elo.get(h, DEFAULT_ELO) + 1

print(f"Final BOS team_starter_elo: {team_starter_elo.get('BOS', DEFAULT_ELO)}")

conn.close()
