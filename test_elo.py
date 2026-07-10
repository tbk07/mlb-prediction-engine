import sqlite3
import pandas as pd
import joblib

conn = sqlite3.connect('mlb.db')
df = pd.read_sql("SELECT * FROM game_features WHERE h_team='BOS' AND season=2026 ORDER BY date DESC LIMIT 1", conn)
print('game_features p_elo:', df['h_p_elo'].iloc[0])

state = joblib.load('pipeline_state.pkl')
print('team_starter_elo BOS:', state['team_starter_elo']['BOS'])
print('pitcher_elo nan:', state['pitcher_elo'].get('nan'))
print('pitcher_elo None:', state['pitcher_elo'].get('None'))
print('pitcher_elo null:', state['pitcher_elo'].get('null'))

games_df = pd.read_sql("SELECT * FROM games WHERE h_team='BOS' AND season=2026 ORDER BY date DESC LIMIT 1", conn)
hp = games_df.iloc[0]['h_p_starter']
print('hp:', hp, type(hp))

conn.close()
