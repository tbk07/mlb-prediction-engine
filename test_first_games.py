import sqlite3
import pandas as pd
import joblib

conn = sqlite3.connect('mlb.db')
df = pd.read_sql("SELECT h_team, h_p_elo, date FROM game_features WHERE season=2026 ORDER BY date ASC LIMIT 10", conn)
print(df)
conn.close()
