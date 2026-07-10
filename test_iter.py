import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('mlb.db')
games_df = pd.read_sql("SELECT * FROM games ORDER BY date", conn)

print("Number of games:", len(games_df))
for i, row in games_df.iterrows():
    vp = row['v_p_starter']
    if pd.isna(vp):
        pass
    elif str(vp) == 'nan':
        print(f"Game {row['gamePk']}: vp={vp} (type {type(vp)}) is NOT pd.isna! str(vp) == 'nan'")

conn.close()
