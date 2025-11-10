import sqlite3
import pandas as pd
import glob
import requests
import json
from datetime import datetime, timedelta
import os

DB_PATH = 'mlb.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS games (
        gamePk TEXT PRIMARY KEY,
        date TEXT,
        season INTEGER,
        v_team TEXT,
        h_team TEXT,
        v_score INTEGER,
        h_score INTEGER,
        park_name TEXT,
        park_id TEXT,
        altitude_flag INTEGER,
        roof_type TEXT,
        temperature INTEGER,
        v_p_starter TEXT,
        h_p_starter TEXT,
        v_p_hand TEXT,
        h_p_hand TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pitcher_game_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gamePk TEXT,
        team TEXT,
        player_id TEXT,
        name TEXT,
        is_starter INTEGER,
        IP REAL,
        ER INTEGER,
        H INTEGER,
        BB INTEGER,
        K INTEGER,
        pitch_count INTEGER,
        UNIQUE(gamePk, player_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS batter_game_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gamePk TEXT,
        team TEXT,
        player_id TEXT,
        name TEXT,
        is_starter INTEGER,
        AB INTEGER,
        H INTEGER,
        HR INTEGER,
        RBI INTEGER,
        BB INTEGER,
        SO INTEGER,
        UNIQUE(gamePk, player_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS odds (
        gamePk TEXT PRIMARY KEY,
        home_ml INTEGER,
        away_ml INTEGER,
        over_under REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        gamePk TEXT PRIMARY KEY,
        model_version TEXT,
        h_win_prob REAL,
        v_win_prob REAL,
        prediction INTEGER,
        actual INTEGER,
        edge REAL
    )
    ''')

    conn.commit()
    conn.close()

def import_retrosheet():
    print("Importing Retrosheet data to DB...")
    conn = sqlite3.connect(DB_PATH)
    files = glob.glob("../data/gl*.txt")
    if not files:
        files = glob.glob("data/gl*.txt")

    cols = {
        0: 'date', 3: 'v_team', 6: 'h_team', 9: 'v_score', 10: 'h_score',
        16: 'park_id', 101: 'v_pitcher', 102: 'v_p_hand', 103: 'h_pitcher', 104: 'h_p_hand'
    }
    
    games_data = []
    for f in files:
        df = pd.read_csv(f, header=None, low_memory=False, usecols=list(cols.keys()))
        df = df.rename(columns=cols)
        
        # Determine season from filename
        year = int(f.split('gl')[-1][:4])
        
        for _, row in df.iterrows():
            date_str = str(row['date'])
            date_fmtd = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            gamePk = f"retro_{row['v_team']}_{row['h_team']}_{date_str}"
            
            # Simple altitude and roof mocks
            altitude = 1 if row['park_id'] == 'DEN02' else 0
            roof = 'open' # Simplified
            
            games_data.append((
                gamePk, date_fmtd, year, row['v_team'], row['h_team'], 
                row['v_score'], row['h_score'], row['park_id'], row['park_id'],
                altitude, roof, 70, row['v_pitcher'], row['h_pitcher'], 
                row['v_p_hand'], row['h_p_hand']
            ))

    cursor = conn.cursor()
    cursor.executemany('''
    INSERT OR IGNORE INTO games 
    (gamePk, date, season, v_team, h_team, v_score, h_score, park_name, park_id, altitude_flag, roof_type, temperature, v_p_starter, h_p_starter, v_p_hand, h_p_hand)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', games_data)
    conn.commit()
    conn.close()
    print(f"Imported {len(games_data)} Retrosheet games.")

def fetch_mlb_api_2026():
    print("Fetching 2026 games from MLB API...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    start_date = "2026-03-20"
    end_date = "2026-11-01"
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}&hydrate=probablePitcher"
    resp = requests.get(url).json()
    
    mapping = {"LAD": "LAN", "SF": "SFN", "NYM": "NYN", "NYY": "NYA", "CHC": "CHN", "CWS": "CHA", "KC": "KCA", "TB": "TBA", "SD": "SDN", "WSH": "WAS"}

    for date in resp.get('dates', []):
        for game in date.get('games', []):
            if game['status']['abstractGameState'] != 'Final': continue
            
            gamePk = str(game['gamePk'])
            cursor.execute("SELECT 1 FROM games WHERE gamePk=?", (gamePk,))
            if cursor.fetchone(): continue
            
            away = game['teams']['away']
            home = game['teams']['home']
            v_team = mapping.get(away['team'].get('abbreviation', 'UNK'), away['team'].get('abbreviation', 'UNK'))
            h_team = mapping.get(home['team'].get('abbreviation', 'UNK'), home['team'].get('abbreviation', 'UNK'))
            
            v_score = away.get('score', 0)
            h_score = home.get('score', 0)
            
            park_name = game.get('venue', {}).get('name', 'UNK')
            park_id = str(game.get('venue', {}).get('id', 'UNK'))
            altitude = 1 if 'Coors' in park_name else 0
            
            v_p = away.get('probablePitcher', {}).get('id', 'UNK')
            h_p = home.get('probablePitcher', {}).get('id', 'UNK')
            
            # Fetch Boxscore
            bs_url = f"https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"
            bs_resp = requests.get(bs_url).json()
            
            v_p_hand = 'R'
            h_p_hand = 'R'
            
            cursor.execute('''
            INSERT INTO games 
            (gamePk, date, season, v_team, h_team, v_score, h_score, park_name, park_id, altitude_flag, roof_type, temperature, v_p_starter, h_p_starter, v_p_hand, h_p_hand)
            VALUES (?, ?, 2026, ?, ?, ?, ?, ?, ?, ?, 'open', 70, ?, ?, ?, ?)
            ''', (gamePk, date['date'], v_team, h_team, v_score, h_score, park_name, park_id, altitude, str(v_p), str(h_p), v_p_hand, h_p_hand))
            
            # Extract player stats
            teams_data = {'away': v_team, 'home': h_team}
            for t_side, t_abbr in teams_data.items():
                box = bs_resp.get('teams', {}).get(t_side, {})
                players = box.get('players', {})
                for p_key, p_data in players.items():
                    p_id = p_data.get('person', {}).get('id')
                    p_name = p_data.get('person', {}).get('fullName')
                    stats = p_data.get('stats', {})
                    pos = p_data.get('position', {}).get('abbreviation')
                    
                    # Pitching
                    p_stats = stats.get('pitching', {})
                    if p_stats and float(p_stats.get('inningsPitched', 0.0)) > 0:
                        ip = float(p_stats.get('inningsPitched', 0.0))
                        is_starter = 1 if str(p_id) == str(v_p) or str(p_id) == str(h_p) else 0
                        cursor.execute('''
                        INSERT OR IGNORE INTO pitcher_game_stats
                        (gamePk, team, player_id, name, is_starter, IP, ER, H, BB, K, pitch_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (gamePk, t_abbr, str(p_id), p_name, is_starter, ip, p_stats.get('earnedRuns', 0), p_stats.get('hits', 0), p_stats.get('baseOnBalls', 0), p_stats.get('strikeOuts', 0), p_stats.get('numberOfPitches', 0)))
                    
                    # Batting
                    b_stats = stats.get('batting', {})
                    if b_stats and b_stats.get('atBats', 0) > 0:
                        cursor.execute('''
                        INSERT OR IGNORE INTO batter_game_stats
                        (gamePk, team, player_id, name, is_starter, AB, H, HR, RBI, BB, SO)
                        VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                        ''', (gamePk, t_abbr, str(p_id), p_name, b_stats.get('atBats', 0), b_stats.get('hits', 0), b_stats.get('homeRuns', 0), b_stats.get('rbi', 0), b_stats.get('baseOnBalls', 0), b_stats.get('strikeOuts', 0)))
                        
            conn.commit()
    conn.close()

def fetch_odds():
    print("Mocking Odds API fetch...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT gamePk FROM games WHERE season = 2026")
    games = cursor.fetchall()
    
    for g in games:
        cursor.execute("INSERT OR IGNORE INTO odds (gamePk, home_ml, away_ml, over_under) VALUES (?, -110, -110, 8.5)", (g[0],))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    import_retrosheet()
    fetch_mlb_api_2026()
    fetch_odds()
    print("Data pipeline complete.")
