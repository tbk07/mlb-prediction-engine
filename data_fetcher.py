import requests
import os
import zipfile
import io
import pandas as pd

DATA_DIR = "data"
RETROSHEET_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"

def download_retrosheet_data(years):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for year in years:
        file_path = f"{DATA_DIR}/gl{year}.txt"
        if os.path.exists(file_path):
            continue
            
        print(f"Downloading {year}...")
        r = requests.get(RETROSHEET_URL.format(year=year))
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(DATA_DIR)
        else:
            print(f"Failed to download {year}")

def load_data(years):
    columns = [
        0, 3, 6, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49,
        101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131,
        132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160
    ]
    
    all_data = []
    for year in years:
        path = f"{DATA_DIR}/gl{year}.txt"
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, low_memory=False)
            all_data.append(df)
            
    return pd.concat(all_data) if all_data else pd.DataFrame()

if __name__ == "__main__":
    years = list(range(2015, 2026))
    download_retrosheet_data(years)
