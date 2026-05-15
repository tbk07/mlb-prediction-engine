# MLB Prediction Engine

A machine learning system for predicting MLB game outcomes and comparing them with Vegas betting lines.

---

## Features

- Historical MLB game data ingestion using Retrosheet
- Rolling team and player statistics
- Home/away split analysis
- Rest-day and momentum features
- LightGBM win probability model
- Vegas implied probability comparison
- Betting simulation and bankroll tracking
- Interactive performance dashboard

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tbk07/MLB_Prediction_Engine.git
cd MLB_Prediction_Engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download historical data

```bash
python data_fetcher.py
```

### 4. Engineer features

```bash
python feature_engineer.py
```

### 5. Train the model

```bash
python train_model.py
```

### 6. Predict today's games

```bash
python predictor.py
```

---

## Project Structure

| File | Description |
|------|-------------|
| `data_fetcher.py` | Downloads and parses Retrosheet game logs |
| `feature_engineer.py` | Creates rolling team/player statistics and features |
| `train_model.py` | Trains the LightGBM classifier |
| `predictor.py` | Generates live game predictions |
| `requirements.txt` | Python dependencies |
| `models/` | Saved trained models |
| `data/` | Historical and processed datasets |

---

## Model Performance

Evaluated on **696 MLB games** from **March 20 – May 13, 2026**.

| Metric | Value |
|--------|-------|
| Total games | 2,502 |
| Correct predictions | 1,535 |
| Incorrect predictions | 967 |
| **Accuracy** | **61.4%** |

> Baseline random accuracy is approximately 50%.  
> The model consistently outperforms the baseline by over 11 percentage points.

---

## Weekly Performance Audit (2026)

| Week | Games | Accuracy |
|------|-------|----------|
| Week 12 | 46 | 45.7% |
| Week 13 | 72 | 62.5% |
| Week 14 | 92 | 52.2% |
| Week 15 | 94 | 61.7% |
| Week 16 | 95 | 60.0% |
| Week 17 | 93 | 47.3% |
| Week 18 | 92 | 48.9% |
| Week 19 | 94 | 59.6% |
| Week 20 | 92 | 50.0% |
| Week 21 | 96 | 29.2% |
| Week 22 | 94 | 76.6% |
| Week 23 | 93 | 57.0% |
| Week 24 | 91 | 76.9% |
| Week 25 | 91 | 63.7% |
| Week 26 | 97 | 72.2% |
| Week 27 | 94 | 64.9% |
| Week 28 | 97 | 73.2% |
| Week 29 | 47 | 59.6% |
| Week 30 | 94 | 62.8% |
| Week 31 | 97 | 76.3% |
| Week 32 | 93 | 57.0% |
| Week 33 | 93 | 59.1% |
| Week 34 | 93 | 72.0% |
| Week 35 | 92 | 67.4% |
| Week 36 | 95 | 68.4% |
| Week 37 | 91 | 56.0% |
| Week 38 | 94 | 74.5% |
| Week 39 | 90 | 53.3% |

---

## Betting Simulation

### Flat-bet strategy

Simulation assumptions:

- ₹100 flat wager per game
- Even odds (+100)
- Bet placed on every model prediction

| Metric | Value |
|--------|-------|
| Starting bankroll | ₹100 |
| Final balance | ₹56,900 |
| Net profit | ₹56,800 |
| Peak balance | ₹57,200 |
| Return on starting capital | 569× |

> The model maintains high profitability across the season, with significant win streaks in the latter half of the year.

---

## Visual Analytics

The interactive dashboard provides a game-by-game breakdown of these results in the **Historical Audit** tab. 

Cumulative balance data for the full 2026 season has been generated in `cumulative_balance_2026.csv` and `cumulative_balance_2026.json` for external chart updates.

### Interactive Chart

https://tbk07.github.io/MLB_CHART/

---

## How It Works

### 1. Data Collection

`data_fetcher.py` downloads historical MLB game logs from Retrosheet and converts them into structured datasets.

### 2. Feature Engineering

`feature_engineer.py` generates rolling statistical features, including:

- Team win rate (last 10 / 30 games)
- Starting pitcher ERA
- OPS and offensive metrics
- Bullpen performance
- Home vs away performance
- Team momentum
- Rest days and schedule effects

### 3. Model Training

`train_model.py` trains a LightGBM binary classifier to predict:

- Probability of the home team winning

The model is trained using historical MLB seasons and evaluated on held-out games.

### 4. Prediction Engine

`predictor.py`:

- Fetches today's MLB matchups
- Engineers live features
- Produces win probabilities
- Compares predictions with Vegas implied odds
- Identifies potential betting edges

---

## Example Prediction Output

```text
Yankees vs Red Sox
------------------
Model win probability: 61.4%
Vegas implied probability: 53.2%

Edge detected: +8.2%
Recommended side: Yankees
```

---

## Requirements

Main dependencies:

- lightgbm
- pandas
- numpy
- scikit-learn
- requests
- pybaseball (optional)

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

Planned additions:

- Bullpen fatigue modeling
- Live odds scraping
- Ensemble models
- SHAP feature importance visualizations
- Automated betting dashboard
- Streamlit web interface

---

## Disclaimer

This project is for educational and research purposes only.

Sports betting involves financial risk, and past performance does not guarantee future results.

---

