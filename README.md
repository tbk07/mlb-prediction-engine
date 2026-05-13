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
| Total games | 696 |
| Correct predictions | 384 |
| Incorrect predictions | 312 |
| **Accuracy** | **55.2%** |

> Baseline random accuracy is approximately 50%.  
> The model consistently outperforms the baseline by roughly 5 percentage points.

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
| Final balance | ₹7,300 |
| Net profit | ₹7,200 |
| Peak balance | ₹7,600 |
| Return on starting capital | 73× |

> The bankroll briefly dips below breakeven during the first ~10 games, so maintaining a larger buffer bankroll is recommended.

---

## Balance Over Time

The chart below shows bankroll growth across all 696 simulated bets.

[![MLB Betting Balance Chart](https://tbk07.github.io/MLB_CHART/chart.png)](https://tbk07.github.io/MLB_CHART/)

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

