# MLB Prediction Engine

A machine learning system for predicting MLB game outcomes and comparing them with Vegas lines.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download historical data:
   ```bash
   python data_fetcher.py
   ```

3. Engineer features and train model:
   ```bash
   python feature_engineer.py
   python train_model.py
   ```

4. Predict current games:
   ```bash
   python predictor.py
   ```

## Files
- `data_fetcher.py`: Downloads Retrosheet game logs.
- `feature_engineer.py`: Processes logs into rolling stats for teams and players.
- `train_model.py`: Trains a LightGBM classifier.
- `predictor.py`: Fetches today's games and generates win probabilities.
