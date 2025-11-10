import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import matplotlib.pyplot as plt

def run_backtest():
    df = pd.read_csv("features_v3.csv")
    
    # In Retrosheet logs, dates are YYYYMMDD
    # Let's split by year. 2024 as test set.
    # Assuming the first 80% is training and last 20% is testing for a quick check
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    print(f"\n--- Backtest Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n--- Top Features ---")
    print(importance.head(10))
    
    return acc

if __name__ == "__main__":
    run_backtest()
