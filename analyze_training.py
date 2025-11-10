import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def analyze_v5():
    df = pd.read_csv("features_v5.csv")
    
    # 2024 games approx (last 2430 games * N years)
    # Let's just take the last 20%
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:]
    
    print(f"Historical Test Set (last 20%):")
    print(f"Home Win Rate: {test_df['target'].mean():.2%}")
    
    # Simple Elo strategy accuracy
    elo_acc = accuracy_score(test_df['target'], (test_df['exp_h'] > 0.5).astype(int))
    print(f"Historical Elo Accuracy: {elo_acc:.4f}")
    
    # Model accuracy
    model = joblib.load("mlb_model_v5.pkl")
    X = test_df.drop('target', axis=1)
    model_acc = accuracy_score(test_df['target'], model.predict(X))
    print(f"Historical Model Accuracy: {model_acc:.4f}")
    
    # Correlation with target
    corr = test_df.corr()['target'].sort_values(ascending=False)
    print("\nTop Correlations with Target:")
    print(corr.head(10))

if __name__ == "__main__":
    analyze_v5()
