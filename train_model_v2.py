import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib

def train():
    df = pd.read_csv("features_v2.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Simple split (could be improved with time-based split as in notebooks)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, verbose=-1, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log Loss: {log_loss(y_test, y_prob):.4f}")
    
    joblib.dump(model, "mlb_model_v2.pkl")
    print("Model saved to mlb_model_v2.pkl")

if __name__ == "__main__":
    train()
