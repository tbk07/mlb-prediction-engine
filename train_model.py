import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train():
    df = pd.read_csv("features.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    joblib.dump(model, "mlb_model.pkl")
    print("Model saved to mlb_model.pkl")

if __name__ == "__main__":
    train()
