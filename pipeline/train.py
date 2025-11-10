import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'mlb.db'

def get_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM game_features", conn)
    conn.close()
    # Ensure ordered by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def get_splits(df):
    splits = {}
    
    # Split A: Train 2015-2022, Val 2023-2024, Test 2026
    train_A = df[df['season'] <= 2022]
    val_A = df[df['season'].isin([2023, 2024])]
    test = df[df['season'] == 2026]
    splits['A'] = (train_A, val_A, test)
    
    # Split B: Train 2015-2024 (excluding last 20% of 2024), Val last 20% 2024
    df_2024 = df[df['season'] == 2024]
    val_B_size = int(len(df_2024) * 0.2)
    val_B = df_2024.iloc[-val_B_size:] if val_B_size > 0 else df_2024
    train_B = df[((df['season'] >= 2015) & (df['season'] < 2024)) | ((df['season'] == 2024) & (~df.index.isin(val_B.index)))]
    splits['B'] = (train_B, val_B, test)
    
    # Split C: Train 2023-2024, Test 2026
    train_C = df[df['season'].isin([2023, 2024])]
    val_C_size = int(len(train_C) * 0.2)
    val_C = train_C.iloc[-val_C_size:] if val_C_size > 0 else train_C
    train_C = train_C.iloc[:-val_C_size] if val_C_size > 0 else train_C
    splits['C'] = (train_C, val_C, test)
    
    # Split D: Train 2025, Test 2026
    train_D = df[df['season'] == 2025]
    val_D_size = int(len(train_D) * 0.2)
    val_D = train_D.iloc[-val_D_size:] if val_D_size > 0 else train_D
    train_D = train_D.iloc[:-val_D_size] if val_D_size > 0 else train_D
    splits['D'] = (train_D, val_D, test)
    
    return splits

def calc_roi(y_true, y_prob, odds=-110):
    # Bet $100 on home if prob > 0.55
    bets = 0
    wins = 0
    profit = 0
    payout = 100 * (100 / abs(odds)) if odds < 0 else 100 * (odds / 100)
    for p, t in zip(y_prob, y_true):
        if p > 0.55:
            bets += 1
            if t == 1:
                profit += payout
            else:
                profit -= 100
    if bets == 0: return 0.0
    return profit

def optimize_lgbm(X_tr, y_tr, X_v, y_v):
    def obj(trial):
        params = {
            'n_estimators': 100,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'verbosity': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        return log_loss(y_v, model.predict_proba(X_v))
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=5)
    best = lgb.LGBMClassifier(**study.best_params, verbosity=-1)
    best.fit(X_tr, y_tr)
    return best

def optimize_xgb(X_tr, y_tr, X_v, y_v):
    def obj(trial):
        params = {
            'n_estimators': 100,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        return log_loss(y_v, model.predict_proba(X_v))
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=5)
    best = xgb.XGBClassifier(**study.best_params, verbosity=0)
    best.fit(X_tr, y_tr)
    return best

def optimize_rf(X_tr, y_tr, X_v, y_v):
    def obj(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 15)
        }
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_tr, y_tr)
        return log_loss(y_v, model.predict_proba(X_v))
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=5)
    best = RandomForestClassifier(**study.best_params, random_state=42)
    best.fit(X_tr, y_tr)
    return best

def optimize_cat(X_tr, y_tr, X_v, y_v):
    def obj(trial):
        params = {
            'iterations': 100,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'verbose': 0
        }
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr)
        return log_loss(y_v, model.predict_proba(X_v))
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=5)
    best = CatBoostClassifier(**study.best_params, verbose=0)
    best.fit(X_tr, y_tr)
    return best

def train_lr(X_tr, y_tr):
    model = make_pipeline(PolynomialFeatures(2), LogisticRegression(max_iter=100))
    model.fit(X_tr, y_tr)
    return model

def train_mlp(X_tr, y_tr):
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=100)
    model.fit(X_tr, y_tr)
    return model

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = get_data()
    drop_cols = ['gamePk', 'season', 'date', 'target', 'h_team', 'v_team']
    
    # Fill NAs
    df = df.fillna(df.mean(numeric_only=True))
    
    splits = get_splits(df)
    results = []
    
    best_loss = float('inf')
    best_model = None
    
    print("Running experiments...")
    for split_name, (train, val, test) in splits.items():
        if len(train) == 0 or len(test) == 0: continue
        
        X_tr = train.drop(columns=drop_cols)
        y_tr = train['target']
        X_v = val.drop(columns=drop_cols)
        y_v = val['target']
        X_te = test.drop(columns=drop_cols)
        y_te = test['target']
        
        print(f"--- Split {split_name} ---")
        
        models = {}
        print("Training LightGBM...")
        models['LightGBM'] = optimize_lgbm(X_tr, y_tr, X_v, y_v)
        print("Training XGBoost...")
        models['XGBoost'] = optimize_xgb(X_tr, y_tr, X_v, y_v)
        print("Training Random Forest...")
        models['RandomForest'] = optimize_rf(X_tr, y_tr, X_v, y_v)
        print("Training CatBoost...")
        models['CatBoost'] = optimize_cat(X_tr, y_tr, X_v, y_v)
        print("Training Logistic Regression...")
        models['LogReg'] = train_lr(X_tr, y_tr)
        print("Training Neural Network...")
        models['MLP'] = train_mlp(X_tr, y_tr)
        
        # Ensembles
        estimators = [('lgb', models['LightGBM']), ('xgb', models['XGBoost']), ('cat', models['CatBoost'])]
        print("Training Stacking Ensemble...")
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        stack.fit(X_tr, y_tr)
        models['Stacking'] = stack
        
        print("Training Soft Voting Ensemble...")
        vote = VotingClassifier(estimators=estimators, voting='soft')
        vote.fit(X_tr, y_tr)
        models['SoftVoting'] = vote
        
        for name, m in models.items():
            tr_pred = m.predict(X_tr)
            te_pred = m.predict(X_te)
            te_prob = m.predict_proba(X_te)[:, 1]
            
            tr_acc = accuracy_score(y_tr, tr_pred)
            te_acc = accuracy_score(y_te, te_pred)
            loss = log_loss(y_te, te_prob)
            brier = brier_score_loss(y_te, te_prob)
            roi = calc_roi(y_te, te_prob)
            
            results.append({
                'Model': name,
                'Split': split_name,
                'Train Acc': tr_acc,
                'Test Acc': te_acc,
                'Log Loss': loss,
                'Brier': brier,
                'Est. ROI': roi
            })
            
            if loss < best_loss:
                best_loss = loss
                best_model = m
                
    res_df = pd.DataFrame(results)
    res_df.to_csv("experiment_results.csv", index=False)
    
    print("\nTop 5 Configurations by Test Accuracy:")
    print(res_df.sort_values('Test Acc', ascending=False).head(5))
    
    joblib.dump(best_model, "best_model.pkl")
    print(f"\nBest model saved as best_model.pkl (Log Loss: {best_loss:.4f})")

if __name__ == "__main__":
    main()
