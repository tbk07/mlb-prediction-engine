import sqlite3
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
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
    # We still use some fixed splits for final evaluation but will use TSCV internally
    splits = {}
    
    # Split A: Train 2015-2024, Test 2026
    train_A = df[df['season'] <= 2024]
    val_A_size = int(len(train_A) * 0.1)
    val_A = train_A.iloc[-val_A_size:] if val_A_size > 0 else train_A
    train_A = train_A.iloc[:-val_A_size] if val_A_size > 0 else train_A
    test = df[df['season'] == 2026]
    splits['A'] = (train_A, val_A, test)
    
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

def optimize_lgbm(X_tr, y_tr):
    ratio = (y_tr == 0).sum() / (y_tr == 1).sum()
    tscv = TimeSeriesSplit(n_splits=3)
    
    def obj(trial):
        params = {
            'n_estimators': 150,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'scale_pos_weight': ratio,
            'verbosity': -1
        }
        
        losses = []
        for train_idx, val_idx in tscv.split(X_tr):
            X_t, X_v = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_t, y_v = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_t, y_t)
            losses.append(log_loss(y_v, model.predict_proba(X_v)))
        return np.mean(losses)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=30) # Reduced slightly for speed with CV
    
    base_model = lgb.LGBMClassifier(**study.best_params, verbosity=-1)
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated.fit(X_tr, y_tr)
    return calibrated

def optimize_xgb(X_tr, y_tr):
    ratio = (y_tr == 0).sum() / (y_tr == 1).sum()
    tscv = TimeSeriesSplit(n_splits=3)
    
    def obj(trial):
        params = {
            'n_estimators': 150,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'scale_pos_weight': ratio,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        losses = []
        for train_idx, val_idx in tscv.split(X_tr):
            X_t, X_v = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_t, y_v = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_t, y_t)
            losses.append(log_loss(y_v, model.predict_proba(X_v)))
        return np.mean(losses)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=30)
    
    base_model = xgb.XGBClassifier(**study.best_params, verbosity=0)
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated.fit(X_tr, y_tr)
    return calibrated

def optimize_rf(X_tr, y_tr):
    tscv = TimeSeriesSplit(n_splits=3)
    def obj(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 15)
        }
        losses = []
        for train_idx, val_idx in tscv.split(X_tr):
            X_t, X_v = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_t, y_v = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_t, y_t)
            losses.append(log_loss(y_v, model.predict_proba(X_v)))
        return np.mean(losses)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=15)
    
    base_model = RandomForestClassifier(**study.best_params, random_state=42)
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated.fit(X_tr, y_tr)
    return calibrated

def optimize_cat(X_tr, y_tr):
    tscv = TimeSeriesSplit(n_splits=3)
    def obj(trial):
        params = {
            'iterations': 150,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'verbose': 0
        }
        losses = []
        for train_idx, val_idx in tscv.split(X_tr):
            X_t, X_v = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_t, y_v = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(X_t, y_t)
            losses.append(log_loss(y_v, model.predict_proba(X_v)))
        return np.mean(losses)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=20)
    
    base_model = CatBoostClassifier(**study.best_params, verbose=0)
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated.fit(X_tr, y_tr)
    return calibrated

def train_lr(X_tr, y_tr):
    model = make_pipeline(PolynomialFeatures(2), LogisticRegression(max_iter=500))
    model.fit(X_tr, y_tr)
    return model

def train_mlp(X_tr, y_tr):
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, early_stopping=True)
    model.fit(X_tr, y_tr)
    return model

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = get_data()
    # Drop columns that are not features. Also drop implied_h_prob to avoid leakage.
    drop_cols = ['gamePk', 'season', 'date', 'target', 'h_team', 'v_team', 'implied_h_prob']
    
    # Fill NAs for features, but drop rows where target is missing
    df = df.dropna(subset=['target'])
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
        print("Training LightGBM (Optimized & Calibrated)...")
        models['LightGBM'] = optimize_lgbm(X_tr, y_tr)
        
        print("Training XGBoost (Optimized & Calibrated)...")
        models['XGBoost'] = optimize_xgb(X_tr, y_tr)
        
        print("Training Random Forest (Optimized & Calibrated)...")
        models['RandomForest'] = optimize_rf(X_tr, y_tr)
        
        print("Training CatBoost (Optimized & Calibrated)...")
        models['CatBoost'] = optimize_cat(X_tr, y_tr)
        
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
            
            print(f"Debug: {name} tr_pred type: {tr_pred.dtype}, te_pred type: {te_pred.dtype}")
            print(f"Debug: {name} te_pred sample: {te_pred[:5]}")
            
            # Ensure te_pred is binary if it's not
            if te_pred.dtype.kind in 'fc':
                te_pred = (te_pred > 0.5).astype(int)
            if tr_pred.dtype.kind in 'fc':
                tr_pred = (tr_pred > 0.5).astype(int)

            tr_acc = accuracy_score(y_tr, tr_pred)
            te_acc = accuracy_score(y_te, te_pred)
            loss = log_loss(y_te, te_prob)
            brier = brier_score_loss(y_te, te_prob)
            
            # Get implied prob from test data for ROI
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
            
            print(f"  {name}: Test Acc={te_acc:.4f}, LogLoss={loss:.4f}, ROI={roi:.2f}%")
            
            if loss < best_loss:
                best_loss = loss
                best_model = m
        
        # Feature Importance Analysis (for LightGBM)
        try:
            lgbm_model = models['LightGBM'].calibrated_classifiers_[0].base_estimator
            importances = pd.DataFrame({'feat': X_tr.columns, 'imp': lgbm_model.feature_importances_})
            print("\nTop 10 Features (LGBM):")
            print(importances.sort_values('imp', ascending=False).head(10))
        except:
            pass
                
    res_df = pd.DataFrame(results)
    res_df.to_csv("experiment_results.csv", index=False)
    
    print("\nTop 5 Configurations by Test Accuracy:")
    print(res_df.sort_values('Test Acc', ascending=False).head(5))
    
    joblib.dump(best_model, "best_model.pkl")
    print(f"\nBest model saved as best_model.pkl (Log Loss: {best_loss:.4f})")

if __name__ == "__main__":
    main()
