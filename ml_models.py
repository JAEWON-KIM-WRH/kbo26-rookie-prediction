"""
KBO 신인왕 예측 모델 - ML 모델 (Scikit-learn)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib
from typing import Dict, Tuple, List

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MLModelTrainer:
    """ML 모델 학습 및 평가"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = ""
        self.results = {}
    
    def _get_models(self) -> Dict:
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, 
                class_weight='balanced', random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', probability=True, 
                class_weight='balanced', random_state=42
            )
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100, max_depth=6,
                scale_pos_weight=8, random_state=42,
                eval_metric='logloss'
            )
        
        return models
    
    def train_all(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """모든 모델 학습"""
        self.models = self._get_models()
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
            cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            
            model.fit(X, y)
            
            self.results[name] = {
                'cv_f1': cv_f1.mean(),
                'cv_auc': cv_auc.mean()
            }
            
            print(f"  F1: {cv_f1.mean():.4f}, AUC: {cv_auc.mean():.4f}")
        
        best_name = max(self.results, key=lambda k: self.results[k]['cv_f1'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        return self.results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """예측"""
        y_pred = self.best_model.predict(X)
        y_prob = self.best_model.predict_proba(X)[:, 1]
        return y_pred, y_prob
    
    def save_model(self, filepath: str):
        joblib.dump(self.best_model, filepath)
    
    def load_model(self, filepath: str):
        self.best_model = joblib.load(filepath)


if __name__ == "__main__":
    from data_processor import KBODataProcessor, create_synthetic_data
    
    processor = KBODataProcessor()
    data = create_synthetic_data(300)
    X, df = processor.prepare_data(data)
    y = data['is_winner'].values
    
    trainer = MLModelTrainer()
    results = trainer.train_all(X, y)
    
    print(f"\nBest Model: {trainer.best_model_name}")
