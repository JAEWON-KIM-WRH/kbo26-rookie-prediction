"""
KBO 신인왕 예측 모델 - SHAP 기반 모델 해석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available")


class ModelInterpreter:
    """SHAP 기반 모델 해석"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def compute_shap_values(self, X: np.ndarray):
        """SHAP 값 계산"""
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return None
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """전역 특성 중요도"""
        if self.shap_values is None:
            return pd.DataFrame()
        
        if isinstance(self.shap_values, list):
            values = self.shap_values[1]
        else:
            values = self.shap_values
        
        importance = np.abs(values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def plot_summary(self, X: np.ndarray, save_path: Optional[str] = None):
        """SHAP Summary Plot"""
        if not SHAP_AVAILABLE or self.shap_values is None:
            return
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, X, 
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def explain_prediction(self, X_single: np.ndarray, idx: int = 0) -> pd.DataFrame:
        """개별 예측 해석"""
        if self.shap_values is None:
            return pd.DataFrame()
        
        if isinstance(self.shap_values, list):
            values = self.shap_values[1][idx]
        else:
            values = self.shap_values[idx]
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_single[idx] if len(X_single.shape) > 1 else X_single,
            'shap_value': values
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return df


if __name__ == "__main__":
    from data_processor import KBODataProcessor, create_synthetic_data
    from ml_models import MLModelTrainer
    
    processor = KBODataProcessor()
    data = create_synthetic_data(300)
    X, df = processor.prepare_data(data)
    y = data['is_winner'].values
    
    trainer = MLModelTrainer()
    trainer.train_all(X, y)
    
    if SHAP_AVAILABLE:
        interpreter = ModelInterpreter(trainer.best_model, processor.feature_columns)
        interpreter.compute_shap_values(X)
        
        importance = interpreter.get_feature_importance()
        print("\nFeature Importance:")
        print(importance)
