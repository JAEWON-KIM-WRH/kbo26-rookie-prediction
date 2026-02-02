"""
KBO 신인왕 예측 모델 - 학습 파이프라인
"""

import os
import pandas as pd
import numpy as np
from data_processor import KBODataProcessor, create_synthetic_data
from ml_models import MLModelTrainer
from dl_models import DLModelTrainer
from model_interpreter import ModelInterpreter, SHAP_AVAILABLE


def main():
    print("=" * 60)
    print("KBO 2026 신인왕 예측 모델 - 학습 파이프라인")
    print("=" * 60)
    
    # 1. 데이터 준비
    print("\n[1/6] 데이터 준비...")
    processor = KBODataProcessor()
    
    # 합성 학습 데이터 생성
    train_data = create_synthetic_data(300)
    X, df = processor.prepare_data(train_data)
    y = train_data['is_winner'].values
    
    print(f"  - 학습 데이터: {X.shape[0]}개")
    print(f"  - 특성 수: {X.shape[1]}개")
    print(f"  - 신인왕 비율: {y.mean():.2%}")
    
    # 2. ML 모델 학습
    print("\n[2/6] ML 모델 학습...")
    ml_trainer = MLModelTrainer()
    ml_results = ml_trainer.train_all(X, y)
    
    # 3. DL 모델 학습
    print("\n[3/6] DL 모델 학습...")
    mlp_trainer = DLModelTrainer('mlp', X.shape[1])
    mlp_trainer.train(X, y, epochs=50)
    
    attn_trainer = DLModelTrainer('attention', X.shape[1])
    attn_trainer.train(X, y, epochs=50)
    
    # 4. SHAP 분석
    print("\n[4/6] SHAP 분석...")
    if SHAP_AVAILABLE:
        interpreter = ModelInterpreter(ml_trainer.best_model, processor.feature_columns)
        interpreter.compute_shap_values(X)
        importance = interpreter.get_feature_importance()
        print("  특성 중요도:")
        print(importance.to_string(index=False))
    
    # 5. 2026 신인 예측
    print("\n[5/6] 2026 신인 예측...")
    rookies_2026 = pd.DataFrame([
        {'name': '박준현', 'team': '키움', 'draft_round': 1, 'draft_pick': 1, 'is_pitcher': 1, 'age': 18, 'controversy_flag': 1},
        {'name': '신재인', 'team': 'NC', 'draft_round': 1, 'draft_pick': 2, 'is_pitcher': 0, 'age': 18, 'controversy_flag': 0},
        {'name': '오재원', 'team': '한화', 'draft_round': 1, 'draft_pick': 3, 'is_pitcher': 0, 'age': 18, 'controversy_flag': 0},
        {'name': '신동건', 'team': '롯데', 'draft_round': 1, 'draft_pick': 4, 'is_pitcher': 1, 'age': 18, 'controversy_flag': 0},
        {'name': '김민준', 'team': 'SSG', 'draft_round': 1, 'draft_pick': 5, 'is_pitcher': 1, 'age': 18, 'controversy_flag': 0},
    ])
    
    X_pred, _ = processor.prepare_data(rookies_2026)
    _, ml_probs = ml_trainer.predict(X_pred)
    _, mlp_probs = mlp_trainer.predict(X_pred)
    
    rookies_2026['ML_Prob'] = ml_probs
    rookies_2026['DL_Prob'] = mlp_probs
    rookies_2026['Ensemble'] = (ml_probs + mlp_probs) / 2
    rookies_2026['Final'] = rookies_2026.apply(
        lambda x: 0 if x['controversy_flag'] == 1 else x['Ensemble'], axis=1
    )
    
    print("\n예측 결과:")
    print(rookies_2026[['name', 'team', 'ML_Prob', 'DL_Prob', 'Final']].to_string(index=False))
    
    # 6. 결과 저장
    print("\n[6/6] 결과 저장...")
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../outputs', exist_ok=True)
    
    ml_trainer.save_model('../models/best_ml_model.pkl')
    rookies_2026.to_csv('../outputs/predictions_2026.csv', index=False)
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
