"""
KBO 신인왕 예측 모델 - 데이터 전처리
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class KBODataProcessor:
    """KBO 신인왕 데이터 전처리 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'draft_score', 'star_factor', 'injury_rate', 
            'pitcher_score', 'batter_score', 'age_normalized',
            'is_pitcher', 'controversy_penalty'
        ]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 엔지니어링"""
        df = df.copy()
        
        # 드래프트 순위 점수
        if 'draft_round' in df.columns and 'draft_pick' in df.columns:
            df['draft_score'] = 1 / (df['draft_round'] + df['draft_pick'] * 0.1)
        
        # 나이 정규화
        if 'age' in df.columns:
            df['age_normalized'] = (df['age'] - 18) / 10
        
        # 부상률
        df['injury_rate'] = df.get('injury_history', 0) * 0.1
        
        # 미디어 노출
        df['star_factor'] = df.get('media_exposure_score', 5) / 10
        
        # 투수/타자 점수 (기본값)
        df['pitcher_score'] = np.where(df['is_pitcher'] == 1, 0.5, 0)
        df['batter_score'] = np.where(df['is_pitcher'] == 0, 0.5, 0)
        
        # 논란 페널티
        df['controversy_penalty'] = df.get('controversy_flag', 0)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """학습/예측용 데이터 준비"""
        df = self.create_features(df)
        
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_columns].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, df


def create_synthetic_data(n_samples: int = 200) -> pd.DataFrame:
    """합성 학습 데이터 생성"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        is_winner = np.random.random() < 0.12
        is_pitcher = np.random.random() < 0.55
        
        if is_winner:
            draft_round = np.random.choice([1, 2], p=[0.8, 0.2])
            draft_pick = np.random.randint(1, 5)
            media_score = np.random.uniform(7, 10)
            controversy = 0
        else:
            draft_round = np.random.choice([1, 2, 3, 4, 5])
            draft_pick = np.random.randint(1, 11)
            media_score = np.random.uniform(3, 8)
            controversy = np.random.choice([0, 1], p=[0.95, 0.05])
        
        data.append({
            'name': f'선수{i+1}',
            'draft_round': draft_round,
            'draft_pick': draft_pick,
            'is_winner': int(is_winner),
            'is_pitcher': int(is_pitcher),
            'age': np.random.randint(18, 25),
            'media_exposure_score': media_score,
            'injury_history': np.random.randint(0, 3),
            'controversy_flag': controversy
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    processor = KBODataProcessor()
    data = create_synthetic_data(200)
    X, df = processor.prepare_data(data)
    print(f"데이터 크기: {X.shape}")
    print(f"신인왕 비율: {data['is_winner'].mean():.2%}")
