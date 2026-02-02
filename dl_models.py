"""
KBO 신인왕 예측 모델 - DL 모델 (PyTorch)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from typing import Dict, List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RookieMLP(nn.Module):
    """MLP 모델"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class AttentionModel(nn.Module):
    """Attention 기반 모델"""
    
    def __init__(self, input_dim: int, embed_dim: int = 32):
        super().__init__()
        
        self.embedding = nn.Linear(1, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.input_dim = input_dim
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x, _ = self.attention(x, x, x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class DLModelTrainer:
    """DL 모델 학습"""
    
    def __init__(self, model_type: str = 'mlp', input_dim: int = 8):
        self.model_type = model_type
        self.input_dim = input_dim
        
        if model_type == 'mlp':
            self.model = RookieMLP(input_dim).to(device)
        else:
            self.model = AttentionModel(input_dim).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.criterion = nn.BCELoss()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """모델 학습"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_loss = self.criterion(val_output, y_val_t).item()
            
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """예측"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            probs = self.model(X_t).cpu().numpy().squeeze()
        
        preds = (probs > 0.5).astype(int)
        return preds, probs
    
    def save_model(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath))


if __name__ == "__main__":
    from data_processor import KBODataProcessor, create_synthetic_data
    
    processor = KBODataProcessor()
    data = create_synthetic_data(300)
    X, df = processor.prepare_data(data)
    y = data['is_winner'].values
    
    print("Training MLP...")
    mlp_trainer = DLModelTrainer('mlp', X.shape[1])
    mlp_trainer.train(X, y, epochs=50)
    
    print("\nTraining Attention...")
    attn_trainer = DLModelTrainer('attention', X.shape[1])
    attn_trainer.train(X, y, epochs=50)
