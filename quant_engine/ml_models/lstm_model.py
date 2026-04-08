"""PyTorch LSTM model for sequential Deep Learning predictions."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
import logging

logger = logging.getLogger(__name__)


class PyTorchLSTM(nn.Module):
    """Underlying PyTorch LSTM neural network architecture."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Bierzemy ukryty stan z ostatniego kroku czasowego sekwencji
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)


class LSTMSwingModel:
    """Wrapper class integrating PyTorch LSTM into the quant engine workflow."""

    FEATURE_COLS = [
        'Ret_1', 'Ret_5', 'Vol_Ratio', 'RSI_14', 'MACD_Hist', 'Z_Score'
    ]

    def __init__(self, sequence_length=30, epochs=15):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        # Automatyczne wsparcie dla akceleracji GPU jeśli dostępne
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer stationary features required for Neural Networks."""
        data = df.copy()
        # Niestacjonarne ceny muszą zostać zlogarytmowane / zróżnicowane
        data['Ret_1'] = data['Close'].pct_change()
        data['Ret_5'] = data['Close'].pct_change(5)
        data['Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1e-8)

        # Wskaźniki
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        data['RSI_14'] = 100 - (100 / (1 + rs))

        ema_fast = data['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = macd - signal

        mean = data['Close'].rolling(50).mean()
        std = data['Close'].rolling(50).std()
        data['Z_Score'] = (data['Close'] - mean) / (std + 1e-8)

        return data

    def _create_sequences(self, data: np.ndarray, target: np.ndarray):
        """Transform 2D tabular data into 3D sequential blocks for LSTM."""
        X_seq = sliding_window_view(data, window_shape=(self.sequence_length, data.shape[1]))
        X_seq = X_seq.reshape(-1, self.sequence_length, data.shape[1])
        y_seq = target[self.sequence_length - 1:]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

    def train(self, df: pd.DataFrame):
        """Train the LSTM model on the provided historical dataset."""
        df_feat = self.build_features(df)

        # Prosty Swing Target: Czy za 10 świec cena będzie wyżej niż teraz?
        df_feat['Target'] = np.where(df_feat['Close'].shift(-10) > df_feat['Close'], 1, 0)
        df_feat = df_feat.dropna()

        if len(df_feat) < self.sequence_length * 2:
            return None

        X = df_feat[self.FEATURE_COLS].values
        y = df_feat['Target'].values

        # Skalowanie wariancji (Krytyczne dla sieci neuronowych)
        X_scaled = self.scaler.fit_transform(X)

        X_tensor, y_tensor = self._create_sequences(X_scaled, y)

        self.model = PyTorchLSTM(input_dim=len(self.FEATURE_COLS)).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.is_trained = True

        predictions = np.full(len(df), 0.5)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
            predictions[-len(preds):] = preds

        df['WF_Prediction'] = predictions
        return df

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Infer probabilities. Fills missing sequential memory with 0.5."""
        if not self.is_trained or len(df_features) < self.sequence_length:
            return np.full(len(df_features), 0.5)

        X = df_features[self.FEATURE_COLS].fillna(0).values
        X_scaled = self.scaler.transform(X)

        X_tensor, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        X_tensor = X_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()

        final_preds = np.full(len(df_features), 0.5)
        final_preds[-len(preds):] = preds
        return final_preds