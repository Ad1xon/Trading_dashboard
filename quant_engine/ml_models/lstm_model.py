"""PyTorch LSTM model for sequential deep-learning swing predictions."""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

class PyTorchLSTM(nn.Module):
    """Two-layer LSTM with dropout followed by a sigmoid head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns sigmoid probability from the last timestep."""
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)


class LSTMSwingModel:
    """Wrapper integrating PyTorch LSTM with expanding-window walk-forward training."""

    def __init__(
        self,
        sequence_length: int = 30,
        epochs: int = 15,
        n_wf_folds: int = 3,
        feature_cols: list | None = None,
    ):
        self.feature_cols = feature_cols or [
            'Ret_1', 'Ret_5', 'Vol_Ratio', 'RSI_14', 'MACD_Hist', 'Z_Score',
        ]
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.n_wf_folds = n_wf_folds
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer stationary features required by the LSTM."""
        data = df.copy()
        data['Ret_1'] = data['Close'].pct_change()
        data['Ret_5'] = data['Close'].pct_change(5)
        data['Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1e-8)

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

        data['Std'] = data['Close'].rolling(20).std()

        return data

    def _create_sequences(self, data: np.ndarray, target: np.ndarray):
        """Transform 2-D tabular data into 3-D sequential blocks for LSTM."""
        X_seq = sliding_window_view(
            data, window_shape=(self.sequence_length, data.shape[1]),
        )
        X_seq = X_seq.reshape(-1, self.sequence_length, data.shape[1])
        y_seq = target[self.sequence_length - 1:]
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1),
        )

    def _train_single_fold(self, X_scaled: np.ndarray, y: np.ndarray):
        """Train the LSTM on a single data fold."""
        if len(X_scaled) < self.sequence_length * 2:
            return

        X_tensor, y_tensor = self._create_sequences(X_scaled, y)

        self.model = PyTorchLSTM(input_dim=len(self.feature_cols)).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)

        self.model.train()
        for _epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def train(self, df: pd.DataFrame):
        """Expanding-window walk-forward training with purge gap."""
        df_feat = self.build_features(df)

        horizon = 10
        purge_gap = horizon
        df_feat['Target'] = np.where(
            df_feat['Close'].shift(-horizon) > df_feat['Close'], 1, 0,
        )
        df_feat = df_feat.dropna()

        if len(df_feat) < self.sequence_length * 2:
            return None

        n = len(df_feat)
        predictions = np.full(len(df), 0.5)
        fold_size = n // (self.n_wf_folds + 1)

        for fold in range(self.n_wf_folds):
            train_end = fold_size * (fold + 1)
            test_start = train_end + purge_gap
            test_end = min(test_start + fold_size, n)

            if train_end < self.sequence_length * 2 or test_start >= test_end:
                continue

            train_df = df_feat.iloc[:train_end]
            test_df = df_feat.iloc[test_start:test_end]

            X_train_raw = train_df[self.feature_cols].values
            self.scaler.fit(X_train_raw)
            X_train_scaled = self.scaler.transform(X_train_raw)
            y_train = train_df['Target'].values

            self._train_single_fold(X_train_scaled, y_train)

            if self.model is not None and len(test_df) >= self.sequence_length:
                X_test_raw = test_df[self.feature_cols].values
                X_test_scaled = self.scaler.transform(X_test_raw)
                X_test_tensor, _ = self._create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)))
                X_test_tensor = X_test_tensor.to(self.device)

                self.model.eval()
                with torch.no_grad():
                    preds = self.model(X_test_tensor).cpu().numpy().flatten()

                test_indices = df_feat.index[test_start:test_end]
                pred_offset = len(test_indices) - len(preds)
                for idx_i in range(len(preds)):
                    orig_pos = df.index.get_loc(test_indices[pred_offset + idx_i])
                    if isinstance(orig_pos, int):
                        predictions[orig_pos] = preds[idx_i]

        self.is_trained = self.model is not None

        all_X = df_feat[self.feature_cols].values
        all_X_scaled = self.scaler.transform(all_X)

        if self.is_trained:
            self.model.eval()
            with torch.no_grad():
                X_all_tensor, _ = self._create_sequences(all_X_scaled, np.zeros(len(all_X_scaled)))
                X_all_tensor = X_all_tensor.to(self.device)
                full_preds = self.model(X_all_tensor).cpu().numpy().flatten()
                predictions[-len(full_preds):] = full_preds

        df['WF_Prediction'] = predictions
        return df

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Infer probabilities for new data."""
        if not self.is_trained or len(df_features) < self.sequence_length:
            return np.full(len(df_features), 0.5)

        X = df_features[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        X_tensor, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        X_tensor = X_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()

        final_preds = np.full(len(df_features), 0.5)
        final_preds[-len(preds):] = preds
        return final_preds
