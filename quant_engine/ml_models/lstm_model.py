"""
PyTorch LSTM model for sequential deep-learning swing predictions.

Walk-forward training: scaler is fit only on the training portion and
applied (transform-only) to the test portion to prevent data leakage.
"""

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
    """Wrapper integrating PyTorch LSTM into the quant-engine workflow.

    Key design decisions:
        * ``StandardScaler`` is fit **only** on the training split to
          avoid look-ahead bias that would inflate backtest performance.
        * A simple train/test split with a purge gap replaces the naïve
          fit-on-everything approach.
        * ``Std`` column is added in ``build_features`` so the backtester
          always has access to a volatility proxy for position sizing.
    """

    FEATURE_COLS = [
        'Ret_1', 'Ret_5', 'Vol_Ratio', 'RSI_14', 'MACD_Hist', 'Z_Score',
    ]

    def __init__(self, sequence_length: int = 30, epochs: int = 15):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer stationary features required by the LSTM.

        Raw prices are converted to returns and z-scores so the network
        receives approximately stationary inputs.  A ``Std`` column is
        added for downstream position-sizing in the backtester.
        """
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
        """Transform 2-D tabular data into 3-D sequential blocks for LSTM.

        Args:
            data:   Scaled feature matrix ``(n_samples, n_features)``.
            target: Binary target array of length ``n_samples``.

        Returns:
            Tuple of ``(X_tensor, y_tensor)`` where X has shape
            ``(n_sequences, seq_len, n_features)``.
        """
        X_seq = sliding_window_view(
            data, window_shape=(self.sequence_length, data.shape[1]),
        )
        X_seq = X_seq.reshape(-1, self.sequence_length, data.shape[1])
        y_seq = target[self.sequence_length - 1:]
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1),
        )

    def train(self, df: pd.DataFrame):
        """Train the LSTM with a proper train/test split.

        The scaler is fit exclusively on the training portion to prevent
        information from future bars leaking into the feature
        normalisation.  A 10-bar purge gap separates the training and
        test sets.

        Args:
            df: OHLCV DataFrame (ideally H1/H4/D1 for swing trading).

        Returns:
            *df* augmented with ``WF_Prediction`` column, or ``None``
            when data is insufficient.
        """
        df_feat = self.build_features(df)

        horizon = 10
        df_feat['Target'] = np.where(
            df_feat['Close'].shift(-horizon) > df_feat['Close'], 1, 0,
        )
        df_feat = df_feat.dropna()

        if len(df_feat) < self.sequence_length * 2:
            return None

        train_end = int(len(df_feat) * 0.70)
        purge_gap = horizon
        test_start = train_end + purge_gap

        if test_start >= len(df_feat):
            test_start = train_end

        train_df = df_feat.iloc[:train_end]
        test_df = df_feat.iloc[test_start:]

        X_train_raw = train_df[self.FEATURE_COLS].values
        y_train_raw = train_df['Target'].values

        self.scaler.fit(X_train_raw)
        X_train_scaled = self.scaler.transform(X_train_raw)

        if len(X_train_scaled) < self.sequence_length * 2:
            return None

        X_tensor, y_tensor = self._create_sequences(X_train_scaled, y_train_raw)

        self.model = PyTorchLSTM(input_dim=len(self.FEATURE_COLS)).to(self.device)
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

        self.is_trained = True

        predictions = np.full(len(df), 0.5)

        all_X = df_feat[self.FEATURE_COLS].values
        all_X_scaled = self.scaler.transform(all_X)

        self.model.eval()
        with torch.no_grad():
            X_all_tensor, _ = self._create_sequences(all_X_scaled, np.zeros(len(all_X_scaled)))
            X_all_tensor = X_all_tensor.to(self.device)
            preds = self.model(X_all_tensor).cpu().numpy().flatten()
            predictions[-len(preds):] = preds

        df['WF_Prediction'] = predictions
        return df

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Infer probabilities for new data.

        Rows that cannot form a full sequence are filled with 0.5
        (neutral probability).
        """
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