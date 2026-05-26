import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class LSTMPredictorNet(nn.Module):
    """PyTorch LSTM neural network for binary sequence classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        # Take the output of the last sequence step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class LSTMPredictorModel:
    """Wrapper class providing scikit-learn style fit/predict_proba for LSTM."""
    def __init__(self, input_dim: int, sequence_length: int = 10, hidden_dim: int = 64, num_layers: int = 2, epochs: int = 20, lr: float = 0.001):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.net = None
        self.model_type = "LSTM"

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_seq = []
        y_seq = []
        if y is not None:
            if hasattr(y, "values"):
                y = y.values
            elif hasattr(y, "to_numpy"):
                y = y.to_numpy()

        # Need at least sequence_length samples
        if len(X) < self.sequence_length:
            # Pad X if too short
            pad_len = self.sequence_length - len(X)
            X = np.pad(X, ((pad_len, 0), (0, 0)), mode="edge")
            if y is not None:
                y = np.pad(y, (pad_len, 0), mode="edge")

        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i : i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length - 1])

        X_seq = np.array(X_seq)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        if y is not None:
            y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)
            return X_tensor, y_tensor
        return X_tensor

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info(f"Training PyTorch LSTM on {len(X)} samples...")
        X_tensor, y_tensor = self._create_sequences(X, y)
        self.net = LSTMPredictorNet(self.input_dim, self.hidden_dim, self.num_layers)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.net.train()
        # Batch training
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.net(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                logger.info(f"LSTM Training Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        logger.info("LSTM training completed.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.array([[0.5, 0.5]])

        self.net.eval()
        with torch.no_grad():
            if len(X) < self.sequence_length:
                pad_len = self.sequence_length - len(X)
                X_seq = np.pad(X, ((pad_len, 0), (0, 0)), mode="edge")
            else:
                X_seq = X[-self.sequence_length :]

            X_tensor = torch.tensor(np.array([X_seq]), dtype=torch.float32)
            prob_up = float(self.net(X_tensor)[0, 0])
            return np.array([[1.0 - prob_up, prob_up]])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
