"""
model.py
Baseline LSTM model for Remaining Useful Life (RUL) prediction from
sensor time-series windows. This is your sensor-only baseline -- the
number you'll later try to beat with the text-fusion model.
"""

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """
    Simple LSTM -> fully connected head that maps a (window_size, num_features)
    sensor sequence to a single scalar RUL prediction.
    """

    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, window_size, num_features)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]          # take the final timestep's hidden state
        rul = self.head(last_hidden)         # (batch, 1)
        return rul.squeeze(-1)               # (batch,)
