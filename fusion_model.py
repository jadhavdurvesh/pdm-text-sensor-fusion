"""
fusion_model.py
Fusion model: LSTM sensor encoder + pretrained transformer text encoder,
combined via concatenation, to predict RUL from BOTH sensor windows and
synthetic maintenance-log text.

This is the model you compare against your sensor-only baseline (model.py).
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class SensorEncoder(nn.Module):
    """Same LSTM architecture as the baseline, but outputs a hidden vector
    instead of a final RUL scalar -- the fusion head does that."""

    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_size

    def forward(self, x):
        # x: (batch, window_size, num_features)
        out, _ = self.lstm(x)
        return out[:, -1, :]  # (batch, hidden_size) -- last timestep's hidden state


class TextEncoder(nn.Module):
    """
    Wraps a pretrained transformer (default: distilbert-base-uncased, small
    and fast -- good for a laptop/Colab-free-tier budget) and projects its
    pooled output down to a fixed embedding size.

    freeze_base=True freezes the transformer's weights and only trains the
    projection layer -- much faster to train and usually enough for a
    fixed-vocabulary domain like maintenance logs. Set False if you have
    time/compute for full fine-tuning.
    """

    def __init__(self, model_name="distilbert-base-uncased", proj_dim=32, freeze_base=True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        self.projection = nn.Linear(self.transformer.config.hidden_size, proj_dim)
        self.output_dim = proj_dim

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over tokens (masking out padding) -- more stable than
        # using only the [CLS] token for short technical text like these logs.
        last_hidden = outputs.last_hidden_state              # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).float()           # (batch, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts                               # (batch, hidden)
        return self.projection(pooled)                         # (batch, proj_dim)


class FusionModel(nn.Module):
    """Combines SensorEncoder + TextEncoder via concatenation, then an MLP
    head predicts the final RUL scalar."""

    def __init__(self, num_sensor_features, text_model_name="distilbert-base-uncased",
                 sensor_hidden=64, text_proj_dim=32, freeze_text=True, dropout=0.2):
        super().__init__()
        self.sensor_encoder = SensorEncoder(num_sensor_features, hidden_size=sensor_hidden)
        self.text_encoder = TextEncoder(text_model_name, proj_dim=text_proj_dim, freeze_base=freeze_text)

        combined_dim = self.sensor_encoder.output_dim + self.text_encoder.output_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, sensor_x, input_ids, attention_mask):
        sensor_vec = self.sensor_encoder(sensor_x)                       # (batch, sensor_hidden)
        text_vec = self.text_encoder(input_ids, attention_mask)          # (batch, text_proj_dim)
        combined = torch.cat([sensor_vec, text_vec], dim=1)              # (batch, combined_dim)
        rul = self.head(combined)
        return rul.squeeze(-1)
