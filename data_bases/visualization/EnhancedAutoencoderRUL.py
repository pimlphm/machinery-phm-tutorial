
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# === Enhanced Model ===
class EnhancedAutoencoderRUL(nn.Module):
    """
    Enhanced LSTM Autoencoder with improved RUL prediction
    """
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.3):
        super().__init__()

        # Enhanced LSTM encoder with more layers
        self.lstm_encoder = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout, bidirectional=True
        )

        # Decoder with attention mechanism
        self.decoder_lstm = nn.LSTM(
            hidden_size * 2, hidden_size,
            num_layers=2, batch_first=True,
            dropout=dropout
        )
        self.decoder_output = nn.Linear(hidden_size, input_size)

        # Enhanced RUL prediction head with multiple layers
        self.rul_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Attention mechanism for RUL prediction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        # x: [batch, time, channels]
        batch_size, seq_len, _ = x.shape

        # Encoder
        encoder_outputs, (h_n, c_n) = self.lstm_encoder(x)  # [B, T, hidden_size*2]

        # Self-attention for better feature extraction
        attn_output, _ = self.attention(encoder_outputs, encoder_outputs, encoder_outputs)
        encoder_outputs = self.layer_norm(encoder_outputs + attn_output)

        # Decoder for reconstruction
        decoder_outputs, _ = self.decoder_lstm(encoder_outputs)  # [B, T, hidden_size]
        recon = self.decoder_output(decoder_outputs)  # [B, T, input_size]

        # RUL prediction using enhanced features
        rul_features = encoder_outputs  # Use attended features
        rul = self.rul_layers(rul_features).squeeze(-1)  # [B, T]

        return recon, rul
