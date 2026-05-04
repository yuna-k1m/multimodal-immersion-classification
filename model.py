from __future__ import annotations

import torch
import torch.nn as nn


def to_channels_first(x: torch.Tensor) -> torch.Tensor:
    """Convert common biosignal tensor shapes to [batch, channels, time]."""
    x = x.float()

    if x.dim() == 2:
        return x.unsqueeze(1)

    if x.dim() == 3:
        batch, dim1, dim2 = x.shape
        if dim1 > dim2 and dim2 <= 64:
            return x.transpose(1, 2)
        return x

    if x.dim() == 4:
        if x.size(1) == 1:
            return x.squeeze(1)
        if x.size(-1) == 1:
            return x.squeeze(-1)

        batch = x.size(0)
        time = x.size(-1)
        return x.reshape(batch, -1, time)

    if x.dim() > 4:
        batch = x.size(0)
        time = x.size(-1)
        return x.reshape(batch, -1, time)

    raise ValueError(f"Expected input with at least 2 dimensions, got shape {tuple(x.shape)}")


class EEGNetEncoder(nn.Module):
    """Compact EEG encoder inspired by EEGNet-style temporal/depthwise filtering."""

    def __init__(self, feature_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LazyConv1d(16, kernel_size=64, padding=32, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=16, padding=8, groups=16, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = to_channels_first(x)
        return self.encoder(x)


class Conv1DEncoder(nn.Module):
    """1D CNN encoder for single- or multi-channel physiological signals."""

    def __init__(self, feature_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LazyConv1d(32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = to_channels_first(x)
        return self.encoder(x)


class ModalityAttention(nn.Module):
    """Attention-based fusion over EEG, GSR, and PPG feature vectors."""

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()

        self.score_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_layer(features)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(features * weights, dim=1)
        return fused, weights


class FusionModel(nn.Module):
    """Multimodal binary classifier using EEG, GSR, and PPG inputs."""

    def __init__(self, feature_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()

        self.eeg_encoder = EEGNetEncoder(feature_dim=feature_dim, dropout=dropout)
        self.gsr_encoder = Conv1DEncoder(feature_dim=feature_dim, dropout=dropout)
        self.ppg_encoder = Conv1DEncoder(feature_dim=feature_dim, dropout=dropout)
        self.fusion = ModalityAttention(feature_dim=feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        gsr: torch.Tensor,
        ppg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eeg_feature = self.eeg_encoder(eeg)
        gsr_feature = self.gsr_encoder(gsr)
        ppg_feature = self.ppg_encoder(ppg)

        features = torch.stack([eeg_feature, gsr_feature, ppg_feature], dim=1)
        fused_feature, attention_weights = self.fusion(features)
        logits = self.classifier(fused_feature)

        return logits, attention_weights
