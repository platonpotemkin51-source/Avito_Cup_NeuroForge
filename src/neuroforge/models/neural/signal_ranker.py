"""Retained B02 top3 model; state_dict remains compatible with measured winners."""
import torch
from torch import nn
from .data_signals import NAMES
from .history_mean import make_ranker


class SignalRanker(nn.Module):
    def __init__(self, category_count, signal='top3_recent_cos', ranks='none', width=64, layers=2, dropout=.1, context='none'):
        super().__init__()
        if signal not in ('recent_cos', 'top3_recent_cos') or ranks != 'none' or context != 'none':
            raise ValueError('Only the retained top3 model and its mean control are supported')
        self.signal = signal
        self.ranker = make_ranker(category_count, width=width, layers=layers, dropout=dropout)
        # Keep all four slots for exact compatibility with trained 93353-parameter models.
        # The two rejected rank/context slots stay zero, as in the measured top3 winners.
        self.ranker.candidate = nn.Linear(146, width)

    def forward(self, features, category, users, padding, history, signals):
        if signals.shape != (*features.shape[:2], len(NAMES)):
            raise ValueError('Invalid retained signal tensor shape')
        extra = torch.zeros((*features.shape[:2], 4), dtype=features.dtype, device=features.device)
        extra[..., 0] = signals[..., 0]
        extra[..., 1] = signals[..., NAMES.index(self.signal)]
        return self.ranker(torch.cat([features, extra], dim=-1), category, users, padding, history)
