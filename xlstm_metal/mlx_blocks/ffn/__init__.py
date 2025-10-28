"""
FFN (Feed-Forward Network) block for xLSTM-7B

Implements the gated FFN from transformers xLSTM implementation.
"""

from .block import xLSTMFeedForwardBlock, GatedFFN

__all__ = ["xLSTMFeedForwardBlock", "GatedFFN"]
