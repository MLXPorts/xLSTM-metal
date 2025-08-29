# User-Provided Working Example (Verbatim Adaptation)

The following listing captures the working prototype shared by the user (with minimal formatting fixes). A runnable variant is placed at `examples/transformer_lnn_example.py`.

```python
# -*- coding: utf-8 -*-
"""# Liquid Neural Networks: A Novel Approach to Dynamic Neural Computation
By Sydney Bach (design/architecture) Claude.ai for his math genius and coding and beautiful math markup.

The concept is to cache blocks of inference into cubes as transformers find their path through the network.

These cubes of data will do transfer learning over time from the transformer and predict the outputs based on past activations. In this hybrid architecture, the LNN gradually influences the outputs of the transformer based, leveraging its strengths with attention and having neuroplastic properties, can adapt over time and make new predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class LiquidTimeConstant(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.time_net = nn.Linear(hidden_size, hidden_size)
        self.state_net_g = nn.Linear(hidden_size, hidden_size)
        self.state_net_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.A = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=-1)
        features = self.backbone(combined)
        f_t = torch.sigmoid(self.time_net(features))
        g_x = self.state_net_g(features)
        h_x = self.state_net_h(features)
        gate = torch.sigmoid(-f_t * t.view(-1, 1))
        h_new = gate * g_x + (1 - gate) * h_x
        return h_new, h_new

class TransformerLNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.ltc = LiquidTimeConstant(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, times: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if times is None:
            times = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            times = times.unsqueeze(0).expand(batch_size, -1)
        h = self.input_proj(x)
        h_att = h.transpose(0, 1)
        h_att, _ = self.attention(h_att, h_att, h_att, attn_mask=mask)
        h_att = h_att.transpose(0, 1)
        h_att = self.norm1(h + h_att)
        ltc_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            ltc_in = h_att[:, t]
            ltc_out, ltc_state = self.ltc(ltc_in, ltc_state, times[:, t])
            outputs.append(ltc_out)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.norm2(outputs + h_att)
        y = self.output_proj(outputs)
        return y
```

See the script for dataset generation, training loop, and plotting utilities.

