#!/usr/bin/env python
"""Debug dimensions issue"""

from xlstm_metal_complete import xLSTMLargeConfig

config = xLSTMLargeConfig(
    embedding_dim=128,
    num_heads=2,
    num_blocks=1,
    vocab_size=100
)

# Calculate dimensions
v_dim = int(config.embedding_dim * config.v_dim_factor)
qk_dim = int(config.embedding_dim * config.qk_dim_factor)

print(f"embedding_dim: {config.embedding_dim}")
print(f"v_dim_factor: {config.v_dim_factor}")
print(f"qk_dim_factor: {config.qk_dim_factor}")
print(f"v_dim: {v_dim}")
print(f"qk_dim: {qk_dim}")
print(f"num_heads: {config.num_heads}")
print(f"v_head_dim: {v_dim // config.num_heads}")
print(f"qk_head_dim: {qk_dim // config.num_heads}")

# The issue: v and qk have different head dimensions!
# v has shape [B, NH, S, v_dim/NH] = [B, 2, S, 64]
# q, k have shape [B, NH, S, qk_dim/NH] = [B, 2, S, 32]