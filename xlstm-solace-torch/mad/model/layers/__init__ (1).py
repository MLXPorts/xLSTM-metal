# channel mixers:
from xlstm_solace_torch.mad.model.layers.mlp import Mlp, SwiGLU, MoeMlp
from xlstm_solace_torch.mad.model.layers.rwkv import channel_mixer_rwkv5_wrapped
from xlstm_solace_torch.mad.model.layers.rwkv import channel_mixer_rwkv6_wrapped
# sequence mixers:
# NOTE: Attention module removed - flash_attn dependency not needed for xLSTM
# from xlstm_solace_torch.mad.model.layers.attention import Attention
from xlstm_solace_torch.mad.model.layers.attention_linear import LinearAttention
from xlstm_solace_torch.mad.model.layers.attention_gated_linear import GatedLinearAttention
from xlstm_solace_torch.mad.model.layers.hyena import HyenaOperator, MultiHeadHyenaOperator, HyenaExpertsOperator
from xlstm_solace_torch.mad.model.layers.mamba import Mamba
from xlstm_solace_torch.mad.model.layers.rwkv import time_mixer_rwkv5_wrapped_bf16
from xlstm_solace_torch.mad.model.layers.rwkv import time_mixer_rwkv6_wrapped_bf16

# xLSTM blocks:
from xlstm_solace_torch.mad.model.layers.xlstm_mlstm import MLSTMBlock
from xlstm_solace_torch.mad.model.layers.xlstm_ffn import GatedFFN
from xlstm_solace_torch.mad.model.layers.xlstm_swa import SlidingWindowAttention, SWABlock