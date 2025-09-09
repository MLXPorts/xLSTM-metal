

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer
from xlstm_official_full.xlstm_large.from_pretrained import load_from_pretrained

# Load model using the official method
model_path = "/Volumes/emberstuff/xLSTM/xlstm_7b_model"
use_mps = torch.backends.mps.is_available()
chunkwise_kernel = "chunkwise--ray_compiled_steps"
sequence_kernel = "native_sequence__metal"
step_kernel = "metal"

model = load_from_pretrained(
    checkpoint_path=model_path,
    backend_mode="inference",
    return_last_states=True,
    chunkwise_kernel_name=chunkwise_kernel,
    sequence_kernel_name=sequence_kernel,
    step_kernel_name=step_kernel,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available. This configuration requires GPU/Metal.")

# Move to MPS
model = model.to("mps")
model.eval()

# Test prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to("mps")

print(f"Prompt: {prompt}")

# Generate using model's generate method
with torch.no_grad():
    tokens, state = model.generate(
        prefill_tokens=input_ids,
        max_length=10,
        sampling_type="greedy"
    )

# Decode
output = tokenizer.decode(tokens[0])
print(f"Output: {output}")
