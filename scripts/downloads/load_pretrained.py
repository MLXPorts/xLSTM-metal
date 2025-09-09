
"""
Load and use the pretrained xLSTM 7B model
"""

import sys
import os
import json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
import argparse

# Add xlstm to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xlstm_impl.models.xlstm import xLSTMLarge, xLSTMLargeConfig
from xlstm_impl.utils.device import DEVICE, get_device_info

def load_pretrained_model(model_path):
    """Load the pretrained xLSTM model from safetensors files."""
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create our config from the loaded config
    config = xLSTMLargeConfig(
        embedding_dim=config_dict['embedding_dim'],
        num_heads=config_dict['num_heads'],
        num_blocks=config_dict['num_blocks'],
        vocab_size=config_dict['vocab_size'],
        use_bias=config_dict.get('use_bias', False),
        norm_eps=config_dict.get('norm_eps', 1e-6),
        norm_reduction_force_float32=config_dict.get('norm_reduction_force_float32', True),
        add_out_norm=config_dict.get('add_out_norm', True),
        qk_dim_factor=config_dict.get('qk_dim_factor', 0.5),
        v_dim_factor=config_dict.get('v_dim_factor', 1.0),
        mode=config_dict.get('mode', 'inference'),
        chunk_size=config_dict.get('chunk_size', 64),
        return_last_states=config_dict.get('return_last_states', True),
        ffn_proj_factor=config_dict.get('ffn_proj_factor', 2.667),
        ffn_round_up_to_multiple_of=config_dict.get('ffn_round_up_to_multiple_of', 64),
        gate_soft_cap=config_dict.get('gate_soft_cap', 15.0),
        output_logit_soft_cap=config_dict.get('output_logit_soft_cap', 30.0),
        weight_mode=config_dict.get('weight_mode', 'single'),
    )
    
    # Create model
    print(f"Creating xLSTM model with config:")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num blocks: {config.num_blocks}")
    print(f"  Vocab size: {config.vocab_size}")
    
    model = xLSTMLarge(config)
    
    # Load weights from safetensors files
    model_index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(model_index_path):
        with open(model_index_path, 'r') as f:
            index = json.load(f)
        
        # Get list of safetensor files
        safetensor_files = set()
        for key, file in index['weight_map'].items():
            safetensor_files.add(file)
        
        print(f"\nLoading weights from {len(safetensor_files)} safetensor files...")
        
        state_dict = {}
        for file in safetensor_files:
            file_path = os.path.join(model_path, file)
            print(f"  Loading {file}...")
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        
        # The pretrained model already matches our naming!
        # Just need to map backbone.embeddings.weight -> embedding.weight
        print("\nLoading weights into model...")
        
        # Fix embedding key
        if 'backbone.embeddings.weight' in state_dict:
            state_dict['embedding.weight'] = state_dict.pop('backbone.embeddings.weight')
        
        # Load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"\nMissing keys: {len(missing_keys)}")
            for key in missing_keys[:5]:
                print(f"  {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"\nUnexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:
                print(f"  {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        print("\n✓ Weights loaded successfully!")
    
    return model, config

def generate_with_model(model, tokenizer, prompt, max_length=100, temperature=0.8, device="cpu"):
    """Generate text using the loaded model."""
    model.eval()
    model = model.to(device)
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating up to {max_length} tokens...")
    print("-" * 50)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get next token
            next_token_logits = logits[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Decode and print
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_text, end='', flush=True)
            
            # Stop if EOS
            if next_token[0] == tokenizer.eos_token_id:
                break
    
    print("\n" + "-" * 50)
    
    # Decode full text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Load and use pretrained xLSTM model")
    parser.add_argument("prompt", type=str, help="Input prompt for generation")
    parser.add_argument("--model_path", type=str, default="/Volumes/emberstuff/xLSTM/xlstm_7b_model",
                        help="Path to pretrained model")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Print device info
    device_info = get_device_info()
    print(f"\n{'='*50}")
    print(f"xLSTM Pretrained Model Inference")
    print(f"{'='*50}")
    print(f"Device: {device_info['device_name']} ({DEVICE})")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, config = load_pretrained_model(args.model_path)
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Generate text
    generated = generate_with_model(
        model, tokenizer, args.prompt, 
        max_length=args.max_length,
        temperature=args.temperature,
        device=DEVICE
    )
    
    print(f"\nFull generated text:")
    print(f"{'='*50}")
    print(generated)
    print(f"{'='*50}")

if __name__ == "__main__":
    main()