#!/usr/bin/env python
"""
Download and use pretrained xLSTM-7B model from HuggingFace
"""

import os
import sys
import torch
from pathlib import Path


def download_xlstm_7b():
    """Download the xLSTM-7B model from HuggingFace"""
    print("="*60)
    print("Downloading xLSTM-7B Pretrained Model")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import login
        
        # Check for HF token
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            print("\n✓ HuggingFace token found, logging in...")
            login(token=hf_token)
        else:
            print("\n⚠ No HF_TOKEN found in environment")
            print("  Some models may require authentication")
        
        print("\nDownloading model from HuggingFace...")
        print("Model: NX-AI/xLSTM-7b")
        print("This is a 7B parameter model, download may take a while...")
        
        # Create cache directory
        cache_dir = Path("./model_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Download tokenizer
        print("\n1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "NX-AI/xLSTM-7b",
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=hf_token if hf_token else None
        )
        print("   ✓ Tokenizer downloaded")
        
        # Download model
        print("\n2. Downloading model weights...")
        print("   This will download ~14GB of data (6 safetensor files)")
        
        model = AutoModelForCausalLM.from_pretrained(
            "NX-AI/xLSTM-7b",
            cache_dir=cache_dir,
            device_map="cpu",  # Start on CPU
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True,
            token=hf_token if hf_token else None
        )
        print("   ✓ Model weights downloaded")
        
        # Model info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nModel info:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Memory size: ~{param_count * 2 / 1e9:.1f}GB (fp16)")
        
        # Save model locally
        local_path = Path("./xlstm_7b_model")
        local_path.mkdir(exist_ok=True)
        
        print(f"\n3. Saving model locally to {local_path}...")
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print("   ✓ Model saved locally")
        
        return model, tokenizer
        
    except ImportError:
        print("\n⚠ transformers library not installed!")
        print("Installing transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "safetensors"])
        print("Please run the script again after installation.")
        return None, None
    
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_pretrained_model(model, tokenizer):
    """Test the pretrained model with sample generation"""
    if model is None or tokenizer is None:
        return
    
    print("\n" + "="*60)
    print("Testing Pretrained Model")
    print("="*60)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\nMoving model to GPU...")
        model = model.to(device)
    else:
        print(f"\nUsing CPU (GPU not available)")
    
    model.eval()
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "xLSTM is a new architecture that",
        "In machine learning, the key to success is"
    ]
    
    print("\nGenerating text with xLSTM-7B:")
    print("-" * 40)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")


def extract_architecture_info(model):
    """Extract and display model architecture information"""
    print("\n" + "="*60)
    print("Model Architecture Analysis")
    print("="*60)
    
    if model is None:
        return
    
    # Analyze model structure
    print("\nModel structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            if hasattr(module, 'weight'):
                weight_shape = tuple(module.weight.shape)
                print(f"  {name}: {module.__class__.__name__} {weight_shape}")
    
    # Count different types of layers
    layer_counts = {}
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        layer_counts[class_name] = layer_counts.get(class_name, 0) + 1
    
    print("\nLayer type summary:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:
            print(f"  {layer_type}: {count}")


def convert_to_our_format():
    """Convert HuggingFace weights to our xLSTM implementation format"""
    print("\n" + "="*60)
    print("Converting Weights to Our Format")
    print("="*60)
    
    local_path = Path("./xlstm_7b_model")
    
    if not local_path.exists():
        print("✗ Model not found locally. Please download first.")
        return
    
    print("\nLoading saved model...")
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    # Extract state dict
    state_dict = model.state_dict()
    
    print(f"\nFound {len(state_dict)} weight tensors")
    print("\nSample weight names:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}: {state_dict[key].shape}")
    
    # Save weights in a format we can load
    weight_path = Path("./xlstm_7b_weights.pt")
    print(f"\nSaving weights to {weight_path}...")
    torch.save(state_dict, weight_path)
    print(f"✓ Weights saved ({weight_path.stat().st_size / 1e9:.1f}GB)")
    
    return state_dict


def main():
    """Main execution"""
    print("xLSTM-7B Pretrained Model Download and Test")
    print("=" * 60)
    
    # Check if model already exists locally
    local_path = Path("./xlstm_7b_model")
    
    if local_path.exists():
        print("\n✓ Model already downloaded locally")
        print("Loading from local cache...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            print("✓ Model loaded from cache")
        except Exception as e:
            print(f"Error loading cached model: {e}")
            model, tokenizer = None, None
    else:
        # Download model
        model, tokenizer = download_xlstm_7b()
    
    if model is not None:
        # Analyze architecture
        extract_architecture_info(model)
        
        # Test generation
        test_pretrained_model(model, tokenizer)
        
        # Convert weights
        convert_to_our_format()
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())