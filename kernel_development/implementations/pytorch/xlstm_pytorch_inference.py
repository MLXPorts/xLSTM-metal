"""
Enhanced PyTorch xLSTM with optimized inference and generation capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class xLSTMInference(nn.Module):
    """xLSTM with optimized inference capabilities"""

    def __init__(self, base_model):
        """Initialize with a base xLSTM model"""
        super().__init__()
        self.model = base_model
        self.vocab_size = base_model.vocab_size
        self.device = next(base_model.parameters()).device

    def forward_step(
            self,
            token_id: torch.Tensor,
            hidden_states: List,
            position: Optional[int] = None
    ) -> Tuple[torch.Tensor, List]:
        """
        Single step forward for generation
        
        Args:
            token_id: Current token ID [batch_size]
            hidden_states: List of hidden states for each block
            position: Current position in sequence (for positional encodings if used)
            
        Returns:
            logits: Output logits [batch_size, vocab_size]
            hidden_states: Updated hidden states
        """
        # Embed single token
        x = self.model.embedding(token_id)  # [batch_size, embed_dim]

        # Apply dropout if in training mode
        if self.model.dropout and self.training:
            x = self.model.dropout(x)

        # Process through blocks with state updates
        for i, block in enumerate(self.model.blocks):
            x, hidden_states[i] = block(x, hidden_states[i])
            if self.model.dropout and self.training and i < len(self.model.blocks) - 1:
                x = self.model.dropout(x)

        # Compute logits
        logits = self.model.head(x)  # [batch_size, vocab_size]

        return logits, hidden_states

    def forward_sequence(
            self,
            input_ids: torch.Tensor,
            hidden_states: Optional[List] = None,
            use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Efficient forward pass for full sequence (prefill)
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            hidden_states: Optional initial hidden states
            use_cache: Whether to return final hidden states
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden_states: Final hidden states (if use_cache=True)
        """
        batch_size, seq_len = input_ids.shape

        # Initialize hidden states if needed
        if hidden_states is None:
            hidden_states = self.model.init_hidden(batch_size)

        # Embed all tokens at once
        x = self.model.embedding(input_ids)  # [batch_size, seq_len, embed_dim]

        if self.model.dropout and self.training:
            x = self.model.dropout(x)

        # Process sequence efficiently (batch processing where possible)
        all_logits = []

        # For each time step
        for t in range(seq_len):
            x_t = x[:, t, :]

            # Process through all blocks
            for i, block in enumerate(self.model.blocks):
                x_t, hidden_states[i] = block(x_t, hidden_states[i])
                if self.model.dropout and self.training and i < len(self.model.blocks) - 1:
                    x_t = self.model.dropout(x_t)

            # Compute logits for this timestep
            logits_t = self.model.head(x_t)
            all_logits.append(logits_t)

        # Stack all logits
        logits = torch.stack(all_logits, dim=1)  # [batch_size, seq_len, vocab_size]

        if use_cache:
            return logits, hidden_states
        else:
            return logits, None

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            generation_config: Optional[GenerationConfig] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            generation_config: Generation configuration
            attention_mask: Optional attention mask
            **kwargs: Additional generation parameters
            
        Returns:
            generated_ids: Generated token IDs [batch_size, total_len]
        """
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        else:
            # Override with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(generation_config, key):
                    setattr(generation_config, key, value)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Prefill: Process input sequence
        _, hidden_states = self.forward_sequence(input_ids)

        # Initialize generation
        generated = input_ids
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Generation loop
        for _ in range(generation_config.max_new_tokens):
            # Get last token
            last_token = generated[:, -1]

            # Forward step
            logits, hidden_states = self.forward_step(last_token, hidden_states)

            # Apply repetition penalty if specified
            if generation_config.repetition_penalty != 1.0:
                self._apply_repetition_penalty(
                    logits, generated, generation_config.repetition_penalty
                )

            # Sample next token
            next_tokens = self._sample_token(logits, generation_config)

            # Update sequences
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=-1)

            # Check for EOS token
            if generation_config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences & (next_tokens != generation_config.eos_token_id)
                if not unfinished_sequences.any():
                    break

        return generated

    def _sample_token(
            self,
            logits: torch.Tensor,
            config: GenerationConfig
    ) -> torch.Tensor:
        """Sample next token from logits"""
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Apply top-k filtering
        if config.top_k is not None and config.top_k > 0:
            logits = self._top_k_filtering(logits, config.top_k)

        # Apply top-p (nucleus) filtering
        if config.top_p is not None and config.top_p < 1.0:
            logits = self._top_p_filtering(logits, config.top_p)

        # Sample or take argmax
        if config.do_sample:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)

        return next_tokens

    def _top_k_filtering(
            self,
            logits: torch.Tensor,
            top_k: int
    ) -> torch.Tensor:
        """Filter logits to keep only top-k tokens"""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    def _top_p_filtering(
            self,
            logits: torch.Tensor,
            top_p: float
    ) -> torch.Tensor:
        """Filter logits using nucleus (top-p) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits

    def _apply_repetition_penalty(
            self,
            logits: torch.Tensor,
            generated: torch.Tensor,
            penalty: float
    ):
        """Apply repetition penalty to logits"""
        for i in range(generated.shape[0]):
            for token_id in set(generated[i].tolist()):
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty

    @torch.no_grad()
    def generate_streaming(
            self,
            input_ids: torch.Tensor,
            generation_config: Optional[GenerationConfig] = None,
            **kwargs
    ):
        """
        Generator for streaming text generation
        
        Yields:
            token_id: Next generated token
            hidden_states: Current hidden states
        """
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)

        # Prefill
        _, hidden_states = self.forward_sequence(input_ids)

        # Yield tokens one by one
        last_token = input_ids[:, -1]

        for _ in range(generation_config.max_new_tokens):
            # Forward step
            logits, hidden_states = self.forward_step(last_token, hidden_states)

            # Sample next token
            next_token = self._sample_token(logits, generation_config)

            yield next_token, hidden_states

            # Check for EOS
            if generation_config.eos_token_id is not None:
                if (next_token == generation_config.eos_token_id).any():
                    break

            last_token = next_token


def create_inference_model(base_model_or_config, device='cpu'):
    """
    Create an xLSTM model optimized for inference
    
    Args:
        base_model_or_config: Either an existing xLSTM model or configuration dict
        device: Device to place model on
        
    Returns:
        xLSTMInference model
    """
    if isinstance(base_model_or_config, dict):
        # Create base model from config
        from xlstm_pytorch import create_xlstm_model
        base_model = create_xlstm_model(**base_model_or_config, device=device)
    else:
        base_model = base_model_or_config

    return xLSTMInference(base_model).to(device)


# Example usage and testing
if __name__ == "__main__":
    import time
    from xlstm_pytorch import create_xlstm_model

    print("Creating xLSTM model for inference...")

    # Create base model
    base_model = create_xlstm_model()

    # Wrap with inference capabilities
    model = xLSTMInference(base_model)
    model.eval()

    # Test generation
    batch_size = 2
    prompt_len = 10
    prompt = torch.randint(0, 1000, (batch_size, prompt_len))

    print(f"\nPrompt shape: {prompt.shape}")

    # Test standard generation
    print("\n1. Testing standard generation...")
    start_time = time.time()

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )

    gen_time = time.time() - start_time
    print(f"Generated shape: {generated.shape}")
    print(f"Generation time: {gen_time:.3f}s")
    print(f"Tokens/sec: {20 / gen_time:.1f}")

    # Test streaming generation
    print("\n2. Testing streaming generation...")
    tokens_generated = []

    for token, _ in model.generate_streaming(
            prompt,
            max_new_tokens=10,
            temperature=0.8
    ):
        tokens_generated.append(token)
        print(f"  Generated token: {token.tolist()}")

    print(f"Total tokens generated: {len(tokens_generated)}")

    # Test prefill performance
    print("\n3. Testing prefill performance...")
    long_sequence = torch.randint(0, 1000, (1, 100))

    start_time = time.time()
    logits, states = model.forward_sequence(long_sequence)
    prefill_time = time.time() - start_time

    print(f"Prefill shape: {logits.shape}")
    print(f"Prefill time: {prefill_time:.3f}s")
    print(f"Tokens/sec: {100 / prefill_time:.1f}")

    print("\n✓ Inference model tests complete!")
