# Full LM Trainer — Plan

Scope: Build a proper LM trainer with vocab head, masking, and integrated ponder loss over token-level ACT telemetry.

## Model
- `HRMXLSTM` + `LMHead` (`nn.Linear(D, V)`) with weight tying to embeddings (optional).
- Sequence packing + attention masks (if attention heads added later).

## Losses
- `L = CE + λ·ponder` where ponder from `act_prob_mean` over active tokens.
- Optional: label smoothing, dropout scheduling.

## Data
- Tokenizer + dataset loader (HF datasets optional), streaming-friendly.
- Curriculum for length, with Z5 alignment.

## Training
- AdamW + cosine schedule; gradient clipping; AMP/bf16.
- Logging via `TelemetryLogger`; aggregator picks up CSV/JSONL.

## Eval
- Perplexity on held-out; telemetry summaries per epoch.

## Tests
- Tiny synthetic dataset; overfit sanity; logging completeness; determinism via `trace_hash`.

