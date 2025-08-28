#!/usr/bin/env python
"""
Run a local xLSTM HF checkpoint on Apple Silicon (MPS) using the compiled backends.

- Loads HF-style sharded safetensors from a local directory (config.json + model-*.safetensors).
- Instantiates xlstm_official_full xLSTMLarge with compiled kernels on MPS.
- Maps known key differences (backbone.embeddings.weight -> embedding.weight).
"""
import argparse
import os
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
import torch.nn.functional as F

from xlstm_official_full.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig
from mlstm_kernels.torch.monitoring.memory import MemoryMonitor
from mlstm_kernels.torch.monitoring.ray_metrics import make_gauges


def load_local_config(config_path: Path) -> xLSTMLargeConfig:
    """Load HF config.json and build xLSTMLargeConfig.

    - Copies all relevant fields from JSON.
    - On MPS, forces compiled MPS backends (Ray, metal, native_sequence__metal).
      Other fields (e.g., chunk_size) honor JSON unless CLI overrides are given.
    """
    cfg = json.loads(config_path.read_text())

    # Base construction from JSON
    mcfg = xLSTMLargeConfig(
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        vocab_size=cfg["vocab_size"],
        use_bias=cfg.get("use_bias", False),
        norm_eps=cfg.get("norm_eps", 1e-6),
        norm_reduction_force_float32=cfg.get("norm_reduction_force_float32", True),
        add_out_norm=cfg.get("add_out_norm", True),
        qk_dim_factor=cfg.get("qk_dim_factor", 0.5),
        v_dim_factor=cfg.get("v_dim_factor", 1.0),
        gate_soft_cap=cfg.get("gate_soft_cap", 15.0),
        output_logit_soft_cap=cfg.get("output_logit_soft_cap", 30.0),
        weight_mode=cfg.get("weight_mode", "single"),
    )

    # Optional tunables that should mirror JSON unless overridden later
    if "chunk_size" in cfg:
        mcfg.chunk_size = int(cfg["chunk_size"])  # JSON default often 64
    if "mode" in cfg:
        mcfg.mode = cfg["mode"]
    if "return_last_states" in cfg:
        mcfg.return_last_states = bool(cfg["return_last_states"])
    if "autocast_kernel_dtype" in cfg:
        mcfg.autocast_kernel_dtype = cfg["autocast_kernel_dtype"]
    if "inference_state_dtype" in cfg:
        mcfg.inference_state_dtype = cfg["inference_state_dtype"]
    if "ffn_proj_factor" in cfg:
        mcfg.ffn_proj_factor = float(cfg["ffn_proj_factor"])
    if "ffn_round_up_to_multiple_of" in cfg:
        mcfg.ffn_round_up_to_multiple_of = int(cfg["ffn_round_up_to_multiple_of"])

    # Backends: on MPS, force compiled defaults (ignore Triton in JSON)
    if torch.backends.mps.is_available():
        # Ray by default unless explicitly overridden via env
        default_chunkwise = os.environ.get("XLSTM_CHUNKWISE_BACKEND", "ray_compiled_steps")
        mcfg.chunkwise_kernel = f"chunkwise--{default_chunkwise}"
        mcfg.sequence_kernel = "native_sequence__metal"
        mcfg.step_kernel = "metal"
        # Sensible inference defaults
        mcfg.mode = "inference"
        mcfg.return_last_states = True
        if "autocast_kernel_dtype" not in cfg:
            mcfg.autocast_kernel_dtype = "bfloat16"
        if "inference_state_dtype" not in cfg:
            mcfg.inference_state_dtype = "float32"
        # Default Ray local_mode unless user chose otherwise
        os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "1")
    else:
        # Non-MPS: respect JSON kernels if present
        if "chunkwise_kernel" in cfg:
            mcfg.chunkwise_kernel = cfg["chunkwise_kernel"]
        if "sequence_kernel" in cfg:
            mcfg.sequence_kernel = cfg["sequence_kernel"]
        if "step_kernel" in cfg:
            mcfg.step_kernel = cfg["step_kernel"]

    return mcfg


def load_local_weights(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    idx = json.loads(index_path.read_text())
    files = sorted({f for f in idx["weight_map"].values()})
    state = {}
    for f in files:
        fp = model_dir / f
        with safe_open(str(fp), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                state[k] = sf.get_tensor(k)
    # Key mapping
    if "backbone.embeddings.weight" in state:
        state["embedding.weight"] = state.pop("backbone.embeddings.weight")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./xlstm_7b_model")
    ap.add_argument("--config", type=str, default=None, help="Optional JSON config to override CLI args")
    ap.add_argument("--prompt", type=str, default=None, help="Inline prompt text (use --prompt-file for long context)")
    ap.add_argument("--prompt-file", type=str, default=None, help="Path to a prompt text file")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    # Optional overrides; if omitted, JSON+MPS defaults are used
    ap.add_argument("--workers", type=int, default=None, help="CPU coordinator threads (queued backend)")
    ap.add_argument("--heads-per-band", type=int, default=None, help="Heads per task (MPS queued/Ray scheduler)")
    ap.add_argument("--streams", type=int, default=None, help="MPS streams (queued backend; defaults to workers)")
    ap.add_argument("--chunk-size", type=int, default=None, help="Chunk size for compiled backends (defaults from JSON)")
    ap.add_argument("--chunkwise-backend", type=str, default=None,
                    help="Override chunkwise backend key (ray_compiled_steps, queued_compiled_steps, native_compiled_autograd)")
    ap.add_argument("--stats-log", type=str, default=None, help="Optional CSV file to log rolling decode stats")
    ap.add_argument("--stats-every", type=int, default=1, help="Log every N decode steps to stats log")
    # Memory monitoring / watchdog (global logging; drivers also have watchdogs)
    ap.add_argument("--mem-log", type=str, default=None, help="Write process/MPS memory CSV (ts,rss,avail,total,mps_alloc,mps_reserved)")
    ap.add_argument("--mem-every", type=int, default=None, help="Memory sampling period in ms (default 200)")
    ap.add_argument("--mem-soft-pct", type=float, default=None, help="Soft memory threshold as fraction of total (default 0.85)")
    ap.add_argument("--mem-hard-pct", type=float, default=None, help="Hard memory threshold as fraction of total (default 0.92)")
    ap.add_argument("--mem-soft-mb", type=float, default=None, help="Soft memory threshold in MB (overrides pct)")
    ap.add_argument("--mem-hard-mb", type=float, default=None, help="Hard memory threshold in MB (overrides pct)")
    ap.add_argument("--mem-action", type=str, default=None, help="Comma actions on soft limit: warn,empty_cache (default warn,empty_cache)")
    ap.add_argument("--min-chunk", type=int, default=None, help="Minimum chunk size when shrinking (default 8)")
    ap.add_argument("--no-mem-shrink", action="store_true", help="Disable chunk-size shrinking on soft limit")
    # Ray / Dashboard controls
    ap.add_argument("--ray-dashboard", action="store_true", help="Start a local Ray head with dashboard enabled (http://127.0.0.1:8265)")
    ap.add_argument("--ray-dashboard-port", type=int, default=8265, help="Ray dashboard port (default 8265)")
    ap.add_argument("--ray-local-mode", type=int, default=None, choices=[0,1], help="Set Ray local_mode explicitly (1=in-process, 0=multiprocess)")
    ap.add_argument("--ray-keep-alive", action="store_true", help="Keep Ray head running after run (dashboard stays up)")
    args = ap.parse_args()

    # Load JSON config (if provided) to override args
    if args.config:
        cfg_path = Path(args.config)
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        cfg = json.loads(cfg_path.read_text())
        # Allowed keys to override
        overrides = {
            "model_path": "model_path",
            "prompt": "prompt",
            "max_new_tokens": "max_new_tokens",
            "workers": "workers",
            "heads_per_band": "heads_per_band",
            "streams": "streams",
            "chunk_size": "chunk_size",
            "chunkwise_backend": "chunkwise_backend",
        }
        for k_src, k_dst in overrides.items():
            if k_src in cfg and getattr(args, k_dst if k_dst != "max_new_tokens" else "max_new_tokens", None) is not None:
                setattr(args, k_dst if k_dst != "max_new_tokens" else "max_new_tokens", cfg[k_src])

    model_dir = Path(args.model_path)
    assert model_dir.is_dir(), f"Not a directory: {model_dir}"
    assert torch.backends.mps.is_available(), "MPS not available; requires Apple Silicon."

    # Load config and model
    mcfg = load_local_config(model_dir / "config.json")
    # Auto-apply last optimizer best.json unless disabled
    if os.environ.get("XLSTM_USE_BEST", "1") == "1":
        try:
            from pathlib import Path as _P
            import json as _json
            runs = sorted((_P("runs/mps_opt").glob("*/best.json")), key=lambda p: p.stat().st_mtime, reverse=True)
            if runs:
                best = _json.loads(runs[0].read_text())
                hpb = best.get("heads_per_band") or best.get("hpb")
                csz = best.get("chunk_size") or best.get("ck")
                if hpb is not None and os.environ.get("XLSTM_MPS_HEADS_PER_BAND") is None and args.heads_per_band is None:
                    os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(int(hpb))
                if csz is not None and args.chunk_size is None:
                    mcfg.chunk_size = int(csz)
        except Exception:
            pass
    # Apply optional CLI overrides (None means: keep JSON/defaults)
    if args.chunk_size is not None:
        mcfg.chunk_size = args.chunk_size
    if args.chunkwise_backend:
        os.environ["XLSTM_CHUNKWISE_BACKEND"] = args.chunkwise_backend
        # Update the config view as well
        mcfg.chunkwise_kernel = f"chunkwise--{args.chunkwise_backend}"
    model = xLSTMLarge(mcfg)
    # Load weights
    print("Loading weights ...")
    sd = load_local_weights(model_dir)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (showing up to 10):")
        for k in missing[:10]:
            print("  ", k)
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} (showing up to 10):")
        for k in unexpected[:10]:
            print("  ", k)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Device setup
    device = torch.device("mps")
    model = model.to(device).eval()

    # Optional: enforce GPU-only if available in this PyTorch build
    try:
        torch._C._set_mps_fallback_enabled(False)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Prepare input
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    else:
        assert args.prompt is not None, "Provide --prompt or --prompt-file"
        prompt_text = args.prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device, dtype=inputs.dtype)
        inputs = torch.cat([bos, inputs], dim=1)

    # Optional tuning knobs for queued/Ray schedulers
    if args.workers is not None:
        os.environ["XLSTM_MPS_WORKERS"] = str(args.workers)
    if args.heads_per_band is not None:
        os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(args.heads_per_band)
    if args.streams is not None:
        os.environ["XLSTM_MPS_STREAMS"] = str(args.streams)

    # Memory watchdog envs (drivers respect these). Also start optional global logger.
    if args.mem_soft_pct is not None:
        os.environ["XLSTM_MEM_SOFT_PCT"] = str(args.mem_soft_pct)
    if args.mem_hard_pct is not None:
        os.environ["XLSTM_MEM_HARD_PCT"] = str(args.mem_hard_pct)
    if args.mem_soft_mb is not None:
        os.environ["XLSTM_MEM_SOFT_MB"] = str(args.mem_soft_mb)
    if args.mem_hard_mb is not None:
        os.environ["XLSTM_MEM_HARD_MB"] = str(args.mem_hard_mb)
    if args.mem_every is not None:
        os.environ["XLSTM_MEM_POLL_MS"] = str(max(50, args.mem_every))
    if args.mem_action is not None:
        os.environ["XLSTM_MEM_ACTION"] = args.mem_action
    if args.min_chunk is not None:
        os.environ["XLSTM_MIN_CHUNK"] = str(int(args.min_chunk))
    if args.no_mem_shrink:
        os.environ["XLSTM_SHRINK_ON_SOFT"] = "0"
    # Enable watchdog by default when logging is requested
    if args.mem_log:
        os.environ.setdefault("XLSTM_MEM_WATCHDOG", "1")
    # Ray dashboard / lifecycle envs
    if args.ray_local_mode is not None:
        os.environ["XLSTM_RAY_LOCAL_MODE"] = "1" if args.ray_local_mode == 1 else "0"
    if args.ray_dashboard:
        os.environ["XLSTM_RAY_DASHBOARD"] = "1"
        # Dashboard implies non-local mode unless user overrode
        os.environ.setdefault("XLSTM_RAY_LOCAL_MODE", "0")
    os.environ["XLSTM_RAY_DASHBOARD_PORT"] = str(int(args.ray_dashboard_port))
    # Ensure Ray cleans up unless the user wants to keep the dashboard alive
    os.environ["XLSTM_RAY_AUTOSHUTDOWN"] = "0" if args.ray_keep_alive else os.environ.get("XLSTM_RAY_AUTOSHUTDOWN", "1")

    print("Generating ...")
    import time
    @torch.no_grad()
    def _greedy_gen_timed(prefill_tokens: torch.Tensor, max_len: int):
        """Return generated tokens, state, prefill_time, decode_time.
        Optionally logs per-step stats to CSV if --stats-log is provided.
        """
        state = None
        B = prefill_tokens.size(0)
        gen = torch.empty((B, max_len), dtype=torch.long, device=device)

        stats_fp = None
        if args.stats_log:
            stats_path = Path(args.stats_log)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_fp = open(stats_path, "w")
            stats_fp.write("step,dt_ms,cum_ms,inst_tok_s,avg_tok_s\n")

        # Prefill step (full prompt)
        gauges = make_gauges()
        t0 = time.time()
        logits, state = model(prefill_tokens, state)
        t1 = time.time()
        # Emit prefill throughput if time is non-trivial
        prompt_len = prefill_tokens.shape[1]
        prefill_tps = prompt_len / max((t1 - t0), 1e-9)
        try:
            gauges["tok_s_prefill"].set(prefill_tps)
        except Exception:
            pass
        next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
        gen[:, 0:1] = next_tok

        # Decode steps
        decode_time = 0.0
        cum = 0.0
        for i in range(1, max_len):
            td0 = time.time()
            logits, state = model(next_tok, state)
            td1 = time.time()
            dt = td1 - td0
            decode_time += dt
            cum += dt
            next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
            gen[:, i:i+1] = next_tok
            if stats_fp and (i % max(args.stats_every, 1) == 0):
                inst = 1.0 / max(dt, 1e-9)
                avg = i / max(cum, 1e-9)
                stats_fp.write(f"{i},{dt*1000.0:.3f},{cum*1000.0:.3f},{inst:.3f},{avg:.3f}\n")
            # Emit decode instantaneous tok/s to Ray dashboard if available
            try:
                gauges["tok_s_decode"].set(1.0 / max(dt, 1e-9))
            except Exception:
                pass
        if stats_fp:
            stats_fp.close()
        return gen, state, (t1 - t0), decode_time

    mem_mon = MemoryMonitor(log_csv_path=args.mem_log).start() if args.mem_log else None
    try:
        tokens, _, t_prefill, t_decode = _greedy_gen_timed(inputs, args.max_new_tokens)
    finally:
        if mem_mon is not None:
            mem_mon.stop()
    dt = t_prefill + t_decode
    print("\nOutput:")
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(text)
    # Metrics: tokens/sec for decode and prefill throughput (tokens per second)
    prompt_len = inputs.shape[1]
    dec_tps = (args.max_new_tokens - 1) / max(t_decode, 1e-9)
    prefill_tps = prompt_len / max(t_prefill, 1e-9)
    print(f"\nTiming: total={dt:.2f}s (prefill={t_prefill:.2f}s, decode={t_decode:.2f}s)")
    print(f"Throughput: prefill={prefill_tps:.1f} tok/s, decode={dec_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
