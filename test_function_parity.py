#!/usr/bin/env python3
"""Function-level parity harness between MLX implementation and canonical math.

This script exercises key pieces of the mLSTM pipeline (embedding norm,
soft-cap gates, multi-head RMSNorm) using real activations captured from
an actual forward pass. Each comparison reports the maximum absolute
error so we can track numerical drift component by component.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tests.parity.canonical_reference import (
    canonical_multihead_rmsnorm,
    canonical_mlstm_recurrent_sequence,
    canonical_rmsnorm,
    canonical_soft_cap,
)

import importlib.util

chunkwise_path = Path(__file__).parent / "quarantine" / "mlx_native" / "blocks" / "mlstm" / "chunkwise_mlx.py"
spec = importlib.util.spec_from_file_location("chunkwise_mlx_ref", chunkwise_path)
chunkwise_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(chunkwise_module)
mlstm_chunkwise_mlx = chunkwise_module.mlstm_chunkwise_mlx


def run_chunkwise_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    i_preact: mx.array,
    f_preact: mx.array,
    chunk_size: int,
    eps: float,
):
    B, NH, S, _ = q.shape
    assert S % chunk_size == 0, "Reference chunkwise path expects full chunks"
    h_ref, state_ref = mlstm_chunkwise_mlx(
        q,
        k,
        v,
        i_preact,
        f_preact,
        chunk_size=chunk_size,
        eps=eps,
        return_last_states=True,
    )
    return h_ref, state_ref
from xlstm_metal.mlx_jit.models.wired_xlstm import WiredxLSTM
from xlstm_metal.mlx_jit.tokenizer import TokenizerBlock, TokenizerConfig
from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards
from diagnostic_weights import MODEL_DIR, build_model  # reuse helpers

PROMPTS = [
    "Hello",
    "The quick brown",
    "Once upon a",
    "2 + 2 =",
]


def compare_arrays(name: str, actual: mx.array, reference: np.ndarray) -> Dict[str, float]:
    actual_np = np.array(actual)
    diff = np.abs(actual_np - reference)
    return {
        "component": name,
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
    }


def main() -> None:
    print("=" * 60)
    print("FUNCTION PARITY DIAGNOSTIC")
    print("=" * 60)

    print("Loading deterministic weights...")
    weights = load_safetensor_shards(str(MODEL_DIR))
    model = build_model(weights)
    tokenizer = TokenizerBlock(TokenizerConfig(model_path=str(MODEL_DIR)))
    print("✓ Model and tokenizer ready\n")

    results: Dict[str, list] = {}

    for prompt in PROMPTS:
        print(f"Prompt: '{prompt}'")
        ids = tokenizer.encode(prompt)
        if ids.ndim == 1:
            ids = mx.expand_dims(ids, axis=0)

        # Embedding output
        embeddings = model.embedding(ids)
        block = model.blocks[0]

        # 1) RMSNorm before mLSTM
        eps = float(block.norm_mlstm._eps.item())
        norm_weight = np.array(block.norm_mlstm.weight) if hasattr(block.norm_mlstm, "weight") else None
        rms_actual = block.norm_mlstm(embeddings)
        rms_ref = canonical_rmsnorm(np.array(embeddings), eps=eps, weight=norm_weight)
        res = compare_arrays("norm_mlstm", rms_actual, rms_ref)
        results.setdefault(res["component"], []).append(res)
        print(f"  RMSNorm Δ (max/mean): {res['max_abs']:.3e} / {res['mean_abs']:.3e}")

        # 2) Soft-cap gates inside projection cell
        proj = block.mlstm_cell.projection_cell
        cap = proj.gate_soft_cap

        proj.gate_soft_cap = None
        q_raw, k_raw, v_raw, i_raw, f_raw = proj(rms_actual)
        proj.gate_soft_cap = cap
        q, k, v, i_capped, f_capped = proj(rms_actual)

        if cap is not None:
            ref_i = canonical_soft_cap(np.array(i_raw), cap)
            ref_f = canonical_soft_cap(np.array(f_raw), cap)
            res_i = compare_arrays("soft_cap_i", i_capped, ref_i)
            res_f = compare_arrays("soft_cap_f", f_capped, ref_f)
            results.setdefault(res_i["component"], []).append(res_i)
            results.setdefault(res_f["component"], []).append(res_f)
            print(f"  Soft-cap Δ (i): {res_i['max_abs']:.3e}")
            print(f"  Soft-cap Δ (f): {res_f['max_abs']:.3e}")

        # 3) Multi-head RMSNorm inside output cell
        kernel_mode = getattr(block.mlstm_cell, "kernel_mode", "parallel")
        h_seq, state_seq = block.mlstm_cell.recurrent_kernel(q, k, v, i_capped, f_capped, None)
        output_cell = block.mlstm_cell.output_cell
        h_heads = h_seq.transpose(0, 2, 1, 3)
        h_norm = output_cell.norm(h_heads)
        mh_weight = np.array(output_cell.norm.weight).reshape(1, 1, -1)
        canon_mh = canonical_multihead_rmsnorm(
            np.array(h_heads),
            eps=float(output_cell.norm.inner._eps.item()),
            weight=mh_weight,
        )
        res_mh = compare_arrays("multihead_rms", h_norm, canon_mh)
        results.setdefault(res_mh["component"], []).append(res_mh)
        print(f"  Multi-head RMS Δ: {res_mh['max_abs']:.3e}")

        # 4) Recurrent kernel parity (sequential reference)
        q_np = np.array(q)
        k_np = np.array(k)
        v_np = np.array(v)
        i_np = np.array(i_capped)
        f_np = np.array(f_capped)
        h_ref, state_ref = canonical_mlstm_recurrent_sequence(
            q_np,
            k_np,
            v_np,
            i_np,
            f_np,
            eps=float(block.mlstm_cell.eps),
        )
        h_res = compare_arrays("recurrent_h", h_seq, h_ref)
        results.setdefault("recurrent_h", []).append(h_res)
        print(f"  Recurrent h Δ: {h_res['max_abs']:.3e}")

        c_mx, n_mx, m_mx = state_seq
        c_res = compare_arrays("state_C", c_mx, state_ref[0])
        n_res = compare_arrays("state_N", n_mx, state_ref[1])
        m_res = compare_arrays("state_M", m_mx, state_ref[2].squeeze(-1))
        for label, entry in zip(["state_C", "state_N", "state_M"], [c_res, n_res, m_res]):
            results.setdefault(label, []).append(entry)
        print(f"  State Δ (C/N/M): {c_res['max_abs']:.3e} / {n_res['max_abs']:.3e} / {m_res['max_abs']:.3e}")

        if kernel_mode == "parallel":
            chunk_size = getattr(block.mlstm_cell, "chunk_size", q.shape[2])
            full_chunks = (q.shape[2] // chunk_size)
            chunk_tokens = full_chunks * chunk_size
            if chunk_tokens > 0:
                head = slice(0, chunk_tokens)
                h_parallel, state_parallel = block.mlstm_cell.parallel_kernel(
                    q[:, :, head, :],
                    k[:, :, head, :],
                    v[:, :, head, :],
                    i_capped[:, :, head],
                    f_capped[:, :, head],
                    None,
                )

                h_chunk_ref, state_chunk_ref = run_chunkwise_reference(
                    q[:, :, head, :],
                    k[:, :, head, :],
                    v[:, :, head, :],
                    i_capped[:, :, head],
                    f_capped[:, :, head],
                    chunk_size=chunk_size,
                    eps=block.mlstm_cell.eps,
                )

                res_chunk = compare_arrays("chunkwise_h", h_parallel, np.array(h_chunk_ref))
                results.setdefault("chunkwise_h", []).append(res_chunk)
                print(f"  Chunkwise h Δ: {res_chunk['max_abs']:.3e}")

                c_res_chunk = compare_arrays("chunk_state_C", state_parallel[0], np.array(state_chunk_ref[0]))
                n_res_chunk = compare_arrays("chunk_state_N", state_parallel[1], np.array(state_chunk_ref[1]))
                m_ref = state_chunk_ref[2]
                if m_ref.ndim == 3:
                    m_ref = m_ref[:, :, -1]
                if state_parallel[2].ndim == 3:
                    m_ref = m_ref[:, :, None]
                m_res_chunk = compare_arrays("chunk_state_M", state_parallel[2], np.array(m_ref))
                results.setdefault("chunk_state_C", []).append(c_res_chunk)
                results.setdefault("chunk_state_N", []).append(n_res_chunk)
                results.setdefault("chunk_state_M", []).append(m_res_chunk)
                print(
                    f"  Chunk states Δ (C/N/M): {c_res_chunk['max_abs']:.3e} / {n_res_chunk['max_abs']:.3e} / {m_res_chunk['max_abs']:.3e}"
                )

        print()

    print("Summary (max across prompts):")
    for component, entries in results.items():
        max_abs = max(e["max_abs"] for e in entries)
        mean_abs = max(e["mean_abs"] for e in entries)
        print(f"  {component:15s} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}")


if __name__ == "__main__":
    main()
