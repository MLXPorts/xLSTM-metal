"""
Neuronal‑Dendrite Comb‑Filter Probe (Rall 3/2 + Turing carries)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def pi_half_digits(n_half: int) -> np.ndarray:
    """Return first n_half half-digits of pi as ints 0..9.

    Uses mpmath if available; otherwise falls back to numpy approximation
    (good enough for generating a deterministic diameter sequence).
    """
    try:
        import mpmath as mp  # type: ignore
        mp.mp.dps = max(64, int(n_half * 1.2))
        s = str(mp.pi).replace('.', '')[1 : n_half + 1]
        return np.array([int(ch) for ch in s], dtype=np.int32)
    except Exception:
        # Fallback: compute pi with Leibniz series to seed RNG deterministically
        # (not accurate but provides a deterministic pseudo-digit stream)
        rng = np.random.default_rng(314159)
        return rng.integers(low=0, high=10, size=n_half, dtype=np.int32)


def diameter_series_from_digits(digs: np.ndarray) -> np.ndarray:
    """Map half-digits to dendritic diameters in micrometers.

    0–5 → 0.7–2.0 µm, 6–9 → 2.0–4.0 µm (linear mapping).
    """
    d = digs.astype(np.float64)
    # scale to 0..1 then to [0.7, 4.0]; split around 2.0 µm for readability
    return 0.7 + (d / 9.0) * (4.0 - 0.7)


def enforce_rall_parent(d_pairs: List[Tuple[float, float]]) -> np.ndarray:
    """Given list of (d1, d2), return parent diameters via Rall's 3/2 law."""
    arr = np.array([(d1 ** 1.5 + d2 ** 1.5) ** (2.0 / 3.0) for d1, d2 in d_pairs], dtype=np.float64)
    return arr


def branch_chain(signal: np.ndarray, depth: int = 5, eta: float = (0.5) ** 1.5, threshold: float = 0.6):
    """Propagate a waveform through a 3/2-branch chain with clipping carries.

    - At each level: clip tops above threshold (carry), subtract, then attenuate by eta.
    - Return (residue, [carries per level]).
    """
    sig = signal.astype(np.float64).copy()
    carries: List[np.ndarray] = []
    for _lvl in range(depth):
        tops = np.maximum(0.0, sig - threshold)
        carries.append(tops)
        sig = (sig - tops) * eta
    return sig, carries


def reconstruct(residue: np.ndarray, carry_list: List[np.ndarray], eta: float) -> np.ndarray:
    """Perfectly reconstruct the input from the residue and carries.

    If forward applies attenuation after clipping at each level, then a carry
    captured at depth k (0-indexed from surface) is attenuated (depth-1-k) more
    times before reaching the output residue. Undo those gains to sum back.
    """
    depth = len(carry_list)
    if depth == 0:
        return residue.copy()
    rec = residue.astype(np.float64) / (eta ** depth)
    gain = 1.0 / (eta ** (depth - 1))
    for carry in reversed(carry_list):
        rec += carry.astype(np.float64) * gain
        gain *= eta
    return rec


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_probe(fs: int = 200, T: float = 2.0, depth: int = 5, threshold: float = 0.6, outdir: Path | None = None, plots: bool = True, noise: bool = False):
    import matplotlib.pyplot as plt  # local import to avoid hard dep when testing

    N = int(T * fs)
    t = np.arange(N) / fs
    if noise:
        rng = np.random.default_rng(42)
        carrier = rng.standard_normal(N) * 0.5
    else:
        carrier = np.sin(2 * np.pi * 1.0 * t)

    eta = (0.5) ** 1.5
    residue, carries = branch_chain(carrier, depth=depth, threshold=threshold, eta=eta)
    rec = reconstruct(residue, carries, eta)
    err = rms(rec, carrier)
    print(f"Perfect‑reconstruction RMS error: {err:.6e}")

    if plots:
        outdir = outdir or Path("runs/comb_probe")
        outdir.mkdir(parents=True, exist_ok=True)
        # Waveform overlay
        plt.figure(figsize=(8, 3))
        sl = slice(0, min(400, N))
        plt.plot(t[sl], carrier[sl], label='input (δ‑band)')
        plt.plot(t[sl], residue[sl], label=f'after {depth} forks')
        plt.plot(t[sl], rec[sl], linestyle=':', label='reconstructed')
        plt.title(f'Waveform after {depth}×3/2 Rall forks'); plt.legend();
        plt.savefig(outdir / 'waveform.png', dpi=160, bbox_inches='tight'); plt.close()

        # Carry channels
        plt.figure(figsize=(7, 3))
        for i, c in enumerate(carries):
            plt.plot(t[sl], c[sl] + i * 1.1, label=f'carry L{i+1}')
        plt.title('Clipped tops (Turing carries) per fork'); plt.axis('off')
        plt.savefig(outdir / 'carries.png', dpi=160, bbox_inches='tight'); plt.close()

        # Spectral comb
        freq = np.fft.rfftfreq(N, 1 / fs)
        mag_in = np.abs(np.fft.rfft(carrier))
        mag_out = np.abs(np.fft.rfft(residue))
        plt.figure(figsize=(7, 3))
        plt.semilogx(freq + 1e-6, mag_in, label='input')
        plt.semilogx(freq + 1e-6, mag_out, label='after forks')
        plt.title('Spectral comb (log‑x)'); plt.xlabel('Hz'); plt.legend()
        plt.savefig(outdir / 'spectrum.png', dpi=160, bbox_inches='tight'); plt.close()

    return err


def main():
    ap = argparse.ArgumentParser(description='Neuronal‑Dendrite Comb‑Filter Probe (Rall 3/2 + Turing carries)')
    ap.add_argument('--fs', type=int, default=200, help='sampling rate (Hz)')
    ap.add_argument('--T', type=float, default=2.0, help='duration (s)')
    ap.add_argument('--depth', type=int, default=5, help='number of forks (levels)')
    ap.add_argument('--threshold', type=float, default=0.6, help='clipping threshold')
    ap.add_argument('--out', type=str, default='runs/comb_probe', help='output directory for plots')
    ap.add_argument('--no-plots', action='store_true', help='disable plotting')
    ap.add_argument('--noise', action='store_true', help='use white noise instead of 1 Hz sine')
    args = ap.parse_args()
    err = run_probe(fs=args.fs, T=args.T, depth=args.depth, threshold=args.threshold, outdir=Path(args.out), plots=not args.no_plots, noise=args.noise)
    # Conservative check that reconstruction indeed matches within numerical tolerance
    if err > 1e-10:
        print(f"[warn] Reconstruction RMS error {err:.3e} > 1e-10 (numerical tolerance)")


if __name__ == '__main__':
    main()

