import numpy as np


def test_reconstruction_rms_small():
    from lab.dendritic_comb_probe import branch_chain, reconstruct, rms

    fs = 200
    T = 1.0
    t = np.arange(int(fs * T)) / fs
    carrier = np.sin(2 * np.pi * 1.0 * t)
    eta = (0.5) ** 1.5
    residue, carries = branch_chain(carrier, depth=4, threshold=0.6, eta=eta)
    rec = reconstruct(residue, carries, eta)
    err = rms(rec, carrier)
    assert err < 1e-10

