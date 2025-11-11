"""
Validate presence and placement of upstream xLSTM files and Metal kernel scaffolding.

This does not import modules; it only inspects the workspace to:
- Confirm upstream-like files exist in xlstm_official (or xlstm_official_full subset)
- Confirm our implementation locations exist
- Confirm Metal kernels and bindings exist and are discoverable

Exit code 0 on success; non-zero if critical files are missing.
Prints a human-readable report to stdout.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_files(base: Path, rels: list[str]) -> tuple[list[str], list[str]]:
    """Checks for the presence of files relative to a base path.

    Args:
        base (Path): The base path to check against.
        rels (list[str]): A list of relative paths to check.

    Returns:
        A tuple containing two lists: the first with the present files and the
        second with the missing files.
    """
    present, missing = [], []
    for rel in rels:
        p = base / rel
        if p.exists():
            present.append(rel)
        else:
            missing.append(rel)
    return present, missing


def main() -> int:
    """Validates the project layout and prints a report.

    This function checks for the presence of critical files and directories in
    the project, including the upstream xLSTM files, the local implementation,
    and the Metal kernels. It prints a report to stdout and returns an exit
    code indicating success or failure.

    Returns:
        0 if the validation is successful, 1 otherwise.
    """
    ok = True
    report = []

    # Upstream mirror checks
    upstream_base = ROOT / "xlstm_official"
    upstream_expected = [
        # xlstm_large subset
        "xlstm_large/components.py",
        "xlstm_large/utils.py",
        "xlstm_large/model.py",
        "xlstm_large/from_pretrained.py",
        "xlstm_large/generate.py",
        # components
        "components/conv.py",
        "components/linear_headwise.py",
        "components/ln.py",
        "components/init.py",
        "components/feedforward.py",
        "components/util.py",
        "components/__init__.py",
        # blocks (present in mirror; slstm/cuda not required to run MPS but should be present)
        "blocks/mlstm/block.py",
        "blocks/mlstm/layer.py",
        "blocks/mlstm/cell.py",
        "blocks/slstm/block.py",
        "blocks/slstm/layer.py",
        "blocks/slstm/cell.py",
    ]
    up_present, up_missing = check_files(upstream_base, upstream_expected)
    report.append(f"[upstream] base={upstream_base}")
    report.append(f"  present: {len(up_present)}")
    if up_missing:
        ok = False
        report.append(f"  missing: {len(up_missing)}")
        for m in up_missing:
            report.append(f"    - {m}")

    # Our implementation checks
    impl_base = ROOT / "xlstm_impl"
    impl_expected = [
        "models/xlstm.py",
        "models/xlstm_block_stack.py",
        "backends/mlstm_backend.py",
        "layers/mlstm/block.py",
        "layers/mlstm/layer.py",
        "utils/device.py",
    ]
    impl_present, impl_missing = check_files(impl_base, impl_expected)
    report.append(f"[impl] base={impl_base}")
    report.append(f"  present: {len(impl_present)}")
    if impl_missing:
        ok = False
        report.append(f"  missing: {len(impl_missing)}")
        for m in impl_missing:
            report.append(f"    - {m}")

    # Kernel placement checks (current state)
    metal_base = ROOT / "kernels/metal"
    metal_expected = [
        "shaders/mlstm_kernels.metal",
        "pytorch_ext/mlstm_metal_backend.mm",
        # optional helper
        "pytorch_ext/setup.py",
    ]
    metal_present, metal_missing = check_files(metal_base, metal_expected)
    # Fallback to archived prototypes if not present in kernels/metal
    if metal_missing:
        metal_base = ROOT / "research_archive/metal_prototypes/kernels_metal"
        metal_present, metal_missing = check_files(metal_base, metal_expected)
    report.append(f"[metal-kernels] base={metal_base}")
    report.append(f"  present: {len(metal_present)}")
    if metal_missing:
        # Not fatal if setup.py absent, but flag
        report.append(f"  missing: {len(metal_missing)}")
        for m in metal_missing:
            report.append(f"    - {m}")

    # Torch kernel registry checks (ensures selectable backends)
    kernels_base = ROOT / "mlstm_kernels" / "torch"
    kernels_expected = [
        "backend_module.py",
        "chunkwise/__init__.py",
        "recurrent/__init__.py",
        # metal forward stub exists today
        "chunkwise/metal/fw.py",
    ]
    krn_present, krn_missing = check_files(kernels_base, kernels_expected)
    report.append(f"[kernel-registry] base={kernels_base}")
    report.append(f"  present: {len(krn_present)}")
    if krn_missing:
        ok = False
        report.append(f"  missing: {len(krn_missing)}")
        for m in krn_missing:
            report.append(f"    - {m}")

    # Print report
    print("\n".join(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
