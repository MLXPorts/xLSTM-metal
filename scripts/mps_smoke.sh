#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.

echo "[mps_smoke] Preflight (MPS + Ray)..."
python - <<'PY'
from src.lnn_hrm.preflight import assert_mps, assert_ray
assert_mps(); assert_ray()
print('Preflight OK: MPS+Ray present')
PY

echo "[mps_smoke] Running wrapper demo..."
python examples/xlstm_hrm_wrapper_demo.py

echo "[mps_smoke] Running tiny trainer..."
python examples/train_with_ponder_demo.py

echo "[mps_smoke] Running unit tests (CPU-safe + MPS-only)..."
pytest -q tests/test_memory_cube_behavior.py tests/test_trace_and_logging.py::test_trace_hash_determinism
pytest -q tests/test_wrapper_multiblock.py tests/test_act_energy_telemetry.py tests/test_trace_and_logging.py::test_ponder_trainer_writes_logs

echo "[mps_smoke] Done. Logs at runs/telem_demo/ and any tmp test dirs."
