# debug

Low-level probes and debug utilities for Apple MPS and tensor plumbing.

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**

Tools
- `debug_dims.py`: Shape/dim checks for typical model paths.
- `debug_mps_memory.py`: Inspect MPS memory behavior.
- `debug_storage_api.py`: Explore tensor storage internals.
- `mps_probe`: Quick platform check for MPS availability/capabilities.

When to use
- Investigating performance hiccups or memory fragmentation.
- Verifying tensor/device properties before running heavy jobs.
