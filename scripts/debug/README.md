# debug

Low-level probes and debug utilities for Apple MPS and tensor plumbing.

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**

Tools
- `mps_probe`: Quick platform check for MPS availability/capabilities.

When to use
- Investigating performance hiccups or memory fragmentation.
- Verifying tensor/device properties before running heavy jobs.
