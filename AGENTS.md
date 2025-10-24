## CLAUDE Guidance (Read Me First)

This repo runs on Apple Silicon with GPU-first compiled backends. These rules are strict so your help stays production‑safe and useful.

### Zero‑Mock Policy (Production Code)
- Don’t add mocks, stubs, or fake implementations to production paths. Use real implementations or leave TODOs in comments with precise next steps.
- If a mock is unavoidable for a demo, place it under test or example folders only, guarded by clear naming and comments. Never wire a mock into default execution.

### Always Reuse Before Rebuilding
- Before implementing anything, search the repo for existing code:
  - `rg -n "symbol|filename|keyword"` to locate reuse targets.
  - Scan `mlstm_kernels/`, `scripts/`, and `docs/` for prior art and conventions.
- Prefer refactoring or adapting existing modules to new needs rather than new, parallel implementations.
- If you must introduce new code, explain why reuse was insufficient and reference your searches.

### Lead With Vulnerability
- Be explicit about what you don’t know or what isn’t working yet.
- State assumptions and risks up front. Offer a short plan with checkpoints.
- After changes, verify and report limits (“what this didn’t solve”).

### Respect the Shell and Filesystem
- Use OS tools for file ops; avoid dumping large files into the terminal:
  - Copy/rename: `cp`, `mv` (not read‑and‑reprint).
  - Search: `rg` (ripgrep) for speed and focus.
  - Inspect selectively: `sed -n '1,200p' <file>` (chunked views only).
- Don’t stream giant artifacts or model weights to stdout. If you need to show something, quote only the relevant lines.

### Memory, GPU, and Process Hygiene
- On macOS/MPS, respect unified memory constraints; prefer our watchdog instead of ad‑hoc guards.
- If you start Ray in multi‑process mode, ensure a clean shutdown:
  - Call `ray.shutdown()` on normal exit.
  - If a crash leaves residue, run `ray stop --force`.
- Don’t leave background processes running. Confirm with `top`, `ps`, or `ray status`.

### Use the Existing Telemetry
- Prefer the repo’s tools instead of rolling new ones:
  - Memory/logging: `MemoryMonitor` and `--mem-*` flags.
  - Terminal monitor: `scripts/xltop.py` (TUI, JSON/NDJSON modes).
  - Dashboard: runner flags `--ray-dashboard [--ray-keep-alive]`.

### Safe Editing Practices
- Keep changes minimal and focused; do not refactor unrelated code.
- Mirror existing style and folder layout.
- Update documentation when adding flags, env vars, or user‑visible behavior.
- Never paste huge files into responses; prefer diffs and file paths.

### Validation Before Confidence
- If tests exist, run them. If a fast sanity check is possible, do it (e.g., import the module, run a short command).
- Show exact commands you used (with `conda run -n base python …` and `PYTHONPATH=.`) so others can reproduce.

### When Creating New Tools/Scripts
- Provide a concise CLI, a short `--help`, and defaults that are safe on Apple/MPS.
- Honor existing env vars (`XLSTM_*`, `PYTORCH_ENABLE_MPS_FALLBACK=0`).
- Add a brief note to `docs/LOGGING_AND_OBSERVABILITY.md` or `AGENTS.md` if user‑facing.

### Don’ts (Common Failure Modes)
- Don’t fabricate behavior or outputs; if something can’t be verified, say so.
- Don’t create alternate copies of existing modules with minor deltas; extend the original instead.
- Don’t spam the terminal with entire files or binary junk.
- Don’t introduce “temporary” shortcuts in production paths (e.g., skipping GPU for convenience).

### Banned/Discouraged Phrases (in production code/comments)
- “simplified”, “for simplicity”, “toy”, “placeholder”, “dummy implementation”, “fake implementation”
- “approximate”, “rough sketch”, “simulate”, “pretend”
- “we won’t implement”, “will not implement” (unless explicitly tracking a TODO with owner/date)

Use precise language: state constraints and the real implementation. If something remains undone, add a TODO with owner/date and the exact next step.

### Quick Command Snippets
- Search quickly: `rg -n "pattern" path/`
- View a file chunk: `sed -n '1,200p' path/to/file.py`
- Copy/rename: `cp src.py dst.py` • `mv old.py new.py`
- Kill a PID: `kill -TERM <pid>` then `kill -KILL <pid>` if needed
- Clean Ray residue: `ray stop --force`
- Memory snapshot: `conda run -n base python scripts/xltop.py --json`

### Ask When Unsure
- If the requirement is ambiguous, ask concise clarifying questions rather than guessing. Provide a proposed plan with options.

—
This file is the behavioral contract for Claude‑family assistants in this repo. Defer to `AGENTS.md` for tooling specifics and to `docs/` for backend details.
