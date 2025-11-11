## Summary

Explain the change in 1–3 sentences. Link any relevant issues or docs.

## Checklist (Policy)

- [ ] Reuse first: I searched for existing implementations (commands I ran):
    - `rg -n "<keywords>" <paths>`
- [ ] No mocks/stubs in production paths (only in tests/examples if any)
- [ ] Avoided “simplified/toy/placeholder/dummy/fake” language in production code/comments
- [ ] If I called `ray.init(...)`, I also ensured an explicit `ray.shutdown()` on normal exit
- [ ] No large artifacts added (>25MB) unless explicitly allowed (and documented)
- [ ] Updated docs/AGENTS.md, LOGGING_AND_OBSERVABILITY.md, or README if user‑visible behavior changed

## Validation

- [ ] I ran a quick sanity check (commands):
    - `conda run -n base python -c "import <module>; print('ok')"`
    - `conda run -n base python scripts/xltop.py --json` (optional)

## Notes / Risks

Call out anything that’s not addressed yet, assumptions, or follow‑ups.

