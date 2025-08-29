# FAQ

**Q: Why residuals instead of full outputs?**
- Better conditioning and safety; easier to cap and audit.

**Q: Can cubes drift and corrupt outputs?**
- Gate hysteresis, audits, and residual clamps limit damage; demotion and purge on failures.

**Q: How big should cubes be?**
- Start small (16k) and scale with memory and hit rates.

**Q: Is this faster than baseline?**
- On repeated patterns, cubes amortize cost; liquid adds light compute; net impact is workload-dependent.

**Q: Do we still need CoT?**
- Not required; HRM provides latent iterative refinement; CoT can be layered if desired.

**Q: Does α ever reach 1.0?**
- Only after sustained confidence and low drift; never in one step due to rate limits.

