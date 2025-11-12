# xLSTM-Metal Documentation

**Project:** xLSTM for Apple Silicon (MLX + Metal)  
**Author:** Sydney Renee (sydney@solace.ofharmony.ai)  
**Status:** Production (January 2025)

This directory contains technical documentation for the working xLSTM-Metal implementation.

## Quick Navigation

### 🏗️ Architecture & Design
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Complete system architecture
  - Component overview
  - NCPS wiring system
  - mLSTM/sLSTM blocks
  - Metal kernel design
  - Implementation details
  - Performance characteristics
  - Design decisions and tradeoffs

### 📚 API Reference
- **[COMPONENTS.md](./COMPONENTS.md)** - Auto-generated API documentation
  - FFN (SwiGLU feed-forward)
  - mLSTM cells and blocks
  - sLSTM cells and blocks
  - Metal forward/backward kernels
  - NCPS wiring infrastructure
  - Normalization layers
  - Utilities (config, weights, loaders)

### 🧠 NCPS Wiring
- **[NCPS_WIRING.md](./NCPS_WIRING.md)** - Neural Circuit Policy wiring system
  - Auto-discovery from safetensors
  - Block structure introspection
  - Sequential execution model
  - Why NCPS over MAD stages

### 🔢 Numerical Stability
- **[NUMERICAL_STABILITY.md](./NUMERICAL_STABILITY.md)** - Precision engineering
  - Float32 requirements
  - Why not bfloat16
  - Critical stability patterns
  - Emberlint enforcement rules
  - Debugging NaN issues

## Documentation Organization

```
docs/
├── README.md                    # This file
├── ARCHITECTURE.md              # System design (main reference)
├── COMPONENTS.md                # API documentation (auto-generated)
├── NCPS_WIRING.md              # Wiring system details
├── NUMERICAL_STABILITY.md      # Precision patterns
└── METAL_KERNEL_GUIDE.md       # (from m2-bert-mlx, Metal programming)
```

## Historical Documentation

Experimental designs, debugging logs, and research notes are preserved in:

```
docs_archive/
├── architecture/        # MAD system (historical), current NCPS analysis
├── debugging/           # Bug fixes (dtype confusion, kernel issues)
├── lessons_learned/     # Distilled debugging principles
├── components/          # Metal kernel implementation notes
├── experiments/         # Flowlang DSL, other prototypes
├── research/            # Future work, academic analysis
├── porting/             # Platform-specific notes
└── project_management/  # Milestones, summaries
```

See `docs_archive/README.md` for navigation guide.

## How to Use This Documentation

**If you want to...**

| Goal | Start Here |
|------|------------|
| Understand the overall system | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| Look up a specific class or function | [COMPONENTS.md](./COMPONENTS.md) |
| Learn about the wiring system | [NCPS_WIRING.md](./NCPS_WIRING.md) |
| Debug numerical issues | [NUMERICAL_STABILITY.md](./NUMERICAL_STABILITY.md) |
| Write custom Metal kernels | [METAL_KERNEL_GUIDE.md](./METAL_KERNEL_GUIDE.md) |
| Understand design decisions | [ARCHITECTURE.md § Design Decisions](./ARCHITECTURE.md#design-decisions) |
| See what didn't work | `docs_archive/debugging/` |
| Find future research directions | `docs_archive/research/` |

## Documentation Philosophy

**Active docs (this directory):**
- Describe the working, production system
- Maintained in sync with code
- Authoritative technical reference
- Suitable for external users

**Archive docs (docs_archive/):**
- Preserve intellectual history
- Document failed experiments (why they failed)
- Future research directions
- Planning notes and milestones
- Internal development reference

**Docstrings (in code):**
- API-level documentation
- Mathematical explanations
- Usage examples
- Extracted to COMPONENTS.md automatically

## Related Projects

This work builds on and contributes to:

- **[ncps-mlx](https://github.com/MLXPorts/ncps-mlx)** - NCPS wiring patterns for MLX
- **[m2-bert-mlx](https://github.com/MLXPorts/m2-bert-mlx)** - Metal kernel design, FFT precision
- **[ember-ml](https://github.com/SolaceHarmony/ember-ml)** - Precision linting (emberlint)
- **[Faiss-mlx](https://github.com/MLXPorts/Faiss-mlx)** - Unified memory patterns

All part of The Solace Project ecosystem.

## Contributing to Documentation

**When adding new components:**
1. Write comprehensive docstrings in code
2. Run `python tools/extract_docstrings.py` to update COMPONENTS.md
3. Add architecture notes to ARCHITECTURE.md if needed
4. Update this README if adding new doc files

**When fixing bugs:**
1. Document the fix in code comments
2. Add summary to `docs_archive/lessons_learned/`
3. Update NUMERICAL_STABILITY.md if precision-related

**Documentation style:**
- Technical and precise
- Include code examples
- Explain *why* not just *what*
- Reference papers/projects where applicable
- Use Sydney's voice (engineering-focused, no fluff)

## Questions?

**Technical issues:** Open GitHub issue  
**Documentation gaps:** PR or issue  
**Design discussions:** sydney@solace.ofharmony.ai

---

*This documentation represents production-tested code. Experimental approaches are preserved in `docs_archive/` for historical reference and future research.*
