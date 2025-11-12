# xLSTM-Metal Documentation Archive

This directory contains historical documentation, design notes, debugging records, and research that informed the development of xLSTM-Metal. This is the project's intellectual history—what we tried, what worked, what failed, and why.

**⚠️ Historical Archive** - For current production documentation, see `../docs/`

## Archive Organization

### 📁 debugging/
Bug investigations and resolutions. Chronological record of issues encountered.

- **dtype_confusion/** - Python/MLX type mismatches, NaN propagation
- **kernel_issues/** - Metal kernel-specific bugs (RMSNorm, etc.)
- **COMPLETE_FIX_SUMMARY.md** - Master timeline of all debugging efforts

**Why it matters:** Shows common pitfalls when porting PyTorch→MLX and working with Metal kernels.

### 📁 lessons_learned/
Distilled knowledge from debugging. These aren't bug reports—they're principles.

- **NUMERIC_STABILITY_TORCH_vs_MLX.md** - Key differences between frameworks
- **MLX_NUMERICS_LAB.md** - Experimental findings on numerical behavior

**Why it matters:** Should inform future development and new documentation.

### 📁 architecture/
Design evolution. What shipped vs. what didn't.

- **current/** - NCPS-based design that shipped
  - XLSTM_MAD_NCPS_DESIGN.md
  - MLSTM_NUMERICAL_STABILITY_ANALYSIS.md
  - Block execution and component architecture
  
- **historical/** - Significant efforts that were replaced
  - **mad_system/** - MAD (Metal Accelerated Dispatch) architecture
    - Was the original design approach
    - Replaced by NCPS wiring for better parallelism
    - Kept for context and potential future use
  - TTT (Test-Time Training) kernel experiments
  - Alternative parallelism strategies

**Why it matters:** MAD represents months of work. It didn't ship, but the ideas may resurface.

### 📁 components/
Implementation deep-dives for specific subsystems.

- **kernels/** - Metal kernel development guides
  - MLX_Metal_Kernel_Guide.md (comprehensive)
  - MLX_KERNEL_PATTERNS.md
  - TRITON_KERNELS_DEEP_DIVE.md
  - MLX_METAL_SHADER_INTEGRATION.md

**Why it matters:** Essential for understanding custom Metal kernels and MLX integration.

### 📁 experiments/
Prototypes and research code that never integrated.

- **flowlang/** - Experimental DSL for flow-based programming
  - Never integrated into production
  - Interesting ideas for future meta-programming
  
- *Future:* liquid_time_constant/, alternative_architectures/

**Why it matters:** Documents what was tried. Saves future developers from repeating experiments.

### 📁 research/
Academic analysis, competitive comparisons, long-term research directions.

- REVERSIBLE_RNN_NOTES.md - Reversible architecture investigations
- HOGWILD_ANALYSIS.md - Lock-free parallelism research
- MEGATRON_MODEL_PARALLELISM_ANALYSIS.md - Scaling strategies
- xLSTM_vs_LFM2_Comparison.md - Competitive analysis
- EXTENDED_CONTEXT_PLAN.md - Long context handling
- STATE_EXPANSION_PRECISION.md - Precision analysis

**Why it matters:** Informs future research directions. Not immediate TODOs but worth revisiting.

### 📁 porting/
Platform-specific implementation notes and cross-platform comparisons.

- **mlx_metal/** - MLX/Metal inference architecture
- **pytorch_mps/** - PyTorch MPS comparison, conv1d differences
- **coreml/** - CoreML/ANE guidance
- **ray_coroutines/** - Ray integration attempts
- **M2BERT_ARCHITECTURE_ANALYSIS.md** - Influenced Metal kernel design

**Why it matters:** Essential context for cross-platform work and understanding design constraints.

### 📁 project_management/
Task completion summaries, milestone tracking, meta-analysis.

- CONFIG_DRIVEN_ARCHITECTURE_VERIFIED.md
- HYPERPROFILE_PYTORCH_MLX_ANALYSIS.md
- DOCSTRING_ENRICHMENT_SUMMARY.md
- RESEARCH_NOTES.md

**Why it matters:** Project velocity tracking and validation checkpoints.

## Navigation Guide

**If you want to...**

- **Understand a bug:** Start with `debugging/COMPLETE_FIX_SUMMARY.md`, drill into specific issues
- **Learn MLX/Metal patterns:** Read `components/kernels/MLX_Metal_Kernel_Guide.md`
- **Understand why NCPS:** See `architecture/current/XLSTM_MAD_NCPS_DESIGN.md`
- **Know what MAD was:** Read `architecture/historical/mad_system/`
- **Compare to PyTorch:** Check `lessons_learned/NUMERIC_STABILITY_TORCH_vs_MLX.md`
- **Port to another platform:** Review `porting/` for your target
- **Research extensions:** Browse `research/` for academic analysis

## Relationship to Production Docs

| Directory | Purpose | Audience |
|-----------|---------|----------|
| `docs/` | Current system documentation | Users, contributors |
| `docs_archive/` | Historical context & research | Maintainers, researchers |

**Golden rule:** If it describes the current working system, it belongs in `docs/`. If it's history, experiments, or bugs we fixed, it belongs here.

## Contributing to the Archive

When archiving new material:

1. **debugging/** - Add dated bug reports with resolution
2. **lessons_learned/** - Extract principles from debugging
3. **architecture/historical/** - Document deprecated designs with "why we moved on"
4. **experiments/** - Failed experiments should explain what didn't work
5. **research/** - External research that informed decisions

Keep the narrative honest. Document failures as thoroughly as successes.

---

**Archive maintained by:** Sydney Renee (sydney@solace.ofharmony.ai)  
**Last reorganization:** 2024-11-12
