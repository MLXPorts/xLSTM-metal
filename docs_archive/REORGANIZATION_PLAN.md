# Documentation Archive Reorganization Plan

## Current Issues
1. Root-level files need categorization
2. "plan/" is ambiguous (research vs todo items)
3. "summaries/" mixes different summary types
4. Historical MAD architecture mixed with production NCPS

## Proposed Structure

```
docs_archive/
├── README.md                          # Navigation guide
│
├── debugging/                         # Bug investigations & fixes
│   ├── dtype_confusion/
│   │   ├── FIX_DTYPE_ISSUE.md
│   │   └── FIX_NAN_ISSUE.md
│   ├── kernel_issues/
│   │   └── FIX_RMSNORM_METAL_KERNEL.md
│   └── COMPLETE_FIX_SUMMARY.md       # Master debugging timeline
│
├── lessons_learned/                   # Hard-won knowledge
│   ├── NUMERIC_STABILITY_TORCH_vs_MLX.md
│   └── MLX_NUMERICS_LAB.md           # Experimental findings
│
├── architecture/                      # Design evolution (keep existing)
│   ├── current/                      # What we shipped
│   │   └── (keep relevant design docs)
│   └── historical/                   # MAD, alternatives explored
│       └── mad_system/               # Move MAD docs here
│
├── components/                        # Implementation deep-dives
│   ├── kernels/                      # Metal kernel guides (keep)
│   └── blocks/                       # Block implementations
│
├── experiments/                       # Prototypes & research code
│   ├── flowlang/                     # DSL experiment (move from root)
│   ├── liquid_time_constant/         # LTC neuron integration attempts
│   └── alternative_architectures/     # Non-NCPS approaches
│
├── research/                          # Academic analysis & comparisons
│   ├── REVERSIBLE_RNN_NOTES.md
│   ├── HOGWILD_ANALYSIS.md
│   ├── MEGATRON_MODEL_PARALLELISM_ANALYSIS.md
│   ├── xLSTM_vs_LFM2_Comparison.md
│   ├── EXTENDED_CONTEXT_PLAN.md
│   └── STATE_EXPANSION_PRECISION.md
│
├── porting/                           # Platform-specific (keep existing)
│   ├── mlx_metal/
│   ├── pytorch_mps/
│   ├── coreml/
│   └── ray_coroutines/
│
└── project_management/                # Tasks, milestones, summaries
    ├── CONFIG_DRIVEN_ARCHITECTURE_VERIFIED.md
    ├── HYPERPROFILE_PYTORCH_MLX_ANALYSIS.md
    ├── DOCSTRING_ENRICHMENT_SUMMARY.md
    └── RESEARCH_NOTES.md              # Meta-notes on research direction
```

## Migration Mapping

### Root files → New locations
- `FIX_DTYPE_ISSUE.md` → `debugging/dtype_confusion/`
- `FIX_NAN_ISSUE.md` → `debugging/dtype_confusion/`
- `FIX_RMSNORM_METAL_KERNEL.md` → `debugging/kernel_issues/`
- `COMPLETE_FIX_SUMMARY.md` → `debugging/` (keep at top level of debugging)
- `NUMERIC_STABILITY_TORCH_vs_MLX.md` → `lessons_learned/`
- `MLX_NUMERICS_LAB.md` → `lessons_learned/`
- `DOCSTRING_ENRICHMENT_SUMMARY.md` → `project_management/`

### plan/ → research/
All files in `plan/` are research notes, move entire folder contents to `research/`

### summaries/ → project_management/
These are project milestone summaries, not architectural summaries

### flowlang/ → experiments/flowlang/
Experimental DSL, never shipped

### components/mad/ → architecture/historical/mad_system/
MAD was replaced by NCPS, archive the design

## Rationale

**debugging/** -chronological bug history, organized by problem type
- Helps future debuggers find similar issues
- Shows evolution of understanding

**lessons_learned/** - distilled knowledge from debugging
- Not bugs themselves, but principles learned
- Should inform new docs/ content

**architecture/current vs historical/** - separate what shipped from what didn't
- MAD system was a significant effort but didn't ship
- Worth keeping for context but clearly marked historical

**experiments/** - clearly labeled prototypes
- Flowlang never integrated
- LTC experiments incomplete
- Document what was tried, why it didn't work

**research/** - academic/comparative analysis
- Not immediate todo items
- Long-term research directions
- Competitive analysis

**project_management/** - tracking work done
- Task completion summaries
- Milestone verifications
- Different from technical documentation

## Delete Candidates

Consider removing:
- Binary files in components/kernels/ (mlx_streams.py - not documentation)
- Duplicate summaries
- Obsolete planning docs superseded by actual implementation

## New README.md

Create comprehensive README.md in docs_archive/ explaining:
- Purpose of archive (historical context, not current docs)
- How to navigate
- What each top-level folder contains
- Relationship to main docs/
