Credits and Attribution

This project builds upon and is inspired by the xLSTM work by NX‑AI and collaborators. We thank the original authors and
the NX‑AI team for their research, open‑source implementations, and kernels.

- Upstream repository: https://github.com/NX-AI/xlstm
- Paper: xLSTM: Extended Long Short-Term Memory (arXiv:2405.04517)
- Paper: xLSTM‑7B: A Recurrent LLM for Fast and Efficient Inference (arXiv:2503.13427)
- Kernels: https://github.com/NX-AI/mlstm_kernels

Authors (xLSTM: Extended Long Short‑Term Memory):

- Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter
  Klambauer, Johannes Brandstetter, Sepp Hochreiter

Authors (xLSTM‑7B paper):

- Maximilian Beck, Korbinian Pöppel, Phillip Lippe, Richard Kurle, Patrick M. Blies, Günter Klambauer, Sebastian Böck,
  Sepp Hochreiter

If you use this repository in academic or industrial work, please cite the xLSTM papers and acknowledge NX‑AI
accordingly. See CITATION.cff for citation entries.

Port and Apple Silicon (MPS) Integration

- This independent port to PyTorch with Metal (MPS), chunkwise scheduling, Ray integration, memory watchdog/telemetry,
  and `xltop` tooling was implemented by Sydney Bach (The Solace Project). Upstream authors and NX‑AI were not involved
  in the engineering of this port.
- Maintainer/GitHub: Sydney Bach — https://github.com/SolaceHarmony
- Project link: https://github.com/SolaceHarmony/xLSTM-Metal

Affiliation & Trademark Notice

- This repository is not affiliated with NX‑AI. "xLSTM" is referenced solely to credit the original work and papers by
  NX‑AI and authors listed above.
