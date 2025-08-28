# Embedding “Homonyms” and Disambiguation at Inference Time

Purpose
- Explain, in practitioner terms, why models sometimes conflate multiple senses (“homonyms”) and how to design inference to reduce ambiguity — without relying on hidden chain‑of‑thought. We focus on observables and levers you can use.

What we mean by “homonyms” here
- Polysemy in embeddings: the same token or phrase maps near multiple meanings depending on context. Latent features superpose senses that the network later separates via attention/gating.
- Practical symptom: small context shifts cause large logit swings; certain prompts lead to oscillation between senses or topic drift.

Observable signals you can log
- Gate tension (I vs F): track distributions of input/forget gates across heads; spikes in I on ambiguous spans can indicate sense switching.
- Readout denominator (QᵀN): near‑zero denominators indicate under‑normalized reads (amplifies noise); clamp and log counts.
- Logit lens on H: probe linear classifiers over intermediate H to see which sense dominates; compare across bands/tiles.
- Entropy/contrast: token‑level entropy spikes around ambiguous spans; KL between successive steps highlights flips.

Design levers to reduce ambiguity
- Context scaffolding: add disambiguating tokens early (“company‑Apple” vs “fruit‑apple”).
- Temperature/penalty tuning: lower temperature near ambiguous spans; modest repetition penalties to avoid sense ping‑pong.
- Head‑band scheduling: ensure bands with disambiguating features aren’t starved; adjust `heads_per_band` to keep key heads in the same band.
- State precision: keep (C,N) stable (fp32 or limb‑precision) on very long contexts to avoid numeric drift that masquerades as semantic drift.

Probing/playbooks
- Minimal pairs: craft prompts differing by one disambiguating token; compare H/entropy traces.
- Controlled ablations: zero out a head‑band for a few steps; observe recovery to identify critical bands.
- Timing windows: slide the disambiguator earlier/later to see causality; log gate I peaks.

What we deliberately do not include
- No “internal chain‑of‑thought” or step‑by‑step latent traces — those are neither reliably extractable nor appropriate to expose. We rely on stable, external observables and ablations instead.

Takeaways
- Ambiguity is expected in superposed embeddings; your job is to give the model room and signal to separate senses (context and scheduling), and to keep numerics stable so you’re not chasing noise.

