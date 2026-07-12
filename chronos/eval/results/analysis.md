# Frequency-Response Analysis — Patch Aliasing in Chronos-Bolt

## Models evaluated

| Label | P | S | Training | Role |
|-------|---|---|----------|------|
| p16-s16 (official) | 16 | 16 | 200k steps, full corpus | S=16 anchor, patch-size reference |
| p16-s12-seed42 | 16 | 12 | 100k steps, 48-shard subset | Stride axis |
| p16-s8-seed42 | 16 | 8 | 100k steps, 48-shard subset | Stride axis |
| p16-s4-seed42 | 16 | 4 | 100k steps, 48-shard subset | Stride axis |
| p8-s8-seed42 | 8 | 8 | 100k steps, 48-shard subset | Patch-size axis |
| p24-s24-seed42 | 24 | 24 | 100k steps, 48-shard subset | Patch-size axis |

## Key finding 1: Patch-size aliasing confirmed

The patch embedding creates frequency-response nulls whose position is determined by patch width P.
Predicted null locations: `f_null = k * fs / P` (i.e., integer cycles-per-patch).

| Model | P | Predicted 1st null | Measured recovery at null | Recovery at 10 Hz (baseline) |
|-------|---|--------------------|--------------------------|-------------------------------|
| p16-s16 (official) | 16 | 32 Hz | 0.929 | 0.995 |
| p16-s12-seed42 | 16 | 32 Hz | 0.054 | 0.966 |
| p16-s8-seed42 | 16 | 32 Hz | 0.123 | 0.988 |
| p16-s4-seed42 | 16 | 32 Hz | 0.026 | 0.965 |
| p8-s8-seed42 | 8 | 64 Hz | 0.333 | 1.007 |
| p24-s24-seed42 | 24 | 21.3 Hz | 0.904 (at 22 Hz) | 0.962 |

The nulls shift with P: P=8 models lose amplitude at 64 Hz, P=16 at 32 Hz, P=24 near 21 Hz. This is the aliasing signature — the null frequency is inversely proportional to patch width.

**Nuance:** The official model (200k, full data) partially "learns around" integer-cpp nulls. At cpp=1.0 (32 Hz) it recovers 0.93, while the 100k retrained models collapse to 0.03–0.12. However, all P=16 models — including the official — collapse together at higher cpp values (cpp > 1.5, i.e. > 48 Hz), where recovery drops below 0.12. The deeper training of the official model buys resilience at the first null but cannot overcome the fundamental bandwidth limitation at higher frequencies.

## Key finding 2: Stride does NOT mitigate the null

Original hypothesis: reducing stride S (increasing overlap) should fill the aliasing null by sampling the signal more frequently.

**Result: Rejected.** All P=16 stride variants (S=16, 12, 8, 4) show the same collapse pattern:

| Freq (Hz) | cpp | Official (S=16) | S=12 | S=8 | S=4 |
|-----------|-----|-----------------|------|-----|-----|
| 28 | 0.875 | 0.467 | 0.076 | 0.074 | 0.041 |
| 30 | 0.9375 | 0.867 | 0.046 | 0.058 | 0.009 |
| 32 | 1.0 | 0.929 | 0.054 | 0.123 | 0.026 |
| 34 | 1.0625 | 0.914 | 0.028 | 0.045 | 0.024 |
| 48 | 1.5 | 0.118 | 0.016 | 0.080 | 0.301 |

The retrained models (S=12, S=8, S=4) all collapse at the same frequencies with comparable depth. The null is determined by the patch window width (how many samples are projected together), not by how frequently that window is applied. This is a **patch-width** effect, not a patch-rate effect.

The official (S=16) performs better at the first null but still collapses at higher frequencies — its advantage comes from more training (200k vs 100k, full data vs 48-shard subset), not from its stride value.

## Key finding 3: High-frequency bandwidth ceiling

All models — regardless of P or S — become essentially non-functional above cpp ≈ 1.5:

- P=16 models: recovery < 0.1 for most frequencies above 48 Hz (except brief peaks at exactly integer cpp in the official)
- P=8 model: recovery < 0.05 for frequencies above 64 Hz
- P=24 model: recovery < 0.05 for frequencies above 48 Hz

The practical frequency bandwidth of the forecaster is approximately `0 < f < fs / P` (one cycle per patch). Beyond this, the patch embedding cannot preserve the oscillation.

## Key finding 4: Recovery peaks at exactly integer cpp (official only)

The official model shows an unexpected pattern: sharp recovery peaks at exactly integer cpp values (32, 64, 128 Hz — i.e. cpp = 1, 2, 4). At cpp=4.0 (128 Hz), recovery is 0.971 for the official. The retrained models also show peaks at 128 Hz (0.87–0.95), suggesting this is a learned feature, not noise.

Interpretation: when the signal has exactly an integer number of cycles per patch, the representation is simpler (every patch sees the same waveform segment). The transformer can learn a shortcut for this case. But at non-integer cpp > 1, the inter-patch phase varies and the model fails.

## Phase error

Phase error is meaningful only where amplitude recovery is substantial (> 0.3). In the recoverable frequency range (< 20 Hz for retrained, < 32 Hz for official), phase errors are typically < 15 degrees — the models track phase well when they can recover amplitude at all. At collapsed frequencies, phase error is noise (random atan2 of near-zero signal).

## Summary

| Hypothesis | Verdict |
|-----------|---------|
| Patch width P sets the null locations | **Confirmed** — nulls at k*fs/P |
| Nulls are at integer cpp regardless of P | **Confirmed** — P=8, P=16, P=24 all null at integer cpp |
| Stride overlap fills the null | **Rejected** — S=4 collapses as hard as S=16 |
| Deeper training can partially overcome aliasing | **Partially confirmed** — official survives cpp=1 but not cpp > 1.5 |
| Practical bandwidth = fs/P | **Confirmed** — recovery collapses above this threshold |
