"""Audit TS-Fault's 21 models under a common scoring reference.

TS-Fault's robustness ratio divides each model's faulted error by its own
clean error. Scoring instead against the trivial baseline's faulted error
turns the same published mean aggregates into the common-reference skill
ratio used in the accompanying paper, with no need to re-run their models.

Both reconstructed rankings below use the MEAN aggregates printed in Table V,
so the scoring reference is the only change between them. TS-Fault's published
robustness rank instead uses the median relative degradation across
configurations and is therefore not expected to match either ranking exactly.

The published r is a mean of within-dataset ratios (Sec. V-D), retained
only for provenance. Reconstructed own r below is faulted mean / clean mean.
Inputs are the Clean MSE, Faulted MSE, and published r columns of Table V in
arXiv:2606.18539, transcribed directly into ``T`` for line-by-line inspection.
"""

from scipy.stats import spearmanr

T = [("Naive", 1.238, 129.5, 105.7), ("SeasonalNaive", 0.905, 118.5, 165.5),
     ("ARIMA", 0.999, 121.7, 120.7), ("ETS", 1.044, 126.7, 122.9),
     ("DLinear", 0.562, 66.85, 197.3), ("NLinear", 0.540, 72.23, 142.1),
     ("N-BEATS", 0.449, 21.59, 54.6), ("LSTM", 0.733, 0.780, 1.07),
     ("GRU", 0.680, 0.775, 1.14), ("TCN", 0.874, 7.389, 7.90),
     ("Autoformer", 0.822, 24.19, 48.0), ("FEDformer", 1.072, 24.29, 22.6),
     ("PatchTST", 0.551, 86.63, 262.6), ("iTransformer", 0.529, 83.16, 272.3),
     ("TimeXer", 0.537, 87.98, 301.2), ("TimeMixer", 0.590, 90.88, 264.2),
     ("TimesNet", 0.571, 53.34, 141.7), ("NSTransformer", 0.608, 55.16, 101.1),
     ("TimesFM", 0.516, 162.7, 555.2), ("Chronos", 0.613, 165.6, 512.5),
     ("Moirai", 0.682, 153.1, 365.6)]
LREF, FREF = 1.238, 129.5          # Naive, the trivial baseline in their pool

if len(T) != 21 or len({row[0] for row in T}) != 21:
    raise RuntimeError("Table V transcription must contain 21 unique models")

own = {t[0]: i + 1 for i, t in enumerate(sorted(T, key=lambda t: t[2] / t[1]))}
shared = {t[0]: i + 1 for i, t in enumerate(sorted(T, key=lambda t: t[2]))}

print(f"{'model':15s} {'clean skill':>11s} {'own r':>8s} {'rank':>5s}"
      f" {'shared skill':>12s} {'rank':>5s} {'move':>5s}")
for n, L, F, source_r in sorted(T, key=lambda t: shared[t[0]]):
    r = F / L
    sk = FREF / F
    print(f"{n:15s} {LREF/L:11.2f} {r:8.1f} {own[n]:5d} {sk:12.2f}"
          f" {shared[n]:5d} {own[n]-shared[n]:+5d}"
          f"{'   below baseline' if sk < 1 else ''}")

cs = [LREF / t[1] for t in T]
a = spearmanr(cs, [-own[t[0]] for t in T])
b = spearmanr(cs, [-shared[t[0]] for t in T])
print(f"\nSpearman(clean accuracy, robustness score)"
      f"\n  own clean-error ratio {a.statistic:+.3f}  (p={a.pvalue:.3f})"
      f"\n  common reference      {b.statistic:+.3f}  (p={b.pvalue:.3f})")
