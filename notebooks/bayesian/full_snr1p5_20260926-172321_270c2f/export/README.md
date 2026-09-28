# Run full_snr1p5_20260926-172321_270c2f

Mode `full`, tone SNR 1.5, 4 chains of 2000 retained draws after 2000 warm-up.

| Where | What |
|---|---|
| `../run_info.json` | creation time, repository commit, one entry per session |
| `../analysis_manifest.json` | every tracked artifact with its SHA-256 |
| `../data/` | the collection: raw shards, merged tables `02_*.parquet`, the signal archive `signals/` |
| `../04_*.nc`, `../05_sens_*.nc` | every posterior, as ArviZ InferenceData (NetCDF) |
| `../tables/*.csv` | every result table; `bayesGates.tex` and `bayesFitSummary.tex` for the report |
| `../figures/*.png`, `../figures/report/*.pdf` | every figure; the report-ready ones as vector PDF |
| `posterior_summaries/` | az.summary of every fit |
| `estimand_draws.parquet`, `prior_draws.parquet` | posterior and prior draws of every estimand |

Reloading a posterior needs only ArviZ:

```python
import arviz as az
idata = az.from_netcdf("04_A_both.nc", engine="h5netcdf")
```

The notebook itself, with `STANDALONE_FIGURES = True` and `RUN_NAME = "full_snr1p5_20260926-172321_270c2f"`, redraws every
figure from these files without Chronos and without sampling.
