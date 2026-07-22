# Apparent criticality without contagion — reproducibility repository

Code and data to reproduce every result, table and figure in:

> M. Herrera-Marín, *Apparent criticality without contagion: common-driver bias and identifiable
> structure in inferred cryptocurrency networks.* Submitted to *Quantitative Finance*, 2026.

Companion preprint (Paper I): *Apparent criticality from common-drive aliasing in self-exciting
networks*, Research Square, 2026. https://doi.org/10.21203/rs.3.rs-10296787/v1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## What is here

```
notebooks/
  crypto_confirmatory_V4.ipynb   Headline K=5 confirmatory analysis (Table 4, Figure 9)
  crypto_V5_extension.ipynb      Cross-sectional scaling K=5..30 (Table 5, Figure 10)
code/
  code_network_scaling.py        Reduced-form large-network simulation (Section 8)
  end_to_end_hawkes_benchmark.py End-to-end binned Hawkes stress test (Section 8)
data/tables/
  calibration.csv                Synthetic zero-witness separation calibration (Table 3)
  network_scaling_results.csv    Reduced-form scaling results
  end_to_end_hawkes_summary.csv  End-to-end benchmark summary
  v3_design_results.csv          Cryptocurrency design audit
  v4_confirmatory_inference.csv  Headline K=5 confirmatory estimates
  v5_phaseA_sweep.csv            Full K x q sweep (108 designs) — main scaling result
  v5_phaseB_confirmatory.csv     Confirmatory inference on selected large-K designs
  v5_economic_consequence.csv    Top-transmitter change under deconfounding
  v5_verdict.csv                 Summary decision statistics
  v5_zero_witness_analytic.csv   Analytic zero-witness separation
figures/
  v5_scaling.pdf                 Figure 10 (cross-sectional scaling)
```

## Data provenance

The high-frequency inputs are **public** aggregate-trade files from the Binance Vision archive
(https://data.binance.vision). They are **not redistributed here**; the notebooks download exactly the
files needed for the event windows in the paper and verify their availability before downloading. The
asset universe (30 pairs, ordered by capitalisation), the six event windows (three crisis episodes and
three matched calm controls) and the event-threshold grid are defined at the top of
`notebooks/crypto_V5_extension.ipynb`.

No proprietary data are used. All randomness is seeded; re-running reproduces the reported numbers.

## How to reproduce

```bash
git clone https://github.com/mherrera-udd/apparent-criticality-crypto.git
cd apparent-criticality-crypto
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                                            # run the notebooks top to bottom
```

The notebooks are ordered and self-checking: early cells run a preflight (verifying every Binance file
exists) and an auto-test on synthetic data (~20 s) before any large download. Do not skip them.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).

## License

Released under the MIT License (see `LICENSE`).
