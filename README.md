# TRAPPIST-1e: coupled Photochem + petitRADTRANS Monte Carlo, JWST comparison and AI emulator

Code, model results and analysis scripts for

> **M. Waqas & T. Hidayat (2026),** *AI-driven Investigation of the TRAPPIST-1e Atmosphere Using Coupled
> 1-D Photochem and petitRADTRANS Codes*, in preparation for *The Astronomical Journal*.

We compare the JWST NIRSpec/PRISM transmission spectrum of TRAPPIST-1e (Espinoza et al. 2025) with
**1,861 self-consistent N₂–CO₂–CH₄ atmospheres**. Each model has a radiative–convective climate and a
photochemical steady state from [Photochem](https://github.com/Nicholaswogan/photochem), and a
transmission spectrum from [petitRADTRANS](https://petitradtrans.readthedocs.io). Partial pressures are
drawn at random from broad priors, so the likelihood-weighted ensemble is a posterior sample. A
random-forest emulator trained on the ensemble reproduces the pipeline in milliseconds.

**Main results**
- A featureless (flat) spectrum fits better than every coupled model, both for the stellar-contamination-corrected
  combined spectrum and for the uncorrected quiet visits 1 and 2. The data contain no atmospheric signal.
- Among coupled atmospheres the data favour muted spectra: low CO₂ (< 0.077 bar at 95% credibility; < 0.12 bar
  from visits 1+2) and high CH₄; N₂ is unconstrained. Through the climate solution this implies a temperate surface,
  T_surf = 288 (+39/−14) K.
- These exclusions hold for cloud-free atmospheres: in a grey-cloud test, CO₂-rich atmospheres (≳0.09 bar)
  become compatible with the data only if an opaque cloud top lies at ≈0.03 bar or lower.
- The emulator reproduces χ² for 1,415 unseen models with R² = 0.984.

---

## Repository layout

```
pipeline/          The physical model: climate -> photochemistry -> petitRADTRANS -> JWST fit
  trappist1e_mc_pipeline.py     Monte Carlo pipeline used for the paper (single notebook cell)
  setup_workshop_inputs.sh      Fetches the Photochem workshop input files and sets 100 layers
analysis/          Analyses of the pipeline output
  statistical_analysis.py       Full statistical analysis (Colab); saves figures + tables as a zip
  flat_line_baseline.py         Featureless-spectrum baseline (chi2, BIC) on the combined spectrum
  refit_visits12.py             Refit of all models to the uncorrected quiet visits 1 and 2
  convergence_check.py          Re-runs a subsample and records whether photochemistry reached steady state
  cloud_test.py                 Grey-cloud robustness test for the best-fitting and excluded models
paper_scripts/     Reproduce every number, table and figure in the paper
  analysis.py  -> derived/results.json and derived tables (posteriors, emulator validation, detectability)
  tp_analysis.py -> derived/tp_diagnostics.csv, tp_stats.json (thermal structure of the 10 best models)
  figures.py, fig_tp.py, fig_corr.py, fig_spectra.py -> figures/
  fig_cloud_test.py, fig_cloud_spectra.py -> figures/ (grey-cloud robustness test)
  retrain.py   -> models/..._v2_all1856.pkl (emulator retrained on all converged models)
emulator_app/      Interactive Streamlit app ("AI agent") that runs the trained emulators
data/
  MC_2000_master_results_flatspec_21.csv   One row per Monte Carlo draw (1,995 rows; see columns below)
  model_fits/                              Binned model spectra (model_XXXX_jwst_fit.csv) of the 10 best and the worst model
  cloud_test/                              Grey-cloud test: results table and binned spectra (20 models x 6 cloud levels)
  atmosphere_profiles/                     Climate profiles (T, P, composition) of the 10 best-fitting models
  analysis_tables/                         Spearman correlation matrix, p-values, FDR-corrected correlations
  refit_visits12.csv                       chi2 of every converged model against uncorrected visits 1+2 (chi2_v12)
  refit_visits12_summary.csv               Flat line vs best model for visits 1+2 (chi2, BIC, Bayes factor)
derived/           Outputs of paper_scripts (already included, so figures can be made without the emulator)
figures/           Paper figures
models/            Emulator pickles go here (download from Zenodo, see models/README.md)
```

## Quick start: reproduce the paper figures

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python paper_scripts/tp_analysis.py
python paper_scripts/fig_tp.py
python paper_scripts/fig_corr.py
python paper_scripts/fig_spectra.py
python paper_scripts/fig_cloud_test.py
python paper_scripts/fig_cloud_spectra.py
python paper_scripts/figures.py
```

To recompute all numbers in the paper from scratch, first download the emulator files into `models/`
(see `models/README.md`), then run `python paper_scripts/analysis.py` before `figures.py`.

## Running the emulator app

```bash
cd emulator_app
pip install -r requirements.txt
# copy the three downloaded .pkl files into emulator_app/models/
streamlit run app.py
```

The emulator is valid only inside its training ranges and only for the spectrum, uncertainties and
offset-only fit used here. It should not be used in the region where the physical models failed to converge
(high CH₄).

## Running the physical pipeline

The pipeline needs Photochem and petitRADTRANS (with its opacity data), which are easiest to install with conda:

```bash
conda create -n trappist -c conda-forge python=3.11 photochem
conda activate trappist
pip install petitRADTRANS threadpoolctl scipy pandas matplotlib
cd pipeline
bash setup_workshop_inputs.sh
```

Then download the spectrum `tspectra_decontaminated_corrected_flat_JWST-CLR-R10000.txt` from the
Espinoza et al. (2025) release (Zenodo, doi:10.5281/zenodo.16125662, folder `espinoza+2025/figure3/`),
set `JWST_FILE` near the top of `trappist1e_mc_pipeline.py` to its location, and run the script
(it was written as a single Jupyter notebook cell). Each model takes minutes; the full 2,000-draw run
takes days on a workstation. Per-model outputs are written to `MC_2000_model_results_flatspec/`.

For `analysis/refit_visits12.py`, set `VISIT_DIR` to the `espinoza+2025/figure3/data/jwst/` folder of the same release.

**Settings used in the paper:** AdiabatClimate with 100 layers, top at 10⁻⁵ bar, 270 bar ocean, relative
humidity 0.5, surface albedo 0.1; EvoAtmosphere with the Zahnle H–N–O–C network, K_zz = 10⁵ cm² s⁻¹,
fixed-pressure lower boundaries for N₂, CO₂ and CH₄; HAZMAT TRAPPIST-1 spectrum scaled to 879 W m⁻²;
petitRADTRANS correlated-k opacities (H₂O, CO₂, CH₄, CO, O₂; Rayleigh N₂, H₂; CIA N₂–N₂, N₂–H₂);
Gaussian line-spread function R = 100; one free additive offset per spectrum.

## Columns of `data/MC_2000_master_results_flatspec_21.csv`

| Column | Meaning |
|---|---|
| `model_id` | Monte Carlo draw number |
| `P_N2`, `P_CO2`, `P_CH4` | Input surface partial pressures (bar) |
| `T_surf_input` | Surface temperature from the first-stage climate sweep (K) |
| `T_surf_climate` | Self-consistent surface temperature from the full pipeline (K) |
| `climate_converged`, `photochemistry_converged` | Convergence flags |
| `n_data`, `dof` | Number of spectral bins used (67) and degrees of freedom (66) |
| `wavelength_min_um`, `wavelength_max_um` | Wavelength range of the fit (µm) |
| `offset_ppm` | Best-fitting additive offset (ppm) |
| `chi2`, `reduced_chi2`, `delta_chi2` | Fit statistics against the combined spectrum |
| `relative_likelihood` | exp(−Δχ²/2) |
| `rms_residual_ppm` | RMS of the residuals (ppm) |
| `bic`, `delta_bic` | Bayesian information criterion |
| `max_feature_snr` | Largest feature significance, SNR_max |
| `n_transits_for_detection` | Transits for a 5σ single-feature detection |
| `spectral_amplitude_ppm`, `max_transit_depth_ppm`, `min_transit_depth_ppm` | Native-resolution spectrum range (ppm) |

## Notes for users

- Set the data paths before running: `JWST_FILE` in `pipeline/trappist1e_mc_pipeline.py` and in `analysis/flat_line_baseline.py`, and `VISIT_DIR` in `analysis/refit_visits12.py`.
- In the pipeline, `photochemistry_converged` records that the photochemical integration completed without error. Whether each run reached steady state is checked separately by `analysis/convergence_check.py`, which re-runs a subsample of models and records the solver's convergence flag.

## Data sources

- JWST data: MAST, doi:10.17909/yzwd-vq54
- Reduced spectra: Espinoza et al. (2025), ApJL 990, L52; doi:10.5281/zenodo.16125662
- Photochem: Wogan et al. (2025); workshop inputs from https://github.com/Nicholaswogan/UofA_Photochem_Workshop

## Citation

Please cite the paper above and this archive (see `CITATION.cff`; Zenodo DOI to be added on release).

## Licence

Code: GNU General Public License v3.0 (see `LICENSE`), the same licence as Photochem.
Data tables and figures: Creative Commons Attribution 4.0 (CC BY 4.0).
