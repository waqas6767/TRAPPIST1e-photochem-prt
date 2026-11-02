# ============================================================
# TRAPPIST-1e MONTE CARLO ATMOSPHERIC MODEL STATISTICAL ANALYSIS
# ATMOS / Photochem + petitRADTRANS Results
#
# Designed for Google Colab (also runs locally: pass the CSV path
# as the first command-line argument, or set CSV_PATH below).
#
# Every table and figure is saved into one results folder, which is
# zipped at the end and downloaded automatically in Colab:
#
#   TRAPPIST1e_analysis_results/
#       figures/   *.png (300 dpi) and *.pdf (vector, for the paper)
#       tables/    *.csv
#       TRAPPIST1e_statistical_analysis.csv   (full per-model table)
#       TRAPPIST1e_statistical_results.xlsx   (all summary tables)
#       analysis_log.txt                      (everything printed)
#       README.txt
#   -> TRAPPIST1e_analysis_results.zip
# ============================================================

# ------------------------------------------------------------
# 1. INSTALL / IMPORT PACKAGES
# ------------------------------------------------------------

import os
import sys
import shutil
import subprocess
from datetime import datetime

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "openpyxl", "seaborn", "statsmodels"], check=False)

import numpy as np
import pandas as pd
import matplotlib
if not IN_COLAB:
    matplotlib.use("Agg")          # no screen needed when running locally
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

try:
    from IPython.display import display
except ImportError:                # plain Python fallback
    def display(x):
        print(x)

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.5g}")

np.random.seed(42)                 # reproducible bootstrap results


# ------------------------------------------------------------
# 1b. OUTPUT FOLDER + SAVE HELPERS
# ------------------------------------------------------------

OUT_DIR = "TRAPPIST1e_analysis_results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

if os.path.exists(OUT_DIR):        # start clean so old files never mix in
    shutil.rmtree(OUT_DIR)
os.makedirs(FIG_DIR)
os.makedirs(TAB_DIR)

saved_files = []                   # list of everything written (for README)


class Tee:
    """Copy everything printed to the screen into analysis_log.txt."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

LOG_PATH = os.path.join(OUT_DIR, "analysis_log.txt")
sys.stdout = Tee(LOG_PATH)


def save_fig(name):
    """Save the current figure as PNG (300 dpi) and PDF, show it, close it."""
    for ext, dpi in (("png", 300), ("pdf", None)):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_files.append(path)
    if IN_COLAB:
        plt.show()
    plt.close()


def save_table(table, name, index=True):
    """Save a DataFrame/Series as CSV in tables/ and display it."""
    path = os.path.join(TAB_DIR, f"{name}.csv")
    table.to_csv(path, index=index)
    saved_files.append(path)
    display(table)


# ------------------------------------------------------------
# 2. LOAD CSV
# ------------------------------------------------------------

print("=" * 70)
print("LOAD YOUR MONTE CARLO CSV FILE")
print("=" * 70)

CSV_PATH = None                    # set a path here to skip the upload dialog

if CSV_PATH is None and len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
    CSV_PATH = sys.argv[1]

if CSV_PATH is None:
    if IN_COLAB:
        from google.colab import files
        uploaded = files.upload()
        CSV_PATH = list(uploaded.keys())[0]
    else:
        raise SystemExit("Give the CSV path: python this_script.py results.csv")

filename = CSV_PATH
df = pd.read_csv(filename)

print(f"\nLoaded file: {filename}")
print(f"Number of models: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

display(df.head())


# ------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"Total models: {len(df)}")
print(f"Total variables: {len(df.columns)}")

print("\nColumn names:")
for col in df.columns:
    print(" -", col)

print("\nMissing values:")
save_table(df.isna().sum().to_frame("Missing values"), "00_missing_values")


# ------------------------------------------------------------
# 4. CHECK CONVERGENCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CONVERGENCE STATISTICS")
print("=" * 70)

convergence_rows = []

if "climate_converged" in df.columns:
    climate_conv = df["climate_converged"].astype(str).str.lower().isin(["true", "1", "yes"])
    print(f"Climate converged: {climate_conv.sum()} / {len(df)} ({100*climate_conv.mean():.2f}%)")
    convergence_rows.append(("climate_converged", climate_conv.sum(), len(df)))

if "photochemistry_converged" in df.columns:
    photo_conv = df["photochemistry_converged"].astype(str).str.lower().isin(["true", "1", "yes"])
    print(f"Photochemistry converged: {photo_conv.sum()} / {len(df)} ({100*photo_conv.mean():.2f}%)")
    convergence_rows.append(("photochemistry_converged", photo_conv.sum(), len(df)))

if "climate_converged" in df.columns and "photochemistry_converged" in df.columns:
    both_conv = climate_conv & photo_conv
    print(f"Both converged: {both_conv.sum()} / {len(df)} ({100*both_conv.mean():.2f}%)")
    df["fully_converged"] = both_conv
    convergence_rows.append(("both_converged", both_conv.sum(), len(df)))

if convergence_rows:
    conv_table = pd.DataFrame(convergence_rows, columns=["stage", "converged", "total"])
    conv_table["percent"] = 100 * conv_table["converged"] / conv_table["total"]
    conv_table.to_csv(os.path.join(TAB_DIR, "01_convergence.csv"), index=False)
    saved_files.append(os.path.join(TAB_DIR, "01_convergence.csv"))


# ------------------------------------------------------------
# 5. SELECT ONLY VALID / CONVERGED MODELS
# ------------------------------------------------------------

if "fully_converged" in df.columns:
    analysis_df = df[df["fully_converged"]].copy()
else:
    analysis_df = df.copy()

if "chi2" in analysis_df.columns:              # a converged model must also have a chi2
    analysis_df = analysis_df[np.isfinite(analysis_df["chi2"])].copy()

print("\nModels used for statistical analysis:", len(analysis_df))


# ------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

numeric_cols = analysis_df.select_dtypes(include=np.number).columns

descriptive = analysis_df[numeric_cols].describe().T
descriptive["median"] = analysis_df[numeric_cols].median()
descriptive["std"] = analysis_df[numeric_cols].std()
descriptive["variance"] = analysis_df[numeric_cols].var()
descriptive["IQR"] = (analysis_df[numeric_cols].quantile(0.75)
                      - analysis_df[numeric_cols].quantile(0.25))

save_table(descriptive, "02_descriptive_statistics")


# ------------------------------------------------------------
# 7. ATMOSPHERIC PARAMETER STATISTICS
# ------------------------------------------------------------

atmospheric_columns = ["P_N2", "P_CO2", "P_CH4", "T_surf_input", "T_surf_climate"]
available_atmospheric = [c for c in atmospheric_columns if c in analysis_df.columns]

print("\n" + "=" * 70)
print("ATMOSPHERIC PARAMETER STATISTICS")
print("=" * 70)

atm_stats = analysis_df[available_atmospheric].describe().T
atm_stats["median"] = analysis_df[available_atmospheric].median()
atm_stats["IQR"] = (analysis_df[available_atmospheric].quantile(0.75)
                    - analysis_df[available_atmospheric].quantile(0.25))

save_table(atm_stats, "03_atmospheric_parameter_statistics")


# ------------------------------------------------------------
# 8. LOG-TRANSFORM PRESSURES
# ------------------------------------------------------------

for col in ["P_N2", "P_CO2", "P_CH4"]:
    if col in analysis_df.columns:
        analysis_df["log10_" + col] = np.log10(analysis_df[col].clip(lower=1e-20))


# ------------------------------------------------------------
# 9. RECOMPUTE CHI-SQUARE STATISTICS
# ------------------------------------------------------------

if "chi2" in analysis_df.columns:
    best_chi2 = analysis_df["chi2"].min()
    analysis_df["delta_chi2_calc"] = analysis_df["chi2"] - best_chi2
    analysis_df["relative_likelihood_calc"] = np.exp(-0.5 * analysis_df["delta_chi2_calc"])
    analysis_df["relative_likelihood_calc"] /= analysis_df["relative_likelihood_calc"].max()

    print("\nBest χ² =", best_chi2)
    print("\nTop 10 models by χ²:")
    save_table(analysis_df.sort_values("chi2").head(10), "04_top10_models_by_chi2", index=False)


# ------------------------------------------------------------
# 10. REDUCED CHI-SQUARE
# ------------------------------------------------------------

if "chi2" in analysis_df.columns and "dof" in analysis_df.columns:
    analysis_df["reduced_chi2_calc"] = analysis_df["chi2"] / analysis_df["dof"]

    print("\n" + "=" * 70)
    print("REDUCED CHI-SQUARE")
    print("=" * 70)
    save_table(analysis_df["reduced_chi2_calc"].describe().to_frame("reduced_chi2"),
               "05_reduced_chi2_summary")


# ------------------------------------------------------------
# 11. AIC AND BIC
#
# NOTE: every model in the grid has the same k, so AIC and BIC rank
# models exactly like chi2 does. They only become informative when
# comparing models with different numbers of free parameters
# (e.g. the atmosphere grid against a flat line).
# ------------------------------------------------------------

if "chi2" in analysis_df.columns:
    n_data = analysis_df["n_data"].iloc[0] if "n_data" in analysis_df.columns else 67

    k = 4   # P_N2 + P_CO2 + P_CH4 + spectral offset

    analysis_df["AIC_calc"] = analysis_df["chi2"] + 2 * k
    analysis_df["BIC_calc"] = analysis_df["chi2"] + k * np.log(n_data)
    analysis_df["delta_AIC"] = analysis_df["AIC_calc"] - analysis_df["AIC_calc"].min()
    analysis_df["delta_BIC_calc"] = analysis_df["BIC_calc"] - analysis_df["BIC_calc"].min()
    analysis_df["BIC_weight"] = np.exp(-0.5 * analysis_df["delta_BIC_calc"])
    analysis_df["BIC_weight"] /= analysis_df["BIC_weight"].sum()


# ------------------------------------------------------------
# 12. BEST-FIT MODEL
# ------------------------------------------------------------

if "chi2" in analysis_df.columns:
    best_model = analysis_df.loc[analysis_df["chi2"].idxmin()]

    print("\n" + "=" * 70)
    print("BEST-FIT MODEL")
    print("=" * 70)
    save_table(best_model.to_frame("Best-fit value"), "06_best_fit_model")


# ------------------------------------------------------------
# 13. BEST MODELS BY DIFFERENT STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST MODELS")
print("=" * 70)

best_by_metric = []
for metric in ["chi2", "reduced_chi2_calc", "AIC_calc", "BIC_calc"]:
    if metric in analysis_df.columns:
        best = analysis_df.loc[analysis_df[metric].idxmin()]
        print(f"\nBest model according to {metric}:")
        print("Model ID:", best.get("model_id", "N/A"))
        print("Value:", best[metric])
        best_by_metric.append({"metric": metric, "model_id": best.get("model_id", np.nan),
                               "value": best[metric]})

if best_by_metric:
    pd.DataFrame(best_by_metric).to_csv(os.path.join(TAB_DIR, "07_best_model_by_metric.csv"), index=False)
    saved_files.append(os.path.join(TAB_DIR, "07_best_model_by_metric.csv"))


# ------------------------------------------------------------
# 14. GOODNESS-OF-FIT P-VALUE
# ------------------------------------------------------------

if "chi2" in analysis_df.columns and "dof" in analysis_df.columns:
    analysis_df["chi2_pvalue"] = stats.chi2.sf(analysis_df["chi2"], analysis_df["dof"])

    print("\n" + "=" * 70)
    print("CHI-SQUARE GOODNESS-OF-FIT")
    print("=" * 70)
    save_table(analysis_df["chi2_pvalue"].describe().to_frame("chi2_pvalue"),
               "08_goodness_of_fit_pvalues")


# ------------------------------------------------------------
# 15. IDENTIFY STATISTICALLY PREFERRED MODELS
# ------------------------------------------------------------

if "delta_BIC_calc" in analysis_df.columns:
    analysis_df["BIC_category"] = pd.cut(
        analysis_df["delta_BIC_calc"],
        bins=[-np.inf, 2, 6, 10, np.inf],
        labels=["Weak evidence", "Positive evidence", "Strong evidence", "Very strong evidence"])

    print("\nBIC model-selection categories:")
    save_table(analysis_df["BIC_category"].value_counts().to_frame("Number of models"),
               "09_BIC_categories")


# ------------------------------------------------------------
# 16. CONFIDENCE REGION USING DELTA CHI-SQUARE
# ------------------------------------------------------------

if "delta_chi2_calc" in analysis_df.columns:
    print("\nModels within Δχ² thresholds:")
    rows = []
    for threshold in [1, 2.3, 4, 6.17, 9]:
        count = (analysis_df["delta_chi2_calc"] <= threshold).sum()
        print(f"Δχ² <= {threshold}: {count} models ({100*count/len(analysis_df):.2f}%)")
        rows.append({"delta_chi2_threshold": threshold, "n_models": count,
                     "percent": 100 * count / len(analysis_df)})
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "10_delta_chi2_thresholds.csv"), index=False)
    saved_files.append(os.path.join(TAB_DIR, "10_delta_chi2_thresholds.csv"))


# ------------------------------------------------------------
# 17. PARAMETER CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SPEARMAN CORRELATION ANALYSIS")
print("=" * 70)

correlation_columns = ["P_N2", "P_CO2", "P_CH4", "T_surf_input", "T_surf_climate", "chi2",
                       "reduced_chi2_calc", "rms_residual_ppm", "max_feature_snr",
                       "n_transits_for_detection", "spectral_amplitude_ppm"]
available_corr = [c for c in correlation_columns if c in analysis_df.columns]

spearman_corr = analysis_df[available_corr].corr(method="spearman")
save_table(spearman_corr, "11_spearman_correlation_matrix")


# ------------------------------------------------------------
# 18. SPEARMAN P-VALUES
# ------------------------------------------------------------

p_matrix = pd.DataFrame(np.ones((len(available_corr), len(available_corr))),
                        index=available_corr, columns=available_corr)

for i in range(len(available_corr)):
    for j in range(len(available_corr)):
        x = analysis_df[available_corr[i]]
        y = analysis_df[available_corr[j]]
        valid = x.notna() & y.notna()
        if valid.sum() >= 3:
            _, p = spearmanr(x[valid], y[valid])
            p_matrix.iloc[i, j] = p

print("\nSpearman correlation p-values:")
save_table(p_matrix, "12_spearman_pvalues")


# ------------------------------------------------------------
# 19. MULTIPLE-TEST CORRECTION (Benjamini-Hochberg FDR)
# ------------------------------------------------------------

pairs = []
for i in range(len(available_corr)):
    for j in range(i + 1, len(available_corr)):
        pairs.append({"Variable 1": available_corr[i], "Variable 2": available_corr[j],
                      "rho": spearman_corr.iloc[i, j], "p_value": p_matrix.iloc[i, j]})

corr_results = pd.DataFrame(pairs)

if len(corr_results) > 0:
    reject, p_corrected, _, _ = multipletests(corr_results["p_value"], method="fdr_bh")
    corr_results["FDR_p"] = p_corrected
    corr_results["significant_FDR"] = reject
    corr_results = corr_results.sort_values("FDR_p")

    corr_results.to_csv(os.path.join(TAB_DIR, "13_correlations_FDR_all.csv"), index=False)
    saved_files.append(os.path.join(TAB_DIR, "13_correlations_FDR_all.csv"))
    print("\nSignificant correlations after FDR correction:")
    save_table(corr_results[corr_results["significant_FDR"]], "14_correlations_FDR_significant",
               index=False)


# ------------------------------------------------------------
# 20. CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(13, 10))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", center=0, square=True)
plt.title("Spearman Rank Correlation Matrix")
plt.tight_layout()
save_fig("01_spearman_correlation_heatmap")


# ------------------------------------------------------------
# 21. DISTRIBUTION OF ATMOSPHERIC PARAMETERS
# ------------------------------------------------------------

for col in ["P_N2", "P_CO2", "P_CH4"]:
    if col in analysis_df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(np.log10(analysis_df[col].clip(lower=1e-20)), bins=30)
        plt.xlabel(f"log10({col} [bar])")
        plt.ylabel("Number of models")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        save_fig(f"02_distribution_{col}")


# ------------------------------------------------------------
# 22. TEMPERATURE DISTRIBUTION
# ------------------------------------------------------------

if "T_surf_climate" in analysis_df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(analysis_df["T_surf_climate"], bins=30)
    plt.xlabel("Climate Surface Temperature (K)")
    plt.ylabel("Number of models")
    plt.title("Distribution of Climate Surface Temperature")
    plt.tight_layout()
    save_fig("03_distribution_T_surf_climate")


# ------------------------------------------------------------
# 23. CHI-SQUARE DISTRIBUTION
# ------------------------------------------------------------

if "reduced_chi2_calc" in analysis_df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(analysis_df["reduced_chi2_calc"], bins=30)
    plt.axvline(1, linestyle="--", label="Reduced χ² = 1")
    plt.xlabel("Reduced χ²")
    plt.ylabel("Number of models")
    plt.title("Distribution of Reduced χ²")
    plt.legend()
    plt.tight_layout()
    save_fig("04_distribution_reduced_chi2")


# ------------------------------------------------------------
# 24. ΔBIC DISTRIBUTION
# ------------------------------------------------------------

if "delta_BIC_calc" in analysis_df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(analysis_df["delta_BIC_calc"], bins=30)
    plt.axvline(2, linestyle="--", label="ΔBIC = 2")
    plt.axvline(6, linestyle="--", label="ΔBIC = 6")
    plt.axvline(10, linestyle="--", label="ΔBIC = 10")
    plt.xlabel("ΔBIC")
    plt.ylabel("Number of models")
    plt.title("Bayesian Information Criterion Comparison")
    plt.legend()
    plt.tight_layout()
    save_fig("05_distribution_delta_BIC")


# ------------------------------------------------------------
# 25-27. χ² VS TEMPERATURE / CO2 / CH4
# ------------------------------------------------------------

if "T_surf_climate" in analysis_df.columns and "chi2" in analysis_df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(analysis_df["T_surf_climate"], analysis_df["chi2"], alpha=0.6)
    plt.xlabel("Climate Surface Temperature (K)")
    plt.ylabel("χ²")
    plt.title("χ² vs Climate Surface Temperature")
    plt.tight_layout()
    save_fig("06_chi2_vs_T_surf")

if "P_CO2" in analysis_df.columns and "chi2" in analysis_df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(np.log10(analysis_df["P_CO2"].clip(lower=1e-20)), analysis_df["chi2"], alpha=0.6)
    plt.xlabel("log10(CO₂ pressure [bar])")
    plt.ylabel("χ²")
    plt.title("χ² vs CO₂ Pressure")
    plt.tight_layout()
    save_fig("07_chi2_vs_CO2")

if "P_CH4" in analysis_df.columns and "chi2" in analysis_df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(np.log10(analysis_df["P_CH4"].clip(lower=1e-20)), analysis_df["chi2"], alpha=0.6)
    plt.xlabel("log10(CH₄ pressure [bar])")
    plt.ylabel("χ²")
    plt.title("χ² vs CH₄ Pressure")
    plt.tight_layout()
    save_fig("08_chi2_vs_CH4")


# ------------------------------------------------------------
# 28. SPECTRAL AMPLITUDE VS TEMPERATURE
# ------------------------------------------------------------

if "spectral_amplitude_ppm" in analysis_df.columns and "T_surf_climate" in analysis_df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(analysis_df["T_surf_climate"], analysis_df["spectral_amplitude_ppm"], alpha=0.6)
    plt.xlabel("Climate Surface Temperature (K)")
    plt.ylabel("Spectral Amplitude (ppm)")
    plt.title("Spectral Amplitude vs Surface Temperature")
    plt.tight_layout()
    save_fig("09_spectral_amplitude_vs_T_surf")


# ------------------------------------------------------------
# 29. SNR VS NUMBER OF TRANSITS
# ------------------------------------------------------------

if "max_feature_snr" in analysis_df.columns and "n_transits_for_detection" in analysis_df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(analysis_df["n_transits_for_detection"], analysis_df["max_feature_snr"], alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Number of Transits Required")
    plt.ylabel("Maximum Feature S/N")
    plt.title("Detectability of Atmospheric Spectral Features")
    plt.tight_layout()
    save_fig("10_snr_vs_transits")


# ------------------------------------------------------------
# 30. DETECTION STATISTICS
# ------------------------------------------------------------

if "max_feature_snr" in analysis_df.columns:
    print("\n" + "=" * 70)
    print("ATMOSPHERIC FEATURE DETECTION")
    print("=" * 70)

    rows = []
    for threshold in [1, 2, 3, 5]:
        n_detect = (analysis_df["max_feature_snr"] >= threshold).sum()
        print(f"S/N >= {threshold}: {n_detect} models ({100*n_detect/len(analysis_df):.2f}%)")
        rows.append({"snr_threshold": threshold, "n_models": n_detect,
                     "percent": 100 * n_detect / len(analysis_df)})
    pd.DataFrame(rows).to_csv(os.path.join(TAB_DIR, "15_detection_snr_thresholds.csv"), index=False)
    saved_files.append(os.path.join(TAB_DIR, "15_detection_snr_thresholds.csv"))


# ------------------------------------------------------------
# 31. TRANSIT REQUIREMENT STATISTICS
# ------------------------------------------------------------

if "n_transits_for_detection" in analysis_df.columns:
    transit_stats = analysis_df["n_transits_for_detection"].describe(
        percentiles=[.05, .10, .25, .50, .75, .90, .95])

    print("\n" + "=" * 70)
    print("NUMBER OF TRANSITS REQUIRED")
    print("=" * 70)
    save_table(transit_stats.to_frame("Number of transits"), "16_transits_required_summary")


# ------------------------------------------------------------
# 32. BEST DETECTABLE ATMOSPHERES
# ------------------------------------------------------------

if "max_feature_snr" in analysis_df.columns:
    print("\n" + "=" * 70)
    print("TOP 20 MODELS BY ATMOSPHERIC FEATURE SNR")
    print("=" * 70)
    top_snr = analysis_df.sort_values("max_feature_snr", ascending=False).head(20)
    save_table(top_snr, "17_top20_models_by_snr", index=False)


# ------------------------------------------------------------
# 33. BEST MODELS BY LOWEST NUMBER OF TRANSITS
# ------------------------------------------------------------

if "n_transits_for_detection" in analysis_df.columns:
    print("\n" + "=" * 70)
    print("TOP 20 MODELS REQUIRING FEWEST TRANSITS")
    print("=" * 70)
    top_transits = analysis_df.sort_values("n_transits_for_detection").head(20)
    save_table(top_transits, "18_top20_models_fewest_transits", index=False)


# ------------------------------------------------------------
# 34. TEMPERATURE REGIME ANALYSIS
# ------------------------------------------------------------

if "T_surf_climate" in analysis_df.columns:
    bins = [-np.inf, 280, 300, 320, 350, 380, np.inf]
    labels = ["<280 K", "280–300 K", "300–320 K", "320–350 K", "350–380 K", ">380 K"]
    analysis_df["temperature_regime"] = pd.cut(analysis_df["T_surf_climate"], bins=bins, labels=labels)

    agg = dict(number_of_models=("model_id", "count"), median_chi2=("chi2", "median"))
    if "reduced_chi2_calc" in analysis_df:
        agg["median_reduced_chi2"] = ("reduced_chi2_calc", "median")
    if "max_feature_snr" in analysis_df:
        agg["median_snr"] = ("max_feature_snr", "median")
    if "n_transits_for_detection" in analysis_df:
        agg["median_transits"] = ("n_transits_for_detection", "median")
    temp_summary = analysis_df.groupby("temperature_regime", observed=True).agg(**agg)

    print("\n" + "=" * 70)
    print("TEMPERATURE REGIME ANALYSIS")
    print("=" * 70)
    save_table(temp_summary, "19_temperature_regimes")


# ------------------------------------------------------------
# 35. COMPOSITION REGIME ANALYSIS
# ------------------------------------------------------------

if "P_CO2" in analysis_df.columns:
    co2_bins = [-np.inf, 0.001, 0.01, 0.1, 0.5, np.inf]
    co2_labels = ["<0.001 bar", "0.001–0.01 bar", "0.01–0.1 bar", "0.1–0.5 bar", ">0.5 bar"]
    analysis_df["CO2_regime"] = pd.cut(analysis_df["P_CO2"], bins=co2_bins, labels=co2_labels)

    agg = dict(number_of_models=("model_id", "count"), median_chi2=("chi2", "median"))
    if "max_feature_snr" in analysis_df:
        agg["median_snr"] = ("max_feature_snr", "median")
    if "n_transits_for_detection" in analysis_df:
        agg["median_transits"] = ("n_transits_for_detection", "median")
    co2_summary = analysis_df.groupby("CO2_regime", observed=True).agg(**agg)

    print("\n" + "=" * 70)
    print("CO₂ REGIME ANALYSIS")
    print("=" * 70)
    save_table(co2_summary, "20_CO2_regimes")


# ------------------------------------------------------------
# 36. MONTE CARLO WEIGHTED STATISTICS
# ------------------------------------------------------------

weighted_table = None
if "relative_likelihood_calc" in analysis_df.columns:
    weights = analysis_df["relative_likelihood_calc"].fillna(0)
    if weights.sum() > 0:
        weighted_results = {}
        for col in ["P_N2", "P_CO2", "P_CH4", "T_surf_climate", "max_feature_snr",
                    "n_transits_for_detection", "spectral_amplitude_ppm"]:
            if col in analysis_df.columns:
                weighted_results[col] = np.average(
                    analysis_df[col].fillna(analysis_df[col].median()), weights=weights)

        print("\n" + "=" * 70)
        print("LIKELIHOOD-WEIGHTED PARAMETERS")
        print("=" * 70)
        weighted_table = pd.Series(weighted_results, name="Likelihood-weighted mean").to_frame()
        save_table(weighted_table, "21_likelihood_weighted_means")


# ------------------------------------------------------------
# 37. PARAMETER RANGES OF TOP 1% MODELS
# ------------------------------------------------------------

top1_stats = None
if "delta_chi2_calc" in analysis_df.columns:
    cutoff = np.percentile(analysis_df["chi2"], 1)
    top_1_percent = analysis_df[analysis_df["chi2"] <= cutoff].copy()

    print("\n" + "=" * 70)
    print("TOP 1% χ² MODEL PARAMETER RANGE")
    print("=" * 70)
    top1_stats = top_1_percent[available_atmospheric].describe().T
    save_table(top1_stats, "22_top1percent_parameter_ranges")


# ------------------------------------------------------------
# 38. BOOTSTRAP CONFIDENCE INTERVAL FOR MEDIANS
# ------------------------------------------------------------

def bootstrap_median(data, n_boot=5000, confidence=0.95):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    medians = [np.median(np.random.choice(data, size=len(data), replace=True))
               for _ in range(n_boot)]
    alpha = 1 - confidence
    return (np.median(data), np.percentile(medians, 100 * alpha / 2),
            np.percentile(medians, 100 * (1 - alpha / 2)))


print("\n" + "=" * 70)
print("BOOTSTRAPPED MEDIAN CONFIDENCE INTERVALS")
print("=" * 70)

bootstrap_results = []
for col in available_atmospheric:
    median, lower, upper = bootstrap_median(analysis_df[col])
    bootstrap_results.append({"Parameter": col, "Median": median,
                              "95% CI lower": lower, "95% CI upper": upper})
bootstrap_results = pd.DataFrame(bootstrap_results)
save_table(bootstrap_results, "23_bootstrap_median_CI", index=False)


# ------------------------------------------------------------
# 39. SAVE COMPLETE STATISTICAL DATASET
# ------------------------------------------------------------

output_csv = os.path.join(OUT_DIR, "TRAPPIST1e_statistical_analysis.csv")
analysis_df.to_csv(output_csv, index=False)
saved_files.append(output_csv)
print(f"\nSaved: {output_csv}")


# ------------------------------------------------------------
# 40. SAVE STATISTICAL SUMMARY TO EXCEL
# ------------------------------------------------------------

output_excel = os.path.join(OUT_DIR, "TRAPPIST1e_statistical_results.xlsx")

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    descriptive.to_excel(writer, sheet_name="Descriptive")
    atm_stats.to_excel(writer, sheet_name="Atmospheric")
    spearman_corr.to_excel(writer, sheet_name="Spearman_Correlation")
    p_matrix.to_excel(writer, sheet_name="Correlation_Pvalues")
    if len(corr_results) > 0:
        corr_results.to_excel(writer, sheet_name="Correlation_Results", index=False)
    bootstrap_results.to_excel(writer, sheet_name="Bootstrap_CI", index=False)
    if "temperature_regime" in analysis_df.columns:
        temp_summary.to_excel(writer, sheet_name="Temperature_Regimes")
    if "CO2_regime" in analysis_df.columns:
        co2_summary.to_excel(writer, sheet_name="CO2_Regimes")
    if weighted_table is not None:
        weighted_table.to_excel(writer, sheet_name="Likelihood_Weighted")
    if top1_stats is not None:
        top1_stats.to_excel(writer, sheet_name="Top1pct_Ranges")

saved_files.append(output_excel)
print(f"Saved: {output_excel}")


# ------------------------------------------------------------
# 41. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Models analyzed: {len(analysis_df)}")

if "chi2" in analysis_df.columns:
    best = analysis_df.loc[analysis_df["chi2"].idxmin()]
    print(f"Best model ID: {best.get('model_id', 'N/A')}")
    print(f"Best χ²: {best['chi2']:.4f}")
    if "reduced_chi2_calc" in analysis_df.columns:
        print(f"Best reduced χ²: {best['reduced_chi2_calc']:.4f}")
    if "BIC_calc" in analysis_df.columns:
        print(f"Best BIC: {best['BIC_calc']:.4f}")

if "max_feature_snr" in analysis_df.columns:
    print(f"Maximum atmospheric feature S/N: {analysis_df['max_feature_snr'].max():.3f}")
if "n_transits_for_detection" in analysis_df.columns:
    print(f"Minimum transits required: {analysis_df['n_transits_for_detection'].min():.2f}")


# ------------------------------------------------------------
# 42. README + ZIP + DOWNLOAD
# ------------------------------------------------------------

readme_path = os.path.join(OUT_DIR, "README.txt")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("TRAPPIST-1e Monte Carlo statistical analysis\n")
    f.write(f"Created: {datetime.now():%Y-%m-%d %H:%M}\n")
    f.write(f"Input CSV: {os.path.basename(filename)}\n")
    f.write(f"Models in file: {len(df)}; models analysed (converged, finite chi2): {len(analysis_df)}\n\n")
    f.write("Contents:\n")
    for p in sorted(saved_files + [LOG_PATH]):
        f.write("  " + os.path.relpath(p, OUT_DIR) + "\n")

sys.stdout.flush()
sys.stdout = sys.stdout.stdout     # stop copying to the log before zipping

zip_path = shutil.make_archive(OUT_DIR, "zip", root_dir=".", base_dir=OUT_DIR)

n_figs = len([p for p in saved_files if p.endswith(".png")])
n_tabs = len([p for p in saved_files if p.endswith(".csv")])
print(f"\nZIP created: {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB; "
      f"{n_figs} figures, {n_tabs} CSV tables, 1 Excel workbook, log, README)")

if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
