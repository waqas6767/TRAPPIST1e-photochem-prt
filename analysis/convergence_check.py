"""
Photochemical convergence check for a subsample of the Monte Carlo models.

The pipeline records that each photochemical integration finished without
error. This script tests whether the runs actually reached steady state.
It re-runs climate and photochemistry with the pipeline's settings for the
ten best-fitting models plus 90 randomly chosen converged models (fixed
seed), and records the convergence flag returned by Photochem's robust
stepper.

For each model it records
    climate_ok, T_surf_rerun   climate convergence and surface temperature
                               (compared with the master table to confirm
                               that the re-run reproduces the original)
    steady_state               True if the photochemical model converged
    gave_up                    True if the solver gave up
    hit_step_cap               True if MAX_STEPS was reached
    n_steps, runtime_s

The fraction of models at steady state is reported with a 95% Wilson
confidence interval. Results are appended after every model, so an
interrupted run can be resumed (finished models are skipped).

Run from the pipeline folder (input/, zahnle_HNOC.yaml,
TRAPPIST1e_stellar_flux.txt, MC_2000_model_results_flatspec/).
Output: convergence_check/convergence_check_results.csv
"""

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from photochem import EvoAtmosphere
from photochem.clima import AdiabatClimate
from threadpoolctl import threadpool_limits

_ = threadpool_limits(limits=4)

# ---------------- settings ----------------
MASTER_CSV = Path("MC_2000_model_results_flatspec/MC_2000_master_results_flatspec.csv")
CHECK_DIR = Path("convergence_check")
OUT_CSV = CHECK_DIR / "convergence_check_results.csv"
N_TOP = 10           # best-fitting models always included
N_RANDOM = 90        # plus this many random converged models
SEED = 42
MAX_STEPS = 50000    # safety cap on robust_step calls (counted as not converged)

# pipeline settings (identical to run_one_model)
EDDY = 1e5
CHECK_DIR.mkdir(exist_ok=True)


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])


# ---------------- choose the models ----------------
master = pd.read_csv(MASTER_CSV)
ok = master[as_bool(master["climate_converged"])
            & as_bool(master["photochemistry_converged"])
            & np.isfinite(master["chi2"])].copy()

top = ok.nsmallest(N_TOP, "chi2")
rest = ok.drop(top.index)
rand = rest.sample(n=min(N_RANDOM, len(rest)), random_state=SEED)
sample = pd.concat([top.assign(group="top10"), rand.assign(group="random")])
sample = sample.sort_values("model_id").reset_index(drop=True)

done = set()
if OUT_CSV.exists():
    done = set(pd.read_csv(OUT_CSV)["model_id"].astype(int))

print("=" * 70)
print("PHOTOCHEMISTRY CONVERGENCE CHECK")
print("=" * 70)
print(f"Converged models in master CSV : {len(ok)}")
print(f"Models to check                : {len(sample)} ({N_TOP} best + {len(rand)} random, seed {SEED})")
print(f"Already done (skipped)         : {len(done & set(sample.model_id))}")
print(f"Output                         : {OUT_CSV}")
print("=" * 70)


# ---------------- one model ----------------
def check_one(row):
    mid = int(row.model_id)
    P_N2, P_CO2, P_CH4 = float(row.P_N2), float(row.P_CO2), float(row.P_CH4)
    rec = {"model_id": mid, "group": row.group, "P_N2": P_N2, "P_CO2": P_CO2, "P_CH4": P_CH4,
           "chi2": float(row.chi2), "T_surf_master": float(row.T_surf_climate),
           "climate_ok": False, "T_surf_rerun": np.nan, "steady_state": False,
           "gave_up": False, "hit_step_cap": False, "n_steps": 0, "runtime_s": np.nan, "error": ""}
    t0 = time.time()

    # 1. climate (same as pipeline)
    c = AdiabatClimate("input/species_climate.yaml", "input/settings_climate.yaml",
                       "TRAPPIST1e_stellar_flux.txt")
    P_i = np.ones(len(c.species_names)) * 1e-10
    P_i[c.species_names.index("H2O")] = 270.0
    P_i[c.species_names.index("N2")] = P_N2
    P_i[c.species_names.index("CO2")] = P_CO2
    P_i[c.species_names.index("CH4")] = P_CH4
    P_i *= 1e6
    c.RH = np.ones(len(c.species_names)) * 0.5
    c.P_top = 1e-5 * 1e6
    c.max_rc_iters = 30
    c.convective_newton_step_size = 0.05
    c.solve_for_T_trop = True
    c.T_trop = c.rad.skin_temperature(0.3)
    c.surface_temperature(P_i, T_guess=300)
    rec["climate_ok"] = bool(c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below))
    rec["T_surf_rerun"] = float(c.T_surf)
    if not rec["climate_ok"]:
        rec["runtime_s"] = time.time() - t0
        return rec

    atm = CHECK_DIR / f"atmosphere_check_{mid:04d}.txt"
    c.out2atmosphere_txt(str(atm), eddy=np.ones(c.z.shape[0]) * EDDY, overwrite=True)

    # 2. photochemistry (same as pipeline)
    pc = EvoAtmosphere("zahnle_HNOC.yaml", "input/settings_photochemistry.yaml",
                       "TRAPPIST1e_stellar_flux.txt", str(atm))
    pc.var.verbose = 0
    pc.var.atol = 1e-23
    pc.var.upwind_molec_diff = True
    pc.set_lower_bc("N2", bc_type="press", press=P_N2 * 1e6)
    pc.set_lower_bc("CO2", bc_type="press", press=P_CO2 * 1e6)
    pc.set_lower_bc("CH4", bc_type="press", press=P_CH4 * 1e6)
    pc.update_vertical_grid(TOA_pressure=1e-7 * 1e6)

    # 3. same loop as utils.find_steady_state, but we KEEP the result
    pc.initialize_robust_stepper(pc.wrk.usol)
    give_up, converged, n = False, False, 0
    while n < MAX_STEPS:
        give_up, converged = pc.robust_step()
        n += 1
        if give_up or converged:
            break
    rec["n_steps"] = n
    rec["gave_up"] = bool(give_up)
    rec["hit_step_cap"] = bool(n >= MAX_STEPS and not (give_up or converged))
    rec["steady_state"] = bool(converged and not give_up)
    rec["runtime_s"] = time.time() - t0
    atm.unlink(missing_ok=True)
    return rec


# ---------------- run ----------------
todo = sample[~sample.model_id.isin(done)]
t_start = time.time()
for k, row in enumerate(todo.itertuples(index=False), 1):
    try:
        rec = check_one(row)
    except Exception as exc:
        rec = {"model_id": int(row.model_id), "group": row.group, "P_N2": row.P_N2,
               "P_CO2": row.P_CO2, "P_CH4": row.P_CH4, "chi2": row.chi2,
               "T_surf_master": row.T_surf_climate, "climate_ok": False, "T_surf_rerun": np.nan,
               "steady_state": False, "gave_up": False, "hit_step_cap": False, "n_steps": 0,
               "runtime_s": np.nan, "error": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc(limit=1)
    pd.DataFrame([rec]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
    el = time.time() - t_start
    eta = el / k * (len(todo) - k) / 60
    print(f"[{k:3d}/{len(todo)}] model {rec['model_id']:04d} ({rec['group']}) | "
          f"climate={rec['climate_ok']} | steady_state={rec['steady_state']} | "
          f"gave_up={rec['gave_up']} | steps={rec['n_steps']} | "
          f"{rec['runtime_s']:.0f}s | ETA {eta:.0f} min")

# ---------------- summary ----------------
r = pd.read_csv(OUT_CSV).drop_duplicates("model_id", keep="last")
r = r[r.model_id.isin(sample.model_id)]
for col in ["climate_ok", "steady_state", "gave_up", "hit_step_cap"]:
    r[col] = as_bool(r[col])


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


n = len(r)
k = int(r.steady_state.sum())
lo, hi = wilson(k, n)
dT = (r.T_surf_rerun - r.T_surf_master).abs()

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Models checked                  : {n}")
print(f"Climate reproduced              : {int(r.climate_ok.sum())} of {n}  "
      f"(max |T_surf difference| = {dT.max():.3f} K)")
print(f"Photochemical steady state      : {k} of {n} = {100*k/n:.1f}%  "
      f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")
print(f"   best 10 models               : {int(r[r.group=='top10'].steady_state.sum())} of {int((r.group=='top10').sum())}")
print(f"   random models                : {int(r[r.group=='random'].steady_state.sum())} of {int((r.group=='random').sum())}")
print(f"Stepper gave up                 : {int(r.gave_up.sum())}")
print(f"Hit the step cap ({MAX_STEPS})      : {int(r.hit_step_cap.sum())}")
print(f"Errors                          : {int((r.error.fillna('') != '').sum())}")
print(f"Median runtime per model        : {r.runtime_s.median():.0f} s")

bad = r[~r.steady_state]
if len(bad):
    print()
    print("NOT at steady state:")
    print(bad[["model_id", "group", "P_N2", "P_CO2", "P_CH4", "chi2", "gave_up", "n_steps", "error"]]
          .to_string(index=False))
    print()
    print("Median log10 P_CH4: not converged "
          f"{np.log10(bad.P_CH4).median():.2f} vs converged {np.log10(r[r.steady_state].P_CH4).median():.2f}")

print()
print("-" * 70)
print("COMPACT SUMMARY")
print("-" * 70)
print(f"checked={n}; steady={k}; ci={100*lo:.1f}-{100*hi:.1f}; top10={int(r[r.group=='top10'].steady_state.sum())}/"
      f"{int((r.group=='top10').sum())}; gave_up={int(r.gave_up.sum())}; cap={int(r.hit_step_cap.sum())}; "
      f"maxdT={dT.max():.3f}; median_s={r.runtime_s.median():.0f}")
print("=" * 70)
