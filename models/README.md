# Emulator files

The trained random-forest emulators are too large for GitHub. Download them from the
Zenodo dataset record (DOI in the main README) and place them here:

| File | Used by |
|---|---|
| `TRAPPIST1e_Climate_Detection_AI_Combined.pkl` | `paper_scripts/analysis.py`, `emulator_app/` |
| `TRAPPIST1e_Detection_AI_COMPLETE.pkl` | `paper_scripts/analysis.py`, `emulator_app/` |
| `TRAPPIST1e_Detection_Statistics_RF.pkl` | `emulator_app/` |
| `TRAPPIST1e_Climate_Detection_AI_Combined_v2_all1856.pkl` | made by `paper_scripts/retrain.py` (no download needed) |

Only load pickle files from a source you trust: unpickling can run code. The scripts in this
repository use a restricted loader that accepts only scikit-learn, NumPy, pandas and SciPy classes.
For the app, copy the three downloaded files into `emulator_app/models/`.
