"""Repository paths shared by the paper scripts (run them from any directory)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "data") + "/"          # master results CSV, atmosphere profiles, analysis tables
DERIVED = str(ROOT / "derived") + "/"    # outputs of analysis.py / tp_analysis.py
MODELS = str(ROOT / "models") + "/"      # emulator pickles (download from Zenodo)
FIGURES = str(ROOT / "figures") + "/"    # paper figures
