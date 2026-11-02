# TRAPPIST-1e Atmosphere Detection Lab (Streamlit)

Interactive explorer for the TRAPPIST-1e emulator. It runs the three trained random
forests directly (no approximation) to predict the surface temperature, the fit to the
JWST spectrum and the number of transits needed to detect a feature, for any N₂, CO₂
and CH₄ partial pressures inside the training range.

## Run
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

The emulator files were saved with scikit-learn 1.6.1, so it is pinned in requirements.txt
(needs Python 3.10 to 3.13). Download the three .pkl files from the Zenodo dataset (see the main README) into `models/`.

Only load pickle files you trust: unpickling can run code. app.py uses a restricted
loader that only allows scikit-learn, NumPy, pandas and SciPy classes.
