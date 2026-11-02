#!/usr/bin/env bash
# Fetch the Photochem workshop files used by the pipeline and apply the one
# change made for this work: 100 atmospheric layers in the climate model
# (the photochemistry settings already use 100 layers).
#
# Run from this folder:  bash setup_workshop_inputs.sh
set -euo pipefail
COMMIT=55fd4ce838bb315c0ef61914e51bef24f8016242   # workshop version used (30 June 2025)
git clone https://github.com/Nicholaswogan/UofA_Photochem_Workshop.git workshop_tmp
git -C workshop_tmp checkout "$COMMIT"
cp -r workshop_tmp/5_TRAPPIST1e/input .
cp workshop_tmp/5_TRAPPIST1e/input_files.py workshop_tmp/5_TRAPPIST1e/utils.py .
sed -i.bak 's/number-of-layers: 50/number-of-layers: 100/' input/settings_climate.yaml
rm -rf workshop_tmp input/*.bak
grep -n "number-of-layers" input/settings_*.yaml
echo "Done: input/, input_files.py and utils.py are ready next to trappist1e_mc_pipeline.py"
