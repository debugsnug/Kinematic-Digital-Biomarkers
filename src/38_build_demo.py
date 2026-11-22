# ============================================================
# STEP 38 - BUILD THE REACH LAB DEMONSTRATION FROM THE FINAL RESULTS
# ============================================================
# Regenerates demo/study.json and presentation/Offline replay.html from the corrected pipeline outputs
# (data/processed, results/model_benchmark, results/nested_selection, results/nested_top3_per_fold).
# The replay's headline is the fully nested selection result; the Random Forest is used only to explain predictions.
import os
import runpy
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "demo"))
runpy.run_path(os.path.join(BASE, "demo", "build.py"), run_name="__main__")
