#!/usr/bin/env python
"""
================================================================================
MASTER EXPERIMENT REPRODUCTION PIPELINE
Course: 23CSE365 Internet of Things – Amrita Vishwa Vidyapeetham
Project: Explainable Kinematic Digital Biomarkers and Functional Resilience
         Modeling for Upper-Limb Performance Assessment in Dyskinetic CP
================================================================================

This master script runs the complete end-to-end experimental pipeline.
Randomness is fixed: every script uses the constant seed 42 (defined in each script;
there is deliberately no --seed option, because a flag that did not reach every script
would be misleading). Changing the seed means editing the SEED / RANDOM_STATE constants.

Usage:
    python run_pipeline.py                  # = --stage all: raw Excel -> features -> models -> audit
    python run_pipeline.py --clean          # first delete ALL derived artifacts (data/processed, results), then run
    python run_pipeline.py --stage features # raw data -> data/processed/experiment_dataset.csv only
    python run_pipeline.py --stage ml       # models, validation, explainability, resilience, audit
                                            #   (re-uses the existing data/processed/experiment_dataset.csv)
    python run_pipeline.py --stage audit    # leakage / split-integrity / methodology audit only
    python run_pipeline.py --reuse-cache    # skip steps 01-02 if data/processed/clean_frames.pkl exists (debug only)

The default 'all' stage ALWAYS re-reads the raw Excel files (steps 01-02); it never silently
re-uses a cached intermediate unless --reuse-cache is given.
"""

import sys
import os
import argparse
import time
import subprocess
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


MARKER_DIR = os.path.join(PROJECT_ROOT, "logs", ".done")
RESUME = False


def run_step(script_name, description, env=None, tag=""):
    script_path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    marker = os.path.join(MARKER_DIR, f"{script_name}{'_' + tag if tag else ''}.ok")
    if RESUME and os.path.exists(marker):
        print(f"\n[resume] skipping {script_name} {tag} (completed earlier)")
        return

    print("\n" + "=" * 78)
    print(f"STEP: {script_name} {tag}- {description}")
    print("=" * 78, flush=True)
    t0 = time.time()

    cmd = [sys.executable, script_path]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env={**os.environ, **(env or {})})

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step {script_name} {tag} failed with exit code {result.returncode} after {elapsed:.1f}s.")
        sys.exit(result.returncode)

    os.makedirs(MARKER_DIR, exist_ok=True)
    open(marker, "w").write(f"{elapsed:.1f}s\n")
    print(f"[OK] {script_name} {tag}finished successfully in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="Master reproduction script for Course Project (23CSE365)"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "features", "ml", "audit"],
        default="all",
        help="Pipeline execution stage (default: 'all' = rebuild everything from the raw data)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete every derived artifact (data/processed/*, results/*) before running"
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Skip raw loading/preprocessing when data/processed/clean_frames.pkl exists (debug only)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps that already finished OK (markers in logs/.done); for restarting an interrupted run"
    )

    args = parser.parse_args()
    global RESUME
    RESUME = args.resume

    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "matplotlib_proj_cache")

    print("=" * 78)
    print("REPRODUCIBLE EXPERIMENT PIPELINE")
    print(f"Course  : 23CSE365 Internet of Things – Amrita Vishwa Vidyapeetham")
    print(f"Stage   : {args.stage}   clean={args.clean}   reuse_cache={args.reuse_cache}   resume={args.resume}")
    print(f"Seed    : 42 (fixed constant in every script)")
    print(f"Root Dir: {PROJECT_ROOT}")
    print("=" * 78)

    if args.clean and args.stage in ("ml", "audit"):
        sys.exit("--clean with --stage ml/audit would delete the inputs they need; use --stage all")
    if args.clean:
        for sub in ("data/processed", "results"):
            target = os.path.join(PROJECT_ROOT, *sub.split("/"))
            if os.path.isdir(target):
                print(f"[clean] removing {target}")
                shutil.rmtree(target)
            os.makedirs(target, exist_ok=True)
        shutil.rmtree(MARKER_DIR, ignore_errors=True)
        if args.stage in ("ml", "audit"):
            sys.exit("--clean with --stage ml/audit would delete the inputs they need; use --stage all")

    start_total = time.time()

    # Stage 1: Feature Extraction & Dataset Construction (always from the raw Excel files)
    if args.stage in ["all", "features"]:
        cache = os.path.join(PROJECT_ROOT, "data", "processed", "clean_frames.pkl")
        if not (args.reuse_cache and os.path.exists(cache)):
            run_step("01_load_raw.py", "Load raw markerless video-derived Excel files")
            run_step("02_preprocess.py", "Clean, filter, and normalize coordinates")

        run_step("03_features.py", "Extract spatial, temporal, angular & quality features")
        run_step("07_repetition_aware.py", "Format repetition trajectories")
        run_step("08_repetition_descriptors.py", "Compute intra-subject repetition descriptors")
        run_step("09_build_experiment_dataset.py", "Assemble datasets: PRIMARY coordinate-only, SECONDARY angle-enhanced, position-error ablation")
        run_step("32_feature_audit.py", "Empirical feature-modality audit (dependency test, angle provenance, counts)")

    # Stage 2: Machine Learning Benchmark, Selection, Validation & Explainability
    SECONDARY = {
        "angle_enhanced": "experiment_dataset_angle_enhanced.csv",
        "coordinate_no_position_error": "experiment_dataset_coord_no_position_error.csv",
        "coordinate_no_duration": "experiment_dataset_coord_no_duration.csv",
        "coordinate_no_duration_no_position_error": "experiment_dataset_coord_no_duration_no_position_error.csv",
        "coordinate_no_timing_coupled": "experiment_dataset_coord_no_timing_coupled.csv",
    }
    REPEAT_SETS = ["angle_enhanced", "coordinate_no_duration", "coordinate_no_timing_coupled"]   # whole-procedure repeats (time-intensive); other sets: single nested run
    if args.stage in ["all", "ml"]:
        # Step A: PRIMARY (coordinate-only) 16-model benchmark + fully nested selection estimate
        run_step("23_model_benchmark.py", "PRIMARY coordinate-only: 16 models, inner-CV tuning and selection")
        run_step("29_nested_selection_estimate.py", "PRIMARY: fully nested estimate of the whole selection procedure")

        # Step A2: SECONDARY / ABLATION feature sets, identical folds and procedure
        for tag, fname in SECONDARY.items():
            sub = f"feature_sets/{tag}"
            env = {"EXPERIMENT_DATASET": fname, "RESULTS_SUBDIR": f"{sub}/model_benchmark"}
            run_step("23_model_benchmark.py", f"{tag}: 16 models", env=env, tag=f"[{tag}] ")
            run_step("29_nested_selection_estimate.py", f"{tag}: fully nested estimate", env=env, tag=f"[{tag}] ")

        # Step B: validation of the selected models (independent Pipeline/GridSearchCV path, cross-checked)
        run_step("16_top3_validation.py", "GLOBAL top-3 diagnostic only (selection-optimistic); cross-check benchmark path")
        run_step("17_confusion_calibration.py", "Compute confusion matrices, MCC & (probability-only) Brier metrics")
        run_step("18_roc_pr_curves.py", "Generate ROC and Precision-Recall curves")

        # Step C: Feature Representation Ablation
        run_step("11_feature_stability.py", "Compute cross-fold feature stability (Jaccard index)")
        run_step("12_ablation.py", "Ablation: Static vs. Repetition-Aware vs. Stable features")
        run_step("22_ablation_statistics.py", "Statistical comparison of ablation representations")

        # Step D: Explainable AI, Functional Resilience & Statistical Significance
        run_step("13_statistical_analysis.py", "Univariate effect sizes (all primary predictors, BH-FDR)")
        run_step("14_generate_figures.py", "Generate publication research figures")
        run_step("15_bootstrap_ci.py", "Compute 10,000 bootstrap confidence intervals")
        run_step("20_shap_oof.py", "Out-of-fold SHAP explainability analysis")
        run_step("31_lime_oof.py", "Out-of-fold LIME explanations + SHAP agreement")
        run_step("30_functional_resilience.py", "Functional Resilience Index from rep1-rep8")

        # Step E: repeated subject-wise partitions
        run_step("27_repeated_subject_cv.py", "Repeated subject-wise nested CV (10 partitions) of the selected families")
        run_step("34_repeated_selection_procedure.py", "Repeat the WHOLE selection procedure over 10 partitions (primary)")
        for tag, fname in SECONDARY.items():
            if tag not in REPEAT_SETS:
                continue
            env = {"EXPERIMENT_DATASET": fname, "FEATURE_SET_DIR": f"feature_sets/{tag}"}
            run_step("34_repeated_selection_procedure.py", f"{tag}: whole procedure over 10 partitions", env=env, tag=f"[{tag}] ")

        # Step F: leakage / confound checks and consolidation
        run_step("26_label_permutation_test.py", "Empirical leakage sanity check: label-permutation test (1000)")
        run_step("28_verify_split_integrity.py", "Re-derive the seed-42 outer/inner splits and assert zero overlap (other seeds: asserted in step 23)")
        run_step("33_acquisition_confound_analysis.py", "Recording-site / acquisition confound investigation")
        run_step("36_nested_top3_per_fold.py", "PRIMARY top-3: select separately within each outer training fold, test on unseen subjects")
        run_step("37_duration_dependence_audit.py", "Recording-duration dependence audit (clip length vs every predictor)")
        run_step("35_feature_set_comparison.py", "Compare coordinate-only vs angle-enhanced vs no-position-error")
        run_step("24_final_comparison.py", "Consolidate final comparison tables and master report")
        run_step("39_nested_pipeline_explanations.py", "Explain the nested winners (permutation importance on held-out subjects)")
        run_step("40_importance_across_feature_sets.py", "SHAP importance across the six feature sets")
        run_step("41_fri_component_analysis.py", "FRI verification and component analysis")
        run_step("42_final_figures.py", "Curated final figures with captions")
        run_step("38_build_demo.py", "Build the Reach Lab study replay and offline page from the final results")

    # Stage 3: Leakage & Methodology Audit
    if args.stage in ["all", "ml", "audit"]:
        run_step("43_verify_reported_numbers.py", "Assert reported headline numbers equal the saved CSVs")
        run_step("25_leakage_audit.py", "Run comprehensive leakage and methodology audit")

    total_time = time.time() - start_total
    print("\n" + "=" * 78)
    print(f"PIPELINE COMPLETE in {total_time:.1f}s ({total_time/60:.2f} minutes)")
    print(f"Summary Report: results/final_comparison/final_results_summary.txt")
    print(f"Audit Report  : results/leakage_audit/leakage_audit_report.txt")
    print("=" * 78)


if __name__ == "__main__":
    main()
