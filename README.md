# Explainable Kinematic Digital Biomarkers for Upper-Limb Assessment in Dyskinetic Cerebral Palsy

**Course:** 23CSE365 Internet of Things — Amrita Vishwa Vidyapeetham  
**Guide:** Dr. Thangam S.  
**Data:** KU Leuven markerless video-derived upper-limb motion capture  
(DOI 10.48804/BZM9SJ; Vanmechelen et al., *J. Biomechanics* 2024)

---

## Quick-Start (run on your machine in < 5 minutes)

```bash
# 1. Clone
git clone <repo-url>
cd <project-folder>

# 2. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies (all pinned for exact reproducibility)
pip install -r requirements.txt

# 4a. Run only the ML steps (uses existing preprocessed CSV — fastest):
python run_pipeline.py --stage ml

# 4b. Run everything from raw Excel files (~3.5 hours):
python run_pipeline.py --clean
```

**The file used for all model training, testing and prediction is:**
```
data/processed/final_preprocessed_features_for_model_training_and_prediction.csv
```
This is the **primary preprocessed dataset** (84 coordinate-only features, 1 row per subject, 51 rows). It is identical to `experiment_dataset.csv` and is kept under this descriptive name for publication clarity.

**To run only the new guide-requested experiments (Steps 44–46):**
```bash
# Train-test split experiments (70:30, 80:20, 90:10)
python src/44_train_test_splits.py

# SHAP + LIME for top-5 models + cross-model importance
python src/45_shap_lime_top5_models.py

# Dedicated SHAP + LIME for the single highest-performing model only (MLP)
python src/47_shap_lime_best_model.py

# Deep learning extensions (exploratory)
python src/46_dl_extensions.py
```

---

## Full audit: **`AUDIT_REPORT.md`** | Demo: `docs/DEMO_GUIDE.md` | Readiness: `SUBMISSION_READINESS.md`

## What is predicted

| Item | Value |
|---|---|
| Task | Binary **classification** |
| Target variable | `target` (built in `src/09_build_experiment_dataset.py` from the clinical group) |
| Classes | `1` = DCP (dyskinetic cerebral palsy), n = 30 subjects; `0` = TD (typically developing), n = 21 subjects |
| Unit of analysis | One row per **subject** (51 rows) |
| Excluded from predictors | `subject`, `side`, `group`, `target` |

## Feature sets (single source of truth: `src/feature_sets.py`, audit table: `data/processed/feature_audit.csv`)

| Set | Predictors | Role |
|---|---|---|
| `coordinate_only` | **84** = 78 key-point-coordinate features + 6 `position_error_*` (target-object) | **PRIMARY** - `data/processed/experiment_dataset.csv` |
| `angle_enhanced` | 168 = the 84 + 84 joint-angle-derived features | SECONDARY (labelled) |
| `coordinate_no_position_error` | 78 | ABLATION (target-object dependence) |
| `coordinate_no_duration` | 78 | ABLATION: drops `movement_duration_s_*` (= recording clip length) |
| `coordinate_no_duration_no_position_error` | 72 | ABLATION: both |
| `coordinate_no_timing_coupled` | 66 | ABLATION (strictest): also drops normalised jerk and SPARC, which track duration (rho 0.96 / -0.82) |

Each predictor is tagged with its source modality (coordinates / coordinates + target object / angle-derived). The assignment is not only declared: step `32_feature_audit.py` re-runs the feature extractor with the angle (or object) columns blanked across all 408 recordings and asserts that exactly the registry's predictors change. **Provenance note:** in this dataset the "angle" columns are 2D angles computed from the video key points (dataset paper section 2.4; reproduced from the key points (median RMSE ~1e-14; exact in 38/51 elbow, 38/51 wrist and 46/50 shoulder rep1 recordings, larger deviations in the rest) in `results/feature_audit/angle_provenance_*.csv`); they are not independent Vicon 3D-motion-analysis angles. Excluding them is still done, as requested, because they are a second-stage derivation of the coordinates.

## How to reproduce everything from the raw data

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --clean          # deletes every derived file, then raw .xlsx -> features -> audit -> 16 models x 3 feature sets -> nested estimate -> validation -> repeated runs -> explainability -> audit
```

`--resume` continues after an interruption; `--reuse-cache` is a debug option; `--stage features|ml|audit` runs one stage. A full run takes about 3.5 hours (208 min in the verified run) on a 2-core machine (log: `logs/clean_run_full.log`).
Environment of the verified run: Python 3.13, numpy 2.3.5, pandas 2.2.3, scipy 1.17.1, scikit-learn 1.6.1, xgboost 3.0.4, shap 0.49.1, matplotlib 3.10.1, openpyxl 3.1.5, joblib 1.4.2, lime 0.2.0.1 (pinned in `requirements.txt`). All randomness uses seed 42 (CV seeds 42-51 for repeated runs). `notebooks/reproduce_all.ipynb` displays the final tables/figures read from `results/` (re-run the pipeline first).

## Headline results (coordinate-only primary, 84 predictors)

| Quantity | Value |
|---|---|
| **Fully nested whole-procedure estimate** (family + hyperparameters chosen inside every outer training fold; seed-42 partition) | accuracy **0.824** [0.706, 0.922], balanced acc. 0.814 [0.699, 0.918], F1 0.852, sensitivity 0.867, specificity 0.762, MCC 0.634 [0.403, 0.840], mean fold ROC-AUC 0.968 |
| Same whole procedure repeated on 10 re-partitions of the SAME 51 subjects | accuracy 0.855 (SD 0.030, range 0.824-0.902) |
| **Per-outer-fold top-3** (chosen inside each training fold; step 36) | rank-1 / rank-2 / rank-3 slot accuracy 0.824 / 0.863 / 0.843; majority vote of the three 0.863 [0.765, 0.941] - selection-unbiased |
| Global top-3 by inner CV (MLP, Extra Trees, RBF-SVM; step 16) pooled subject-level accuracy | 0.882 / 0.882 / 0.902 (fold-mean 0.884 / 0.884 / 0.904) - **optimistic** (ranking pools inner CV over all folds), kept for the course requirement only |
| Angle-enhanced (168), nested | 0.863 [0.765, 0.941] |
| Without `position_error` (78), nested | 0.882 [0.784, 0.961] |
| Without duration (78) / duration + position_error (72) / strictest timing-coupled (66), nested | 0.843 / 0.824 / 0.843 |

Differences between feature sets are not statistically distinguishable (paired subject-level bootstrap / exact McNemar, `results/feature_sets/`). Removing `position_error`, the duration features, or all timing-coupled features does **not** lower performance, so neither the target-object feature nor duration alone drives the result. This does not make duration harmless: duration cannot be separated from clip length/trimming (below), and many other predictors correlate with it.
The earlier 168-predictor full-feature nested estimate (0.824) is kept only as a historical comparison; the coordinate-only run is the headline.

## Demonstration and slides (Reach Lab)

`present.py` / `Present.command` start a local receiver and dashboard: **Study replay** of any of the 51 participants' eight reaches with held-out results, and a **Live session** that accepts a wrist marker from a webcam or timestamped coordinates POSTed by another device (the IoT part of the course project). `presentation/` holds the 5-slide deck (PDF copy and a standalone offline page). The replay is generated from the final results by step 38 and shows the **nested-selection** result (42/51) as its headline; see `docs/DEMO_GUIDE.md` for the rehearsal script, device format and what each number means. Contributions: pipeline and audit (this project's research core) plus the demo/deck/device interface from the teammate's `reach-lab` repository, re-pointed to the corrected data.

## Honest limitations (read before quoting any number)

* **No external test set.** "Test" always means outer cross-validation folds on 51 subjects from one dataset. Bootstrap CIs and repeated partitions re-use the same 51 subjects and do **not** measure generalisation to new sites, cameras or clinics.
* **Clip length / duration.** `movement_duration_s` = (frames-1)/25 s, i.e. exactly the clip length (r = 1.0). Clip length alone gives balanced accuracy 0.76 (logistic; 0.65 RF); DCP clips are longer on average (65 vs 29 frames, p = 6e-5). Whether this is genuine motor slowness or recording/trimming difference cannot be decided from the supplied metadata (`results/duration_audit/`).
* **Recording-setup confound is unresolved.** The data contain no site/session/camera variable; subject-number blocks are structured by group and a subject-number-only classifier reaches balanced accuracy 0.85-0.87 with a Random Forest (0.64 with a linear model); proxy classifiers: camera-frame positions 0.67, apparent body size 0.60, clip length 0.76; evidence of a specific shortcut is mixed, not absent (`results/confound_analysis/`). Group and recording batch cannot be separated, so high accuracy may partly reflect the recording setup. The position-error ablation does not remove this confound. An independent, site-balanced cohort is required before any clinical claim.
* Selection is by inner-CV accuracy; at n = 51 the top models are statistically indistinguishable, and the selected family changes from fold to fold. LightGBM and CatBoost are **not** implemented (16 models; adding them would give 18 and needs the guide's approval).
* The Functional Resilience Index is a **proposed, exploratory** index (equal weights, not clinically validated). It excludes recording duration, but its group signal (AUC 0.75) mostly disappears (AUC 0.64, FDR p = 0.10) when normalised jerk and SPARC - which track clip length - are removed too. Repetition-aware descriptors showed no statistically supported advantage over static means.
* No calibrated probabilities are claimed for any model: SVM scores are used only for rank metrics; Brier score is reported only for `predict_proba` models and no calibration curve/ECE is computed (Naive Bayes and Extra Trees are often miscalibrated).
* AUC: tables from steps 15/17/18 pool out-of-fold scores across the 5 folds (pooled AUC); nested/benchmark tables report the mean of per-fold AUCs. Values differ slightly (small differences between the two).
* Hyperparameter grids were widened (6-18 configurations per model, times k in {20, 40}); they are still modest because of the sample size (n = 51). Exhaustive tuning would increase selection noise.

## Method in brief

* **Split**: `StratifiedGroupKFold` on subject, outer 5-fold, inner 4-fold, shuffled, seed 42; zero subject overlap is asserted for every outer *and* inner split (step 23) and re-verified on saved assignments (step 28).
* **Leakage control**: median imputer -> StandardScaler -> `SelectKBest(mutual_info_classif)` (k in {20, 40}) -> model, fit on training folds only. Hyperparameters and model choice use inner CV only. The benchmark pre-fits transforms per fold; step 16 re-runs **all 16 families** through a real `Pipeline`+`GridSearchCV` and confirms 80/80 model-fold predictions and best parameters are identical.
* **One model catalog** (`src/model_catalog.py`) defines all 16 models, grids, class weights and preprocessing, used by steps 16, 20, 23, 27, 29, 31, 34. Linear SVM is `SVC(kernel="linear")` everywhere (an earlier benchmark/validation mismatch with `LinearSVC` and different Random-Forest settings was removed).
* **Benchmark (`23`)**: 16 classifiers with hyperparameter grids; best parameters per fold are saved in `model_benchmark_fold_results.csv`; metrics computed per fold and re-derivable from saved out-of-fold predictions.
* **Selected-model validation (`16`)**: global top 3 by inner-CV accuracy (`selected_top_models.json`) evaluated on outer folds (optimistic). **`36`** selects the top 3 separately inside each outer training fold and evaluates them on that fold's unseen subjects (selection-unbiased).
* **Nested estimate (`29`)**, **repeated whole-procedure (`34`)**, **repeated selected-model CV (`27`)**, **per-fold top-3 (`36`)**, **duration audit (`37`)**, **feature-set comparison (`35`)**, **confound analysis (`33`)**, **feature audit (`32`)**, **label-permutation sanity check (`26`)**, **FRI (`30`)**, **SHAP (`20`)**, **LIME (`31`)**, **audit (`25`)**.
* **Metrics**: accuracy, balanced accuracy, F1, sensitivity, specificity, ROC-AUC (never pooled across different model families), PR-AUC, MCC, Brier (probability models only).

## Folder layout

```
run_pipeline.py        single entry point
present.py, demo/      Reach Lab receiver, dashboard, device sender (docs/DEMO_GUIDE.md)
presentation/          5-slide deck (pptx + pdf) and offline replay page
docs/                  demo guide; archive_teammate_review/ (historical, superseded)
src/                   numbered pipeline steps + model_catalog.py + feature_sets.py
data/raw/              original per-repetition Excel files (408)
data/processed/        generated: features, experiment datasets, feature_audit.csv
results/               generated: one folder per analysis (see AUDIT_REPORT.md)
results/figures/       older per-step diagnostic plots (Figure_1 ... Figure_10; several explain the exploratory RF)
results/figures_final/ curated report figures F1-F8 + captions.md (step 42)
tests/                 dashboard_test.js (Playwright; 51 participants x 8 reps vs the CSVs)
notebooks/             reproduce_all.ipynb
logs/                  clean-run log
```

## Updated interpretation and validation note (2026-11-07)

**Primary top-3 validation is step 36 (`results/nested_top3_per_fold/`).** In each outer fold, the 16 model families are ranked by inner-CV accuracy computed using only that fold's training subjects. The three selected families are then evaluated on that fold's held-out subjects. Model identities may differ by fold; that is expected and is reported rather than hiding selection instability. The script now writes both subject-level bootstrap intervals and fold-level mean/SD summaries (`per_fold_top3_fold_summary.csv`).

**Step 16 (`results/top3_validation/`) is a diagnostic cross-check only.** Its single global top-three list pools inner-CV rankings across outer folds. A subject held out in one fold can be present in other folds' training data and can therefore influence that global list. These step-16 outer-fold numbers are selection-optimistic and must not be described as an independent final test estimate. Step 16 is retained to verify the independent sklearn Pipeline/GridSearchCV implementation against the benchmark path.

Preprocessing's `t_norm` is a per-repetition time-axis rescaling from 0 to 1. It is **not** framewise trajectory alignment: the current repetition descriptors compare scalar summaries across repetitions. Do not claim that framewise trajectory alignment or DTW has been performed.

The acquisition/recording confounding limitation remains unresolved by the available dataset. Duration-excluded ablations show that clip duration alone does not account for all classification performance, but correlated movement-quality features and structured participant identifiers mean that the cohort cannot establish site-independent clinical generalisation. FRI remains a proposed exploratory index, not a clinically validated biomarker.


## Architecture

```
data/raw (408 xlsx = 51 subjects x 8 reps)
  -> 01 load (fail-closed 408/51/8) -> 02 preprocess -> 03/07/08 per-rep features + 6 repetition descriptors
  -> 09 experiment datasets (6 feature sets; one row per subject)          [src/feature_sets.py]
  -> 23 16-model benchmark: outer 5-fold x inner 4-fold StratifiedGroupKFold, subjects never split  [src/model_catalog.py]
        imputer -> scaler -> SelectKBest(k) -> model, fitted on training folds only
  -> 29 fully nested estimate (PRIMARY) | 36 per-fold top-3 vote | 16 global top-3 (optimistic, diagnostic)
  -> 33/37 confound + duration audits | 35 feature-set ablation | 34 repeated partitions | 26 label permutation
  -> 20/31 SHAP + LIME (exploratory RF) | 39 explanations of the nested winners | 40 importance across feature sets
  -> 30/41 Functional Resilience Index + component analysis (exploratory)
  -> 42 figures | 43 number check | 25 leakage audit | 38 Reach Lab replay (present.py, demo/)
```

## What every major script does

| Step | Script | Purpose |
|---|---|---|
| 01-09 | `01_load_raw` ... `09_build_experiment_dataset` | raw Excel -> per-repetition kinematics -> subject-level experiment datasets |
| 11-15, 22 | stability, ablation, statistics, figures, bootstrap | descriptor ablation and univariate tests (Mann-Whitney, Cohen d, BH-FDR) |
| 16-18 | top-3 validation, confusion/calibration, ROC/PR | global top-3 (optimistic cross-check only), curves |
| 20, 31 | `shap_oof`, `lime_oof` | out-of-fold SHAP / LIME of an exploratory Random Forest |
| 23 | `model_benchmark` | 16 models x hyper-parameter grids, leak-free nested CV; cross-checked against a real sklearn Pipeline |
| 24, 25 | `final_comparison`, `leakage_audit` | consolidated tables; content-based leakage and methodology checks |
| 26-28 | permutation test, repeated CV, split integrity | sanity checks of the split and the signal |
| 29 | `nested_selection_estimate` | **primary** fully nested estimate |
| 30, 41 | `functional_resilience`, `fri_component_analysis` | exploratory FRI; independent recomputation and per-component analysis |
| 32-34, 37 | feature audit, acquisition confound, repeated selection, duration audit | modality registry, acquisition proxies, clip-length dependence |
| 35, 36 | feature-set comparison, per-fold top-3 | ablation across six sets; selection-unbiased top-3 |
| 38 | `build_demo` | regenerates `demo/study.json` and the offline replay from final results |
| 39 | `nested_pipeline_explanations` | permutation importance of each fold's nested winner (refit, asserted equal to saved accuracy) |
| 40 | `importance_across_feature_sets` | out-of-fold SHAP per feature set, rank agreement, modality shares |
| 42, 43 | `final_figures`, `verify_reported_numbers` | curated figures; asserts every headline number equals its CSV |

## Explainability: what explains what

* **Exploratory Random Forest** (steps 20, 31, 40): SHAP/LIME top features are path efficiency, lateral deviation, trajectory length (SHAP-LIME Spearman 0.84, top-15 overlap 13/15). Interpretation: straighter, less sideways-deviating, steadier reaches are associated with the TD group. Importance ranks are very stable across the six feature sets (Spearman 0.96-1.00).
* **Nested pipeline** (step 39): each fold's winner (HistGradientBoosting, Extra Trees x2, Gaussian NB, Linear SVM) is refitted and permuted on about 10 held-out subjects. Effects are small and unstable and **disagree** with the SHAP ranking. So the SHAP story explains the exploratory forest, not the validated procedure, and no single feature is established as driving the nested result.

## FRI validation (exploratory)

Step 41 recomputes the FRI independently (equal to step 30 for all 51 subjects). FRI separates groups (AUC 0.754, BH-FDR p 0.0068, d = -0.85), but path efficiency and position error carry most of it; removing SPARC and normalised jerk gives AUC 0.643 (FDR p 0.13 in the step-41 family), and max lateral deviation works against the index. The signal therefore depends on timing-coupled metrics and is not established independently of acquisition differences.

## Reach Lab dashboard and its test

Study replay is labelled *Research view*; Live session is labelled *Exploratory demo* (3-feature wrist-only logistic model, not the validated 84-feature pipeline). A participant summary shows all eight reaches (efficiency, duration, endpoint error). Test: `NODE_PATH=$(npm root -g) PY=python3 node tests/dashboard_test.js .` - checks 51 participants x 8 reps against `rep_features.csv` and the nested held-out call/winner against `nested_selection_predictions.csv` (last run: 1,275 values, 0 failures, 0 page errors). Needs Node + Playwright (not in requirements.txt).

## Verification status of this release (honest scope)

* `run_pipeline.py --clean` (208 min, `logs/clean_run_full.log`) produced the primary results and predates steps 39-43, the dashboard edits and tests. Steps 39-43 were run afterwards on the existing results (additive; no primary result changed; a baseline copy was kept).
* A fresh virtualenv from `requirements.txt` + a project copy without `results/` and `data/processed/` ran `python run_pipeline.py --stage features` in 15 s and reproduced `experiment_dataset*.csv` and `rep_features.csv` byte-for-byte. The ML stage was not re-run from scratch for this release.
* Nothing here resolves the acquisition confound (no site/camera variable); no external validation; FRI is not clinically validated.
