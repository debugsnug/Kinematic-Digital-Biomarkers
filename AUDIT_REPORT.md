# Audit report - six course requirements, feature audit, and limitations

Project: explainable kinematic digital biomarkers, DCP vs TD, KU Leuven markerless dataset (51 subjects: 30 DCP, 21 TD).
Every number below is read from files created by `python run_pipeline.py --clean` (log: `logs/clean_run_full.log`).

## 1. Verdicts on the six requirements

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 1 | Train/compare 15-16 models with varied hyperparameters | **MET (16 models; grids modest)** | `src/model_catalog.py` defines 16 models, each with a hyperparameter grid (6-18 configurations x k in {20, 40}; widened in the latest run for LR, RBF-SVM, KNN, LDA, Gradient Boosting, AdaBoost, NB, MLP, XGBoost; still small because n = 51); all execute in `results/model_benchmark/` (every model has 5 folds with saved `best_parameters`). **LightGBM and CatBoost are NOT implemented**; adding them would make 18 models and requires the guide's approval. |
| 2 | Classification task | **MET** | Binary classification, DCP vs TD. |
| 3 | State target and classes clearly | **MET** | `target` = 1 for DCP (30), 0 for TD (21); built once in `09_build_experiment_dataset.py`; consistent in all outputs (audit check "Target distribution"). |
| 4 | Prevent data leakage | **MET, with documented limits** | Imputer/scaler/SelectKBest fit on training folds only; identifiers (`subject`, `side`, `group`) excluded; repetition features computed within-subject; angle features excluded from the primary set; 80/80 model-fold cross-check (all 16 families) between benchmark and real Pipeline path; label-permutation sanity test (null 0.48, p = 0.001). **Not eliminated:** recording-setup confounding (Section 4) - leakage controls do not address it. |
| 5 | Subject-only splits | **MET** | One row per subject; `StratifiedGroupKFold` on subject; zero overlap asserted for all 5 outer **and** all inner (4 per outer) splits inside step 23, saved assignments re-verified in step 28. |
| 6 | Validate 2-3 highest-accuracy models chosen without final test data | **MET via per-fold selection; global top-3 is optimistic** | Step 36 chooses the top 3 families **separately inside each outer training fold** (inner CV only) and evaluates them on that fold's unseen subjects: rank-1/2/3 slots 0.824 / 0.863 / 0.843 accuracy, majority vote 0.863 [0.765, 0.941] (`nested_top3_per_fold/`). The global top 3 (MLP, Extra Trees, RBF-SVM; step 16, `top3_validation/`) pools inner-CV scores over all outer folds, so their outer-fold accuracies (0.88-0.90) are **optimistic** and are not presented as unbiased. No claim is made that these are the globally best models. |

## 2. Requirements not (fully) satisfiable with the supplied data

* **Independent test data**: none exists. With 51 subjects an external hold-out was not carved out; "final test" = outer CV folds.
* **Site/camera balance**: not checkable or adjustable (no variable). See Section 4.
* **LightGBM/CatBoost**: not implemented (above). The course slide lists them; options are (a) ask the guide to accept 16 models without them, (b) replace two weak/redundant models (e.g. Polynomial SVM at 0.644, Decision Tree) with them to stay at 16, or (c) add them for 18. This is a decision for you and your guide; nothing here changes if (b) is chosen except those two rows and the downstream reruns.

## 3. Coordinate-only primary model: feature audit

* Registry: `src/feature_sets.py`; per-predictor table `data/processed/feature_audit.csv` (predictor, base feature, descriptor, source modality, membership in each set).
* All 168 predictors audited: **84 coordinate-derived** (78 coordinates + 6 `position_error_*`, which depend on the target object position) and **84 angle-derived** (14 base angle features x 6 descriptors, including the Movement Stability Index inputs). Primary set = 84; secondary = 168; ablation = 78 (`results/feature_audit/feature_set_sizes.csv`).
* Verified empirically, not by naming: blanking angle (or object) columns in all 408 recordings changes exactly the registered predictors (`base_feature_dependency.csv`; the script asserts equality).
* `position_error` reflects distance between the end-point and the target object: it depends on the target, not on joint angles. Ablation shows no loss without it (below).
* Angle provenance: 2D angles from video key points (reproduced from raw key points, median RMSE ~1e-14, exact in 38/51 elbow, 38/51 wrist and 46/50 shoulder rep1 recordings; dataset paper 2.4), so they are not independent 3D-MA measurements. The instruction to keep them out of the primary model is followed; they are reported as "2D video-derived angle features" (secondary).
* How the quoted figures (Linear SVM 0.902, Extra Trees 0.882 vs 0.878/0.867) relate: no saved run reproduces both pairs under a single configuration. The earlier benchmark and validation used different definitions (Linear SVM as `LinearSVC` vs `SVC`; different Random-Forest class weights/`min_samples_leaf`), which is why numbers differed between steps. Everything now uses one catalog; the numbers in this report are the recomputed ones (Linear SVM 0.904, Extra Trees 0.884 outer-fold accuracy on the coordinate-only set), not reused.

## 4. Site / session / object confounding - what is and is not established

(`results/confound_analysis/confound_summary.txt`)

* Dataset documentation: recruitment in Belgium and the Netherlands; each laboratory trained its own DeepLabCut model. It does **not** state which child was recorded where, and there is no camera/session variable. The earlier statement "all TD in Belgian labs, 6 DCP in Amsterdam" could not be verified and has been **removed**.
* Subject-number blocks are structured by group (25-42 and 67-89 only DCP); a subject-number-only Random Forest reaches balanced accuracy 0.85-0.87 (a linear model 0.64); proxy classifiers: camera-frame positions 0.67, apparent body size 0.60 (an earlier summary omitted clip length, 0.76) and the primary model is least accurate (0.625) on DCP in the 25-42 block, so the evidence for a specific shortcut is mixed. Absence of proof is not proof of absence. Filenames and Excel metadata carry no site information. Arm side (TD 18 left / 3 right) and sex are also unbalanced between groups.
* The `position_error_*` ablation (78 predictors, same outer folds, same nested procedure): nested accuracy 0.882 [0.784, 0.961] vs 0.824 for the primary set; paired difference +0.059 (bootstrap CI 0.000 to +0.137, exact McNemar p = 0.25). Performance does not depend on the target-object feature.
* **Recording duration (new).** `movement_duration_s` is computed as (n_frames - 1) / 25, so it is exactly the clip length (r = 1.000, `results/duration_audit/`). Clip length alone separates the groups with balanced accuracy 0.76 (logistic; 0.65 RF); DCP clips are longer (65 vs 29 frames, Mann-Whitney p = 6e-5). Normalised jerk (rho 0.96) and SPARC (rho -0.82) track clip length as well. Duration may be genuine motor slowness or a recording/trimming difference; the metadata cannot decide. Duration-excluded ablations (Section 5.4) keep nested accuracy at 0.82-0.84, so the classifier is not carried by duration alone.
* **Conclusion: confounding is not fixed.** Group and recording setup cannot be separated here; high accuracy may partly reflect the recording setup. An independent, site-balanced cohort is needed.

## 5. Results (coordinate-only, 84 predictors)

### 5.1 Benchmark (outer-fold mean +/- SD over 5 folds; ranked by inner-CV accuracy) - full table in `results/final_comparison/results_table.md`

Top 6 by inner CV (outer-fold test accuracy in brackets): MLP 0.927 (0.884), Extra Trees 0.922 (0.884), RBF-SVM 0.922 (0.904), Gaussian NB 0.922 (0.922), Linear SVM 0.918 (0.904), Random Forest 0.917 (0.924). Weakest: Decision Tree 0.842 (0.862), Polynomial SVM 0.644 (0.635). Differences among the top ~7 are within one fold SD.

### 5.2 Headline: fully nested estimate (`results/nested_selection/`)

| Metric | Estimate [95% subject-level bootstrap CI] |
|---|---|
| Accuracy | 0.824 [0.706, 0.922] |
| Balanced accuracy | 0.814 [0.699, 0.918] |
| F1 | 0.852 [0.745, 0.935] |
| Sensitivity | 0.867 [0.733, 0.970] |
| Specificity | 0.762 [0.565, 0.941] |
| MCC | 0.634 [0.403, 0.840] |
| ROC-AUC (mean of 5 fold AUCs) | 0.968 (SD 0.037) |

Winners per fold differ (HistGradientBoosting, Extra Trees x2, Gaussian NB, Linear SVM): the selection step is itself unstable. Best parameters per outer fold: `nested_selection_folds.csv`; benchmark and top-3: `model_benchmark_fold_results.csv`, `top3_best_parameters.csv`.

### 5.3 Kinds of uncertainty (do not mix)

| Source | Value | What it covers |
|---|---|---|
| Fold mean +/- SD | 0.824 +/- 0.083 | spread over 5 folds of ~10 subjects |
| Pooled out-of-fold + subject bootstrap | CI above | subject-sampling noise with trained models fixed |
| Repeated partitions, whole procedure (10 seeds) | 0.855 +/- 0.030 (0.824-0.902) | split sensitivity on the same 51 subjects; seed 42 is tied for the lowest of the ten |
| Repeated nested CV of the 3 globally selected families (`27`) | 0.900-0.906 | optimistic (families chosen using all subjects) |

None of these measures generalisation to new sites or cohorts.

### 5.4 Feature-set comparison (same outer folds; `results/feature_sets/`)

| Set | Predictors | Nested accuracy [CI] | Paired diff vs primary [bootstrap CI] | Repeated 10x mean |
|---|---|---|---|---|
| coordinate_only (primary) | 84 | 0.824 [0.706, 0.922] | - | 0.855 |
| angle_enhanced (secondary) | 168 | 0.863 [0.765, 0.941] | +0.039 [-0.059, +0.137] | 0.853 |
| coordinate_no_position_error | 78 | 0.882 [0.784, 0.961] | +0.059 [0.000, +0.137] | not run |
| coordinate_no_duration | 78 | 0.843 [0.745, 0.941] | +0.020 [0.000, +0.059] | 0.867 |
| coordinate_no_duration_no_position_error | 72 | 0.824 [0.706, 0.922] | 0.000 [-0.059, +0.059] | not run |
| coordinate_no_timing_coupled (no duration, norm. jerk, SPARC) | 66 | 0.843 [0.725, 0.941] | +0.020 [-0.039, +0.078] | 0.880 |

No difference is statistically supported (exact McNemar p >= 0.25 for all). The result does not depend on `position_error` or on duration/timing-coupled features alone. It also does not show these features are harmless: duration is entangled with clip length (Section 4), and removing them leaves other predictors that correlate with it.

### 5.5 Explainability and biomarkers

* SHAP (out-of-fold, catalog Random Forest) and LIME (out-of-fold) were recomputed on the 84-predictor set; Spearman rho between them 0.84, top-15 overlap 13/15. Strongest features: path efficiency, lateral deviation, trajectory-length variability, peak-velocity CV.
* Biomarkers (`03`/`30` feature engineering reviewed): trajectory length, path efficiency, lateral deviation, velocity/acceleration, smoothness (normalised jerk), speed variability, position error, plus repetition descriptors (SD, CV, slope, early-late change, monotonicity) are genuinely computed from the key-point trajectories.
* Functional Resilience Index (`30`): **proposed/exploratory**; equal-weight mean of a consistency score and a maintenance score over path efficiency, lateral deviation, SPARC, normalised jerk and position error (recording duration is **excluded** because it equals clip length). Primary FRI: TD median 0.943 vs DCP 0.916; Cohen d -0.85 [-1.44, -0.35]; AUC 0.75 [0.60, 0.88]; BH-FDR p = 0.007 (without position error: AUC 0.69, FDR p = 0.037). **Sensitivity:** the legacy duration-inclusive version gives AUC 0.78 (diagnostic only); a version that also drops normalised jerk and SPARC (which track clip length, rho 0.96 / -0.82) has AUC 0.64 [0.48, 0.80], FDR p = 0.10 - i.e. the FRI's group signal largely depends on timing-coupled smoothness metrics and is **not** duration-free in that sense. Adding the indices to static means: balanced accuracy 0.938 -> 0.959 (+0.021, repeats share the same 51 subjects; not an independent test). The Movement Stability Index uses angle-derived inputs and is labelled as such.

## 6. Checks run (`results/leakage_audit/leakage_audit_report.txt`)

PASS = 22, INFO = 5, LIMITATION = 5, FAIL = 0. Substantive PASS checks read file contents and re-derive the claim: outer-fold assignments identical across all 6 feature sets; 11 saved outer assignments each place every subject in exactly one test fold; nested winners re-derived as the max-inner-accuracy family per fold; benchmark vs real Pipeline predictions and best parameters identical in 80/80 model-folds (all 16 families); predictor counts 84/168/78/78/72/66. Step 23 asserts zero subject overlap in every outer and inner split at run time (not re-derivable from saved files). The FRI and LIME entries only confirm that an output exists (INFO), and the label-permutation test uses a single fixed Random Forest (sanity check only). LIMITATIONS (unresolved, not failed checks): optimistic global top-3; acquisition/duration confounding; no external test set; conditional uncertainty intervals; repetition-aware features not shown to help.

## 7. Change summary versus the previous submission

* Primary feature set changed from 168 to 84 coordinate-only predictors; angle-enhanced kept as labelled secondary; full benchmark, nested estimate, validation, repeated runs, ablations, SHAP/LIME and FRI **re-run** (not re-labelled).
* One shared model catalog replaces per-script model definitions (Linear SVM and Random-Forest mismatches fixed).
* Pooled AUC across different model families removed (steps 15/17/18 still report per-model pooled AUC across folds, labelled as such); Brier only for probability models; SVM scores are not presented as calibrated probabilities.
* Unverifiable site claim removed; confound analysis added; feature-set and position-error comparisons added.
* New: per-fold top-3 selection (step 36), duration audit (37) and three duration ablations, widened grids, Pipeline cross-check on all 16 families, audit checks that re-derive their claims (outer-fold identity across all 6 sets, saved split integrity over 11 assignments, nested winner re-derivation); FRI/LIME presence checks relabelled INFO.
* Fixes found while re-running: hard-coded model names in ROC/PR step, Brier logic in confusion/calibration step, stale predictor count in audit.

## 8. Corrections from the second independent review (2026-11-06)

* `01_load_raw.py` now fails closed unless all 408 files load, every subject has repetitions 1-8 with consistent group/side, and the cohort is 51 subjects (30 DCP / 21 TD).
* `28_verify_split_integrity.py` checks saved outer assignments and prediction-to-fold consistency directly; exact inner/outer splitter reconstruction runs only under the pinned scikit-learn 1.6.1 (the verified run).
* FRI no longer includes `movement_duration_s`; duration-inclusive and timing-coupled-excluded variants are reported as sensitivity analyses only.
* `.python-version` / `requirements.txt` record the verified runtime (CPython 3.13.18, pinned packages). Everything in this report comes from a single uninterrupted `--clean` run (208 min) made **after** these edits (`logs/clean_run_full.log`); results other than FRI were unchanged by them.
* Only the performing arm is present per participant, so bilateral symmetry cannot be estimated.

## 9. Integration with the Reach Lab demonstration (2026-11-20)

* The teammate's repository contributed the dashboard/receiver/device sender (`demo/`, `present.py`), the slide deck and the case-comparison figure. They were re-pointed to `data/processed` and to this pipeline's shared model catalog, and are rebuilt by step 38; audit check "Reach Lab demo matches final results" verifies the replay data hash and that all 51 headline calls equal `results/nested_selection` predictions.
* The deck and dashboard previously headlined 48/51 (Random Forest chosen after seeing all folds, 168 descriptors incl. joint angles). They now headline the fully nested 42/51 (0.824 [0.706, 0.922]) and label the Random Forest as explanation-only (47/51, optimistic) and the three-feature wrist model as exploratory (45/51, features chosen knowing the cohort).
* The teammate's older research scripts and benchmark are archived under `docs/archive_teammate_review/` as superseded; they are not part of the pipeline.
* Teammate-reported statements that did not carry over: "participants 46 and 67 correctly classified" (under nested selection 46, a TD participant, is a false alarm); the 168-descriptor benchmark numbers (superseded by the 84-predictor coordinate-only primary analysis).
* The demo adds no new validation. The live camera path has not been compared with any independent reference.

---
## Addendum (post clean-run additions)
Steps 39-43, `results/figures_final/`, the dashboard edits and `tests/dashboard_test.js` were added after the 208-minute clean run and are NOT covered by `logs/clean_run_full.log`. They only read existing results; no primary result changed (baseline copy kept). Findings: nested-winner permutation importances (step 39) are small and disagree with SHAP ranks; FRI (step 41) recomputes exactly, but without SPARC and normalised jerk its AUC falls 0.754 -> 0.643 (not significant after FDR); step 43 asserts every headline number against its CSV; the dashboard test matched 1,275 displayed values to the dataset with 0 failures. A fresh venv + clean copy reproduced all experiment datasets byte-for-byte (features stage only).
