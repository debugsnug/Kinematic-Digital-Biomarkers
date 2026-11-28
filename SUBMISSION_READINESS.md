# Submission readiness and requirement audit

Project: Explainable kinematic digital biomarkers and functional resilience for markerless upper-limb assessment in dyskinetic cerebral palsy

Course: 23CSE365 Internet of Things, Amrita Vishwa Vidyapeetham

## Prediction task

The implemented task is binary subject-level classification.  The target variable is `target`: `1` denotes dyskinetic cerebral palsy (DCP, 30 subjects) and `0` denotes typically developing (TD, 21 subjects).  Each of the 51 rows represents one subject, aggregated from that subject's eight reaching-sideways repetitions.

## Course requirement audit

| Requirement | Verdict | Implementation evidence | Submission evidence |
|---|---|---|---|
| Train and compare 15 to 16 models with varied hyperparameters | **Met** | `src/model_catalog.py:48-105` defines exactly 16 classifier families and their parameter grids. `src/23_model_benchmark.py:54-57` asserts the catalog size, and `:86-116` evaluates every grid in each outer fold. | `results/model_benchmark/model_benchmark_summary.csv` and `results/model_benchmark/model_benchmark_fold_results.csv` |
| Classification task | **Met** | `src/23_model_benchmark.py:45-52` loads `target` and reports TD and DCP class counts before classification. | `results/final_comparison/final_results_summary.txt`, section 2 |
| Clearly state target variable and its classes | **Met** | `src/09_build_experiment_dataset.py:162-192` retains `target` with subject metadata while creating the experiment datasets. `src/25_leakage_audit.py:231-259` verifies TD=21 and DCP=30. | This document, `README.md` under “What is predicted,” and `AUDIT_REPORT.md` section 1 |
| Prevent data leakage | **Met, with an external-validity limitation** | `src/model_catalog.py:114-141` places imputation, scaling, feature selection, and model fitting inside a `Pipeline` and nested grouped CV. `src/23_model_benchmark.py:60-79` fits transformations only on each training partition and asserts no outer or inner subject overlap. `src/23_model_benchmark.py:45-50` excludes `target`, `subject`, `side`, and `group` from predictors. | `results/leakage_audit/leakage_audit_report.txt`; `results/leakage_audit/label_permutation_test.csv` |
| Split only by subject | **Met** | `src/model_catalog.py:131-134` uses `StratifiedGroupKFold` for both outer and inner CV. `src/28_verify_split_integrity.py:69-84` reconstructs the pinned splits and asserts zero overlap. | `results/model_benchmark/outer_fold_assignment.csv`; `results/leakage_audit/leakage_audit_report.txt` |
| Validate the highest-accuracy 2 to 3 models | **Met** | `src/36_nested_top3_per_fold.py:4-10,25-31` selects the top three families by inner-CV accuracy separately inside every outer training fold, then evaluates them on unseen subjects. `:39-104` produces per-rank and majority-vote metrics with subject-level bootstrap intervals. | `results/nested_top3_per_fold/per_fold_top3_selection.csv`; `per_fold_top3_pooled_metrics.csv`; `per_fold_top3_fold_summary.csv` |

The global top-three analysis in `src/16_top3_validation.py:1-14` is intentionally labelled diagnostic only because its ranking pools inner-CV information across outer folds. Do not use those results as the final unbiased estimate.

## Requirements from the supplied project brief

The broader brief is implemented as follows:

- Preprocessing covers missing-value interpolation, Savitzky-Golay smoothing, coordinate normalization, and per-repetition time normalization in `src/02_preprocess.py`. The time rescaling is not trajectory alignment, as correctly documented at `src/02_preprocess.py:163-166`.
- Spatial, temporal, angular, and movement-quality features feed the experiment datasets; `src/09_build_experiment_dataset.py:157-192` creates the coordinate-only primary set and labelled angle-enhanced/ablation sets.
- The Functional Resilience Index uses each subject's eight repetitions without group-derived scaling. Its proposed formula, sensitivity analyses, and limits are documented in `src/30_functional_resilience.py:4-56`; recording duration is deliberately excluded at `:12-13,81-84`.
- SHAP uses out-of-fold explanations (`src/20_shap_oof.py:1-25`), and LIME uses only the outer-training background distribution before explaining held-out subjects (`src/31_lime_oof.py:2-9,42-67`).

## Results to present to the mentor

Use the per-outer-fold top-three procedure as the required selected-model validation: its majority vote attained accuracy 0.863 (95% subject-bootstrap CI 0.765 to 0.941) and balanced accuracy 0.862 (0.758 to 0.956). The fully nested estimate for the entire model-family and hyperparameter selection procedure was accuracy 0.824 (0.706 to 0.922). These answer different questions and must not be substituted for one another.

## Publication boundary

The codebase is reproducible and suitable for a course-project submission and a transparent exploratory/pilot manuscript. It is **not yet ready for a clinical-performance conference claim**. The supplied cohort has no independent external test set and no site/session/camera metadata; group and recording/acquisition effects cannot be separated. Clip duration exactly equals recording length and differs by group. See `results/confound_analysis/confound_summary.txt` and `results/duration_audit/duration_audit_summary.txt`.

Any manuscript or presentation must therefore describe this as internal, subject-wise cross-validation on 51 participants; call the Functional Resilience Index proposed/exploratory; and avoid claims of clinical deployment, external generalisation, or a validated digital biomarker. A separate site-balanced external cohort, protocol metadata, and clinical-scale validation are required before such claims.

## Demonstration and slides

The Reach Lab dashboard/device receiver (`present.py`, `demo/`) and the 5-slide deck (`presentation/`) are included and generated from the final results by step 38 (`docs/DEMO_GUIDE.md`). The deck and dashboard headline the fully nested 42/51 result, not the superseded 48/51. Camera/live scores are exploratory and unvalidated on any new camera.

## Final submission checklist

1. Submit the source code, `requirements.txt`, `README.md`, `AUDIT_REPORT.md`, this audit, and the `results/` folder together.
2. In the report and presentation, state the target/classes and the subject-wise grouped split before showing accuracy.
3. Show the 16-model benchmark table, the per-fold top-three selection table, and the leakage audit.
4. Include the external-validity limitation verbatim or in equivalent plain language.
5. `logs/clean_run_full.log` is the log of one uninterrupted `python run_pipeline.py --clean` made after the demo integration; re-run it in the pinned Python 3.13 environment before any final publication submission.

---
## Addendum: status after the checklist round
Done: FRI component analysis, importance across feature sets, explanation of the nested winners, curated figures, dashboard labelling/summary/test, README architecture and script table, number-verification script, checklist mapping (`docs/FEEDBACK_CHECKLIST.md`), report and deck.
Still open (cannot be closed with the supplied data): acquisition/site confound, external validation, clinical validity of FRI; LightGBM/CatBoost remains the guide's decision; the full ML stage was not re-run from scratch in a fresh environment for this release.
