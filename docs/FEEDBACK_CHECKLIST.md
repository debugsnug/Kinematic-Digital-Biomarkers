# Teammate checklist -> evidence

Legend: **existing** = already produced by the validated pipeline (unchanged); **new** = added in this release (additive).

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Acquisition confound | existing + extended | `results/confound_analysis/confound_summary.txt`: clip length alone bal. acc. 0.76, subject number 0.85; unresolved, stated as limitation |
| 1 | Duration-independent evaluation | existing | `results/feature_sets/` coordinate_no_duration 0.843, strictest (no duration/jerk/SPARC) 0.843 vs 0.824 primary |
| 1 | 5-way ablation | existing + new figure | `feature_set_headline_comparison.csv`, `paired_subject_level_comparison.csv` (all McNemar p >= 0.25), figure F3 |
| 1 | FRI verification / components | **new** | step 41 -> `functional_resilience/fri_component_analysis.csv`; recomputation equals step 30; timing-removed AUC 0.643 |
| 1 | Sensitivity (features x models) | existing | `feature_sets/benchmark_16_models_by_feature_set.csv`, `repeated_selection/`, FRI weight sensitivity |
| 2 | Leakage-free preprocessing/tuning | existing | pipeline fitted per fold; 80/80 model-folds identical to sklearn Pipeline+GridSearchCV; `leakage_audit/` (0 FAIL); `28_verify_split_integrity` |
| 2 | Nested = primary; global ranking separated | existing | README headline table; step 16 marked optimistic |
| 2 | 82.4 / 81.4 / 86.7 / 76.2 / 85.2 | existing | `nested_selection_ci.csv`; asserted by **new** step 43 |
| 2 | Per-fold top-3 vote 86.3% | existing | `nested_top3_per_fold/`; method in step 36 header; CI 0.765-0.941 |
| 2 | CIs, fold-wise | existing + new figure | CI files; F2 fold-wise |
| 3 | SHAP/LIME outputs inspectable | existing | `results/shap_oof/`, `results/lime_oof/` |
| 3 | Importance across feature sets | **new** | step 40, figure F6 |
| 3 | Explanations vs final selected models | **new** | step 39 (nested winners refit and permuted), figure F7; disagreement with SHAP documented |
| 4 | Dashboard labelling, 8 reps, summaries, errors | **new** (edits) | `demo/`; badges, participant summary, clearer messages |
| 4 | Test vs dataset | **new** | `tests/dashboard_test.js`: 51 x 8 reps, 1,275 values, 0 failures |
| 5 | Figures, captions, stats | **new** | `results/figures_final/` F1-F8 + `captions.md`; `statistical_analysis/` (p, FDR, d) |
| 6 | README, scripts table, requirements | **new** README sections; requirements existing | README |
| 6 | Fresh-environment check | partial | features stage verified byte-identical; full ML stage not re-run (3.5 h) |
| 6 | Backup before any clean run | done | clean pipeline not re-run; baseline copy kept outside the project |
| 7 | Report and final deck | **new** | `report/`, `presentation/` |

Not done / open: external validation; site-resolved confound; LightGBM/CatBoost (16 models kept; guide's decision); clinical validation of FRI.
