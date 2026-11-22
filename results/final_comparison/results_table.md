# Results summary

Task: binary classification, DCP (1, n=30) vs TD (0, n=21); unit = subject; subject-wise nested CV (outer 5-fold / inner 4-fold StratifiedGroupKFold, seed 42).

## Table 1 - all 16 models (outer-fold test metrics, mean +/- SD over 5 folds; ranked by inner-CV accuracy)

| Rank | Model | Inner-CV acc | Test acc | Test bal. acc | Test F1 | Test ROC-AUC | Selected |
|---|---|---|---|---|---|---|---|
| 1 | MLP | 0.927 +/- 0.018 | 0.884 +/- 0.037 | 0.899 +/- 0.018 | 0.888 +/- 0.022 | 0.904 +/- 0.065 | YES |
| 2 | Extra_Trees | 0.922 +/- 0.028 | 0.884 +/- 0.037 | 0.908 +/- 0.013 | 0.898 +/- 0.016 | 0.972 +/- 0.039 | YES |
| 2 | RBF_SVM | 0.922 +/- 0.022 | 0.904 +/- 0.064 | 0.919 +/- 0.049 | 0.910 +/- 0.055 | 0.964 +/- 0.036 | YES |
| 4 | Gaussian_NB | 0.922 +/- 0.034 | 0.922 +/- 0.083 | 0.931 +/- 0.086 | 0.937 +/- 0.070 | 0.983 +/- 0.037 |  |
| 5 | Linear_SVM | 0.918 +/- 0.029 | 0.904 +/- 0.064 | 0.919 +/- 0.049 | 0.910 +/- 0.055 | 0.972 +/- 0.026 |  |
| 6 | Random_Forest | 0.917 +/- 0.029 | 0.924 +/- 0.077 | 0.941 +/- 0.055 | 0.939 +/- 0.058 | 0.992 +/- 0.019 |  |
| 7 | Logistic_L2 | 0.899 +/- 0.026 | 0.904 +/- 0.064 | 0.924 +/- 0.044 | 0.916 +/- 0.049 | 0.932 +/- 0.077 |  |
| 8 | KNN | 0.894 +/- 0.026 | 0.924 +/- 0.077 | 0.944 +/- 0.052 | 0.939 +/- 0.058 | 0.953 +/- 0.056 |  |
| 9 | AdaBoost | 0.892 +/- 0.052 | 0.884 +/- 0.037 | 0.908 +/- 0.013 | 0.898 +/- 0.016 | 0.978 +/- 0.050 |  |
| 10 | LDA | 0.888 +/- 0.029 | 0.805 +/- 0.094 | 0.827 +/- 0.079 | 0.809 +/- 0.075 | 0.921 +/- 0.084 |  |
| 11 | Logistic_L1 | 0.888 +/- 0.025 | 0.884 +/- 0.080 | 0.903 +/- 0.075 | 0.889 +/- 0.090 | 0.944 +/- 0.043 |  |
| 12 | HistGradientBoosting | 0.882 +/- 0.092 | 0.824 +/- 0.083 | 0.858 +/- 0.068 | 0.844 +/- 0.079 | 0.979 +/- 0.036 |  |
| 13 | Gradient_Boosting | 0.868 +/- 0.084 | 0.862 +/- 0.057 | 0.886 +/- 0.050 | 0.872 +/- 0.067 | 0.909 +/- 0.085 |  |
| 14 | XGBoost | 0.854 +/- 0.079 | 0.844 +/- 0.088 | 0.874 +/- 0.071 | 0.866 +/- 0.079 | 0.981 +/- 0.027 |  |
| 15 | Decision_Tree | 0.842 +/- 0.094 | 0.862 +/- 0.115 | 0.886 +/- 0.097 | 0.876 +/- 0.110 | 0.886 +/- 0.097 |  |
| 16 | Polynomial_SVM | 0.644 +/- 0.075 | 0.635 +/- 0.288 | 0.716 +/- 0.182 | 0.523 +/- 0.401 | 0.964 +/- 0.036 |  |

## Table 1b - PRIMARY unbiased estimate: whole procedure (16 models tuned by inner CV, best family by inner-CV accuracy) selected INSIDE each outer training fold

| Fold | Winner (inner-CV) | Inner acc | Test acc | Test ROC-AUC |
|---|---|---|---|---|
| 1 | HistGradientBoosting | 0.975 | 0.700 | 0.979 |
| 2 | Extra_Trees | 0.952 | 0.900 | 1.000 |
| 3 | Extra_Trees | 0.925 | 0.900 | 1.000 |
| 4 | Gaussian_NB | 0.955 | 0.800 | 0.917 |
| 5 | Linear_SVM | 0.950 | 0.818 | 0.944 |
| **Mean +/- SD** | | | **0.824 +/- 0.083** | **0.968** |

Pooled over the 51 held-out subjects, 95% subject-level bootstrap CI (10,000 resamples):

| Metric | Estimate [95% CI] |
|---|---|
| Accuracy | 0.824 [0.706, 0.922] |
| Balanced_Accuracy | 0.814 [0.699, 0.918] |
| F1 | 0.852 [0.745, 0.935] |
| Sensitivity | 0.867 [0.733, 0.970] |
| Specificity | 0.762 [0.565, 0.941] |
| MCC | 0.634 [0.403, 0.840] |

## Table 2 - the 3 selected model families, validation stage (pooled out-of-fold predictions, 10,000 subject-level bootstrap resamples, 95% CI). OPTIMISTIC as an estimate of the selection procedure (families ranked with all folds); see Table 1b.

| Model | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---|---|---|---|
| MLP | 0.882 [0.784, 0.961] | 0.900 [0.817, 0.967] | 0.889 [0.776, 0.966] | 0.902 [0.798, 0.981] |
| Extra_Trees | 0.882 [0.784, 0.961] | 0.886 [0.788, 0.967] | 0.897 [0.807, 0.967] | 0.975 [0.933, 0.998] |
| RBF_SVM | 0.902 [0.824, 0.980] | 0.917 [0.850, 0.983] | 0.909 [0.824, 0.983] | 0.959 [0.900, 0.998] |

## Table 3 - selected models, repeated subject-wise nested CV (10 partitions; mean [95% CI of repeat means])

_Caveat: the 10 repeats re-partition the same 51 subjects, so these intervals capture split randomness only and are much narrower than the true generalisation uncertainty; they are not confidence intervals for new participants._

| Model | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---|---|---|---|
| Extra_Trees | 0.906 [0.891, 0.920] | 0.909 [0.893, 0.926] | 0.918 [0.905, 0.930] | 0.968 [0.958, 0.977] |
| MLP | 0.904 [0.890, 0.918] | 0.913 [0.900, 0.925] | 0.913 [0.899, 0.927] | 0.952 [0.929, 0.975] |
| RBF_SVM | 0.900 [0.881, 0.919] | 0.909 [0.890, 0.929] | 0.910 [0.892, 0.927] | 0.969 [0.959, 0.979] |

## Table 4 - representation ablation (single Random Forest per representation, nested CV)

| Representation | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Repetition_Aware | 0.924 +/- 0.077 | 0.939 +/- 0.058 | 1.000 +/- 0.000 |
| Stable_Features | 0.904 +/- 0.064 | 0.916 +/- 0.049 | 0.970 +/- 0.048 |
| Static | 0.942 +/- 0.053 | 0.948 +/- 0.051 | 0.989 +/- 0.025 |

Ablation conclusion: Repetition-aware descriptors did NOT show a statistically supported advantage over static per-subject means (accuracy 0.924 vs 0.942; ROC-AUC 1.000 vs 0.989; smallest FDR-adjusted Wilcoxon p = 1.00, n = 5 paired folds, i.e. under-powered). The claim that repetition-derived features add discriminative value is therefore not demonstrated for DCP-vs-TD classification on this dataset.

## Table 5 - Functional Resilience Index and related repetition indices (TD vs DCP)

| Index | TD median | DCP median | Cohen d (DCP-TD) [95% CI] | AUC, higher=TD [95% CI] | BH-FDR p |
|---|---|---|---|---|---|
| Functional_Resilience_Index | 0.943 | 0.916 | -0.85 [-1.44, -0.35] | 0.75 [0.60, 0.88] | 0.0068 |
| Functional_Resilience_Index_no_position_error | 0.938 | 0.907 | -0.68 [-1.22, -0.19] | 0.69 [0.54, 0.83] | 0.0369 |
| Consistency_Score | 0.900 | 0.875 | -0.90 [-1.38, -0.52] | 0.75 [0.61, 0.87] | 0.0068 |
| Movement_Stability_Index | 0.860 | 0.840 | -0.60 [-1.08, -0.15] | 0.66 [0.50, 0.80] | 0.0762 |
| Adaptive_Motor_Control_Index | 0.023 | 0.048 | 0.32 [-0.17, 0.79] | 0.42 [0.27, 0.59] | 0.3633 |
| FRI_no_timing_coupled_sensitivity | 0.940 | 0.921 | -0.34 [-1.07, 0.24] | 0.64 [0.48, 0.80] | 0.1012 |
| FRI_duration_included_legacy_sensitivity_only | 0.945 | 0.918 | -0.97 [-1.47, -0.53] | 0.78 [0.65, 0.90] | 0.0044 |

Incremental classification value (10 repeated subject-wise 5-fold partitions, RF, same pipeline):

| Feature set | Accuracy | ROC-AUC | Note |
|---|---|---|---|
| Resilience_indices_only | 0.616 +/- 0.062 | 0.646 |  |
| Static_means | 0.937 +/- 0.015 | 0.954 |  |
| Static_means+Resilience_indices | 0.953 +/- 0.014 | 0.959 | paired diff in balanced acc vs static: +0.021 (SD over repeats 0.014; repeats share the same 51 subjects) |

FRI is a proposed index (equal weights for consistency and maintenance; not clinically validated). Weight sensitivity (consistency weight 0.25 / 0.50 / 0.75): Cohen d = -0.63 / -0.85 / -0.95; AUC = 0.72 / 0.75 / 0.77.

## Table 6 - Out-of-fold LIME vs SHAP agreement

Spearman rho (all features) = 0.84; top-15 overlap = 13/15 (Jaccard 0.76). Top LIME features: path_efficiency_mean, max_lateral_deviation_mm_mean, path_efficiency_cv, mean_lateral_deviation_mm_mean, trajectory_length_mm_cv.

## How to read the uncertainty numbers

- Fold SD (+/-): spread of 5 outer folds of ~10 subjects each; coarse (10-point steps), not a CI.
- Repeated-split variability (Table 3): re-partitions the SAME 51 subjects; shows sensitivity to the split only. Intervals are too narrow to be read as generalisation uncertainty.
- Bootstrap CIs on pooled out-of-fold predictions (Tables 1b, 2): resample the 51 subjects, so they reflect subject-sampling noise in the test predictions, but they hold the trained models fixed and (for Table 2) the model-family choice fixed; they do not include training-set or model-selection variability. Table 1b includes selection inside every fold but still not the variability of re-drawing a different training cohort.
- None of them measure generalisation to new sites, scanners, clinicians or populations: all data come from one dataset (site/acquisition confound unresolved).
- 'Final test' = outer cross-validation folds. There is NO independent hold-out or external test set (impossible with n = 51 without discarding ~20% of subjects).
