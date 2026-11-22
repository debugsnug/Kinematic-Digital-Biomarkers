# Final figure captions

**F1.** Accuracy under four evaluation procedures. Blue = primary (fold-wise nested selection; 95% subject bootstrap CI). Red = global top-3 ranking, which uses pooled inner scores and is optimistic. Dashed = majority-class rate. Single cohort, 51 subjects: no external validation.

**F2.** Each outer fold chooses its own model family and hyper-parameters from training subjects only; ~10 test subjects per fold, so single-fold values move in steps of ~0.1.

**F3.** Same nested procedure on six feature sets (predictor counts in brackets). Removing duration, position error or both timing-coupled smoothness measures does not lower accuracy beyond the CI width; the differences are not statistically distinguishable at n=51. Accuracy is not carried by duration alone, but this does not remove the acquisition confound.

**F4.** Clip length (equal to movement_duration_s) and subject number alone separate DCP from TD far above chance (dashed = 0.5). The supplied data has no site/camera variable, so the confound cannot be resolved here.

**F5.** AUC of the exploratory FRI, of FRI built from one metric, and with one metric left out. Removing SPARC and normalised jerk (red) reduces AUC to 0.64 (BH-FDR p=0.13); max lateral deviation works against the index (leaving it out raises AUC). The FRI is not clinically validated.

**F6.** SHAP of a fixed Random Forest (k=40) per feature set, held-out subjects only. Rank agreement between sets is high (Spearman 0.96-1.00; rank_agreement_between_sets.csv). Explains this exploratory forest, not the nested winners.

**F7.** Each fold's winning pipeline permuted on ~10 held-out subjects. Effects are small relative to their spread and disagree with the SHAP ranking; no single feature is established as driving the nested model. Treat as descriptive.

**F8.** Twelve largest |d| among the FDR-significant features. Group differences are descriptive; duration-related features (clip length) are confounded with acquisition.
