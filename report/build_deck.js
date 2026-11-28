const pptxgen = require("pptxgenjs"); const sizeOf = null;
const P = "/home/claude/work/Integrated/Project/"; const F = P + "results/figures_final/";
const dims = JSON.parse(process.env.DIMS);
const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.title = "Explainable kinematic biomarkers for upper-limb assessment in DCP"; pres.author = "B. Alekya";
const NAVY = "132E43", TEAL = "2A7FA8", AMB = "D08A2B", BG = "F3F6F7", INK = "192F40", MUT = "5B7080", RED = "C4572B", GRN = "2A8C6A";
let n = 0;
function base(title, sub) {
  const s = pres.addSlide(); n++; s.background = { color: "FFFFFF" };
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 13.33, h: 0.12, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText(title, { x: 0.6, y: 0.3, w: 12.1, h: 0.7, fontFace: "Calibri", fontSize: 28, bold: true, color: NAVY, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.6, y: 0.98, w: 12.1, h: 0.4, fontFace: "Calibri", fontSize: 14, color: MUT, margin: 0, isTextBox: true });
  s.addText("IoT 23CSE365 · DCP vs TD kinematic biomarkers", { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 9, color: MUT, margin: 0, isTextBox: true });
  s.addText(String(n), { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 9, color: MUT, align: "right", margin: 0, isTextBox: true });
  return s;
}
function bullets(s, items, o = {}) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
    Object.assign({ x: 0.6, y: 1.6, w: 12, h: 5, fontFace: "Calibri", fontSize: 21, color: INK, valign: "top", margin: 0, isTextBox: true }, o));
}
function fig(s, file, x, y, maxw, maxh) {
  const [w, h] = dims[file]; const r = Math.min(maxw / w, maxh / h); s.addImage({ path: file.startsWith("/") ? file : P + file, x, y, w: w * r, h: h * r });
}
function card(s, x, y, w, h, big, small, col) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: BG }, line: { color: "D3DDE1", width: 0.75 } });
  s.addText(big, { x, y: y + 0.12, w, h: h * 0.55, align: "center", fontSize: 30, bold: true, color: col || NAVY, margin: 0, isTextBox: true });
  s.addText(small, { x: x + 0.1, y: y + h * 0.6, w: w - 0.2, h: h * 0.38, align: "center", fontSize: 12, color: MUT, margin: 0, valign: "top", isTextBox: true });
}
// 1 title
{ const s = pres.addSlide(); n++; s.background = { color: NAVY };
  s.addText("Explainable Kinematic Digital Biomarkers for Upper-Limb Assessment in Cerebral Palsy", { x: 0.8, y: 1.5, w: 11.7, h: 1.8, fontSize: 38, bold: true, color: "FFFFFF", fontFace: "Calibri", margin: 0, isTextBox: true });
  s.addText("Subject-wise classification of dyskinetic CP vs typically developing children from markerless reaching kinematics, with explainability and an exploratory Functional Resilience Index", { x: 0.8, y: 3.4, w: 11, h: 1, fontSize: 18, color: "C7D9E0", margin: 0, isTextBox: true });
  s.addText("B. Alekya  ·  Guide: Dr. Thangam S., Amrita School of Computing, Bangalore\nCourse: 23CSE365 Internet of Things", { x: 0.8, y: 5.3, w: 11, h: 0.9, fontSize: 16, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText("Data: Haberfehlner et al., KU Leuven RDR, doi:10.48804/BZM9SJ (CC BY 4.0)", { x: 0.8, y: 6.5, w: 11, h: 0.4, fontSize: 12, color: "9FB7C3", margin: 0, isTextBox: true }); }
// 2 problem
{ const s = base("Problem statement and motivation");
  bullets(s, ["Upper-limb function in children with cerebral palsy is mostly judged by clinician observation and ordinal scales: subjective and hard to repeat.",
    "Markerless pose estimation (DeepLabCut) can measure reaching from ordinary video, but published work focuses on measurement accuracy, not on turning movement into interpretable biomarkers.",
    "Problem: can interpretable features of eight repeated sideways reaches separate dyskinetic CP (DCP) from typically developing (TD) children, with leakage-free subject-wise evaluation and honest uncertainty?",
    "IoT angle: wrist coordinates from a camera or another device can be streamed to a dashboard (Reach Lab) for live and replayed analysis."]); }
// 3 objectives
{ const s = base("Objectives and course requirements");
  bullets(s, ["Target and classes stated explicitly: DCP = 1 (30 subjects), TD = 0 (21 subjects); one prediction per subject.",
    "Binary classification with 16 models and varied hyper-parameter grids (single shared catalog).",
    "No leakage: every transform, feature selection and tuning step is fitted on training subjects only.",
    "Subject-only splits: StratifiedGroupKFold, no subject in both train and test.",
    "Top-model validation chosen without test data (inner CV only), reported separately from the global ranking.",
    "Explainability (SHAP, LIME, permutation) and an exploratory Functional Resilience Index (FRI)."]); }
// 4 literature
{ const s = base("Literature review: what is known and the gap");
  bullets(s, ["Vanmechelen et al. 2024 (J Biomech 173:112233): DeepLabCut-based markerless analysis of reaching sideways is valid against 3D motion analysis in DCP. It validates measurement; it does not build a classifier or biomarkers.",
    "Mathis et al. 2018: DeepLabCut markerless pose estimation from video.",
    "Smoothness metrics: Balasubramanian et al. 2015 (SPARC, normalised jerk) are standard movement-quality measures.",
    "Evaluation: selecting a model on the same folds that estimate its error is optimistic (Varma & Simon 2006; Cawley & Talbot 2010), so nested cross-validation is the primary estimate here.",
    "Explainability: SHAP (Lundberg & Lee 2017) and LIME (Ribeiro et al. 2016).",
    "Gap addressed: leakage-controlled, subject-wise classification with explanations, ablations for acquisition artefacts, and a resilience index across repetitions."], { fontSize: 19 }); }
// 5 dataset
{ const s = base("Dataset", "KU Leuven markerless reaching-sideways dataset (Haberfehlner et al.), CC BY 4.0");
  card(s, 0.6, 1.7, 2.9, 1.5, "51", "children (30 DCP, 21 TD)"); card(s, 3.7, 1.7, 2.9, 1.5, "408", "files = 51 × 8 repetitions"); card(s, 6.8, 1.7, 2.9, 1.5, "84", "primary predictors per subject"); card(s, 9.9, 1.7, 2.9, 1.5, "25 fps", "2D key points (DeepLabCut)");
  bullets(s, ["Each repetition: x/y trajectories of shoulder, elbow, wrist, target object; joint angles are 2D video-derived.",
    "Loader fails closed unless exactly 408 files / 51 subjects / 8 repetitions are found.",
    "Group imbalance to keep in mind: arm side (TD 18 left / 3 right) and sex differ between groups; left-arm videos were mirrored.",
    "No site, camera or session variable is supplied (see acquisition confound)."], { y: 3.6, h: 3.3, fontSize: 20 }); }
// 6 architecture
{ const s = base("System architecture", "Research pipeline (left) and Reach Lab IoT dashboard (right)");
  const steps = ["Raw Excel (408)", "Load + clean + per-rep kinematics", "6 repetition descriptors → 1 row per subject", "16 models · nested subject-wise CV", "Nested estimate · per-fold top-3 · audits", "SHAP · LIME · permutation · FRI"];
  steps.forEach((t, i) => { const y = 1.55 + i * 0.88; s.addShape(pres.ShapeType.roundRect, { x: 0.6, y, w: 6.6, h: 0.65, fill: { color: i === 3 ? TEAL : BG }, line: { color: "B8C9D1" }, rectRadius: 0.08 });
    s.addText(t, { x: 0.75, y, w: 6.3, h: 0.65, fontSize: 15, color: i === 3 ? "FFFFFF" : INK, bold: i === 3, margin: 0, valign: "middle", isTextBox: true });
    if (i < 5) s.addText("↓", { x: 3.7, y: y + 0.62, w: 0.4, h: 0.25, fontSize: 12, color: MUT, margin: 0, align: "center", isTextBox: true }); });
  const d = ["Camera / device wrist coordinates", "present.py local receiver (/stream, /sessions)", "Dashboard: 51-child replay, 8 reps, speed, summaries", "Research view = validated nested result", "Live view = exploratory 3-feature wrist model"];
  d.forEach((t, i) => { const y = 1.55 + i * 1.0; s.addShape(pres.ShapeType.roundRect, { x: 7.7, y, w: 5.0, h: 0.8, fill: { color: i === 4 ? "FDE7C8" : i === 3 ? "D9EEE4" : BG }, line: { color: "B8C9D1" }, rectRadius: 0.08 });
    s.addText(t, { x: 7.85, y, w: 4.7, h: 0.8, fontSize: 14, color: INK, margin: 0, valign: "middle", isTextBox: true }); }); }
// 7 features
{ const s = base("Feature engineering");
  bullets(s, ["Per repetition: path efficiency, trajectory length, reach distance, lateral deviation, velocity/acceleration, SPARC and normalised jerk, duration, position error to the target.",
    "Across the eight repetitions: mean, SD, CV, slope, early-late change, monotonicity → 14 base features × 6 descriptors = 84 coordinate-only predictors (primary).",
    "Six feature sets are compared: coordinate-only (84), angle-enhanced (168), without position error (78), without duration (78), without both (72), and the strictest set without duration, normalised jerk and SPARC (66).",
    "Joint-angle features are excluded from the primary model because they are derived from the same 2D key points and are missing for many shoulders.",
    "Duration equals clip length ((frames−1)/25), so it is treated as a possible acquisition proxy."], { fontSize: 20 }); }
// 8 evaluation
{ const s = base("Evaluation design and leakage control");
  bullets(s, ["Outer 5-fold × inner 4-fold StratifiedGroupKFold, seed 42; one row per subject, so subjects never cross folds.",
    "Pipeline: median imputer → StandardScaler → SelectKBest(mutual information, k ∈ {20, 40}) → model; all fitted on training subjects only.",
    "16 models: Logistic L1/L2, LDA, Gaussian NB, KNN, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, linear/RBF/polynomial SVM, MLP.",
    "Primary = fully nested: model family and hyper-parameters are chosen inside every outer training fold. Per-fold top-3 majority vote is selection-unbiased. Global top-3 is optimistic and diagnostic only.",
    "Checks: fast benchmark cross-checked against a real sklearn Pipeline + GridSearchCV (80/80 model-folds identical); label-permutation test; automated leakage audit."], { fontSize: 19 }); }
// 9 primary result
{ const s = base("Primary result: fully nested selection", "Subject-level, 51 children, seed-42 partition");
  card(s, 0.6, 1.6, 2.35, 1.4, "82.4%", "accuracy\n95% CI 70.6–92.2"); card(s, 3.1, 1.6, 2.35, 1.4, "81.4%", "balanced accuracy\n95% CI 69.9–91.8"); card(s, 5.6, 1.6, 2.35, 1.4, "86.7%", "sensitivity (DCP)\n26 / 30"); card(s, 8.1, 1.6, 2.35, 1.4, "76.2%", "specificity (TD)\n16 / 21"); card(s, 10.6, 1.6, 2.1, 1.4, "85.2%", "F1-score");
  fig(s, "results/figures_final/F1_evidence_hierarchy.png", 0.6, 3.15, 7.2, 3.8);
  bullets(s, ["Per-fold top-3 majority vote: 86.3% (CI 76.5–94.1), reported separately.", "Repeated nested runs on 10 re-partitions of the same 51 subjects: mean 85.5%, SD 3.0, range 82.4–90.2.", "Global top-3 (0.88–0.90) uses pooled inner scores and is not an unbiased estimate."], { x: 8.0, y: 3.3, w: 4.8, h: 3.5, fontSize: 14 }); }
// 10 foldwise
{ const s = base("Fold-wise performance", "Different outer folds select different winners; this instability is reported, not hidden");
  fig(s, "results/figures_final/F2_foldwise_performance.png", 0.6, 1.6, 7.2, 4.6);
  bullets(s, ["Winners: HistGradientBoosting (fold 1), Extra Trees (2, 3), Gaussian NB (4), Linear SVM (5).", "Fold accuracies 0.70–0.90 with about 10 test subjects each: one subject moves a fold by 10 points.", "Mean fold ROC-AUC 0.968 (per-family AUCs are not pooled)."], { x: 8.1, y: 1.7, w: 4.7, h: 4.5, fontSize: 15 }); }
// 11 ablation
{ const s = base("Feature ablation and duration-independent evaluation");
  fig(s, "results/figures_final/F3_feature_set_ablation.png", 0.6, 1.4, 7.6, 4.2);
  bullets(s, ["Removing position error, duration, or duration + jerk + SPARC does not lower nested accuracy (0.82–0.88 vs 0.82).", "Paired subject-level tests: no difference is significant (exact McNemar p ≥ 0.25).", "So accuracy is not carried by duration or the target-object feature alone.", "It does not remove the acquisition concern: 6 of 84 predictors still correlate |ρ| ≥ 0.7 with clip length, and other correlates remain."], { x: 8.4, y: 1.5, w: 4.5, h: 5, fontSize: 14 }); }
// 12 confound
{ const s = base("Acquisition confounding: unresolved");
  fig(s, "results/figures_final/F4_acquisition_proxies.png", 0.6, 1.4, 6.8, 3.4);
  bullets(s, ["Clip length alone gives balanced accuracy 0.76; subject number alone 0.85 (number blocks are group-structured).", "Mean clip length: DCP 65.2 frames (SD 81.4) vs TD 29.4 (SD 7.3), Mann-Whitney p = 5.8e-05.", "No site, camera or session variable is supplied, so site and group cannot be separated.", "Conclusion: report as a limitation; clinical claims need an independent, site-balanced cohort."], { x: 7.7, y: 1.5, w: 5.2, h: 5, fontSize: 14 }); }
// 13 SHAP
{ const s = base("Explainability 1: SHAP and LIME of an exploratory Random Forest");
  fig(s, "results/figures_final/F6_importance_across_feature_sets.png", 0.6, 1.4, 7.5, 4.4);
  bullets(s, ["Top features: path efficiency, max and mean lateral deviation, trajectory length, acceleration.", "Upper-limb reading: DCP reaches are less direct (lower efficiency), deviate more sideways and vary more between repetitions.", "SHAP vs LIME agreement: Spearman 0.84, 13/15 top features shared.", "Ranking is stable across the six feature sets (Spearman 0.96–1.00).", "This explains the exploratory forest, not the nested pipeline."], { x: 8.3, y: 1.5, w: 4.6, h: 5.3, fontSize: 14 }); }
// 14 nested explanation
{ const s = base("Explainability 2: the nested winners themselves");
  fig(s, "results/figures_final/F7_nested_pipeline_importance.png", 0.6, 1.4, 7.2, 4.4);
  bullets(s, ["Each fold's winner was refitted (accuracy reproduced exactly) and features permuted on held-out subjects.", "Effects are small and unstable and disagree with SHAP: no single feature is established as driving the validated result.", "Consistent with a signal spread over many correlated movement-quality features."], { x: 8.1, y: 1.5, w: 4.8, h: 5, fontSize: 15 }); }
// 15 stats
{ const s = base("Statistical analysis of single features");
  fig(s, "results/figures_final/F8_univariate_effects.png", 0.6, 1.4, 7.6, 4.8);
  bullets(s, ["Mann-Whitney tests with Benjamini-Hochberg FDR and Cohen's d.", "Path efficiency mean: d = −1.45 (DCP lower), FDR p = 9.8e-07.", "Differences are descriptive: duration-related features are confounded with acquisition."], { x: 8.4, y: 1.5, w: 4.5, h: 5, fontSize: 15 }); }
// 16 FRI
{ const s = base("Functional Resilience Index (exploratory)");
  fig(s, "results/figures_final/F5_FRI_components.png", 0.6, 1.4, 7.4, 5.2);
  bullets(s, ["FRI = 0.5·consistency + 0.5·maintenance over eight repetitions; label-free, duration excluded.", "Independently recomputed: equals the pipeline for all 51 children.", "Group AUC 0.754, d = −0.85, BH-FDR p = 0.0068.", "Mostly driven by path efficiency and position error; without SPARC and normalised jerk AUC falls to 0.643 (not significant after FDR).", "Not clinically validated; do not read as a biomarker."], { x: 8.2, y: 1.5, w: 4.7, h: 5.3, fontSize: 14 }); }
// 17 dashboard
{ const s = base("Reach Lab dashboard", "Research view is labelled separately from the exploratory live camera");
  fig(s, "tests/dashboard_study.png", 0.6, 1.5, 7.4, 5.4);
  bullets(s, ["Replay of all 51 children × 8 reaches with trajectory, speed plot and per-reach measurements.", "Held-out result = nested call and fold winner for that child; RF explanation labelled as explaining only that forest.", "Live session: wrist-marker tracking, 3-feature model, banner: not the validated 84-feature pipeline.", "Automated test: 1,275 displayed values matched the dataset, 0 failures."], { x: 8.2, y: 1.6, w: 4.7, h: 5.2, fontSize: 14 }); }
// 18 demo sequence
{ const s = base("Demonstration sequence (about 4 minutes)");
  bullets(s, ["Start: python present.py (or open Offline replay.html). Study replay, pick a TD child and a DCP child.", "Dataset replay: play one reach; show trajectory and wrist-speed plot.", "Eight repetitions: toggle 'All eight trails'; use buttons 1–8 and the participant summary table.", "Participant-level analysis: reveal the held-out call, fold winner and RF explanation.", "Model results: open the validation panel (42 / 51 nested) and F1 / F2 figures.", "Live session: enable camera, point out the exploratory banner; optional device stream."], { fontSize: 21 }); }
// 19 limitations
{ const s = base("Limitations");
  bullets(s, ["Single cohort of 51 children; no external validation. CIs reflect resampling of these subjects only.", "Acquisition/site confound cannot be separated from group (clip length 0.76, subject number 0.85).", "Small outer folds (~10 subjects) make single-fold values coarse.", "Explanations disagree between the exploratory forest (SHAP) and the nested winners (permutation).", "FRI is a proposed index, not clinically validated; signal weakens without timing-coupled metrics.", "Live camera model is exploratory (3 wrist features) and unvalidated on new devices.", "Not run: LightGBM / CatBoost (16 models used); full ML stage not re-run from scratch in a fresh environment."], { fontSize: 20 }); }
// 20 future
{ const s = base("Future scope");
  bullets(s, ["Collect or obtain site/camera metadata and test leave-one-site-out generalisation.", "External validation on an independent cohort; calibrate and report decision thresholds.", "Link kinematic indices and FRI to clinical scales (e.g. MACS, upper-limb tests) before any clinical use.", "Framewise trajectory alignment (e.g. DTW) for within-subject variability.", "Edge deployment of a validated model on the IoT device with on-device feature extraction.", "Guide's decision: add LightGBM / CatBoost to the catalog."], { fontSize: 21 }); }
// 21 conclusion
{ const s = base("Conclusion");
  bullets(s, ["Leakage-controlled, subject-wise nested evaluation gives 82.4% accuracy (balanced 81.4%, sensitivity 86.7%, specificity 76.2%) for DCP vs TD on 51 children.", "The result survives removal of duration, position error and timing-coupled smoothness features, but the acquisition confound is unresolved.", "Path efficiency and lateral deviation are the most consistent explanatory features of the exploratory forest; the nested winners show no single dominant feature.", "FRI is a transparent exploratory index; the Reach Lab dashboard demonstrates the IoT workflow with clear research vs exploratory labelling.", "Claims are limited to this dataset until externally validated."], { fontSize: 21 }); }
// 22 refs
{ const s = base("References and dataset attribution");
  s.addText([
    "Haberfehlner H. et al. KU Leuven Markerless Motion Capture Dataset (reaching sideways). KU Leuven RDR. doi:10.48804/BZM9SJ. CC BY 4.0.",
    "Vanmechelen I. et al. (2024). Markerless motion analysis to assess reaching-sideways in individuals with dyskinetic cerebral palsy: a validity study. J Biomech 173:112233.",
    "Mathis A. et al. (2018). DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci 21:1281–1289.",
    "Balasubramanian S. et al. (2015). On the analysis of movement smoothness. J NeuroEng Rehabil 12:112.",
    "Varma S., Simon R. (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7:91.",
    "Cawley G.C., Talbot N.L.C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. JMLR 11:2079–2107.",
    "Lundberg S.M., Lee S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.",
    "Ribeiro M.T., Singh S., Guestrin C. (2016). “Why should I trust you?” Explaining the predictions of any classifier. KDD.",
    "Benjamini Y., Hochberg Y. (1995). Controlling the false discovery rate. J R Stat Soc B 57:289–300.",
    "Pedregosa F. et al. (2011). Scikit-learn: machine learning in Python. JMLR 12:2825–2830."
  ].map((t, i, a) => ({ text: t, options: { breakLine: i < a.length - 1, paraSpaceAfter: 6 } })), { x: 0.6, y: 1.4, w: 12.1, h: 5.5, fontSize: 15, color: INK, margin: 0, valign: "top", isTextBox: true }); }
pres.writeFile({ fileName: P + "presentation/Final_Project_Presentation.pptx" }).then(() => console.log("ok", n));
