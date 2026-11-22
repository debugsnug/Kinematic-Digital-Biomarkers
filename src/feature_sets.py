# ============================================================
# FEATURE SETS / MODALITY REGISTRY
# ============================================================
# Every predictor is "<base feature>_<descriptor>" where descriptor is one of the six repetition
# descriptors (mean, sd, cv, slope, early_late_change, monotonicity). The base feature decides the
# source modality. The registry is VERIFIED empirically by 32_feature_audit.py (it perturbs the angle
# columns / the target-object columns of the cleaned frames and checks which base features change).
#
#   coordinates                : computed from the 2D key-point (x, y) pixel trajectories only
#   angle_derived              : depends on the per-frame joint-angle columns (R/L-Shoulder, -Elbow, -Wrist).
#                                Per the dataset paper (Vanmechelen et al. 2024, sect. 2.4) these are 2D frontal-plane
#                                angles computed from the DeepLabCut key points; 32_feature_audit.py re-derives them
#                                from the key points to confirm. They are still excluded from the PRIMARY model.
#   coordinates_plus_target    : depends on the tracked target-object key point (position_error)

DESCRIPTORS = ["mean", "sd", "cv", "slope", "early_late_change", "monotonicity"]

ANGLE_BASE = [
    "shoulder_rom_deg", "elbow_rom_deg", "wrist_rom_deg",
    "shoulder_angvel_mean", "elbow_angvel_mean", "wrist_angvel_mean",
    "shoulder_angvel_peak", "elbow_angvel_peak", "wrist_angvel_peak",
    "shoulder_angacc_mean", "elbow_angacc_mean", "wrist_angacc_mean",
    "shoulder_angacc_peak", "elbow_angacc_peak", "wrist_angacc_peak",
    "coordination_elbow_wrist", "coordination_shoulder_elbow",
    "shoulder_angle_sd", "elbow_angle_sd", "wrist_angle_sd",
]
TARGET_BASE = ["position_error_mm"]
# movement_duration_s = (n_frames - 1) / 25 fps, i.e. it IS the clip length (corr = 1.0). Clip length alone separates the groups
# (balanced accuracy ~0.76, results/confound_analysis), so duration cannot be told apart from recording/trimming differences.
DURATION_BASE = ["movement_duration_s"]
# Duration-coupled smoothness measures (step 37: |Spearman| with clip length 0.82-0.96): dimensionless normalised jerk scales with
# duration by construction and SPARC is strongly duration-correlated. Removed together with duration in the strictest ablation.
TIMING_COUPLED_BASE = DURATION_BASE + ["smoothness_norm_jerk", "smoothness_sparc"]

SETS = {
    "coordinate_only": "PRIMARY: key-point coordinates + target object; no angle-derived feature",
    "angle_enhanced": "SECONDARY: coordinate_only + 2D-video-derived joint-angle features",
    "coordinate_no_position_error": "ABLATION: coordinate_only without position_error_* (target-object dependent)",
    "coordinate_no_duration": "ABLATION: coordinate_only without movement_duration_s_* (equals recording clip length)",
    "coordinate_no_duration_no_position_error": "ABLATION: coordinate_only without duration and position_error",
    "coordinate_no_timing_coupled": "ABLATION (strictest): coordinate_only without duration, normalised jerk and SPARC",
}


def split_name(col):
    for d in sorted(DESCRIPTORS, key=len, reverse=True):
        if col.endswith("_" + d):
            return col[: -(len(d) + 1)], d
    return col, ""


def modality(col):
    base, _ = split_name(col)
    if base in ANGLE_BASE:
        return "angle_derived"
    if base in TARGET_BASE:
        return "coordinates_plus_target"
    return "coordinates"


def columns_for(set_name, feature_columns):
    if set_name == "angle_enhanced":
        return list(feature_columns)
    cols = [c for c in feature_columns if modality(c) != "angle_derived"]
    if set_name == "coordinate_only":
        return cols
    if set_name == "coordinate_no_position_error":
        return [c for c in cols if modality(c) != "coordinates_plus_target"]
    if set_name == "coordinate_no_duration":
        return [c for c in cols if split_name(c)[0] not in DURATION_BASE]
    if set_name == "coordinate_no_duration_no_position_error":
        return [c for c in cols if split_name(c)[0] not in DURATION_BASE and modality(c) != "coordinates_plus_target"]
    if set_name == "coordinate_no_timing_coupled":
        return [c for c in cols if split_name(c)[0] not in TIMING_COUPLED_BASE]
    raise ValueError(set_name)
