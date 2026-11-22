from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_sets as fs


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "repetition_descriptors.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiment_dataset.csv"
)


def main():

    print("=" * 70)
    print("STEP 9 — EXPERIMENT DATASET CONSTRUCTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 08_repetition_descriptors.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"subject": str}
    )

    print(f"\nInput shape: {df.shape}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if df["subject"].duplicated().any():
        raise ValueError(
            "Duplicate subject records detected."
        )

    if df["group"].isna().any():
        raise ValueError(
            "Missing group labels detected."
        )

    # --------------------------------------------------------
    # Encode target
    # DCP = 1
    # TD  = 0
    # --------------------------------------------------------

    valid_groups = {"DCP", "TD"}

    if not set(df["group"].unique()).issubset(
        valid_groups
    ):
        raise ValueError(
            f"Unexpected group labels: "
            f"{df['group'].unique()}"
        )

    df["target"] = (
        df["group"] == "DCP"
    ).astype(int)

    # --------------------------------------------------------
    # Remove metadata that must not be used as predictors
    # --------------------------------------------------------

    metadata_columns = [
        "subject",
        "side",
        "group",
        "target",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in metadata_columns
    ]

    # --------------------------------------------------------
    # Remove completely missing features
    #
    # IMPORTANT:
    # This is only structural filtering.
    # No imputation or scaling is performed here.
    # --------------------------------------------------------

    complete_missing = [
        col
        for col in feature_columns
        if df[col].isna().all()
    ]

    if complete_missing:
        print(
            "\nRemoving completely missing features:"
        )

        for col in complete_missing:
            print(f"  - {col}")

        feature_columns = [
            col
            for col in feature_columns
            if col not in complete_missing
        ]

    # --------------------------------------------------------
    # Construct experiment dataset
    # --------------------------------------------------------

    experiment_df = df[
        [
            "subject",
            "side",
            "group",
            "target",
        ] + feature_columns
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Round predictors to 5 significant digits. Raw-to-feature recomputation on a
    # different machine/BLAS differs at ~1e-10 relative; the downstream
    # stability-selection / random-forest steps are chaotic at that scale (one ablation
    # fold flipped 1.0 -> 0.9 from such a difference). 5 s.f. is far finer than the
    # measurement precision (2 mm/pixel pose estimates) and makes the dataset bit-identical
    # across machines (verified by rebuilding from raw data).
    experiment_df[feature_columns] = experiment_df[feature_columns].apply(
        lambda col: col.map(lambda v: float(f"{v:.5g}") if pd.notna(v) else v)
    )

    # --------------------------------------------------------
    # Feature sets. PRIMARY dataset (experiment_dataset.csv) = coordinate-only:
    # no predictor that depends on the joint-angle columns. The full set is kept as the
    # SECONDARY angle-enhanced comparison; a third file drops position_error_* (ablation).
    # --------------------------------------------------------
    meta = ["subject", "side", "group", "target"]
    out_dir = OUTPUT_FILE.parent
    audit_rows = []
    for c in feature_columns:
        base, desc = fs.split_name(c)
        audit_rows.append({
            "predictor": c, "base_feature": base, "descriptor": desc, "source_modality": fs.modality(c),
            "in_coordinate_only": c in fs.columns_for("coordinate_only", feature_columns),
            "in_angle_enhanced": True,
            "in_coordinate_no_position_error": c in fs.columns_for("coordinate_no_position_error", feature_columns),
            "in_coordinate_no_duration": c in fs.columns_for("coordinate_no_duration", feature_columns),
            "in_coordinate_no_duration_no_position_error": c in fs.columns_for("coordinate_no_duration_no_position_error", feature_columns),
            "in_coordinate_no_timing_coupled": c in fs.columns_for("coordinate_no_timing_coupled", feature_columns),
        })
    pd.DataFrame(audit_rows).to_csv(out_dir / "feature_audit.csv", index=False)

    files = {
        "angle_enhanced": out_dir / "experiment_dataset_angle_enhanced.csv",
        "coordinate_only": OUTPUT_FILE,
        "coordinate_no_position_error": out_dir / "experiment_dataset_coord_no_position_error.csv",
        "coordinate_no_duration": out_dir / "experiment_dataset_coord_no_duration.csv",
        "coordinate_no_duration_no_position_error": out_dir / "experiment_dataset_coord_no_duration_no_position_error.csv",
        "coordinate_no_timing_coupled": out_dir / "experiment_dataset_coord_no_timing_coupled.csv",
    }
    for set_name, path in files.items():
        cols = fs.columns_for(set_name, feature_columns)
        experiment_df[meta + cols].to_csv(path, index=False)
        print(f"{set_name:32s}: {len(cols):3d} predictors -> {path.name}")
    feature_columns_full = list(feature_columns)
    feature_columns = fs.columns_for("coordinate_only", feature_columns_full)
    experiment_df = experiment_df[meta + feature_columns]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT DATASET COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput shape: "
        f"{experiment_df.shape}"
    )

    print(
        f"Subjects: "
        f"{experiment_df['subject'].nunique()}"
    )

    print(
        f"Predictor features: "
        f"{len(feature_columns)}"
    )

    print("\nTarget distribution:")

    print(
        experiment_df["target"]
        .value_counts()
        .sort_index()
    )

    print("\nGroup distribution:")

    print(
        experiment_df["group"]
        .value_counts()
    )

    print(
        "\nCompletely missing predictors remaining:",
        experiment_df[
            feature_columns
        ].isna().all().sum()
    )

    print(
        "\nPredictors containing some missing values:",
        experiment_df[
            feature_columns
        ].isna().any().sum()
    )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    print("\nFirst 5 metadata rows:")
    print(
        experiment_df[
            ["subject", "side", "group", "target"]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()