"""
01_load_raw.py

Load all 408 raw DeepLabCut per-repetition Excel files from data/raw/
into one clean per-frame dataframe.

Keeps only the performing arm's landmarks and joint-angle columns.
"""

from pathlib import Path
import re

import openpyxl
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUT_FILE = PROCESSED_DIR / "raw_frames.pkl"


# ============================================================
# DATASET FILE FORMAT
# Example:
# 01_left_TD_rep1.xlsx
# 25_left_DCP_rep8.xlsx
# ============================================================

FNAME_RE = re.compile(
    r"^(\d+)_(left|right)_(TD|DCP)_rep([1-8])\.xlsx$",
    re.IGNORECASE,
)


def parse_filename(filename):
    """Extract subject, performing side, group and repetition."""

    match = FNAME_RE.match(filename)

    if not match:
        return None

    subject, side, group, rep = match.groups()

    return (
        subject,
        side.lower(),
        group.upper(),
        int(rep),
    )


# ============================================================
# LOAD ONE EXCEL FILE
# ============================================================

def load_one(filepath, side):
    """Load one repetition and keep only the performing arm."""

    workbook = openpyxl.load_workbook(
        filepath,
        read_only=True,
        data_only=True,
    )

    worksheet = workbook[workbook.sheetnames[0]]

    rows = list(worksheet.iter_rows(values_only=True))

    if not rows:
        raise ValueError("Excel file is empty")

    header = rows[0]
    data = rows[1:]

    df = pd.DataFrame(data, columns=header)

    # Performing arm
    letter = "R" if side == "right" else "L"

    keep = {
        "index": "frame",

        # Wrist
        f"('wrist_{side}', 'x')": "wrist_x",
        f"('wrist_{side}', 'y')": "wrist_y",

        # Elbow
        f"('elbow_{side}', 'x')": "elbow_x",
        f"('elbow_{side}', 'y')": "elbow_y",

        # Shoulder
        f"('shoulder_{side}', 'x')": "shoulder_x",
        f"('shoulder_{side}', 'y')": "shoulder_y",

        # MCP3
        f"('MCP3_{side}', 'x')": "mcp3_x",
        f"('MCP3_{side}', 'y')": "mcp3_y",

        # Target
        "('object', 'x')": "object_x",
        "('object', 'y')": "object_y",

        # Joint angles
        f"{letter}-Shoulder": "shoulder_angle",
        f"{letter}-Elbow": "elbow_angle",
        f"{letter}-Wrist": "wrist_angle",
    }

    missing = [
        column
        for column in keep
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[list(keep.keys())].rename(columns=keep)

    # Convert everything to numeric
    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    workbook.close()

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 1 — LOADING RAW DATA")
    print("=" * 60)

    # Check raw directory
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw directory not found:\n{RAW_DIR}"
        )

    # Create processed directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find Excel files
    files = sorted(
        RAW_DIR.glob("*.xlsx")
    )

    print(f"\nRaw directory:")
    print(RAW_DIR)

    print(f"\nExcel files found: {len(files)}")

    # Dataset should contain exactly 408 files
    if len(files) != 408:
        raise RuntimeError(f"Expected exactly 408 raw Excel files, found {len(files)}. Refusing partial dataset execution.")

    all_frames = []
    errors = []

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    for number, filepath in enumerate(files, start=1):

        filename = filepath.name

        parsed = parse_filename(filename)

        if parsed is None:

            errors.append(
                (filename, "Invalid filename format")
            )

            continue

        subject, side, group, rep = parsed

        try:

            df = load_one(
                filepath,
                side,
            )

            # Add metadata
            df.insert(
                0,
                "subject",
                subject,
            )

            df.insert(
                1,
                "side",
                side,
            )

            df.insert(
                2,
                "group",
                group,
            )

            df.insert(
                3,
                "rep",
                rep,
            )

            df.insert(
                4,
                "source_file",
                filename,
            )

            all_frames.append(df)

        except Exception as error:

            errors.append(
                (
                    filename,
                    str(error),
                )
            )

        # Progress
        if number % 50 == 0 or number == len(files):

            print(
                f"Processed {number}/{len(files)} files"
            )

    # --------------------------------------------------------
    # Check whether anything loaded
    # --------------------------------------------------------

    if not all_frames:

        raise RuntimeError(
            "No files were successfully loaded."
        )

    # --------------------------------------------------------
    # Fail closed: every expected file must load and each subject must
    # contribute exactly repetitions 1..8 with one consistent diagnosis/side.
    # --------------------------------------------------------

    if errors:
        preview = "; ".join(f"{name}: {err}" for name, err in errors[:10])
        raise RuntimeError(f"{len(errors)} raw file(s) failed validation; refusing partial dataset. First errors: {preview}")
    if len(all_frames) != 408:
        raise RuntimeError(f"Expected 408 successfully loaded files, got {len(all_frames)}")

    full = pd.concat(all_frames, ignore_index=True)
    file_meta = full[["subject", "side", "group", "rep", "source_file"]].drop_duplicates()
    if file_meta["source_file"].duplicated().any():
        raise RuntimeError("A source file produced inconsistent metadata")
    if file_meta.duplicated(["subject", "rep"]).any():
        raise RuntimeError("Duplicate subject/repetition detected")
    expected_reps = set(range(1, 9))
    for subject, part in file_meta.groupby("subject"):
        if set(part["rep"].astype(int)) != expected_reps or len(part) != 8:
            raise RuntimeError(f"Subject {subject} does not have exactly one file for each repetition 1..8")
        if part["group"].nunique() != 1 or part["side"].nunique() != 1:
            raise RuntimeError(f"Subject {subject} has inconsistent group or performing side metadata")
    if file_meta["subject"].nunique() != 51:
        raise RuntimeError(f"Expected 51 subjects, found {file_meta['subject'].nunique()}")
    counts = file_meta.groupby("group")["subject"].nunique().to_dict()
    if counts != {"DCP": 30, "TD": 21}:
        raise RuntimeError(f"Unexpected subject class counts: {counts}; expected DCP=30, TD=21")

    # --------------------------------------------------------
    # Combine all repetitions
    # --------------------------------------------------------

    full = pd.concat(
        all_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    full.to_pickle(
        OUT_FILE
    )

    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RAW DATA LOADING COMPLETE")
    print("=" * 60)

    print(
        f"\nFiles successfully loaded : {len(all_frames)}"
    )

    print(
        f"Files with errors         : {len(errors)}"
    )

    print(
        f"Total rows                : {len(full):,}"
    )

    print(
        f"Subjects                  : "
        f"{full['subject'].nunique()}"
    )

    print(
        f"Groups                    : "
        f"{full.groupby('group')['subject'].nunique().to_dict()}"
    )

    print(
        f"Repetitions               : "
        f"{sorted(full['rep'].unique())}"
    )

    print(
        f"Output                    : {OUT_FILE}"
    )

    print("\nColumns:")
    print(list(full.columns))

    # --------------------------------------------------------
    # File-level validation
    # --------------------------------------------------------

    file_summary = (
        full[
            [
                "subject",
                "side",
                "group",
                "rep",
                "source_file",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "subject",
                "rep",
            ]
        )
    )

    print(
        f"\nUnique repetition files represented: "
        f"{len(file_summary)}"
    )

    # --------------------------------------------------------
    # Error report
    # --------------------------------------------------------

    if errors:

        print("\nERRORS:")

        for error in errors[:20]:

            print(
                f"  {error[0]} -> {error[1]}"
            )

        if len(errors) > 20:

            print(
                f"... and {len(errors) - 20} more."
            )

    else:

        print(
            "\nNo file-loading errors."
        )

    print("\nFirst 5 rows:")
    print(full.head())


if __name__ == "__main__":
    main()