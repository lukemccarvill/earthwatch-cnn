from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "merged_metadata.xlsx"

QUALITY_ORDER = {
    "Good": 1,
    "Moderate": 2,
    "Poor": 3,
}

ORDER_TO_QUALITY = {value: key for key, value in QUALITY_ORDER.items()}


NITRATE_QUALITY = {
    "<0.2": "Good",
    "0.2-0.5": "Good",
    "0.5-1": "Good",
    "1-2": "Moderate",
    "2-5": "Poor",
    "5-10": "Poor",
    ">10": "Poor",
}

PHOSPHATE_QUALITY = {
    "<0.02": "Good",
    "0.02-0.05": "Good",
    "0.05-0.1": "Good",
    "0.1-0.2": "Moderate",
    "0.2-0.5": "Poor",
    "0.5-1": "Poor",
    ">1": "Poor",
}


def clean_bin(value):
    if pd.isna(value):
        return pd.NA
    return str(value).strip()


def worst_quality(*qualities):
    known = [quality for quality in qualities if pd.notna(quality)]
    if not known:
        return pd.NA
    return max(known, key=lambda quality: QUALITY_ORDER[quality])


def apply_threshold_logic(df):
    df = df.copy()

    df["nitrate_bin_clean"] = df["Nitrate (mg/L)"].map(clean_bin)
    df["phosphate_bin_clean"] = df["Phosphate (mg/L)"].map(clean_bin)

    df["nitrate_quality_from_threshold"] = df["nitrate_bin_clean"].map(NITRATE_QUALITY)
    df["phosphate_quality_from_threshold"] = df["phosphate_bin_clean"].map(PHOSPHATE_QUALITY)
    df["threshold_quality"] = df.apply(
        lambda row: worst_quality(
            row["nitrate_quality_from_threshold"],
            row["phosphate_quality_from_threshold"],
        ),
        axis=1,
    )
    df["threshold_matches_feedback"] = df["threshold_quality"].eq(df["Feedback Rating"])
    return df


def print_summary(df):
    print(f"Rows checked: {len(df):,}")
    print(f"Matches: {df['threshold_matches_feedback'].sum():,}")
    print(f"Mismatches: {(~df['threshold_matches_feedback']).sum():,}")
    print(f"Match rate: {df['threshold_matches_feedback'].mean() * 100:.2f}%")
    print()

    print("Earthwatch Feedback Rating counts:")
    print(df["Feedback Rating"].value_counts(dropna=False).to_string())
    print()

    print("Threshold-derived quality counts:")
    print(df["threshold_quality"].value_counts(dropna=False).to_string())
    print()

    print("Confusion matrix: rows=Earthwatch, columns=threshold-derived")
    print(pd.crosstab(df["Feedback Rating"], df["threshold_quality"], dropna=False).to_string())
    print()

    print("Nitrate threshold mapping used:")
    for bin_label, quality in NITRATE_QUALITY.items():
        print(f"  {bin_label}: {quality}")
    print()

    print("Phosphate threshold mapping used:")
    for bin_label, quality in PHOSPHATE_QUALITY.items():
        print(f"  {bin_label}: {quality}")


def print_mismatches(df, max_rows=30):
    mismatches = df.loc[
        ~df["threshold_matches_feedback"],
        [
            "GlobalID",
            "Site Name",
            "Nitrate (mg/L)",
            "Phosphate (mg/L)",
            "nitrate_quality_from_threshold",
            "phosphate_quality_from_threshold",
            "threshold_quality",
            "Feedback Rating",
            "feedback_eng",
        ],
    ]

    if mismatches.empty:
        print()
        print("No mismatches found.")
        return

    print()
    print(f"First {min(max_rows, len(mismatches))} mismatches:")
    print(mismatches.head(max_rows).to_string(index=False))


def main():
    df = pd.read_excel(METADATA_PATH)
    checked = apply_threshold_logic(df)
    print_summary(checked)
    print_mismatches(checked)


if __name__ == "__main__":
    main()
