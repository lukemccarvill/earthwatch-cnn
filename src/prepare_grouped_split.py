from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit

from data_audit import load_audited_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "audited_metadata_with_split.csv"

GROUP_KEY = "GlobalID"
TEST_SIZE = 0.2
RANDOM_STATE = 42

SAFE_METADATA_FEATURES = [
    "Freshwater body type",
    "What is the main land use within 50m?",
    "What is the main bank vegetation? (select all that apply)",
    "Is there any of the following on the water surface?",
    "Estimate the water colour",
]

TEMPORAL_OR_COLLECTION_ABLATION_FEATURES = [
    "Sample Date",
    "Sample Time",
    "Total number of participants",
]

LOCATION_ABLATION_FEATURES = [
    "Country",
    "County",
    "RBD_NAME",
    "MNCAT_NAME",
]

EXACT_LOCATION_ABLATION_FEATURES = [
    "x",
    "y",
    "geometry_x",
    "geometry_y",
]

LEAKY_OR_RESULT_FEATURES = [
    "Feedback Rating",
    "feedback_score",
    "feedback_eng",
    "feedback_core",
    "Nitrate (mg/L)",
    "Phosphate (mg/L)",
    "Nitrate (mg/L) MID",
    "Phosphate (mg/L) MID",
    "sort_order_nitrates",
    "sort_order_phosphates",
    "sort_order_turbidity",
]


def print_duplicate_group_summary(df):
    print("=== Duplicate / grouping checks ===")
    print(f"Rows: {len(df):,}")
    print(f"Unique {GROUP_KEY}: {df[GROUP_KEY].nunique(dropna=False):,}")
    print(f"Duplicate {GROUP_KEY} rows: {df[GROUP_KEY].duplicated().sum():,}")
    print()

    repeated_globalids = (
        df.groupby(GROUP_KEY)
        .size()
        .rename("rows")
        .reset_index()
        .query("rows > 1")
        .sort_values("rows", ascending=False)
    )
    print(f"{GROUP_KEY}s with multiple photos/rows: {len(repeated_globalids):,}")
    if not repeated_globalids.empty:
        print(repeated_globalids.head(20).to_string(index=False))
    print()

    site_datetime_cols = ["Site Name", "Sample Date", "Sample Time"]
    site_datetime_groups = (
        df.groupby(site_datetime_cols, dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .query("rows > 1")
        .sort_values("rows", ascending=False)
    )
    print(f"Site/date/time groups with multiple rows: {len(site_datetime_groups):,}")
    if not site_datetime_groups.empty:
        print(site_datetime_groups.head(20).to_string(index=False))
    print()


def create_grouped_split(df, group_key=GROUP_KEY):
    df = df.copy().reset_index(drop=True)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            y=df["binary_label"],
            groups=df[group_key],
        )
    )

    df["split"] = "train"
    df.iloc[val_idx, df.columns.get_loc("split")] = "val"
    return df


def mark_readable_images(df):
    df = df.copy()
    readable = []
    errors = []

    for path in df["local_image_path"]:
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).verify()
            readable.append(True)
            errors.append("")
        except Exception as exc:
            readable.append(False)
            errors.append(f"{type(exc).__name__}: {exc}")

    df["image_readable"] = readable
    df["image_error"] = errors
    return df


def print_split_summary(df):
    print("=== Split summary ===")
    print(df["split"].value_counts().to_string())
    print()
    print("Binary label counts by split:")
    print(pd.crosstab(df["split"], df["binary_label_name"]).to_string())
    print()
    print("Three-class Feedback Rating counts by split:")
    print(pd.crosstab(df["split"], df["Feedback Rating"]).to_string())
    print()

    train_groups = set(df.loc[df["split"].eq("train"), GROUP_KEY])
    val_groups = set(df.loc[df["split"].eq("val"), GROUP_KEY])
    overlap = train_groups.intersection(val_groups)
    print(f"Overlapping {GROUP_KEY}s between train and val: {len(overlap):,}")
    print()


def print_feature_lists():
    print("=== Feature lists ===")
    print("Safe default metadata features:")
    for feature in SAFE_METADATA_FEATURES:
        print(f"- {feature}")
    print()

    print("Location ablation features:")
    for feature in LOCATION_ABLATION_FEATURES:
        print(f"- {feature}")
    print()

    print("Temporal/collection ablation features:")
    for feature in TEMPORAL_OR_COLLECTION_ABLATION_FEATURES:
        print(f"- {feature}")
    print()

    print("Exact-location ablation features:")
    for feature in EXACT_LOCATION_ABLATION_FEATURES:
        print(f"- {feature}")
    print()

    print("Leaky/result columns to exclude from predictive baselines:")
    for feature in LEAKY_OR_RESULT_FEATURES:
        print(f"- {feature}")
    print()


def main():
    df = load_audited_metadata()
    df = df[df["image_exists"]].copy()
    df = mark_readable_images(df)

    unreadable = df[~df["image_readable"]]
    print("=== Image readability checks ===")
    print(f"Readable images: {df['image_readable'].sum():,}")
    print(f"Unreadable images: {len(unreadable):,}")
    if not unreadable.empty:
        print(unreadable[["saved_filename", "local_image_path", "image_error"]].to_string(index=False))
    print()

    df = df[df["image_readable"]].copy()

    print_duplicate_group_summary(df)
    print_feature_lists()

    split_df = create_grouped_split(df)
    print_split_summary(split_df)

    split_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
