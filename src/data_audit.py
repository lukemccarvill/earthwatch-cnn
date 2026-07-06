from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "Great_UK_WaterBlitz"
METADATA_PATH = DATA_DIR / "merged_metadata.xlsx"

ACCEPTABLE_LABEL = 0
UNACCEPTABLE_LABEL = 1


def resolve_image_path(row, image_dir=IMAGE_DIR):
    """Return the local image path for a metadata row."""
    filename = row.get("saved_filename")

    if pd.isna(filename) or str(filename).strip() == "":
        filename = Path(str(row.get("image_path", ""))).name

    return image_dir / str(filename)


def add_local_image_paths(df, image_dir=IMAGE_DIR):
    df = df.copy()
    df["local_image_path"] = df.apply(resolve_image_path, axis=1, image_dir=image_dir)
    df["image_exists"] = df["local_image_path"].map(Path.exists)
    return df


def add_binary_labels(df):
    df = df.copy()
    rating = df["Feedback Rating"].astype("string").str.strip().str.lower()

    df["binary_label"] = pd.NA
    df.loc[rating.eq("good"), "binary_label"] = ACCEPTABLE_LABEL
    df.loc[rating.isin(["moderate", "poor"]), "binary_label"] = UNACCEPTABLE_LABEL
    df["binary_label_name"] = df["binary_label"].map(
        {
            ACCEPTABLE_LABEL: "Acceptable",
            UNACCEPTABLE_LABEL: "Unacceptable",
        }
    )
    return df


def load_audited_metadata(metadata_path=METADATA_PATH, image_dir=IMAGE_DIR):
    df = pd.read_excel(metadata_path)
    df = add_local_image_paths(df, image_dir=image_dir)
    df = add_binary_labels(df)
    return df


def print_audit_summary(df):
    print(f"Rows: {len(df):,}")
    print(f"Images found: {df['image_exists'].sum():,}")
    print(f"Images missing: {(~df['image_exists']).sum():,}")
    print()
    print("Original three-class quality counts:")
    three_class_counts = df["Feedback Rating"].value_counts(dropna=False)
    print(three_class_counts)
    print()
    print("Collapsed two-class quality counts:")
    two_class_counts = df["binary_label_name"].value_counts(dropna=False)
    print(two_class_counts)
    print()
    print("Duplicate saved filenames:", df["saved_filename"].duplicated().sum())
    print("Duplicate feature GlobalIDs:", df["GlobalID"].duplicated().sum())
    print("Duplicate site/date/time groups:", df[["Site Name", "Sample Date", "Sample Time"]].duplicated().sum())
    print()
    print("Non-empty metadata columns:")
    non_empty_columns = [
        column
        for column in df.columns
        if not df[column].isna().all()
    ]
    for column in non_empty_columns:
        print(f"- {column}")


if __name__ == "__main__":
    audited = load_audited_metadata()
    print_audit_summary(audited)

    missing = audited.loc[~audited["image_exists"], ["saved_filename", "image_path", "local_image_path"]]
    if not missing.empty:
        print()
        print("First missing images:")
        print(missing.head(20).to_string(index=False))
