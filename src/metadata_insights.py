from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "merged_metadata.xlsx"


def print_non_empty_column_summary(df):
    non_empty_columns = [column for column in df.columns if not df[column].isna().all()]

    print(f"Rows: {len(df):,}")
    print(f"Non-empty columns: {len(non_empty_columns):,}")
    print()

    for column in non_empty_columns:
        series = df[column]
        missing_pct = series.isna().mean() * 100
        unique_count = series.nunique(dropna=True)
        sample_values = series.dropna().astype(str).head(3).tolist()

        print(f"Column: {column}")
        print(f"  Missing: {missing_pct:.1f}%")
        print(f"  Unique values: {unique_count:,}")
        print(f"  Sample values: {sample_values}")
        print()


def print_quality_cross_tabs(df):
    columns_to_check = [
        "Nitrate (mg/L)",
        "Phosphate (mg/L)",
        "sort_order_nitrates",
        "sort_order_phosphates",
        "Estimate the water colour",
        "Freshwater body type",
        "What is the main land use within 50m?",
        "Country",
    ]

    for column in columns_to_check:
        print(f"\n## {column}")
        print("Top value counts:")
        print(df[column].value_counts(dropna=False).head(12).to_string())
        print()

        print("Quality mix within each value:")
        cross_tab = pd.crosstab(
            df[column],
            df["Feedback Rating"],
            normalize="index",
        ).round(3)
        print(cross_tab.head(20).to_string())


def print_numeric_quality_summaries(df):
    numeric_columns = [
        "Nitrate (mg/L) MID",
        "Phosphate (mg/L) MID",
    ]

    print("\nNumeric summaries by Feedback Rating:")
    for column in numeric_columns:
        print(f"\n## {column}")
        print(df.groupby("Feedback Rating")[column].describe().round(3).to_string())


def main():
    df = pd.read_excel(METADATA_PATH)

    print("=== Non-empty metadata column summary ===")
    print_non_empty_column_summary(df)

    print("\n=== Quality relationships ===")
    print_quality_cross_tabs(df)
    print_numeric_quality_summaries(df)


if __name__ == "__main__":
    main()
