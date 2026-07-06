from itertools import product
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "merged_metadata.xlsx"

QUALITY_ORDER = {
    "Good": 1,
    "Moderate": 2,
    "Poor": 3,
}
QUALITY_NAMES = tuple(QUALITY_ORDER)
ORDER_TO_QUALITY = {value: key for key, value in QUALITY_ORDER.items()}


def clean_bin(value):
    if pd.isna(value):
        return pd.NA
    return str(value).strip()


def worst_quality(nitrate_quality, phosphate_quality):
    return ORDER_TO_QUALITY[max(QUALITY_ORDER[nitrate_quality], QUALITY_ORDER[phosphate_quality])]


def load_data():
    df = pd.read_excel(METADATA_PATH)
    df = df.copy()
    df["nitrate_bin_clean"] = df["Nitrate (mg/L)"].map(clean_bin)
    df["phosphate_bin_clean"] = df["Phosphate (mg/L)"].map(clean_bin)
    return df


def check_pair_rule_possible(df):
    pair_counts = (
        df.groupby(["nitrate_bin_clean", "phosphate_bin_clean"])["Feedback Rating"]
        .nunique()
        .reset_index(name="n_feedback_ratings")
    )
    conflicting_pairs = pair_counts[pair_counts["n_feedback_ratings"] > 1]

    print("=== Pair-based nitrate/phosphate rule check ===")
    print(f"Observed nitrate/phosphate pairs: {len(pair_counts):,}")
    print(f"Pairs with conflicting Feedback Rating: {len(conflicting_pairs):,}")

    if conflicting_pairs.empty:
        print("A 100% lookup rule based only on the nitrate/phosphate pair is possible.")
    else:
        print("No deterministic pair-only rule can reach 100%, because these pairs conflict:")
        details = df.merge(
            conflicting_pairs[["nitrate_bin_clean", "phosphate_bin_clean"]],
            on=["nitrate_bin_clean", "phosphate_bin_clean"],
            how="inner",
        )
        print(
            pd.crosstab(
                [details["nitrate_bin_clean"], details["phosphate_bin_clean"]],
                details["Feedback Rating"],
            ).to_string()
        )

    print()
    print("Most common Feedback Rating for each observed pair:")
    pair_majority = (
        df.groupby(["nitrate_bin_clean", "phosphate_bin_clean", "Feedback Rating"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["nitrate_bin_clean", "phosphate_bin_clean", "count"],
            ascending=[True, True, False],
        )
    )
    print(pair_majority.to_string(index=False))
    print()


def search_worst_of_two_mappings(df):
    nitrate_bins = sorted(df["nitrate_bin_clean"].dropna().unique())
    phosphate_bins = sorted(df["phosphate_bin_clean"].dropna().unique())
    pair_rating_counts = (
        df.groupby(["nitrate_bin_clean", "phosphate_bin_clean", "Feedback Rating"])
        .size()
        .rename("count")
        .reset_index()
    )
    pair_records = pair_rating_counts.to_dict("records")

    best = {
        "matches": -1,
        "nitrate_mapping": None,
        "phosphate_mapping": None,
        "predictions": None,
    }

    nitrate_mapping_candidates = [
        dict(zip(nitrate_bins, quality_choices))
        for quality_choices in product(QUALITY_NAMES, repeat=len(nitrate_bins))
    ]
    phosphate_mapping_candidates = [
        dict(zip(phosphate_bins, quality_choices))
        for quality_choices in product(QUALITY_NAMES, repeat=len(phosphate_bins))
    ]

    print("=== Search for 100% worst-of-nitrate/phosphate mapping ===")
    print(f"Nitrate bins: {nitrate_bins}")
    print(f"Phosphate bins: {phosphate_bins}")
    print(f"Candidate combinations: {len(nitrate_mapping_candidates) * len(phosphate_mapping_candidates):,}")

    for nitrate_mapping in nitrate_mapping_candidates:
        for phosphate_mapping in phosphate_mapping_candidates:
            matches = 0
            for record in pair_records:
                prediction = worst_quality(
                    nitrate_mapping[record["nitrate_bin_clean"]],
                    phosphate_mapping[record["phosphate_bin_clean"]],
                )
                if prediction == record["Feedback Rating"]:
                    matches += record["count"]

            if matches > best["matches"]:
                predictions = [
                    worst_quality(
                        nitrate_mapping[nitrate_value],
                        phosphate_mapping[phosphate_value],
                    )
                    for nitrate_value, phosphate_value in zip(
                        df["nitrate_bin_clean"],
                        df["phosphate_bin_clean"],
                    )
                ]
                best = {
                    "matches": matches,
                    "nitrate_mapping": nitrate_mapping,
                    "phosphate_mapping": phosphate_mapping,
                    "predictions": predictions,
                }

            if matches == len(df):
                print("Found a 100% mapping.")
                print_mapping(nitrate_mapping, phosphate_mapping)
                return best

    print("No 100% worst-of-two mapping found.")
    print(f"Best matches: {best['matches']:,} / {len(df):,} ({best['matches'] / len(df) * 100:.2f}%)")
    print_mapping(best["nitrate_mapping"], best["phosphate_mapping"])
    return best


def print_mapping(nitrate_mapping, phosphate_mapping):
    print()
    print("Nitrate mapping:")
    for bin_label, quality in nitrate_mapping.items():
        print(f"  {bin_label}: {quality}")

    print()
    print("Phosphate mapping:")
    for bin_label, quality in phosphate_mapping.items():
        print(f"  {bin_label}: {quality}")
    print()


def print_best_confusion(df, predictions):
    result = df.copy()
    result["searched_logic_quality"] = predictions
    result["searched_logic_matches_feedback"] = result["searched_logic_quality"].eq(result["Feedback Rating"])

    print("Confusion matrix: rows=Earthwatch, columns=searched logic")
    print(pd.crosstab(result["Feedback Rating"], result["searched_logic_quality"], dropna=False).to_string())
    print()

    mismatches = result.loc[
        ~result["searched_logic_matches_feedback"],
        [
            "GlobalID",
            "Site Name",
            "Nitrate (mg/L)",
            "Phosphate (mg/L)",
            "searched_logic_quality",
            "Feedback Rating",
            "feedback_eng",
        ],
    ]
    if mismatches.empty:
        print("No mismatches found.")
    else:
        print(f"First {min(30, len(mismatches))} mismatches:")
        print(mismatches.head(30).to_string(index=False))


def main():
    df = load_data()
    check_pair_rule_possible(df)
    best = search_worst_of_two_mappings(df)
    print_best_confusion(df, best["predictions"])


if __name__ == "__main__":
    main()
