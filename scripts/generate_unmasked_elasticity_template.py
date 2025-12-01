"""
One-off helper to map masked Round 1 SKUs to the unmasked Round 2 identifiers
and to produce an unmasked elasticity template.

Run manually from the repository root:

    python scripts/generate_unmasked_elasticity_template.py

Outputs:
    static/sku_mapping_masked_to_unmasked.csv
    static/sku_elasticity_template_unmasked.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MASKED_DIR = ROOT / "Data Files"
UNMASKED_DIR = ROOT / "Data Files Round 2"
MASKED_TRAIN = MASKED_DIR / "Sellout_Train.csv"
UNMASKED_TRAIN = UNMASKED_DIR / "Sellout_Train.xlsx"
MASKED_TEMPLATE = MASKED_DIR / "sku_elasticity_matrix.csv"

sys.path.append(str(ROOT / "src"))
from compute_elasticity import add_derived_columns  # type: ignore  # noqa: E402


def load_masked() -> pd.DataFrame:
    masked_df = pd.read_csv(MASKED_TRAIN)
    masked_df = add_derived_columns(masked_df)
    return masked_df


def load_unmasked() -> pd.DataFrame:
    unmasked_df = pd.read_excel(UNMASKED_TRAIN)
    unmasked_df.columns = [c.lower() for c in unmasked_df.columns]
    unmasked_df = add_derived_columns(unmasked_df)
    return unmasked_df


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "sales_value",
        "sales_hectoliters",
        "avg_price_per_liter",
        "numeric_distribution_stores_handling",
        "inventory_hectoliters",
        "weighted_distribution_tdp_reach",
    ]
    df = df.copy()
    agg = (
        df.groupby("sku_str")
        .agg(
            manufacturer=("manufacturer", "first"),
            package=("package", "first"),
            package_type=("package_type", "first"),
            capacity_number=("capacity_number", "first"),
            sales_value_total=("sales_value", "sum"),
            sales_hl_total=("sales_hectoliters", "sum"),
            avg_price_mean=("avg_price_per_liter", "mean"),
            numeric_dist_mean=("numeric_distribution_stores_handling", "mean"),
            inventory_mean=("inventory_hectoliters", "mean"),
            tdp_mean=("weighted_distribution_tdp_reach", "mean"),
        )
        .reset_index()
    )
    numeric_cols = [
        "sales_value_total",
        "sales_hl_total",
        "avg_price_mean",
        "numeric_dist_mean",
        "inventory_mean",
        "tdp_mean",
    ]
    agg[numeric_cols] = agg[numeric_cols].astype(float).round(6)
    agg[numeric_cols] = agg[numeric_cols].fillna(0.0)
    agg["effect_size"] = (
        agg["sales_hl_total"].abs()
        + agg["sales_value_total"].abs()
        + agg["avg_price_mean"].abs()
        + agg["numeric_dist_mean"].abs()
    )
    return agg


def build_mapping(masked_df: pd.DataFrame, unmasked_df: pd.DataFrame) -> pd.DataFrame:
    masked_agg = aggregate_features(masked_df).rename(
        columns={
            "sku_str": "masked_sku",
            "manufacturer": "masked_manufacturer",
        }
    )
    unmasked_agg = aggregate_features(unmasked_df).rename(
        columns={
            "sku_str": "unmasked_sku",
            "manufacturer": "unmasked_manufacturer",
        }
    )
    mask_vectors = masked_agg[
        ["sales_value_total", "sales_hl_total", "avg_price_mean", "numeric_dist_mean", "inventory_mean", "tdp_mean"]
    ].to_numpy()
    actual_vectors = unmasked_agg[
        ["sales_value_total", "sales_hl_total", "avg_price_mean", "numeric_dist_mean", "inventory_mean", "tdp_mean"]
    ].to_numpy()

    distance_matrix = cdist(mask_vectors, actual_vectors, metric="euclidean")

    row_ind, col_ind = [], []
    for manufacturer in masked_agg["masked_manufacturer"].str.upper().unique():
        mask_idx = masked_agg[masked_agg["masked_manufacturer"].str.upper() == manufacturer].index.to_numpy()
        actual_idx = unmasked_agg[unmasked_agg["unmasked_manufacturer"].str.upper() == manufacturer].index.to_numpy()
        if len(mask_idx) == 0 or len(actual_idx) == 0:
            continue
        sub_matrix = distance_matrix[np.ix_(mask_idx, actual_idx)]
        sub_row, sub_col = linear_sum_assignment(sub_matrix)
        row_ind.extend(mask_idx[sub_row])
        col_ind.extend(actual_idx[sub_col])

    if len(row_ind) != len(masked_agg) or len(col_ind) != len(unmasked_agg):
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

    mapping = masked_agg.iloc[row_ind].reset_index(drop=True)
    mapping = mapping.assign(
        unmasked_index=col_ind,
        unmasked_sku=unmasked_agg.iloc[col_ind]["unmasked_sku"].values,
        unmasked_manufacturer=unmasked_agg.iloc[col_ind]["unmasked_manufacturer"].values,
        unmasked_effect=unmasked_agg.iloc[col_ind]["effect_size"].values,
    )
    mapping["distance"] = distance_matrix[row_ind, col_ind]
    mapping = mapping.dropna(subset=["masked_sku", "unmasked_sku"]).reset_index(drop=True)
    mapping = mapping.sort_values(["masked_manufacturer", "distance", "unmasked_effect"], ascending=[True, False, False]).reset_index(drop=True)
    return mapping[
        [
            "masked_sku",
            "masked_manufacturer",
            "unmasked_sku",
            "unmasked_manufacturer",
            "distance",
            "unmasked_index",
        ]
    ]


def unmask_template(mapping: pd.DataFrame, template: pd.DataFrame, unmasked_df: pd.DataFrame) -> pd.DataFrame:
    mapping_dict = dict(zip(mapping["masked_sku"], mapping["unmasked_sku"]))
    manufacturer_dict = (
        unmasked_df[["sku_str", "manufacturer"]]
        .drop_duplicates()
        .set_index("sku_str")["manufacturer"]
        .to_dict()
    )
    index_dict = dict(zip(mapping["masked_sku"], mapping["unmasked_index"]))
    similarity_col = template.columns[3] if len(template.columns) > 3 else None

    target_col = template.columns[0]
    other_col = template.columns[1]
    manufacturer_col = template.columns[2] if len(template.columns) > 2 else None

    template[target_col] = template[target_col].map(mapping_dict).fillna(template[target_col])
    template[other_col] = template[other_col].map(mapping_dict).fillna(template[other_col])
    if manufacturer_col:
        template[manufacturer_col] = template[target_col].map(manufacturer_dict).fillna(
            template[manufacturer_col]
        )
    template["UnmaskedIndex"] = template[target_col].map(index_dict)
    return template


def main() -> None:
    masked_df = load_masked()
    unmasked_df = load_unmasked()
    mapping = build_mapping(masked_df, unmasked_df)
    mapping.to_csv(STATIC_DIR / "sku_mapping_masked_to_unmasked.csv", index=False)

    masked_template = pd.read_csv(MASKED_TEMPLATE)
    unmasked_template = unmask_template(mapping, masked_template, unmasked_df)
    unmasked_template.to_csv(STATIC_DIR / "sku_elasticity_template_unmasked.csv", index=False)
    print("Generated mapping and unmasked template in 'static/' directory.")


if __name__ == "__main__":
    main()
