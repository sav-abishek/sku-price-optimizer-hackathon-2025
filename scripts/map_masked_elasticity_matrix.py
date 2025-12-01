"""
One-off helper to convert the Round 1 masked elasticity matrix to its unmasked
representation by applying the SKU mapping generated earlier.

Run from the repository root:

    python scripts/map_masked_elasticity_matrix.py

Inputs:
    outputs/elasticity_matrix_masked.csv
    static/sku_mapping_masked_to_unmasked.csv
    Data Files Round 2/Sellout_Train.xlsx

Output:
    outputs/elasticity_matrix.csv  (overwrites the existing file)
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from compute_elasticity import add_derived_columns  # type: ignore

MASKED_MATRIX_PATH = ROOT / "outputs" / "elasticity_matrix_masked.csv"
MAPPING_PATH = ROOT / "static" / "sku_mapping_masked_to_unmasked.csv"
UNMASKED_SELL_OUT_PATH = ROOT / "Data Files Round 2" / "Sellout_Train.xlsx"
OUTPUT_PATH = ROOT / "outputs" / "elasticity_matrix.csv"


def load_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING_PATH)
    if mapping["masked_sku"].isna().any() or mapping["unmasked_sku"].isna().any():
        raise ValueError("Mapping file contains NaN entries; please regenerate the mapping.")
    return mapping


def load_unmasked_manufacturers() -> pd.Series:
    df = pd.read_excel(UNMASKED_SELL_OUT_PATH)
    df.columns = [c.lower() for c in df.columns]
    df = add_derived_columns(df)
    manufacturer_series = (
        df[["sku_str", "manufacturer"]]
        .drop_duplicates()
        .set_index("sku_str")["manufacturer"]
    )
    return manufacturer_series


def map_matrix() -> None:
    mapping = load_mapping()
    manufacturer_lookup = load_unmasked_manufacturers()

    masked_matrix = pd.read_csv(MASKED_MATRIX_PATH)
    target_col = "Target SKU"
    other_col = "Other SKU"
    manufacturer_col = "Manufacturer" if "Manufacturer" in masked_matrix.columns else None

    map_dict = dict(zip(mapping["masked_sku"], mapping["unmasked_sku"]))

    orig_target = masked_matrix[target_col].copy()
    orig_other = masked_matrix[other_col].copy()

    masked_matrix[target_col] = orig_target.map(map_dict)
    masked_matrix[other_col] = orig_other.map(map_dict)

    missing_targets = masked_matrix[target_col].isna().sum()
    missing_others = masked_matrix[other_col].isna().sum()
    if missing_targets or missing_others:
        print(
            f"Warning: leaving {missing_targets} target SKUs and {missing_others} other SKUs with masked names "
            "because they were not found in the mapping."
        )
        masked_matrix[target_col].fillna(orig_target, inplace=True)
        masked_matrix[other_col].fillna(orig_other, inplace=True)

    if manufacturer_col:
        masked_matrix[manufacturer_col] = masked_matrix[target_col].map(manufacturer_lookup).fillna(
            masked_matrix[manufacturer_col]
        )

    masked_matrix.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Wrote unmasked elasticity matrix with {len(masked_matrix)} rows to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    map_matrix()
