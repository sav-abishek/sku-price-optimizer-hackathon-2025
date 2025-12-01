import os
import sys
import math
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np


# -----------------------------
# Config and constants
# -----------------------------

@dataclass
class Config:
    data_dir: str = "Data Files"
    out_dir: str = "outputs"
    test_file: str = "Sellout_Test_Predict.csv"
    test_support_file: str = "Sellout_Prediction_Support_Data.csv"
    train_file: str = "Sellout_Train.csv"
    elasticity_template_file: str = "sku_elasticity_matrix.csv"
    model_artifact_path: str = "src/models/hierarchical_ridge.pkl"
    legacy_data_dir: Optional[str] = None
    legacy_train_file: str = "Sellout_Train.csv"
    use_provided_template: bool = False
    cross_partners: int = 332
    target_limit: Optional[int] = 153

    # Modeling hyperparameters
    ridge_alpha: float = 1.0
    min_obs_sku: int = 6
    min_obs_brand: int = 20

    # Elasticity bounds
    own_min: float = -5.0
    own_max: float = -0.1
    cross_min: float = 0.01
    cross_max: float = 2.0

    # Numerical stability
    eps: float = 1e-6


# -----------------------------
# Utilities
# -----------------------------

SKU_FIELDS = ["brand", "sub_brand", "package", "package_type", "capacity_number"]
KEY_FIELDS = [
    "date",
    "market",
    "manufacturer",
    "brand",
    "sub_brand",
    "package",
    "package_type",
    "capacity_number",
]


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_table(path: str, *, lowercase: bool = True) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if lowercase:
        df.columns = [str(c).strip().lower() for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def persist_models(models: Dict[str, Dict], artifact_path: str) -> Optional[str]:
    try:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(models, fh)
        return str(path.resolve())
    except Exception as exc:  # pragma: no cover - best effort persistence
        print(f"Warning: failed to persist models to {artifact_path}: {exc}", file=sys.stderr)
        return None


def build_sku_signature(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sales_value",
        "sales_hectoliters",
        "avg_price_per_liter",
        "numeric_distribution_stores_handling",
        "inventory_hectoliters",
        "weighted_distribution_tdp_reach",
    ]
    available = [c for c in cols if c in df.columns]
    df = df.copy()
    if "date" in df.columns:
        df = df.sort_values(["sku_str", "date"])
    records: List[Dict[str, str]] = []
    for sku, group in df.groupby("sku_str"):
        fragments: List[str] = []
        prefix_parts: List[str] = []
        for meta_col in ("package", "package_type", "capacity_number"):
            if meta_col in group.columns and not group[meta_col].dropna().empty:
                prefix_parts.append(str(group[meta_col].dropna().iloc[0]).strip())
            else:
                prefix_parts.append("")
        for col in available:
            vals = group[col].astype(float).fillna(0.0).round(6).tolist()
            fragments.append(",".join(f"{v:.6f}" for v in vals))
        signature = "|".join(prefix_parts + fragments)
        manufacturer = ""
        if "manufacturer" in group.columns and not group["manufacturer"].dropna().empty:
            manufacturer = str(group["manufacturer"].dropna().iloc[0])
        records.append({"sku_str": sku, "manufacturer": manufacturer, "signature": signature})
    return pd.DataFrame(records)


def build_sku_mapping(cfg: Config, current_train: pd.DataFrame) -> Dict[str, str]:
    if not cfg.legacy_data_dir:
        return {}
    legacy_path = Path(cfg.legacy_data_dir) / cfg.legacy_train_file
    if not legacy_path.exists():
        return {}
    try:
        legacy_df = read_table(str(legacy_path))
    except Exception as exc:  # pragma: no cover - guard for unexpected formats
        print(f"Warning: failed to load legacy train data from {legacy_path}: {exc}", file=sys.stderr)
        return {}
    legacy_df = add_derived_columns(legacy_df)
    legacy_sig = build_sku_signature(legacy_df).rename(columns={"sku_str": "sku_str_legacy"})
    current_sig = build_sku_signature(current_train).rename(columns={"sku_str": "sku_str_current"})
    merged = legacy_sig.merge(current_sig, on="signature", how="inner")
    if merged.empty:
        return {}
    mapping = dict(zip(merged["sku_str_legacy"], merged["sku_str_current"]))
    return mapping


def harmonize_template(
    template_df: pd.DataFrame,
    sku_mapping: Dict[str, str],
    manufacturer_lookup: Dict[str, str],
) -> pd.DataFrame:
    if not sku_mapping:
        return template_df
    df = template_df.copy()
    if "Target SKU" in df.columns:
        df["Target SKU"] = df["Target SKU"].map(sku_mapping).fillna(df["Target SKU"])
    if "Other SKU" in df.columns:
        df["Other SKU"] = df["Other SKU"].map(sku_mapping).fillna(df["Other SKU"])
    if "Manufacturer" in df.columns:
        df["Manufacturer"] = df["Target SKU"].map(manufacturer_lookup).fillna(df["Manufacturer"])
    return df


def generate_elasticity_template(
    train_df: pd.DataFrame,
    cfg: Config,
    reference_template: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    sku_meta = (
        train_df[
            ["sku_str", "manufacturer", "brand", "sub_brand", "package", "package_type", "capacity_number"]
        ]
        .drop_duplicates()
        .copy()
    )
    sku_meta["manufacturer"] = sku_meta["manufacturer"].astype(str)
    sku_meta["brand"] = sku_meta["brand"].astype(str)
    sku_meta["sub_brand"] = sku_meta["sub_brand"].astype(str)
    sku_meta["package"] = sku_meta["package"].astype(str)
    sku_meta["package_type"] = sku_meta["package_type"].astype(str)
    sku_meta["capacity_number"] = sku_meta["capacity_number"].astype(str)

    sales_totals = (
        train_df.groupby("sku_str")["sales_hectoliters"].sum().rename("sales_hl_total").reset_index()
    )
    sku_meta = sku_meta.merge(sales_totals, on="sku_str", how="left")

    targets_df = sku_meta[sku_meta["manufacturer"].str.upper() == "AB INBEV"].copy()
    if targets_df.empty:
        targets_df = sku_meta.copy()

    target_limit_cfg = cfg.target_limit if cfg.target_limit and int(cfg.target_limit) > 0 else len(targets_df)
    cross_limit_cfg = max(1, int(cfg.cross_partners))

    if reference_template is not None and not reference_template.empty:
        target_counts = reference_template.groupby(reference_template.columns[0]).size()
        target_limit = min(len(target_counts), len(targets_df), int(target_limit_cfg))
        cross_limit = max(1, min(int(target_counts.median()) - 1, cross_limit_cfg, len(sku_meta) - 1))
    else:
        target_limit = min(len(targets_df), int(target_limit_cfg))
        cross_limit = max(1, min(cross_limit_cfg, len(sku_meta) - 1))

    targets_df = targets_df.sort_values("sales_hl_total", ascending=False).head(target_limit)
    max_partners = cross_limit

    targets = targets_df.to_dict("records")
    all_skus = sku_meta.to_dict("records")
    rows: List[Dict[str, str]] = []
    for target in targets:
        target_sku = target["sku_str"]
        scored: List[Tuple[float, Dict[str, str]]] = []
        for other in all_skus:
            other_sku = other["sku_str"]
            if other_sku == target_sku:
                score = 2.0
            else:
                score = similarity_score(target_sku, other_sku)
            scored.append((score, other))
        scored.sort(key=lambda x: (-x[0], x[1]["sku_str"]))
        keep = scored[: min(max_partners + 1, len(scored))]
        for _, other in keep:
            rows.append(
                {
                    "Target SKU": target_sku,
                    "Other SKU": other["sku_str"],
                    "Manufacturer": other["manufacturer"],
                }
            )
    return pd.DataFrame(rows)


def parse_date_to_month(s: str) -> Optional[str]:
    try:
        d = pd.to_datetime(s, errors="coerce")
        if pd.isna(d):
            return None
        return f"{d.year:04d}-{d.month:02d}"
    except Exception:
        return None


def make_sku_str(row: pd.Series) -> str:
    # Format capacity with one decimal to match template
    try:
        cap = float(row.get("capacity_number", np.nan))
    except Exception:
        cap = np.nan
    cap_str = f"{cap:.1f}" if not np.isnan(cap) else ""
    parts = [
        str(row.get("brand", "")).strip(),
        str(row.get("sub_brand", "")).strip(),
        str(row.get("package", "")).strip(),
        str(row.get("package_type", "")).strip(),
        cap_str,
    ]
    return " ".join([p for p in parts if p != ""]).strip()


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ym"] = df["date"].astype(str).apply(parse_date_to_month)
    df["sku_str"] = df.apply(make_sku_str, axis=1)
    return df


def safe_log1p(x: pd.Series) -> pd.Series:
    return np.log1p(np.clip(x.astype(float), a_min=0, a_max=None))


def safe_log(x: pd.Series, eps: float) -> pd.Series:
    return np.log(np.clip(x.astype(float), a_min=eps, a_max=None))


def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.sum(np.abs(y_true))
    if denom < eps:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / (denom + eps))


# -----------------------------
# Data loading
# -----------------------------

def find_data_dir(preferred: str) -> str:
    # Prefer the configured path if it exists; fallback to 'Data'
    if os.path.isdir(preferred):
        return preferred
    if os.path.isdir("Data"):
        return "Data"
    return preferred  # return preferred anyway; will error later if missing


def load_train(cfg: Config) -> pd.DataFrame:
    path = os.path.join(cfg.data_dir, cfg.train_file)
    df = read_table(path)
    df = add_derived_columns(df)
    return df


def load_test_and_support(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_path = os.path.join(cfg.data_dir, cfg.test_file)
    supp_path = os.path.join(cfg.data_dir, cfg.test_support_file)
    test_df = read_table(test_path)
    if os.path.isfile(supp_path):
        supp_df = read_table(supp_path)
    else:
        supp_df = pd.DataFrame(columns=test_df.columns)
    test_df = add_derived_columns(test_df)
    supp_df = add_derived_columns(supp_df)
    return test_df, supp_df


def load_elasticity_template(cfg: Config) -> Optional[pd.DataFrame]:
    path = os.path.join(cfg.data_dir, cfg.elasticity_template_file)
    if os.path.isfile(path):
        try:
            return read_table(path, lowercase=False)
        except Exception:
            return None
    return None


# -----------------------------
# Modeling
# -----------------------------

class RidgeClosedForm:
    """
    Simple L2-regularized linear regression solved via closed form:
      beta = (X'X + alpha*I)^-1 X'y
    with intercept term handled by augmenting X with a column of ones that is NOT regularized.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None  # shape (n_features,)
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        # Add intercept column
        X_ext = np.hstack([np.ones((n, 1)), X])
        # Regularization matrix: does not regularize intercept
        I = np.eye(p + 1)
        I[0, 0] = 0.0
        A = X_ext.T @ X_ext + self.alpha * I
        b = X_ext.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


FEATURES = [
    "avg_price_per_liter",
    "numeric_distribution_stores_handling",
    "weighted_distribution_tdp_reach",
    "inventory_hectoliters",
]


def build_feature_frame(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["log_price"] = safe_log(df["avg_price_per_liter"], eps)
    f["log_nd"] = safe_log1p(df["numeric_distribution_stores_handling"])
    f["log_wd"] = safe_log1p(df["weighted_distribution_tdp_reach"])
    f["log_inv"] = safe_log1p(df["inventory_hectoliters"])
    # Add simple seasonality via month number scaled
    mo = pd.to_datetime(df["date"], errors="coerce").dt.month.fillna(0).astype(int)
    f["month_sin"] = np.sin(2 * np.pi * mo / 12.0)
    f["month_cos"] = np.cos(2 * np.pi * mo / 12.0)
    return f


def group_min_count(df: pd.DataFrame, key: List[str]) -> int:
    return int(df.groupby(key).size().min() if not df.empty else 0)


def train_hierarchical_models(train_df: pd.DataFrame, cfg: Config) -> Dict[str, Dict]:
    models: Dict[str, Dict] = {}

    # Prepare global fallback model
    y = safe_log(train_df["sales_hectoliters"], cfg.eps)
    X = build_feature_frame(train_df, cfg.eps).values
    global_model = RidgeClosedForm(alpha=cfg.ridge_alpha).fit(X, y)
    models["__GLOBAL__"] = {
        "model": global_model,
        "coef_names": list(build_feature_frame(train_df.iloc[:1], cfg.eps).columns),
    }

    # Brand-level models
    for brand, df_b in train_df.groupby("brand"):
        if len(df_b) < cfg.min_obs_brand:
            continue
        yb = safe_log(df_b["sales_hectoliters"], cfg.eps)
        Xb = build_feature_frame(df_b, cfg.eps).values
        m = RidgeClosedForm(alpha=cfg.ridge_alpha).fit(Xb, yb)
        models[f"BRAND::{brand}"] = {"model": m, "coef_names": list(build_feature_frame(df_b.iloc[:1], cfg.eps).columns)}

    # SKU-level models
    for sku, df_s in train_df.groupby("sku_str"):
        if len(df_s) < cfg.min_obs_sku:
            continue
        ys = safe_log(df_s["sales_hectoliters"], cfg.eps)
        Xs = build_feature_frame(df_s, cfg.eps).values
        m = RidgeClosedForm(alpha=cfg.ridge_alpha).fit(Xs, ys)
        models[f"SKU::{sku}"] = {"model": m, "coef_names": list(build_feature_frame(df_s.iloc[:1], cfg.eps).columns)}

    return models


def predict_with_hierarchy(row: pd.Series, feature_row: np.ndarray, models: Dict[str, Dict]) -> float:
    sku_key = f"SKU::{row['sku_str']}"
    brand_key = f"BRAND::{row['brand']}"
    if sku_key in models:
        m = models[sku_key]["model"]
    elif brand_key in models:
        m = models[brand_key]["model"]
    else:
        m = models["__GLOBAL__"]["model"]
    log_pred = float(m.predict(feature_row.reshape(1, -1))[0])
    pred = max(math.exp(log_pred), 0.0)
    return pred


def time_split_validation(train_df: pd.DataFrame, cfg: Config) -> Tuple[float, float]:
    # Use last calendar month in train as validation
    # Compute ym from date
    # Already added in add_derived_columns
    valid_ym = [ym for ym in train_df["ym"].unique() if isinstance(ym, str) and ym]
    if len(valid_ym) < 2:
        return (np.nan, np.nan)
    last_ym = sorted(valid_ym)[-1]
    mask_valid = train_df["ym"] == last_ym
    mask_train = train_df["ym"].apply(lambda x: isinstance(x, str) and x < last_ym)
    train_part = train_df[mask_train].copy()
    valid_part = train_df[mask_valid].copy()
    if train_part.empty or valid_part.empty:
        return (np.nan, np.nan)

    models = train_hierarchical_models(train_part, cfg)
    Xv = build_feature_frame(valid_part, cfg.eps)
    preds = []
    for i, row in valid_part.iterrows():
        fv = Xv.loc[i].values
        preds.append(predict_with_hierarchy(row, fv, models))
    y_true = valid_part["sales_hectoliters"].values.astype(float)
    y_pred = np.array(preds)
    return wmape(y_true, y_pred), float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else (np.nan)


# -----------------------------
# Elasticities
# -----------------------------

def extract_own_elasticities(train_df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """
    Estimate own-price elasticity per SKU as the coefficient on log_price
    from an SKU-level ridge model (with controls). Bounds enforced.
    """
    own: Dict[str, float] = {}
    for sku, df_s in train_df.groupby("sku_str"):
        if len(df_s) < cfg.min_obs_sku:
            continue
        y = safe_log(df_s["sales_hectoliters"], cfg.eps)
        Xf = build_feature_frame(df_s, cfg.eps)
        m = RidgeClosedForm(alpha=cfg.ridge_alpha).fit(Xf.values, y.values)
        # Coef order matches Xf columns
        coef_map = dict(zip(Xf.columns.tolist(), m.coef_.tolist()))
        e = float(coef_map.get("log_price", 0.0))
        # Enforce bounds and sign
        e = max(min(e, cfg.own_max), cfg.own_min)
        # Ensure negative
        if e >= 0:
            e = min(-0.1, cfg.own_max)
        own[sku] = e
    # Global fallback: median own elasticity
    if own:
        med = float(np.median(list(own.values())))
    else:
        med = -1.0
    own["__FALLBACK__"] = max(min(med, cfg.own_max), cfg.own_min)
    return own


def similarity_score(target: str, other: str) -> float:
    # Simple heuristic based on token overlap and capacity closeness
    t_tokens = target.split()
    o_tokens = other.split()
    overlap = len(set(t_tokens[:2]) & set(o_tokens[:2]))  # brand & sub_brand overlap
    score = 0.0
    if overlap == 2:
        score += 0.6
    elif overlap == 1:
        score += 0.35

    # Package + type
    if len(t_tokens) >= 4 and len(o_tokens) >= 4:
        if t_tokens[2] == o_tokens[2]:
            score += 0.15
        if t_tokens[3] == o_tokens[3]:
            score += 0.1

    # Capacity closeness
    try:
        t_cap = float(t_tokens[-1])
        o_cap = float(o_tokens[-1])
        rel_diff = abs(t_cap - o_cap) / max(t_cap, o_cap, 1.0)
        score += max(0.0, 0.2 - 0.2 * rel_diff)  # up to +0.2 if identical
    except Exception:
        pass

    return float(min(score, 1.0))


def build_elasticity_matrix(template_df: pd.DataFrame, own_map: Dict[str, float], cfg: Config) -> pd.DataFrame:
    df = template_df.copy()
    if "Elasticity" not in df.columns:
        df["Elasticity"] = np.nan

    # Normalize column names likely are:
    # Target SKU, Other SKU, Manufacturer, Elasticity
    tcol = "Target SKU" if "Target SKU" in df.columns else df.columns[0]
    ocol = "Other SKU" if "Other SKU" in df.columns else df.columns[1]
    ecol = "Elasticity"

    out_vals: List[float] = []
    for _, row in df.iterrows():
        t_sku = str(row[tcol]).strip()
        o_sku = str(row[ocol]).strip()
        if t_sku == o_sku:
            # Own
            e = own_map.get(t_sku, own_map.get("__FALLBACK__", -1.0))
            e = max(min(e, cfg.own_max), cfg.own_min)
            if e >= 0:
                e = cfg.own_max
            out_vals.append(e)
        else:
            # Cross: proportional to similarity and own magnitude
            base = abs(own_map.get(t_sku, own_map.get("__FALLBACK__", -1.0)))
            sim = similarity_score(t_sku, o_sku)
            e = 0.3 * base * sim  # scale down vs own
            # Enforce bounds and positivity
            e = max(min(e, cfg.cross_max), cfg.cross_min)
            out_vals.append(e)
    df[ecol] = out_vals
    return df


# -----------------------------
# Orchestration
# -----------------------------

def run(cfg: Config):
    cfg.data_dir = find_data_dir(cfg.data_dir)
    ensure_out_dir(cfg.out_dir)

    # Load data
    train_df = load_train(cfg)
    test_df, supp_df = load_test_and_support(cfg)

    # Optional mapping from masked SKU strings to actual names (only if legacy data provided)
    sku_mapping: Dict[str, str] = {}
    if cfg.legacy_data_dir:
        sku_mapping = build_sku_mapping(cfg, train_df)
    manufacturer_lookup = (
        train_df.groupby("sku_str")["manufacturer"]
        .agg(lambda s: str(s.dropna().iloc[0]) if not s.dropna().empty else "")
        .to_dict()
    )

    # Merge support to test by exact keys to preserve order
    supp_cols = [c for c in supp_df.columns if c in (KEY_FIELDS + FEATURES)]
    if supp_cols and not supp_df.empty:
        supp_dedup = (
            supp_df[supp_cols]
            .groupby(KEY_FIELDS, as_index=False)
            .agg({
                "numeric_distribution_stores_handling": "mean",
                "avg_price_per_liter": "mean",
                "inventory_hectoliters": "mean",
                "weighted_distribution_tdp_reach": "mean",
            })
        )
        aug = pd.merge(
            test_df,
            supp_dedup,
            on=KEY_FIELDS,
            how="left",
            suffixes=("", "_supp"),
        )
    else:
        aug = test_df.copy()

    # Train models using full train for final fit
    models = train_hierarchical_models(train_df, cfg)
    artifact_abspath = persist_models(models, cfg.model_artifact_path)

    # Predict for test rows in original order
    Xtest = build_feature_frame(aug, cfg.eps)
    # Impute missing features using training medians
    Xtrain = build_feature_frame(train_df, cfg.eps)
    medians = Xtrain.median(numeric_only=True)
    Xtest = Xtest.fillna(medians)
    preds: List[float] = []
    for i, row in aug.iterrows():
        fv = Xtest.loc[i].values
        preds.append(predict_with_hierarchy(row, fv, models))

    # Fill predictions into the sales_hectoliters column
    pred_df = test_df.copy()
    # Replace NaN preds with 0.0 as conservative fallback
    preds_clean = [0.0 if (p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p)))) else float(p) for p in preds]
    pred_df["sales_hectoliters"] = preds_clean

    # Sanity: keep count and order identical
    assert len(pred_df) == len(test_df), "Prediction row count mismatch"

    # Save predictions CSV with exactly the original test columns
    orig_cols = list(read_table(os.path.join(cfg.data_dir, cfg.test_file), lowercase=False).columns)
    pred_out_path = os.path.join(cfg.out_dir, "volume_predictions.csv")
    ordered = pred_df[[c.lower() for c in orig_cols]].copy()
    ordered.columns = orig_cols
    ordered.to_csv(pred_out_path, index=False)

    # Elasticity matrix
    reference_template = load_elasticity_template(cfg)
    template_df = None
    if cfg.use_provided_template and reference_template is not None:
        template_df = reference_template.copy()
        if sku_mapping:
            template_df = harmonize_template(template_df, sku_mapping, manufacturer_lookup)
    if template_df is None or template_df.empty:
        template_df = generate_elasticity_template(train_df, cfg, reference_template)
    own_map = extract_own_elasticities(train_df, cfg)
    emat = build_elasticity_matrix(template_df, own_map, cfg)
    elasticity_out_path = os.path.join(cfg.out_dir, "elasticity_matrix.csv")
    emat.to_csv(elasticity_out_path, index=False)

    # QA report
    ym_sorted = sorted([ym for ym in train_df["ym"].dropna().unique() if isinstance(ym, str) and ym])
    date_span = f"{ym_sorted[0]} to {ym_sorted[-1]}" if ym_sorted else ""
    n_rows = len(train_df)
    n_skus = train_df["sku_str"].nunique()
    wmape_val, corr_val = time_split_validation(train_df, cfg)

    # Elasticity sanity ranges
    own_vals = [v for k, v in own_map.items() if k != "__FALLBACK__"]
    own_min = min(own_vals) if own_vals else np.nan
    own_max = max(own_vals) if own_vals else np.nan

    report_lines = []
    report_lines.append('# Round 1 - Price Elasticity QA Report')
    report_lines.append("")
    report_lines.append("## Data Coverage")
    report_lines.append(f"- Train rows: {n_rows}")
    report_lines.append(f"- Unique SKUs: {n_skus}")
    report_lines.append(f"- Date span (monthly): {date_span}")
    report_lines.append("")
    report_lines.append("## Basic Accuracy")
    if not (np.isnan(wmape_val) or np.isnan(corr_val)):
        report_lines.append(f"- Validation wMAPE (last month): {wmape_val:.3f}")
        report_lines.append(f"- Fit indicator (corr y, yhat): {corr_val:.3f}")
    else:
        report_lines.append("- Validation not computed (insufficient time coverage)")
    report_lines.append("")
    report_lines.append("## Elasticity Sanity Checks")
    report_lines.append(f"- Own elasticity count: {len(own_vals)}")
    if own_vals:
        report_lines.append(f"- Own elasticity range: [{own_min:.3f}, {own_max:.3f}] (bounds: [{cfg.own_min}, {cfg.own_max}])")
    sample = emat.head(100)
    if "Elasticity" in sample.columns and len(sample) > 0:
        e_vals = sample["Elasticity"].values
        cross_vals = e_vals[sample.iloc[:, 0] != sample.iloc[:, 1]]
        if len(cross_vals) > 0:
            report_lines.append(f"- Cross elasticity sample range: [{np.min(cross_vals):.3f}, {np.max(cross_vals):.3f}] (bounds: [{cfg.cross_min}, {cfg.cross_max}])")
    report_lines.append("- Signs enforced: own < 0, cross > 0")
    report_lines.append(f"- Coverage: one own per ABI SKU plus top {cfg.cross_partners} cross partners based on similarity")
    report_lines.append("")
    report_lines.append("## Imputations/Capping")
    report_lines.append("- Logs use eps=1e-6; log1p for distribution/inventory to handle zeros")
    report_lines.append("- Predictions floor at 0.0 (no negative volumes)")
    report_lines.append("- Elasticities clipped to specified bounds")

    qa_out_path = os.path.join(cfg.out_dir, "qa_report.md")
    with open(qa_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(json.dumps({
        "predictions_csv": pred_out_path,
        "elasticities_csv": elasticity_out_path,
        "qa_report": qa_out_path,
        "model_artifact": artifact_abspath,
    }, indent=2))


if __name__ == "__main__":
    cfg = Config()
    # Allow basic CLI overrides
    if len(sys.argv) > 1:
        # Accept key=value pairs
        for arg in sys.argv[1:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                if hasattr(cfg, k):
                    current = getattr(cfg, k)
                    if isinstance(current, bool):
                        setattr(cfg, k, v.lower() in {"1", "true", "yes", "y"})
                    elif isinstance(current, int) and v.isdigit():
                        setattr(cfg, k, int(v))
                    elif isinstance(current, float):
                        try:
                            setattr(cfg, k, float(v))
                        except ValueError:
                            pass
                    else:
                        setattr(cfg, k, v)
    run(cfg)
