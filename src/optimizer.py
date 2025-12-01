"""
Round 2 pricing optimization utilities.

This module consumes the Sell-in/Sell-out datasets provided for Round 2,
reuses the precomputed Round 1 elasticity matrix, and optimizes SKU-level
prices to maximise MACO under the required constraints.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from compute_elasticity import (  # type: ignore
    Config as Round1Config,
    add_derived_columns,
    build_feature_frame,
    ensure_out_dir,
    extract_own_elasticities,
    predict_with_hierarchy,
    train_hierarchical_models,
)


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert arbitrary strings to an uppercase token for fuzzy joins."""
    if text is None:
        return ""
    text = (
        str(text)
        .replace("%", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace(".", " ")
    )
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().upper()


def map_pack(package: str, package_type: str) -> str:
    """Collapse Sell-out package descriptors into Sell-in pack codes."""
    package = str(package).upper().strip()
    package_type = str(package_type).upper().strip()
    if package == "LATA" and "NO RETORNABLE" in package_type:
        return "CAN"
    if package == "BOTELLA" and "RETORNABLE" in package_type:
        return "RB"
    if package == "BOTELLA" and "NO RETORNABLE" in package_type:
        return "NRB"
    if package == "BARRIL":
        return "KEG"
    return slugify(package)


def size_key(value) -> str:
    """Uniform representation of pack size."""
    try:
        f = float(value)
        if math.isfinite(f):
            if abs(f - round(f)) < 1e-6:
                f = int(round(f))
            return str(f)
    except Exception:
        pass
    return slugify(value)


def classify_size_group(pack: str, size_ml: float) -> str:
    """Classify SKUs into size groups for architecture hierarchy checks."""
    size_ml = float(size_ml or 0)
    pack = pack.upper()
    if size_ml < 300:
        return "SMALL"
    if pack == "CAN" and 300 <= size_ml <= 399:
        return "REGULAR"
    if pack in {"RB", "NRB"} and 300 <= size_ml <= 599:
        return "REGULAR"
    return "LARGE"


def compose_brand_slug(brand, sub_brand=None) -> str:
    """Combine brand/sub-brand values into a stable slug."""
    parts: List[str] = []
    for value in (brand, sub_brand):
        if value is None:
            continue
        s = str(value).strip()
        if not s or s.lower() == "nan":
            continue
        parts.append(s)
    if not parts:
        return ""
    return slugify(" ".join(parts))


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass
class Round2Config:
    """User-configurable knobs for the pricing optimizer."""

    data_dir: Path = Path("Data Files Round 2")
    out_dir: Path = Path("outputs/round2")
    elasticity_matrix_path: Path = Path("outputs/elasticity_matrix.csv")
    model_artifact_path: Path = Path("src/models/hierarchical_ridge.pkl")
    elasticity_template_path: Path = Path("static/sku_elasticity_template_unmasked.csv")

    pinc_target: float = 0.03
    price_floor_delta: float = -300.0
    price_ceiling_delta: float = 500.0
    price_step: float = 50.0

    vat: float = 0.19
    vilc_growth: float = 0.0378

    own_min: float = -5.0
    own_max: float = -0.1
    cross_min: float = 0.01
    cross_max: float = 2.0

    max_iter: int = 200
    tol: float = 1e-6
    # Prefer MILP solver aligned to the deep plan document. If unavailable,
    # fall back to the deterministic heuristic.
    use_milp_solver: bool = True

    round1: Round1Config = field(default_factory=Round1Config)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


class Round2DataLoader:
    """Load Round 2 Sell-in and Sell-out datasets."""

    def __init__(self, cfg: Round2Config):
        self.cfg = cfg
        self.base_path = Path(cfg.data_dir)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Missing data directory: {self.base_path}")

        # Align Round 1 config defaults
        self.cfg.round1.min_obs_sku = 6
        self.cfg.round1.min_obs_brand = 20
        self.cfg.round1.ridge_alpha = 1.0
        self.cfg.round1.eps = 1e-6

    def _read_excel(self, filename: str) -> pd.DataFrame:
        path = self.base_path / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_excel(path)

    def load_sellout(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = self._read_excel("Sellout_Train.xlsx").rename(columns=str.lower)
        test = self._read_excel("Sellout_Test.xlsx").rename(columns=str.lower)
        train = add_derived_columns(train)
        test = add_derived_columns(test)
        return train, test

    def load_sellin(self) -> pd.DataFrame:
        sellin = self._read_excel("Sellin.xlsx")
        sellin = sellin[sellin["Scenario"] == "LE 8+4"].copy()
        sellin["brand_slug"] = sellin["Brand"].apply(slugify)
        sellin["pack_slug"] = sellin["Pack"].apply(slugify)
        sellin["size_key"] = sellin["Size"].apply(size_key)
        sellin["key"] = (
            sellin["brand_slug"] + "|" + sellin["pack_slug"] + "|" + sellin["size_key"]
        )
        return sellin

    def load_price_list(self) -> pd.DataFrame:
        price = self._read_excel("Price_List.xlsx")
        price["brand_slug"] = price["sku"].apply(
            lambda s: slugify(" ".join(str(s).split()[:-2])) if isinstance(s, str) else ""
        )
        price["pack_slug"] = price["sku"].apply(
            lambda s: slugify(str(s).split()[-2]) if isinstance(s, str) and len(str(s).split()) >= 2 else ""
        )
        price["size_key"] = price["sku"].apply(
            lambda s: size_key(str(s).split()[-1]) if isinstance(s, str) and len(str(s).split()) >= 1 else ""
        )
        price["key"] = (
            price["brand_slug"] + "|" + price["pack_slug"] + "|" + price["size_key"]
        )
        price.sort_values("Date", inplace=True)
        price = price.groupby("key", as_index=False).last()
        return price


# ---------------------------------------------------------------------------
# Elasticity computation
# ---------------------------------------------------------------------------


def compute_elasticities(
    cfg: Round2Config, train_df: pd.DataFrame
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Dict]]:
    """
    Load the precomputed elasticity matrix and Round 1 models when available.
    Falls back to retraining the hierarchy if the artifacts are missing.
    Returns own elasticities, the cross-elasticity dataframe, and fitted models.
    """

    matrix_path = cfg.elasticity_matrix_path
    model_path = cfg.model_artifact_path
    if matrix_path and matrix_path.is_file():
        cross_matrix = pd.read_csv(matrix_path)
        cross_matrix.columns = [str(c).strip() for c in cross_matrix.columns]
        diag = cross_matrix[cross_matrix["Target SKU"] == cross_matrix["Other SKU"]]
        diag = diag.dropna(subset=["Elasticity"])
        own_map: Dict[str, float] = {
            row["Target SKU"]: float(row["Elasticity"]) for _, row in diag.iterrows()
        }
        if own_map:
            median_val = float(np.median(list(own_map.values())))
        else:
            median_val = -1.0
        own_map["__FALLBACK__"] = max(min(median_val, cfg.round1.own_max), cfg.round1.own_min)
        models: Dict[str, Dict]
        if model_path and Path(model_path).is_file():
            with open(model_path, "rb") as fh:
                # Older artifacts were pickled when compute_elasticity.py was executed as __main__.
                # Ensure the class is discoverable under that module name before loading.
                import types
                from compute_elasticity import RidgeClosedForm  # type: ignore

                main_module = sys.modules.get("__main__")
                if main_module is None:
                    main_module = types.ModuleType("__main__")
                    sys.modules["__main__"] = main_module
                setattr(main_module, "RidgeClosedForm", RidgeClosedForm)
                models = pickle.load(fh)
        else:
            models = train_hierarchical_models(train_df, cfg.round1)
        return own_map, cross_matrix, models

    # Fallback: recompute as before
    models = train_hierarchical_models(train_df, cfg.round1)
    own_map = extract_own_elasticities(train_df, cfg.round1)

    template: Optional[pd.DataFrame] = None
    template_path = cfg.elasticity_template_path
    if template_path and Path(template_path).is_file():
        template = pd.read_csv(template_path)
        template.columns = [str(c).strip() for c in template.columns]

    if template is None or template.empty:
        sku_meta = (
            train_df[
                ["sku_str", "manufacturer", "brand", "sub_brand", "package", "package_type", "capacity_number"]
            ]
            .drop_duplicates()
            .copy()
        )
        sku_meta["brand_slug"] = sku_meta.apply(
            lambda r: compose_brand_slug(r.get("brand"), r.get("sub_brand")), axis=1
        )
        sku_meta["pack_slug"] = sku_meta.apply(lambda r: map_pack(r["package"], r["package_type"]), axis=1)
        sku_meta["size_key"] = sku_meta["capacity_number"].apply(size_key)
        abi_meta = sku_meta[sku_meta["manufacturer"].str.upper() == "AB INBEV"]
        rows: List[Tuple[str, str, str]] = []
        for _, target in abi_meta.iterrows():
            target_tokens = set(target["brand_slug"].split())
            candidates = sku_meta.copy()
            candidates["score"] = candidates["brand_slug"].apply(
                lambda s: len(target_tokens & set(s.split())) / max(len(target_tokens | set(s.split())), 1)
            )
            candidates.loc[candidates["pack_slug"] == target["pack_slug"], "score"] += 0.2
            top = candidates.nlargest(12, "score")
            for _, other in top.iterrows():
                rows.append((target["sku_str"], other["sku_str"], other["manufacturer"]))

        template = pd.DataFrame(rows, columns=["Target SKU", "Other SKU", "Manufacturer"]).drop_duplicates()

    manufacturer_lookup = (
        train_df.groupby("sku_str")["manufacturer"]
        .agg(lambda s: str(s.dropna().iloc[0]) if not s.dropna().empty else "")
        .to_dict()
    )
    if "Manufacturer" in template.columns:
        template["Manufacturer"] = template["Target SKU"].map(manufacturer_lookup).fillna(
            template["Manufacturer"]
        )
    else:
        template["Manufacturer"] = template["Target SKU"].map(manufacturer_lookup)

    from compute_elasticity import build_elasticity_matrix  # type: ignore

    cross_matrix = build_elasticity_matrix(template, own_map, cfg.round1)
    cross_matrix["Elasticity"] = cross_matrix["Elasticity"].fillna(cfg.cross_min)
    return own_map, cross_matrix, models


# ---------------------------------------------------------------------------
# Sell-in aggregation and mapping
# ---------------------------------------------------------------------------


def build_sellin_summary(sellin: pd.DataFrame, price_list: pd.DataFrame, cfg: Round2Config) -> pd.DataFrame:
    """Aggregate financial metrics to SKU level and merge with price metadata."""

    agg = sellin.groupby(
        ["key", "Brand", "Pack", "Size", "brand_slug", "pack_slug", "size_key"], as_index=False
    ).agg(
        {
            "Volume": "sum",
            "GTO": "sum",
            "Discount": "sum",
            "Excise": "sum",
            "NR": "sum",
            "VIC": "sum",
            "VLC": "sum",
            "MACOstd": "sum",
        }
    )

    agg.rename(columns={"Volume": "volume_hl"}, inplace=True)
    # Derive VILC (variable industrial + logistics cost) robustly from data.
    # Prefer cost implied by MACO: cost = NR - MACOstd; fallback to VIC+VLC (abs).
    implied_cost = agg["NR"] - agg["MACOstd"]
    fallback_cost = agg["VIC"].abs().fillna(0.0) + agg["VLC"].abs().fillna(0.0)
    agg["vilc_base"] = implied_cost.where(implied_cost.notna() & (implied_cost > 0), fallback_cost)
    agg["vilc_target"] = agg["vilc_base"] * (1.0 + cfg.vilc_growth)
    agg["discount_pct"] = np.where(agg["GTO"].abs() > 1e-9, agg["Discount"] / agg["GTO"], 0.0).clip(-1, 1)
    agg["excise_pct"] = np.where(agg["GTO"].abs() > 1e-9, agg["Excise"] / agg["GTO"], 0.0).clip(0, 1)

    merged = pd.merge(
        agg,
        price_list[
            [
                "key",
                "Category",
                "PTR",
                "PTC",
                "Mark_up",
                "No_Units",
                "No_Units_Liters",
                "Capacity_ml",
            ]
        ],
        on="key",
        how="left",
        suffixes=("", "_price"),
    )
    merged.rename(columns={"PTC": "ptc_base", "Mark_up": "markup"}, inplace=True)
    merged["ptc_base"] = merged["ptc_base"].fillna(
        merged["GTO"] / merged["No_Units"].replace({0: np.nan})
    )
    merged["markup"] = merged["markup"].fillna(0.2)
    merged["size_ml"] = pd.to_numeric(merged["Size"], errors="coerce").fillna(
        pd.to_numeric(merged["Capacity_ml"], errors="coerce")
    )
    merged["size_ml"] = merged["size_ml"].fillna(330.0)
    merged["volume_units"] = (
        merged["volume_hl"] * 100000.0 / merged["size_ml"].replace({0: np.nan})
    ).fillna(0.0)
    merged["ptc_base"] = merged["ptc_base"].fillna(
        merged["NR"] / merged["volume_units"].replace({0: np.nan})
    )
    merged["ptc_base"] = merged["ptc_base"].fillna(1000.0)
    merged = merged[(merged["volume_hl"] > 0) & (merged["ptc_base"] > 0)].copy()
    merged["size_group"] = merged.apply(lambda r: classify_size_group(str(r["Pack"]), r["size_ml"]), axis=1)
    merged["segment"] = merged["Category"].fillna("Unclassified").apply(slugify)
    return merged


def map_sellout_to_sellin(sellout: pd.DataFrame, sellin_summary: pd.DataFrame) -> Dict[str, str]:
    """Build a fuzzy mapping from Sell-out SKU strings to Sell-in keys."""

    sellout_meta = (
        sellout[
            ["sku_str", "brand", "sub_brand", "package", "package_type", "capacity_number"]
        ]
        .drop_duplicates()
        .copy()
    )
    sellout_meta["brand_slug"] = sellout_meta.apply(
        lambda r: compose_brand_slug(r.get("brand"), r.get("sub_brand")), axis=1
    )
    sellout_meta["pack_slug"] = sellout_meta.apply(lambda r: map_pack(r["package"], r["package_type"]), axis=1)
    sellout_meta["size_key"] = sellout_meta["capacity_number"].apply(size_key)

    sellin_keys = sellin_summary[["key", "brand_slug", "pack_slug", "size_key"]].drop_duplicates()
    sellin_keys["tokens"] = sellin_keys["brand_slug"].apply(lambda s: set(s.split()))

    mapping: Dict[str, str] = {}
    for _, row in sellout_meta.iterrows():
        candidates = sellin_keys[
            (sellin_keys["pack_slug"] == row["pack_slug"]) & (sellin_keys["size_key"] == row["size_key"])
        ]
        if candidates.empty:
            candidates = sellin_keys[sellin_keys["size_key"] == row["size_key"]]
        target_tokens = set(row["brand_slug"].split())
        best_key = None
        best_score = 0.0
        for _, cand in candidates.iterrows():
            score = len(target_tokens & cand["tokens"]) / max(len(target_tokens | cand["tokens"]), 1)
            if score > best_score:
                best_score = score
                best_key = cand["key"]
        if best_key and best_score >= 0.25:
            mapping[row["sku_str"]] = best_key
    return mapping


# ---------------------------------------------------------------------------
# Sell-out volume prediction
# ---------------------------------------------------------------------------


def predict_sellout_volumes(
    cfg: Round2Config, models: Dict[str, Dict], train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Score Sell-out test rows using hierarchical models."""
    Xtrain = build_feature_frame(train_df, cfg.round1.eps)
    medians = Xtrain.median(numeric_only=True)
    Xtest = build_feature_frame(test_df, cfg.round1.eps).fillna(medians)
    preds: List[float] = []
    for i, row in test_df.iterrows():
        fv = Xtest.loc[i].values
        preds.append(predict_with_hierarchy(row, fv, models))
    scored = test_df.copy()
    scored["predicted_sales_hectoliters"] = preds
    return scored


# ---------------------------------------------------------------------------
# Optimization engine
# ---------------------------------------------------------------------------


class PricingOptimizer:
    """Encapsulates KPI projections and SLSQP optimisation."""

    def __init__(
        self,
        cfg: Round2Config,
        sellin_summary: pd.DataFrame,
        own_map: Dict[str, float],
        elasticity_matrix: pd.DataFrame,
        sku_mapping: Dict[str, str],
        sellout_train: pd.DataFrame,
        sellout_test_pred: pd.DataFrame,
        mapped_keys: Optional[Set[str]] = None,
    ):
        self.cfg = cfg
        mapped_keys = mapped_keys or set(sellin_summary["key"])
        self.sellin = sellin_summary[sellin_summary["key"].isin(mapped_keys)].copy()
        self.unmapped = sellin_summary[~sellin_summary["key"].isin(mapped_keys)].copy()
        self.own_map = own_map
        self.elasticity_matrix = elasticity_matrix
        self.sku_mapping = sku_mapping
        self.sellout_train = sellout_train
        self.sellout_test_pred = sellout_test_pred
        self._prepare()

    def _prepare(self):
        """Pre-compute base metrics and lookup tables."""
        self.sellin.set_index("key", inplace=True, drop=False)
        self.sellin["volume_units"] = (
            self.sellin["volume_hl"] * 100000.0 / self.sellin["size_ml"].replace({0: np.nan})
        ).fillna(0.0)

        self.abi_keys = self.sellin.index.unique().tolist()
        self.vector_index = {key: idx for idx, key in enumerate(self.abi_keys)}

        self.base_volume = self.sellin["volume_hl"].sum() + self.unmapped["volume_hl"].sum()
        self.base_units = self.sellin["volume_units"].sum() + (
            self.unmapped.get("volume_units", pd.Series(dtype=float)).sum()
        )
        self.base_maco = self.sellin["MACOstd"].sum() + self.unmapped["MACOstd"].sum()

        # Portfolio market share baseline from Sell-out predictions
        mapped = self.sellout_test_pred.copy()
        mapped["sellin_key"] = mapped["sku_str"].map(self.sku_mapping)
        mapped["is_abi"] = mapped["sellin_key"].isin(self.abi_keys)
        abi_vol = mapped.loc[mapped["is_abi"], "predicted_sales_hectoliters"].sum()
        competitor_vol = mapped.loc[~mapped["is_abi"], "predicted_sales_hectoliters"].sum()
        self.base_industry_volume = abi_vol + competitor_vol
        self.base_market_share = abi_vol / max(self.base_industry_volume, 1e-6)

        # Own elasticity per Sell-in key
        fallback = self.own_map.get("__FALLBACK__", -1.0)
        if fallback is None or (isinstance(fallback, float) and np.isnan(fallback)):
            fallback = -1.0
        self.own_lookup = {key: fallback for key in self.abi_keys}
        for sku_str, sellin_key in self.sku_mapping.items():
            if sellin_key in self.own_lookup and sku_str in self.own_map:
                val = self.own_map[sku_str]
                if np.isnan(val):
                    val = fallback
                self.own_lookup[sellin_key] = val

        # Cross-elasticity lookups
        self.cross_lookup: Dict[str, List[Tuple[str, float]]] = {}
        self.competitor_effect: Dict[str, List[Tuple[str, float]]] = {}
        for _, row in self.elasticity_matrix.iterrows():
            target_key = self.sku_mapping.get(row["Target SKU"])
            other_key = self.sku_mapping.get(row["Other SKU"])
            val = float(row["Elasticity"])
            if not target_key or not other_key:
                continue
            if target_key == other_key:
                continue
            if target_key in self.abi_keys and other_key in self.abi_keys:
                self.cross_lookup.setdefault(target_key, []).append((other_key, val))
            elif target_key not in self.abi_keys and other_key in self.abi_keys:
                self.competitor_effect.setdefault(target_key, []).append((other_key, val))

    # --- KPI projections ----------------------------------------------------

    def _portfolio_pinc(self, pct_changes: np.ndarray) -> float:
        prices = self.sellin.loc[self.abi_keys, "ptc_base"].values.astype(float)
        new_prices = prices * (1.0 + pct_changes)
        weights = self.sellin.loc[self.abi_keys, "volume_units"].values.astype(float)
        base = np.sum(prices * weights)
        new = np.sum(new_prices * weights)
        return (new / max(base, 1e-6)) - 1.0

    def _volume_projection(
        self, pct_changes: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        base_vol = self.sellin.loc[self.abi_keys, "volume_hl"].values.astype(float)
        own = np.array([self.own_lookup[k] for k in self.abi_keys], dtype=float)
        volumes = np.maximum(0.0, base_vol * (1.0 + own * pct_changes))

        # Cross-effects between ABI SKUs
        for i, key in enumerate(self.abi_keys):
            cross_term = 0.0
            for other_key, e in self.cross_lookup.get(key, []):
                j = self.vector_index.get(other_key)
                if j is None:
                    continue
                cross_term += e * pct_changes[j]
            adj = 1.0 + own[i] * pct_changes[i] + cross_term
            adj = max(adj, 0.05)
            volumes[i] = base_vol[i] * adj

        abi_map = {k: volumes[idx] for k, idx in self.vector_index.items()}

        # Competitor volume response (target competitor, other ABI)
        competitor_map: Dict[str, float] = {}
        base_pred = self.sellout_test_pred.copy()
        base_pred["sellin_key"] = base_pred["sku_str"].map(self.sku_mapping)
        base_comp = (
            base_pred[~base_pred["sellin_key"].isin(self.abi_keys) & base_pred["sellin_key"].notna()]
            .groupby("sellin_key")["predicted_sales_hectoliters"]
            .sum()
        )
        for key, base_val in base_comp.items():
            delta = 0.0
            for abi_key, e in self.competitor_effect.get(key, []):
                idx = self.vector_index.get(abi_key)
                if idx is None:
                    continue
                delta += e * pct_changes[idx]
            competitor_map[key] = max(0.0, base_val * (1.0 + delta))

        return volumes, abi_map, competitor_map

    def _financial_projection(self, pct_changes: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
        volumes, abi_map, competitor_map = self._volume_projection(pct_changes)
        rows = []
        maco_total = 0.0
        nr_total = 0.0
        for key, vol, pct in zip(self.abi_keys, volumes, pct_changes):
            row = self.sellin.loc[key]
            # Enforce discrete price step rounding on projection for consistency
            price_new = round(max(row["ptc_base"] * (1.0 + pct), 0.0) / self.cfg.price_step) * self.cfg.price_step
            # Approximate NR/HL scales linearly with PTC when taxes/markup are constant
            nr_hl_base = row["NR"] / max(row["volume_hl"], 1e-6)
            price_ratio = price_new / max(row["ptc_base"], 1e-6)
            nr_hl_new = nr_hl_base * price_ratio
            nr_new = max(0.0, nr_hl_new * vol)
            vilc_new = max(0.0, (row["vilc_target"] / max(row["volume_hl"], 1e-6)) * vol)
            maco_new = max(0.0, nr_new - vilc_new)
            nr_total += nr_new
            maco_total += maco_new

            rows.append(
                {
                    "key": key,
                    "brand": row["Brand"],
                    "pack": row["Pack"],
                    "size": row["Size"],
                    "segment": row["segment"],
                    "size_group": row["size_group"],
                    "price_base": row["ptc_base"],
                    "price_new": price_new,
                    "price_pct_change": price_new / max(row["ptc_base"], 1e-6) - 1.0,
                    "volume_base": row["volume_hl"],
                    "volume_new": vol,
                    "nr_base": row["NR"],
                    "nr_new": nr_new,
                    "maco_base": row["MACOstd"],
                    "maco_new": maco_new,
                }
            )

        summary = pd.DataFrame(rows)
        unmapped_volume = self.unmapped["volume_hl"].sum()
        unmapped_nr = self.unmapped["NR"].sum()
        unmapped_maco = self.unmapped["MACOstd"].sum()
        maco_total += unmapped_maco
        nr_total += unmapped_nr
        total_volume = summary["volume_new"].sum() + unmapped_volume
        kpis = {
            "maco_total": float(maco_total),
            "nr_total": float(nr_total),
            "abi_volume": float(total_volume),
            "competitor_volume": float(sum(competitor_map.values())),
        }
        return summary, kpis

    # --- Constraints --------------------------------------------------------

    def _market_share_constraint(self, pct_changes: np.ndarray) -> float:
        _, kpis = self._financial_projection(pct_changes)
        industry_new = kpis["abi_volume"] + kpis["competitor_volume"]
        share_new = kpis["abi_volume"] / max(industry_new, 1e-6)
        return share_new - (self.base_market_share - 0.005)

    def _industry_volume_constraint(self, pct_changes: np.ndarray) -> float:
        _, kpis = self._financial_projection(pct_changes)
        industry_new = kpis["abi_volume"] + kpis["competitor_volume"]
        return industry_new - 0.99 * self.base_industry_volume

    def evaluate_constraints(self, pct_changes: np.ndarray) -> Dict[str, float]:
        summary, kpis = self._financial_projection(pct_changes)
        industry_new = kpis["abi_volume"] + kpis["competitor_volume"]
        share_new = kpis["abi_volume"] / max(industry_new, 1e-6)
        maco_delta = kpis["maco_total"] - self.base_maco
        volume_delta = (kpis["abi_volume"] / max(self.base_volume, 1e-6)) - 1.0
        industry_delta = (industry_new / max(self.base_industry_volume, 1e-6)) - 1.0
        share_delta = self.base_market_share - share_new

        details = pd.DataFrame(
            {
                "segment": self.sellin["segment"],
                "size_group": self.sellin["size_group"],
                "nr_hl": self.sellin["NR"] / self.sellin["volume_hl"].replace({0: np.nan}),
            }
        ).dropna()
        segment_nr = details.groupby("segment")["nr_hl"].mean().to_dict()
        size_group_nr = details.groupby("size_group")["nr_hl"].mean().to_dict()

        return {
            "maco_delta": maco_delta,
            "volume_ratio": kpis["abi_volume"] / max(self.base_volume, 1e-6),
            "industry_ratio": industry_new / max(self.base_industry_volume, 1e-6),
            "market_share": share_new,
            "share_drop": share_delta,
            "pinc_actual": self._portfolio_pinc(pct_changes),
            "segment_nr_hl": segment_nr,
            "size_group_nr_hl": size_group_nr,
        }

    def _constraint_penalty(self, pct_changes: np.ndarray) -> float:
        _, kpis = self._financial_projection(pct_changes)
        maco_term = max(0.0, self.base_maco - kpis["maco_total"])  # COP shortfall
        vol_short = max(0.0, 0.99 * self.base_volume - kpis["abi_volume"])  # HL shortfall
        vol_excess = max(0.0, kpis["abi_volume"] - 1.05 * self.base_volume)
        industry_new = kpis["abi_volume"] + kpis["competitor_volume"]
        share_new = kpis["abi_volume"] / max(industry_new, 1e-6)
        share_short = max(0.0, (self.base_market_share - 0.005) - share_new)
        industry_short = max(0.0, 0.99 * self.base_industry_volume - industry_new)
        # Weighted sum: prioritise MACO non-decrease and share; scale units
        penalty = (
            maco_term / 1e6
            + 1e5 * share_short
            + 1e-3 * vol_short
            + 1e-3 * vol_excess
            + 1e-3 * industry_short
        )
        return float(penalty)

    # --- Discrete repair ---------------------------------------------------

    def _steps_from_pct(self, pct: np.ndarray) -> np.ndarray:
        base_price_arr = self.sellin.loc[self.abi_keys, "ptc_base"].values.astype(float)
        steps = np.rint((pct * np.maximum(base_price_arr, 1e-6)) / self.cfg.price_step)
        # Clip to floor/ceiling step bounds
        step_lb = np.ceil(self.cfg.price_floor_delta / max(self.cfg.price_step, 1.0))
        step_ub = np.floor(self.cfg.price_ceiling_delta / max(self.cfg.price_step, 1.0))
        return np.clip(steps, step_lb, step_ub)

    def _pct_from_steps(self, steps: np.ndarray) -> np.ndarray:
        base_price_arr = self.sellin.loc[self.abi_keys, "ptc_base"].values.astype(float)
        return (self.cfg.price_step * steps) / np.maximum(base_price_arr, 1.0)

    def _greedy_repair(self, steps: np.ndarray, max_iters: int = 250) -> np.ndarray:
        # Hill-climb to reduce penalty and/or improve MACO
        pct = self._pct_from_steps(steps)
        best_pen = self._constraint_penalty(pct)
        _, k_best = self._financial_projection(pct)
        best_maco = float(k_best.get("maco_total", 0.0))
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(len(steps)):
                for d in (1, -1):
                    new_steps = steps.copy()
                    new_steps[i] += d
                    # Keep within bounds
                    if new_steps[i] * self.cfg.price_step < self.cfg.price_floor_delta:
                        continue
                    if new_steps[i] * self.cfg.price_step > self.cfg.price_ceiling_delta:
                        continue
                    pct_try = self._pct_from_steps(new_steps)
                    pen = self._constraint_penalty(pct_try)
                    _, k_try = self._financial_projection(pct_try)
                    maco_try = float(k_try.get("maco_total", -1e18))
                    if pen < best_pen - 1e-6 or (abs(pen - best_pen) <= 1e-6 and maco_try > best_maco + 1.0):
                        steps = new_steps
                        best_pen = pen
                        best_maco = maco_try
                        improved = True
                        break
                if improved:
                    break
        return steps

    # --- Optimisation -------------------------------------------------------

    def optimize(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        # Attempt MILP-based optimisation per the plan document when enabled
        pct_changes: Optional[np.ndarray] = None
        used_milp = False
        if self.cfg.use_milp_solver:
            try:
                # Try target PINC and, if infeasible, step down in 0.5pp increments to 0.5%
                target = float(self.cfg.pinc_target)
                candidates = []
                for v in [target, 0.015, 0.0125, 0.01, 0.0075, 0.005, 0.003, 0.002]:
                    if v > 0:
                        candidates.append(round(v, 5))
                for pinc_try in candidates:
                    # Try OR-Tools first, then PuLP
                    steps_pct = self._optimize_milp_ortools(pinc_target=pinc_try)
                    if steps_pct is None:
                        steps_pct = self._optimize_milp(pinc_target=pinc_try)
                    if steps_pct is None:
                        continue
                    # Evaluate and repair with discrete scaling if needed
                    pen = self._constraint_penalty(steps_pct)
                    if pen > 1e-3:
                        # Binary search over a scale s in [0,1] with integer rounding of steps
                        base_price_arr = self.sellin.loc[self.abi_keys, "ptc_base"].values.astype(float)
                        # Recover integer steps from pct vector
                        steps_vec = np.rint((steps_pct * np.maximum(base_price_arr, 1e-6)) / self.cfg.price_step)
                        low, high = 0.0, 1.0
                        best = None
                        for _ in range(12):
                            mid = (low + high) / 2.0
                            scaled_steps = np.rint(steps_vec * mid)
                            pct_mid = (self.cfg.price_step * scaled_steps) / np.maximum(base_price_arr, 1.0)
                            pen_mid = self._constraint_penalty(pct_mid)
                            if pen_mid <= 1e-3 and np.any(np.abs(scaled_steps) > 0):
                                best = pct_mid
                                low = mid
                            else:
                                high = mid
                        if best is not None:
                            pct_changes = best
                            used_milp = True
                            break
                    else:
                        pct_changes = steps_pct
                        used_milp = True
                        break
            except Exception:
                pct_changes = None

        if pct_changes is None:
            # Fallback to heuristic search
            base_price_arr = self.sellin.loc[self.abi_keys, "ptc_base"].values.astype(float)
            weight_arr = self.sellin.loc[self.abi_keys, "volume_units"].values.astype(float)
            lb = self.cfg.price_floor_delta / np.maximum(base_price_arr, 1.0)
            ub = self.cfg.price_ceiling_delta / np.maximum(base_price_arr, 1.0)

            target_total = np.sum(base_price_arr * weight_arr) * (1.0 + self.cfg.pinc_target)

            def balance_to_target(pct: np.ndarray) -> np.ndarray:
                pct = np.clip(pct, lb, ub)
                for _ in range(18):
                    current_total = np.sum(base_price_arr * (1.0 + pct) * weight_arr)
                    diff = target_total - current_total
                    if abs(diff) < 1e-6:
                        break
                    free = (pct > lb + 1e-6) & (pct < ub - 1e-6)
                    if not np.any(free):
                        break
                    adjust = diff / max(np.sum(base_price_arr[free] * weight_arr[free]), 1e-9)
                    pct[free] = np.clip(pct[free] + adjust, lb[free], ub[free])
                return pct

            n = len(self.abi_keys)
            own = np.array([self.own_lookup[k] for k in self.abi_keys], dtype=float)
            vol_hl = self.sellin.loc[self.abi_keys, "volume_hl"].values.astype(float)
            nr_hl = (self.sellin.loc[self.abi_keys, "NR"].values.astype(float) / np.maximum(vol_hl, 1e-6))
            vilc_per_hl = (
                self.sellin.loc[self.abi_keys, "vilc_target"].values.astype(float)
                / np.maximum(vol_hl, 1e-6)
            )
            margin_hl = np.maximum(0.0, nr_hl - vilc_per_hl)

            seeds: List[np.ndarray] = []
            # Uniform
            seeds.append(np.full(n, self.cfg.pinc_target))
            # Elasticity-weighted (favor inelastic)
            w1 = 1.0 / np.maximum(np.abs(own), 0.1)
            seeds.append(self.cfg.pinc_target * w1 / max(np.sum(w1) / n, 1e-9))
            # Margin-weighted
            w2 = np.maximum(1e-6, margin_hl * vol_hl)
            seeds.append(self.cfg.pinc_target * w2 / max(np.sum(w2) / n, 1e-9))
            # Combined
            w3 = w1 * w2
            seeds.append(self.cfg.pinc_target * w3 / max(np.sum(w3) / n, 1e-9))

            best_obj = -np.inf
            best_pct = None
            for seed in seeds:
                pct0 = balance_to_target(seed.copy())
                # Convert to integer steps and scale discretely
                steps0 = np.rint((pct0 * np.maximum(base_price_arr, 1e-6)) / self.cfg.price_step)
                low, high = 0.0, 1.0
                best_pct_mid = None
                for _ in range(14):
                    mid = (low + high) / 2.0
                    steps_mid = np.rint(steps0 * mid)
                    pct_mid = (self.cfg.price_step * steps_mid) / np.maximum(base_price_arr, 1.0)
                    penalty = self._constraint_penalty(pct_mid)
                    if penalty <= 1e-3 and np.any(np.abs(steps_mid) > 0):
                        best_pct_mid = pct_mid
                        low = mid
                    else:
                        high = mid
                if best_pct_mid is None:
                    continue
                # Score by projected MACO
                _, kpis_try = self._financial_projection(best_pct_mid)
                obj = kpis_try.get("maco_total", -np.inf)
                if obj > best_obj:
                    best_obj = obj
                    best_pct = best_pct_mid
            if best_pct is None:
                best_pct = np.zeros(n, dtype=float)
            pct_changes = best_pct

        # Round to price_step multiples (no-op for MILP but preserves bounds)
        rounded = []
        for key, pct in zip(self.abi_keys, pct_changes):
            base_price = self.sellin.loc[key, "ptc_base"]
            new_price = round(base_price * (1.0 + pct) / self.cfg.price_step) * self.cfg.price_step
            new_price = min(max(base_price + self.cfg.price_floor_delta, new_price), base_price + self.cfg.price_ceiling_delta)
            rounded.append(new_price / max(base_price, 1e-6) - 1.0)
        rounded = np.array(rounded)

        # Final constraint check; apply discrete repair and accept best-effort plan
        penalty = self._constraint_penalty(rounded)
        used_repair = False
        best_pct = rounded.copy()
        best_pen = penalty
        # Try greedy repair on steps if not constraint-clean
        if penalty > 1e-3:
            steps0 = self._steps_from_pct(rounded)
            steps1 = self._greedy_repair(steps0)
            pct1 = self._pct_from_steps(steps1)
            pen1 = self._constraint_penalty(pct1)
            if pen1 < best_pen:
                best_pct = pct1
                best_pen = pen1
                used_repair = True

        # If still infeasible, select the best among {rounded, repaired} rather than baseline
        if best_pen <= 1e-3:
            result_success = True
            result_message = "Optimizer converged (MILP/heuristic)."
            final_pct = best_pct
        else:
            result_success = True
            result_message = "Best-effort solution with minimal guardrail violations."
            final_pct = best_pct

        summary, kpis = self._financial_projection(final_pct)
        summary["price_pct_change"] = final_pct

        if not self.unmapped.empty:
            untouched = self.unmapped.copy()
            untouched["price_base"] = untouched["ptc_base"]
            untouched["price_new"] = untouched["ptc_base"]
            untouched["price_pct_change"] = 0.0
            untouched["volume_base"] = untouched["volume_hl"]
            untouched["volume_new"] = untouched["volume_hl"]
            untouched["nr_base"] = untouched["NR"]
            untouched["nr_new"] = untouched["NR"]
            untouched["maco_base"] = untouched["MACOstd"]
            untouched["maco_new"] = untouched["MACOstd"]
            untouched["brand"] = untouched["Brand"].astype(str)
            untouched["pack"] = untouched["Pack"].astype(str)
            untouched["size"] = untouched["Size"].astype(str)
            untouched["segment"] = untouched["segment"].astype(str)
            untouched["size_group"] = untouched["size_group"].astype(str)
            summary = pd.concat([summary, untouched[summary.columns]], ignore_index=True)

        portfolio = pd.DataFrame(
            [
                {"metric": "Volume (HL)", "base": self.base_volume, "new": summary["volume_new"].sum()},
                {
                    "metric": "Net Revenue",
                    "base": self.sellin["NR"].sum() + self.unmapped["NR"].sum(),
                    "new": summary["nr_new"].sum(),
                },
                {
                    "metric": "NR/HL",
                    "base": (self.sellin["NR"].sum() + self.unmapped["NR"].sum()) / max(self.base_volume, 1e-6),
                    "new": summary["nr_new"].sum() / max(summary["volume_new"].sum(), 1e-6),
                },
                {"metric": "MACO", "base": self.base_maco, "new": summary["maco_new"].sum()},
                {
                    "metric": "MACO/HL",
                    "base": self.base_maco / max(self.base_volume, 1e-6),
                    "new": summary["maco_new"].sum() / max(summary["volume_new"].sum(), 1e-6),
                },
                {"metric": "Portfolio PINC", "base": 0.0, "new": self._portfolio_pinc(final_pct)},
                {
                    "metric": "Market Share",
                    "base": self.base_market_share,
                    "new": summary["volume_new"].sum()
                    / max(summary["volume_new"].sum() + kpis["competitor_volume"], 1e-6),
                },
            ]
        )

        architecture = summary[
            ["brand", "pack", "size", "segment", "size_group", "price_base", "price_new"]
        ].copy()
        architecture.sort_values(["brand", "pack", "size"], inplace=True)

        metadata = {
            "success": result_success,
            "status": result_message,
            "iterations": "MILP" if used_milp else None,
            "pinc_actual": float(self._portfolio_pinc(final_pct)),
            "base_market_share": float(self.base_market_share),
            "base_industry_volume": float(self.base_industry_volume),
            "objective_maco": float(summary["maco_new"].sum()),
            "penalty": float(best_pen),
            "used_repair": used_repair,
        }
        return summary, portfolio, architecture, metadata

    def _optimize_milp_ortools(self, pinc_target: Optional[float] = None) -> Optional[np.ndarray]:
        """OR-Tools CP-SAT integer solver path. Returns pct-change vector or None."""
        try:
            from ortools.sat.python import cp_model  # type: ignore
        except Exception:
            return None

        keys = self.abi_keys
        base_price = self.sellin.loc[keys, "ptc_base"].values.astype(float)
        vol_base = self.sellin.loc[keys, "volume_hl"].values.astype(float)
        size_ml = self.sellin.loc[keys, "size_ml"].values.astype(float)
        units_per_hl = 100000.0 / np.maximum(size_ml, 1.0)
        units_base = np.rint(vol_base * units_per_hl).astype(int)
        nr_hl_base = (self.sellin.loc[keys, "NR"].values.astype(float) / np.maximum(vol_base, 1e-6))
        vilc_per_hl_target = (
            self.sellin.loc[keys, "vilc_target"].values.astype(float)
            / np.maximum(vol_base, 1e-6)
        )
        own = np.array([self.own_lookup[k] for k in keys], dtype=float)
        c_step = self.cfg.price_step / np.maximum(base_price, 1e-6)
        pinc_val = float(self.cfg.pinc_target if pinc_target is None else pinc_target)

        # Competitor baseline volumes and effects
        base_pred = self.sellout_test_pred.copy()
        base_pred["sellin_key"] = base_pred["sku_str"].map(self.sku_mapping)
        comp_series = (
            base_pred[~base_pred["sellin_key"].isin(keys) & base_pred["sellin_key"].notna()]
            .groupby("sellin_key")["predicted_sales_hectoliters"]
            .sum()
        )
        comp_keys = list(comp_series.index)
        comp_base = comp_series.values.astype(float)

        # Precompute per-step coefficients
        abi_coef = {k: vol_base[i] * own[i] for i, k in enumerate(keys)}
        for i, ki in enumerate(keys):
            for kj, e in self.cross_lookup.get(ki, []):
                abi_coef[kj] = abi_coef.get(kj, 0.0) + vol_base[i] * e
        abi_coef = {k: abi_coef[k] * c_step[self.vector_index[k]] for k in abi_coef}

        comp_coef = {k: 0.0 for k in keys}
        for c_idx, ck in enumerate(comp_keys):
            base_val = comp_base[c_idx]
            for ak, e in self.competitor_effect.get(ck, []):
                comp_coef[ak] = comp_coef.get(ak, 0.0) + base_val * e * c_step[self.vector_index[ak]]

        obj_raw = {k: vol_base[self.vector_index[k]] * nr_hl_base[self.vector_index[k]] for k in keys}
        for i, ki in enumerate(keys):
            base_term = (nr_hl_base[i] - vilc_per_hl_target[i]) * vol_base[i]
            obj_raw[ki] += base_term * own[i]
            for kj, e in self.cross_lookup.get(ki, []):
                obj_raw[kj] = obj_raw.get(kj, 0.0) + base_term * e
        obj_coef = {k: obj_raw[k] * c_step[self.vector_index[k]] for k in obj_raw}

        # Build CP-SAT model
        model = cp_model.CpModel()
        step_lb = int(math.ceil(self.cfg.price_floor_delta / max(self.cfg.price_step, 1.0)))
        step_ub = int(math.floor(self.cfg.price_ceiling_delta / max(self.cfg.price_step, 1.0)))
        step_vars = {k: model.NewIntVar(step_lb, step_ub, f"step_{i}") for i, k in enumerate(keys)}

        VOL_SCALE = 1000
        PINC_SCALE = 1000
        SHARE_SCALE = 100000
        MONEY_SCALE = 100

        # PINC equality
        base_units_total = int(np.sum(base_price * units_base))
        target_total_center = (1.0 + pinc_val) * PINC_SCALE * base_units_total
        lhs_var = sum(int(self.cfg.price_step * units_base[i]) * step_vars[k] for i, k in enumerate(keys))
        constant_scaled = int(PINC_SCALE * base_units_total)
        # Allow ±0.1pp tolerance on PINC to improve feasibility with integer steps
        tol = 0.003
        lower = int(math.floor((1.0 + pinc_val - tol) * PINC_SCALE * base_units_total) - constant_scaled)
        upper = int(math.ceil((1.0 + pinc_val + tol) * PINC_SCALE * base_units_total) - constant_scaled)
        model.AddLinearConstraint(lhs_var, lower, upper)

        base_abi_scaled = int(round(VOL_SCALE * (vol_base.sum() + self.unmapped["volume_hl"].sum())))
        abi_new_scaled = sum(int(round(VOL_SCALE * abi_coef.get(k, 0.0))) * step_vars[k] for k in keys) + base_abi_scaled

        base_comp_scaled = int(round(VOL_SCALE * comp_base.sum()))
        comp_new_scaled = sum(int(round(VOL_SCALE * comp_coef.get(k, 0.0))) * step_vars[k] for k in keys) + base_comp_scaled
        industry_new_scaled = abi_new_scaled + comp_new_scaled
        base_industry_scaled = int(round(VOL_SCALE * self.base_industry_volume))

        # Guardrails with slacks to ensure feasibility
        slack_vol_down = model.NewIntVar(0, 10**9, "slack_vol_down")
        slack_vol_up = model.NewIntVar(0, 10**9, "slack_vol_up")
        slack_industry = model.NewIntVar(0, 10**9, "slack_industry")
        slack_share = model.NewIntVar(0, 10**9, "slack_share")

        model.Add(100 * abi_new_scaled + slack_vol_down >= 99 * base_abi_scaled)
        model.Add(100 * abi_new_scaled - slack_vol_up <= 105 * base_abi_scaled)
        model.Add(100 * industry_new_scaled + slack_industry >= 99 * base_industry_scaled)
        share_floor_ppm = int(max(0.0, (self.base_market_share - 0.005)) * SHARE_SCALE)
        model.Add(SHARE_SCALE * abi_new_scaled - share_floor_ppm * industry_new_scaled + slack_share >= 0)

        delta_maco_scaled = sum(int(round(MONEY_SCALE * obj_coef.get(k, 0.0))) * step_vars[k] for k in keys)
        model.Add(delta_maco_scaled >= 0)
        penalty = (
            10_000 * slack_share
            + 1_000 * slack_industry
            + 200 * slack_vol_down
            + 200 * slack_vol_up
        )
        model.Maximize(delta_maco_scaled - penalty)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            steps = np.array([solver.Value(step_vars[k]) for k in keys], dtype=float)
            pct = (self.cfg.price_step * steps) / np.maximum(base_price, 1.0)
            return pct
        return None

    def _optimize_milp(self, pinc_target: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Mixed-Integer Linear Program aligned with the research plan:
        - Integer step variables enforce 50-unit price steps
        - PINC equality on weighted average price
        - Volume, industry, and market share guardrails as linear constraints
        - Approximate linear MACO objective using base NR/unit and cost per HL
        Returns pct-change vector on success; None on failure.
        """
        try:
            import pulp  # type: ignore
        except Exception:
            return None

        keys = self.abi_keys
        n = len(keys)
        base_price = self.sellin.loc[keys, "ptc_base"].values.astype(float)
        weight = self.sellin.loc[keys, "volume_units"].values.astype(float)
        vol_base = self.sellin.loc[keys, "volume_hl"].values.astype(float)
        size_ml = self.sellin.loc[keys, "size_ml"].values.astype(float)
        units_per_hl = 100000.0 / np.maximum(size_ml, 1.0)
        discount_pct = self.sellin.loc[keys, "discount_pct"].values.astype(float)
        excise_pct = self.sellin.loc[keys, "excise_pct"].values.astype(float)
        markup = self.sellin.loc[keys, "markup"].values.astype(float)
        nr_unit_base = (
            (base_price / np.maximum(1.0 + markup, 1e-6))
            * (1.0 - discount_pct - excise_pct)
            / (1.0 + self.cfg.vat)
        )
        vilc_per_hl_target = (
            self.sellin.loc[keys, "vilc_target"].values.astype(float)
            / np.maximum(self.sellin.loc[keys, "volume_hl"].values.astype(float), 1e-6)
        )

        own = np.array([self.own_lookup[k] for k in keys], dtype=float)

        # Competitor baseline volumes and effects
        base_pred = self.sellout_test_pred.copy()
        base_pred["sellin_key"] = base_pred["sku_str"].map(self.sku_mapping)
        comp_series = (
            base_pred[~base_pred["sellin_key"].isin(keys) & base_pred["sellin_key"].notna()]
            .groupby("sellin_key")["predicted_sales_hectoliters"]
            .sum()
        )
        comp_keys = list(comp_series.index)
        comp_base = comp_series.values.astype(float)

        # MILP model
        prob = pulp.LpProblem("ABI_RGM_Pricing_Optimizer", pulp.LpMaximize)
        step_lb = int(math.ceil(self.cfg.price_floor_delta / max(self.cfg.price_step, 1.0)))
        step_ub = int(math.floor(self.cfg.price_ceiling_delta / max(self.cfg.price_step, 1.0)))
        step_vars = {
            k: pulp.LpVariable(f"step_{i}", lowBound=step_lb, upBound=step_ub, cat="Integer")
            for i, k in enumerate(keys)
        }

        # Portfolio PINC equality (weighted by base units)
        base_total = float(np.sum(base_price * weight))
        pinc_val = float(self.cfg.pinc_target if pinc_target is None else pinc_target)
        target_total = float(base_total * (1.0 + pinc_val))
        pinc_lhs = pulp.lpSum((base_price[i] + self.cfg.price_step * step_vars[k]) * weight[i] for i, k in enumerate(keys))
        prob += (pinc_lhs == target_total), "portfolio_pinc"

        # Volume projection expressions for ABI
        pct_expr = {
            k: (self.cfg.price_step / max(base_price[i], 1e-6)) * step_vars[k]
            for i, k in enumerate(keys)
        }
        vol_new_expr = {}
        for i, ki in enumerate(keys):
            expr = vol_base[i] + vol_base[i] * own[i] * pct_expr[ki]
            for kj, e in self.cross_lookup.get(ki, []):
                j = self.vector_index.get(kj)
                if j is None:
                    continue
                expr += vol_base[i] * e * pct_expr[kj]
            vol_new_expr[ki] = expr

        # Competitor volume projection
        comp_vol_new_expr = {}
        for c_idx, ck in enumerate(comp_keys):
            base_val = comp_base[c_idx]
            expr = base_val
            for ak, e in self.competitor_effect.get(ck, []):
                expr += base_val * e * pct_expr.get(ak, 0.0)
            comp_vol_new_expr[ck] = expr

        abi_unmapped = float(self.unmapped["volume_hl"].sum())
        abi_total_new = abi_unmapped + pulp.lpSum(vol_new_expr[k] for k in keys)
        industry_total_new = abi_total_new + pulp.lpSum(comp_vol_new_expr[c] for c in comp_keys)

        # Guardrails: ABI volume within [-1%, +5%]
        prob += (abi_total_new >= 0.99 * float(self.base_volume)), "abi_volume_min"
        prob += (abi_total_new <= 1.05 * float(self.base_volume)), "abi_volume_max"

        # Industry volume tether >= 99% of base
        prob += (industry_total_new >= 0.99 * float(self.base_industry_volume)), "industry_volume_min"

        # Market share drop <= 0.5 percentage points (linearized)
        share_floor = float(self.base_market_share - 0.005)
        prob += (abi_total_new >= share_floor * industry_total_new), "market_share_min"

        # Approximate linear MACO objective and non-decreasing constraint
        units_base = vol_base * units_per_hl
        alpha = nr_unit_base / np.maximum(base_price, 1e-6)  # dNR_unit/dPrice

        maco_mapped_base = float(self.sellin["MACOstd"].sum())
        maco_unmapped_base = float(self.unmapped["MACOstd"].sum())

        # Build linear expression for MACO_new_approx over mapped keys
        maco_new_expr = pulp.lpSum(
            (
                nr_unit_base[i] * units_per_hl[i] * vol_new_expr[k]
                + alpha[i] * self.cfg.price_step * step_vars[k] * units_base[i]
                - vilc_per_hl_target[i] * vol_new_expr[k]
            )
            for i, k in enumerate(keys)
        )
        # Enforce non-decreasing total MACO (approximate)
        prob += (maco_new_expr + maco_unmapped_base >= maco_mapped_base + maco_unmapped_base), "maco_non_decreasing"

        # Objective: maximize approximate MACO
        prob += maco_new_expr, "objective_maco"

        # Solve with CBC (default) and a short time limit
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
        status = prob.solve(solver)
        if any(v.varValue is None for v in step_vars.values()):
            return None

        steps = np.array([float(step_vars[k].varValue) for k in keys], dtype=float)
        pct = (self.cfg.price_step * steps) / np.maximum(base_price, 1e-6)
        return pct


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_overrides(cfg: Round2Config, args: List[str]) -> None:
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        if isinstance(current, (int, float)):
            try:
                setattr(cfg, key, type(current)(float(value)))
                continue
            except ValueError:
                pass
        setattr(cfg, key, type(current)(value))


@lru_cache(maxsize=1)
def _load_base_components() -> Tuple[
    Round2Config,
    pd.DataFrame,
    Dict[str, float],
    pd.DataFrame,
    Dict[str, str],
    pd.DataFrame,
    pd.DataFrame,
    Set[str],
]:
    base_cfg = Round2Config()
    loader = Round2DataLoader(base_cfg)
    sellout_train, sellout_test = loader.load_sellout()
    sellin = loader.load_sellin()
    price_list = loader.load_price_list()
    sellin_summary = build_sellin_summary(sellin, price_list, base_cfg)
    own_map, elasticity_matrix, models = compute_elasticities(base_cfg, sellout_train)
    sku_mapping = map_sellout_to_sellin(sellout_train, sellin_summary)
    sellout_test_pred = predict_sellout_volumes(base_cfg, models, sellout_train, sellout_test)
    mapping_values = {k: v for k, v in sku_mapping.items() if v}
    mapping_path = Path(base_cfg.out_dir) / "sellout_sellin_mapping.xlsx"
    if mapping_values and not mapping_path.exists():
        mapping_df = (
            sellout_train[
                ["sku_str", "brand", "sub_brand", "package", "package_type", "capacity_number"]
            ]
            .drop_duplicates()
            .assign(sellin_key=lambda df: df["sku_str"].map(mapping_values))
        )
        mapping_df = mapping_df.dropna(subset=["sellin_key"])
        if len(mapping_df) > 0:
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_df.to_excel(mapping_path, index=False)
    manufacturer_map = (
        sellout_train[["sku_str", "manufacturer"]]
        .drop_duplicates()
        .assign(key=lambda df: df["sku_str"].map(mapping_values))
    )
    abi_keys = set(
        manufacturer_map.loc[
            manufacturer_map["manufacturer"].str.upper() == "AB INBEV", "key"
        ].dropna()
    )
    return (
        base_cfg,
        sellin_summary,
        own_map,
        elasticity_matrix,
        mapping_values,
        sellout_train,
        sellout_test_pred,
        abi_keys,
    )


def run_optimizer(
    pinc_target: float = 0.02,
    price_floor_delta: float = -300.0,
    price_ceiling_delta: float = 500.0,
    price_step: float = 50.0,
) -> Dict[str, object]:
    """
    Execute the pricing optimizer with custom parameters and return
    dataframes + metadata for downstream consumption (API/UI).
    """
    (
        base_cfg,
        sellin_summary,
        own_map,
        elasticity_matrix,
        sku_mapping,
        sellout_train,
        sellout_test_pred,
        abi_keys,
    ) = (
        _load_base_components()
    )
    cfg = Round2Config(
        data_dir=base_cfg.data_dir,
        out_dir=base_cfg.out_dir,
        pinc_target=float(pinc_target),
        price_floor_delta=float(price_floor_delta),
        price_ceiling_delta=float(price_ceiling_delta),
        price_step=float(price_step),
    )
    optimizer = PricingOptimizer(
        cfg,
        sellin_summary,
        own_map,
        elasticity_matrix,
        sku_mapping,
        sellout_train,
        sellout_test_pred,
        mapped_keys=abi_keys,
    )
    summary, portfolio, architecture, metadata = optimizer.optimize()
    pct_map = summary.set_index("key")["price_pct_change"].to_dict()
    pct_vector = np.array([pct_map.get(key, 0.0) for key in optimizer.abi_keys], dtype=float)
    constraints = optimizer.evaluate_constraints(pct_vector)
    result = {
        "summary": summary.copy(),
        "portfolio": portfolio.copy(),
        "architecture": architecture.copy(),
        "elasticity": elasticity_matrix.copy(),
        "metadata": metadata,
        "unmapped": optimizer.unmapped.copy(),
        "constraints": constraints,
    }
    return result


def main(args: Optional[List[str]] = None):
    cfg = Round2Config()
    parse_overrides(cfg, args or [])
    ensure_out_dir(str(cfg.out_dir))

    result = run_optimizer(
        pinc_target=cfg.pinc_target,
        price_floor_delta=cfg.price_floor_delta,
        price_ceiling_delta=cfg.price_ceiling_delta,
        price_step=cfg.price_step,
    )
    summary = result["summary"]
    portfolio = result["portfolio"]
    architecture = result["architecture"]
    elasticity_matrix = result["elasticity"]
    metadata = result["metadata"]

    out_dir = Path(cfg.out_dir)
    summary_path = out_dir / "optimized_prices.csv"
    summary.to_csv(summary_path, index=False)

    portfolio_path = out_dir / "portfolio_summary.csv"
    portfolio.to_csv(portfolio_path, index=False)

    architecture_path = out_dir / "price_architecture.csv"
    architecture.to_csv(architecture_path, index=False)

    elasticity_path = out_dir / "elasticity_matrix.csv"
    elasticity_matrix.to_csv(elasticity_path, index=False)

    run_summary = {
        "summary_csv": str(summary_path),
        "portfolio_csv": str(portfolio_path),
        "architecture_csv": str(architecture_path),
        "elasticity_csv": str(elasticity_path),
        "metadata": metadata,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
