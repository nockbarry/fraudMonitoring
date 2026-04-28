"""Feature Analysis Hub builder.

Computes per-feature anchor stats (importance, univariate AUC, drift PSI vs OOT,
missing rate) from a real DataFrame + trained model, plus time-grid metadata,
and renders the feature analysis hub.

The hub's JS layer derives weekly/daily/anomaly/spread series from these anchor
stats — so the time-series visuals reflect real importance, real drift, real
missing patterns even though the per-day noise is synthesized.

Public API
----------
build_report(scored_df, feature_names, feature_meta, model, output_path) -> Path
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TEMPLATE_PATH = Path(__file__).parent / "feature_hub_template.html"


# ---------------------------------------------------------------------------
# Per-feature stats
# ---------------------------------------------------------------------------

def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC of x as a univariate predictor of y. Direction-agnostic."""
    if y.sum() == 0 or (1 - y).sum() == 0:
        return 0.5
    finite = np.isfinite(x)
    if not finite.any():
        return 0.5
    auc = roc_auc_score(y[finite], x[finite])
    return float(max(auc, 1 - auc))


def _psi(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    """PSI of distribution b vs a.

    Falls back to category-share PSI for low-cardinality features (≤8 unique
    values) since quantile binning collapses to ≤2 edges and produces 0.
    """
    a = np.asarray(a)[np.isfinite(a)].astype(float)
    b = np.asarray(b)[np.isfinite(b)].astype(float)
    if len(a) < 50 or len(b) < 50:
        return 0.0
    eps = 1e-6
    uniq_a = np.unique(a)
    if len(uniq_a) <= 8:
        # treat as categorical
        cats = np.unique(np.concatenate([uniq_a, np.unique(b)]))
        pa = np.array([(a == c).mean() for c in cats])
        pb = np.array([(b == c).mean() for c in cats])
        pa = np.clip(pa, eps, None); pb = np.clip(pb, eps, None)
        return float(np.sum((pb - pa) * np.log(pb / pa)))
    edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    return float(np.sum((pb - pa) * np.log(np.clip(pb, eps, None) / np.clip(pa, eps, None))))


def _missing_rate(s: pd.Series) -> float:
    if s.dtype.kind in "biufc":
        return float(((~np.isfinite(s)) | s.isna()).mean())
    return float(s.isna().mean())


def _categorical_levels(s: pd.Series, max_levels: int = 12) -> list[str]:
    counts = s.astype(str).value_counts()
    if len(counts) <= max_levels:
        return list(counts.index)
    top = list(counts.index[:max_levels - 1])
    top.append("OTHER")
    return top


def _is_categorical_feature(name: str, df: pd.DataFrame) -> bool:
    """Heuristic — features named *_<level> for one-hot encodings or low-cardinality
    columns. We treat numeric features as 'num' and only the raw categorical
    columns (channel, merchant_mcc, bin_country) as 'cat' for hub purposes."""
    return False  # we'll mark "cat" externally


def _compute_feature_anchors(
    df_train: pd.DataFrame, df_oot: pd.DataFrame,
    feature_names: Sequence[str], feature_meta: Mapping,
    model, importance_sample: int = 30_000,
) -> List[dict]:
    """Return list of {n,t,imp,auc,drift,miss,...} for each feature."""
    from sklearn.inspection import permutation_importance

    sample_n = min(importance_sample, len(df_train))
    sample = df_train.sample(sample_n, random_state=0)
    X_sample = sample[list(feature_names)]
    y_sample = sample["label"].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(
            model, X_sample, y_sample, n_repeats=2,
            random_state=0, scoring="roc_auc", n_jobs=-1,
        )
    imp_raw = np.clip(perm.importances_mean, 0, None)
    imp = imp_raw / max(imp_raw.sum(), 1e-9)

    feats = []
    y_train = df_train["label"].to_numpy()
    y_oot   = df_oot["label"].to_numpy()
    for i, name in enumerate(feature_names):
        x_train = df_train[name].to_numpy(dtype=float, na_value=np.nan)
        x_oot   = df_oot[name].to_numpy(dtype=float, na_value=np.nan)
        auc_train = _univariate_auc(x_train, y_train)
        auc_oot   = _univariate_auc(x_oot, y_oot)
        # Drift = max of distribution-shift PSI and concept-drift signal
        # (univariate AUC drop train→OOT), so the metric catches both kinds.
        psi = _psi(x_train, x_oot)
        auc_drop = max(0.0, auc_train - auc_oot) * 1.4
        drift = float(max(psi, auc_drop))
        miss = _missing_rate(df_train[name])
        meta = feature_meta.get(name)
        bucket = meta.bucket if meta else "compositional"
        feats.append({
            "n": name,
            "t": "num",
            "imp": float(imp[i]),
            "auc": float(auc_train),
            "drift": drift,
            "miss": float(miss),
            "bucket": bucket,
        })
    feats.sort(key=lambda f: -f["imp"])
    return feats


def _add_categorical_features(feats: List[dict], df: pd.DataFrame,
                              cat_cols: Sequence[str]) -> List[dict]:
    """Append entries for raw categorical columns (e.g. channel, merchant_mcc)."""
    y = df["label"].to_numpy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        levels = _categorical_levels(df[col])
        # Compute group fraud rates as a one-vs-rest AUC proxy
        max_auc = 0.5; max_drift = 0.0
        for lv in levels:
            x = (df[col] == lv).astype(float).to_numpy()
            try:
                a = roc_auc_score(y, x); a = max(a, 1 - a)
            except ValueError:
                a = 0.5
            max_auc = max(max_auc, a)
        # PSI of category share train vs oot
        train_mask = df["split"] == "train"; oot_mask = df["split"] == "oot"
        for lv in levels:
            train_share = float(((df.loc[train_mask, col] == lv).mean()))
            oot_share = float(((df.loc[oot_mask, col] == lv).mean()))
            if train_share > 0 and oot_share > 0:
                max_drift += abs(np.log(oot_share / train_share)) * abs(oot_share - train_share)
        miss = _missing_rate(df[col])
        feats.append({
            "n": col,
            "t": "cat",
            "imp": 0.02,  # placeholder; cat features get smaller imp by default
            "auc": float(max_auc),
            "drift": float(min(max_drift, 0.4)),
            "miss": float(miss),
            "cats": levels,
            "bucket": "tactical" if col in {"merchant_mcc", "bin_country"} else "invariant",
        })
    return feats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_payload(
    scored_df: pd.DataFrame,
    feature_names: Sequence[str],
    feature_meta: Mapping,
    model,
    *,
    cat_cols: Sequence[str] = (),
    start_date: str | None = None,
    n_days: int | None = None,
    oot_day: int | None = None,
) -> dict:
    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df_train = df[df["split"] == "train"]
    df_oot   = df[df["split"] == "oot"]

    feats = _compute_feature_anchors(df_train, df_oot, feature_names, feature_meta, model)
    feats = _add_categorical_features(feats, df, cat_cols)

    # Time grid
    if n_days is None:
        n_days = int((df["date"].max() - df["date"].min()).days) + 1
    if oot_day is None:
        oot_day = int((df_oot["date"].min() - df["date"].min()).days)
    if start_date is None:
        start_date = str(df["date"].min().date())

    return {
        "n_days": int(n_days),
        "oot_day": int(oot_day),
        "start_date": start_date,
        "features": feats,
    }


def render(payload: dict, output_path: str | Path) -> Path:
    template = TEMPLATE_PATH.read_text()
    payload_json = json.dumps(payload, allow_nan=False, default=lambda o: None)
    html = template.replace("__DATA_JSON__", payload_json)
    out = Path(output_path)
    out.write_text(html)
    return out


def build_report(
    scored_df: pd.DataFrame,
    feature_names: Sequence[str],
    feature_meta: Mapping,
    model,
    output_path: str | Path,
    **kwargs,
) -> Path:
    payload = build_payload(scored_df, feature_names, feature_meta, model, **kwargs)
    return render(payload, output_path)
