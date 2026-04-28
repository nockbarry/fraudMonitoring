"""Model Performance Hub builder.

Consumes a scored DataFrame (with columns: split, label, score, date) and one or
more trained models, computes the per-split metrics required by the model
performance hub schema, and renders the standalone HTML report.

Schema produced (matches the hub's expectations):

    DATA = {
      schema_version, run_id, run_timestamp, model_name, champion_id,
      cohort_dims: [str],
      feature_importance: [{name, gain, weight, cover}, ...],
      models: {<model_id>: {
        id, name, color,
        splits: {<split>: {
          name, n, fraud, fraud_rate,
          roc: {fpr, tpr, auc},
          pr: {recall, precision, auprc, baseline},
          calibration: {bins:[{pred,actual,count}], brier, ece},
          lift_gains: {rows, baseline},
          score_hist: {edges, fraud, legit},
          daily: [...],
          by_depth: {<depth>: {tp, fp, alerts, fcr, precision, gf, score_threshold}},
          daily_by_depth: {<depth>: [...]},
          cohorts: {<col>: {<val>: {n, fraud, auc, fcr_at_depth}}},
        }}
      }},
      sweep, hp_search, shap, extras
    }
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

TEMPLATE_PATH = Path(__file__).parent / "model_perf_template.html"

DEPTHS = (0.005, 0.01, 0.02, 0.05, 0.10)
MODEL_COLORS = {"gbm": "#6366f1", "lr": "#0ea5e9", "rf": "#10b981", "xgb": "#f59e0b"}


# ---------------------------------------------------------------------------
# Per-split metric helpers
# ---------------------------------------------------------------------------

def _downsample(x: np.ndarray, y: np.ndarray, max_pts: int = 200) -> tuple[list, list]:
    if len(x) <= max_pts:
        return x.tolist(), y.tolist()
    idx = np.linspace(0, len(x) - 1, max_pts).astype(int)
    return x[idx].tolist(), y[idx].tolist()


def _roc_payload(y: np.ndarray, s: np.ndarray) -> dict:
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"fpr": [0, 1], "tpr": [0, 1], "auc": float("nan")}
    fpr, tpr, _ = roc_curve(y, s)
    fpr_d, tpr_d = _downsample(fpr, tpr, 200)
    return {"fpr": [round(v, 5) for v in fpr_d],
            "tpr": [round(v, 5) for v in tpr_d],
            "auc": float(roc_auc_score(y, s))}


def _pr_payload(y: np.ndarray, s: np.ndarray) -> dict:
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"recall": [0, 1], "precision": [0, 0], "auprc": float("nan"), "baseline": float(y.mean())}
    p, r, _ = precision_recall_curve(y, s)
    r_d, p_d = _downsample(r[::-1], p[::-1], 200)
    return {"recall": [round(v, 5) for v in r_d],
            "precision": [round(v, 5) for v in p_d],
            "auprc": float(average_precision_score(y, s)),
            "baseline": float(y.mean())}


def _calibration_payload(y: np.ndarray, s: np.ndarray, n_bins: int = 12) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            bins.append({"pred": float((edges[b] + edges[b + 1]) / 2),
                         "actual": 0.0, "count": 0})
            continue
        pred = float(s[m].mean())
        actual = float(y[m].mean())
        ece += m.mean() * abs(pred - actual)
        bins.append({"pred": pred, "actual": actual, "count": int(m.sum())})
    brier = float(brier_score_loss(y, s)) if y.sum() and (1 - y).sum() else float("nan")
    return {"bins": bins, "brier": brier, "ece": float(ece)}


def _lift_gains_payload(y: np.ndarray, s: np.ndarray, n_buckets: int = 10) -> dict:
    n = len(y)
    if n == 0 or y.sum() == 0:
        return {"rows": [], "baseline": float(y.mean()) if n else 0.0}
    order = np.argsort(-s)
    y_sorted = y[order]
    bucket_size = math.ceil(n / n_buckets)
    rows = []
    cum_fraud = 0; cum_n = 0; total_fraud = int(y.sum())
    base = y.mean()
    for b in range(n_buckets):
        lo = b * bucket_size
        hi = min((b + 1) * bucket_size, n)
        seg = y_sorted[lo:hi]
        cum_fraud += int(seg.sum())
        cum_n += len(seg)
        rows.append({
            "bucket": b + 1,
            "depth": cum_n / n,
            "fraud_rate": float(seg.mean()),
            "lift": float(seg.mean() / base) if base > 0 else 0.0,
            "cum_capture": float(cum_fraud / total_fraud),
        })
    return {"rows": rows, "baseline": float(base)}


def _score_hist_payload(y: np.ndarray, s: np.ndarray, n_bins: int = 30) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    fraud_counts, _ = np.histogram(s[y == 1], edges)
    legit_counts, _ = np.histogram(s[y == 0], edges)
    return {"edges": edges.round(4).tolist(),
            "fraud": fraud_counts.tolist(),
            "legit": legit_counts.tolist()}


def _by_depth_payload(y: np.ndarray, s: np.ndarray, depths: Iterable[float]) -> dict:
    out = {}
    n = len(y); total_fraud = int(y.sum())
    if n == 0:
        return {f"{d:.3f}".rstrip("0").rstrip("."): {"tp": 0, "fp": 0, "alerts": 0,
                                                     "fcr": 0, "precision": 0, "gf": 0,
                                                     "score_threshold": 0} for d in depths}
    order = np.argsort(-s)
    y_sorted = y[order]; s_sorted = s[order]
    for d in depths:
        k = max(1, int(round(n * d)))
        thr = float(s_sorted[k - 1])
        tp = int(y_sorted[:k].sum())
        fp = k - tp
        fcr = tp / total_fraud if total_fraud else 0.0
        precision = tp / k if k else 0.0
        gf = (k - tp) / max(tp, 1)  # good-to-fraud ratio
        key = f"{d:.3f}".rstrip("0").rstrip(".")
        out[key] = {"tp": tp, "fp": fp, "alerts": k, "fcr": float(fcr),
                    "precision": float(precision), "gf": float(gf),
                    "score_threshold": thr}
    return out


def _daily_payload(df: pd.DataFrame, depths: Iterable[float]) -> tuple[list, dict]:
    daily = []
    daily_by_depth = {f"{d:.3f}".rstrip("0").rstrip("."): [] for d in depths}
    for day, sub in df.groupby("date", sort=True):
        y = sub["label"].to_numpy(); s = sub["score"].to_numpy()
        if len(y) == 0:
            continue
        try:
            auc = roc_auc_score(y, s) if y.sum() and (1 - y).sum() else float("nan")
        except ValueError:
            auc = float("nan")
        # Aggregate
        daily.append({
            "date": str(day.date() if hasattr(day, "date") else day),
            "n": int(len(sub)),
            "fraud": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "auc": float(auc) if not math.isnan(auc) else None,
            "mean_score": float(s.mean()),
        })
        # Per-depth
        order = np.argsort(-s); y_sorted = y[order]
        n_day = len(y); total_fraud = int(y.sum())
        for d in depths:
            k = max(1, int(round(n_day * d)))
            tp = int(y_sorted[:k].sum())
            fp = k - tp
            fcr = tp / total_fraud if total_fraud else 0.0
            precision = tp / k if k else 0.0
            key = f"{d:.3f}".rstrip("0").rstrip(".")
            daily_by_depth[key].append({
                "date": str(day.date() if hasattr(day, "date") else day),
                "tp": tp, "fp": fp, "alerts": k,
                "fcr": float(fcr), "precision": float(precision),
                "fraud_rate": float(y.mean()),
            })
    return daily, daily_by_depth


def _cohort_payload(df: pd.DataFrame, cohort_dims: Sequence[str], depth: float = 0.02) -> dict:
    out = {}
    for col in cohort_dims:
        if col not in df.columns:
            continue
        vals = {}
        for v, sub in df.groupby(col, sort=False):
            if len(sub) < 200:
                continue
            y = sub["label"].to_numpy(); s = sub["score"].to_numpy()
            try:
                auc = roc_auc_score(y, s) if y.sum() and (1 - y).sum() else float("nan")
            except ValueError:
                auc = float("nan")
            order = np.argsort(-s); y_sorted = y[order]
            k = max(1, int(round(len(sub) * depth)))
            tp = int(y_sorted[:k].sum())
            fcr = tp / max(int(y.sum()), 1)
            vals[str(v)] = {
                "n": int(len(sub)),
                "fraud": int(y.sum()),
                "fraud_rate": float(y.mean()),
                "auc": float(auc) if not math.isnan(auc) else None,
                "fcr_at_depth": float(fcr),
            }
        if vals:
            out[col] = vals
    return out


def _split_payload(df: pd.DataFrame, name: str, cohort_dims: Sequence[str]) -> dict:
    y = df["label"].to_numpy(); s = df["score"].to_numpy()
    return {
        "name": name,
        "n": int(len(df)),
        "fraud": int(y.sum()),
        "fraud_rate": float(y.mean()) if len(df) else 0.0,
        "roc": _roc_payload(y, s),
        "pr": _pr_payload(y, s),
        "calibration": _calibration_payload(y, s),
        "lift_gains": _lift_gains_payload(y, s),
        "score_hist": _score_hist_payload(y, s),
        "daily": _daily_payload(df, DEPTHS)[0],
        "by_depth": _by_depth_payload(y, s, DEPTHS),
        "daily_by_depth": _daily_payload(df, DEPTHS)[1],
        "cohorts": _cohort_payload(df, cohort_dims),
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _feature_importance(model, feature_names: Sequence[str],
                        X_sample: pd.DataFrame, y_sample: np.ndarray) -> List[dict]:
    from sklearn.inspection import permutation_importance
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X_sample, y_sample, n_repeats=2,
            random_state=0, scoring="roc_auc", n_jobs=-1,
        )
    raw = np.clip(result.importances_mean, 0, None)
    s = raw.sum() or 1.0
    gains = raw / s
    out = [{"name": n, "gain": float(g), "weight": None, "cover": None}
           for n, g in zip(feature_names, gains)]
    out.sort(key=lambda x: -x["gain"])
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_payload(
    scored_models: Mapping[str, dict],
    cohort_dims: Sequence[str],
    feature_importance_data: List[dict],
    *,
    model_name: str = "Fraud Demo",
    champion_id: str = "gbm",
    extras: dict | None = None,
) -> dict:
    """
    scored_models is a mapping {model_id: {"name": ..., "df_by_split": {split: df}}}
    where each df has columns ['label', 'score', 'date', ...cohort_dims].
    """
    models = {}
    for mid, info in scored_models.items():
        splits = {}
        for sname, df in info["df_by_split"].items():
            splits[sname] = _split_payload(df, sname, cohort_dims)
        models[mid] = {
            "id": mid,
            "name": info["name"],
            "color": MODEL_COLORS.get(mid, "#6366f1"),
            "splits": splits,
        }

    return {
        "schema_version": "1.0",
        "run_id": f"fraud_demo_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "champion_id": champion_id,
        "cohort_dims": list(cohort_dims),
        "feature_importance": feature_importance_data,
        "models": models,
        "sweep": (extras or {}).get("sweep"),
        "hp_search": (extras or {}).get("hp_search"),
        "shap": (extras or {}).get("shap"),
        "extras": extras or {},
    }


def render(payload: dict, output_path: str | Path) -> Path:
    template = TEMPLATE_PATH.read_text()
    payload_json = json.dumps(payload, allow_nan=False, default=lambda o: None)
    html = template.replace("__DATA_JSON__", payload_json)
    out = Path(output_path)
    out.write_text(html)
    return out


def build_report(
    scored_models: Mapping[str, dict],
    cohort_dims: Sequence[str],
    feature_importance_data: List[dict],
    output_path: str | Path,
    **kwargs,
) -> Path:
    payload = build_payload(scored_models, cohort_dims, feature_importance_data, **kwargs)
    return render(payload, output_path)


# Re-export for convenience
feature_importance = _feature_importance
