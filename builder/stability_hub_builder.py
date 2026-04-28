"""Stability & Calibration Monitoring Hub — report builder.

Takes a fraud-modeling DataFrame (transactions + features + scores + labels)
plus per-feature stability metadata, computes the metrics for each section of
the stability hub, and renders a self-contained HTML report.

The report is the same one in ``stability_monitoring_hub.html`` — this module
swaps in real-data-derived metrics for the hand-authored synthetic block.

Public API
----------
build_report(scored_df, feature_meta, output_path, **kwargs) -> Path

Required columns in scored_df
-----------------------------
- date            : pd.Timestamp (transaction authorization date)
- score           : float in [0,1] from the production model
- label           : 0/1 fraud outcome (mature labels only)
- split           : one of {'train','cal','oot','recent'}
- features...     : numeric model features matching feature_meta keys
- cohort columns  : optional categorical columns named in `cohort_dims`
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

TEMPLATE_PATH = Path(__file__).parent / "template.html"


# ---------------------------------------------------------------------------
# Stability metadata
# ---------------------------------------------------------------------------

VALID_BUCKETS = {"invariant", "tactical", "compositional"}


@dataclass(frozen=True)
class FeatureMeta:
    bucket: str            # "invariant" | "tactical" | "compositional"
    stable_target: bool    # True if target relationship has held across past regime shifts


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def feature_importance(
    model,
    feature_names: Sequence[str],
    feature_meta: Mapping[str, FeatureMeta],
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
) -> List[dict]:
    """Permutation importance with bucket + stable_target tags."""
    from sklearn.inspection import permutation_importance

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X_sample, y_sample, n_repeats=3,
            random_state=0, scoring="roc_auc", n_jobs=-1,
        )

    raw = np.clip(result.importances_mean, a_min=0, a_max=None)
    s = raw.sum() or 1.0
    gains = raw / s

    feats = []
    for name, gain in zip(feature_names, gains):
        meta = feature_meta.get(name, FeatureMeta("compositional", False))
        feats.append({
            "name": name,
            "bucket": meta.bucket,
            "gain": float(gain),
            "stable_target": bool(meta.stable_target),
            "drift_psi": 0.0,  # filled below
        })
    feats.sort(key=lambda f: -f["gain"])
    return feats


def population_stability_index(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    """PSI of distribution b vs reference a using a's quantile edges.

    For low-cardinality features (≤8 unique values), falls back to
    category-share PSI since quantile binning collapses on binary/integer data.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return 0.0
    eps = 1e-6
    uniq_a = np.unique(a)
    if len(uniq_a) <= 8:
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
    pa = np.clip(pa, eps, None); pb = np.clip(pb, eps, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def conditional_psi(
    df_train: pd.DataFrame,
    df_oot: pd.DataFrame,
    feature: str,
    cohort_col: str,
    n_bins: int = 10,
) -> float:
    """Population-weighted average PSI per cohort segment — controls for mix shifts."""
    cohorts = sorted(set(df_train[cohort_col].unique()) | set(df_oot[cohort_col].unique()))
    weighted = 0.0
    total_w = 0
    for c in cohorts:
        a = df_train.loc[df_train[cohort_col] == c, feature].to_numpy()
        b = df_oot.loc[df_oot[cohort_col] == c, feature].to_numpy()
        if len(a) < 200 or len(b) < 200:
            continue
        w = len(b)
        weighted += population_stability_index(a, b, n_bins) * w
        total_w += w
    return float(weighted / total_w) if total_w else 0.0


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width ECE."""
    if len(scores) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = scores[m].mean()
        acc = labels[m].mean()
        ece += (m.mean()) * abs(conf - acc)
    return float(ece)


def daily_series(df_scored: pd.DataFrame, top_decile_threshold: float, window: int = 7) -> List[dict]:
    """Per-day fraud_rate and mean_score, plus rolling-window AUC/ECE/top_decile.

    Per-day AUC/ECE estimates are too noisy at fraud rates ~0.4% (≈10–30 fraud
    labels per day). A rolling window of `window` days gives stable estimates
    while still surfacing regime shifts.
    """
    days = sorted(df_scored["date"].unique())
    by_day: Dict[pd.Timestamp, pd.DataFrame] = {d: g for d, g in df_scored.groupby("date")}
    out = []
    for i, day in enumerate(days):
        sub = by_day[day]
        s = sub["score"].to_numpy(); y = sub["label"].to_numpy()
        # Window over [day-window+1, day] for AUC/ECE/top_decile
        lo = max(0, i - window + 1)
        win = pd.concat([by_day[days[k]] for k in range(lo, i + 1)], ignore_index=True)
        ws = win["score"].to_numpy(); wy = win["label"].to_numpy()
        try:
            auc = roc_auc_score(wy, ws) if wy.sum() and (1 - wy).sum() else float("nan")
        except ValueError:
            auc = float("nan")
        ece = expected_calibration_error(ws, wy) if len(ws) > 200 else float("nan")
        top_mask = ws >= top_decile_threshold
        top_decile = float(wy[top_mask].mean()) if top_mask.any() else float("nan")
        out.append({
            "date": str(day.date() if hasattr(day, "date") else day),
            "fraud_rate": float(y.mean()),
            "mean_score": float(s.mean()),
            "auc": float(auc) if not math.isnan(auc) else None,
            "ece": float(ece) if not math.isnan(ece) else None,
            "top_decile": float(top_decile) if not math.isnan(top_decile) else None,
        })
    return out


def adversarial_validation(
    X_train: pd.DataFrame, X_oot: pd.DataFrame,
    feature_names: Sequence[str], feature_meta: Mapping[str, FeatureMeta],
    n_top: int = 10, sample_n: int = 50_000, random_state: int = 0,
) -> Tuple[float, List[dict]]:
    """Train binary classifier to distinguish train vs OOT; return AUC and top features."""
    rng = np.random.default_rng(random_state)
    n = min(sample_n, len(X_train), len(X_oot))
    tr_idx = rng.choice(len(X_train), size=n, replace=False)
    oo_idx = rng.choice(len(X_oot), size=n, replace=False)
    X = pd.concat([X_train.iloc[tr_idx], X_oot.iloc[oo_idx]], ignore_index=True)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    perm = rng.permutation(len(X))
    X = X.iloc[perm].reset_index(drop=True); y = y[perm]
    cut = int(len(X) * 0.7)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y[:cut], y[cut:]

    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=random_state)
    clf.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

    # Permutation importance to identify top discriminators
    from sklearn.inspection import permutation_importance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            clf, Xte.iloc[:5000], yte[:5000],
            n_repeats=3, random_state=random_state, scoring="roc_auc", n_jobs=-1,
        )
    raw = np.clip(result.importances_mean, 0, None)
    s = raw.sum() or 1.0
    imp = raw / s
    order = np.argsort(-imp)[:n_top]

    out = []
    for j in order:
        name = feature_names[j]
        meta = feature_meta.get(name, FeatureMeta("compositional", False))
        note = f"{meta.bucket}, " + ("stable target" if meta.stable_target else "drifting target")
        out.append({"name": name, "imp": float(imp[j]), "note": note})
    return auc, out


def decile_drift(df_scored: pd.DataFrame, cal_edges: np.ndarray) -> List[dict]:
    """Per-decile daily fraud rates using calibration-set quantile edges."""
    s = df_scored["score"].to_numpy()
    bins = np.clip(np.digitize(s, cal_edges[1:-1]), 0, 9)
    df_scored = df_scored.assign(_decile=bins)
    out = []
    days = sorted(df_scored["date"].unique())
    for d in range(10):
        sub = df_scored[df_scored["_decile"] == d]
        series = []
        for day in days:
            sd = sub[sub["date"] == day]
            series.append(float(sd["label"].mean()) if len(sd) > 0 else None)
        out.append({"d": d + 1, "series": series})
    return out


def quantile_edges(scores: np.ndarray, k: int = 10) -> List[float]:
    if len(scores) == 0:
        return [0.0] + [(i + 1) / k for i in range(k)]
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(scores, qs)
    edges[0], edges[-1] = 0.0, 1.0
    return edges.tolist()


def score_histogram(scores: np.ndarray, n_bins: int = 30) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(scores, edges)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def apply_edges(scores: np.ndarray, edges: List[float]) -> List[float]:
    """Return per-bin population share when `edges` are applied to `scores`."""
    edges = np.asarray(edges)
    bins = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    total = counts.sum() or 1
    return (counts / total).tolist()


def isotonic_then_bin_stability(
    df_cal: pd.DataFrame,
    df_recent_weeks: pd.DataFrame,
    n_bins: int = 10,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Compare raw-score binning vs isotonic-then-bin per-bin populations across
    weekly windows of the recent data."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(df_cal["score"].to_numpy(), df_cal["label"].to_numpy())
    raw_edges = quantile_edges(df_cal["score"].to_numpy(), n_bins)
    cal_isos = iso.transform(df_cal["score"].to_numpy())
    iso_edges = quantile_edges(cal_isos, n_bins)

    raw_bins, iso_bins = [], []
    df_recent_weeks = df_recent_weeks.copy()
    df_recent_weeks["_week"] = ((df_recent_weeks["date"] - df_recent_weeks["date"].min()).dt.days // 7).astype(int)
    weeks = sorted(df_recent_weeks["_week"].unique())[:12]
    for w in weeks:
        sub = df_recent_weeks[df_recent_weeks["_week"] == w]
        if len(sub) == 0:
            raw_bins.append([0.1] * n_bins); iso_bins.append([0.1] * n_bins); continue
        s = sub["score"].to_numpy()
        raw_bins.append(apply_edges(s, raw_edges))
        s_iso = iso.transform(s)
        iso_bins.append(apply_edges(s_iso, iso_edges))
    while len(raw_bins) < 12:
        raw_bins.append([0.1] * n_bins); iso_bins.append([0.1] * n_bins)
    return raw_bins, iso_bins


def cohort_table(
    df_train: pd.DataFrame, df_oot: pd.DataFrame, cohort_col: str,
    fcr_threshold: float,
) -> List[dict]:
    out = []
    cohorts = sorted(set(df_train[cohort_col].unique()) | set(df_oot[cohort_col].unique()),
                     key=lambda x: -len(df_oot[df_oot[cohort_col] == x]))
    for c in cohorts:
        tr = df_train[df_train[cohort_col] == c]
        oo = df_oot[df_oot[cohort_col] == c]
        if len(oo) < 200 or len(tr) < 200:
            continue
        try:
            tr_auc = roc_auc_score(tr["label"], tr["score"]) if tr["label"].sum() and (1 - tr["label"]).sum() else float("nan")
            oo_auc = roc_auc_score(oo["label"], oo["score"]) if oo["label"].sum() and (1 - oo["label"]).sum() else float("nan")
        except ValueError:
            continue
        oo_top = oo[oo["score"] >= fcr_threshold]
        fcr = float(oo_top["label"].sum()) / max(int(oo["label"].sum()), 1)
        out.append({
            "name": str(c),
            "n": int(len(oo)),
            "train_fraud": float(tr["label"].mean()),
            "oot_fraud": float(oo["label"].mean()),
            "train_auc": float(tr_auc),
            "oot_auc": float(oo_auc),
            "fcr_at_2pct": float(fcr),
        })
    return out


def maturation_curve(days_to_chargeback: np.ndarray, max_days: int = 91) -> List[float]:
    """Empirical CDF of label-arrival latency from a sample of mature chargebacks."""
    days = np.clip(days_to_chargeback, 0, max_days)
    cdf = []
    for d in range(max_days):
        cdf.append(float((days <= d).mean()))
    return cdf


def drift_monitor(df_scored: pd.DataFrame, cal_edges: np.ndarray, baseline_label_rates: np.ndarray) -> List[dict]:
    """Per-day input drift (KL on bin volumes vs uniform) and concept drift
    (per-bin fraud-rate divergence vs calibration baseline)."""
    s = df_scored["score"].to_numpy()
    bins_all = np.clip(np.digitize(s, cal_edges[1:-1]), 0, 9)
    df_scored = df_scored.assign(_decile=bins_all)
    days = sorted(df_scored["date"].unique())
    out = []
    eps = 1e-6
    target_vol = np.full(10, 0.1)
    for i, day in enumerate(days):
        sub = df_scored[df_scored["date"] == day]
        if len(sub) == 0:
            continue
        vol = np.bincount(sub["_decile"], minlength=10) / max(len(sub), 1)
        input_drift = float(np.sum(vol * np.log((vol + eps) / target_vol)))
        rates = []
        for d in range(10):
            m = sub["_decile"] == d
            rates.append(float(sub.loc[m, "label"].mean()) if m.any() else float("nan"))
        rates = np.array(rates)
        valid = ~np.isnan(rates) & (baseline_label_rates > 0)
        if valid.any():
            ratio = rates[valid] / baseline_label_rates[valid]
            concept_drift = float(np.mean(np.abs(np.log(np.clip(ratio, 1e-3, 1e3)))))
        else:
            concept_drift = 0.0
        out.append({"i": i, "inputDrift": max(input_drift, 0.0), "conceptDrift": concept_drift})
    return out


# ---------------------------------------------------------------------------
# Build payload + render
# ---------------------------------------------------------------------------

def build_payload(
    scored_df: pd.DataFrame,
    feature_meta: Mapping[str, FeatureMeta],
    feature_names: Sequence[str],
    model,
    cohort_col: str,
    days_to_chargeback: np.ndarray,
    fcr_threshold: float = 0.5,
    importance_sample: int = 30_000,
    shift_date: Optional[str] = None,
) -> dict:
    """Compute the full data payload consumed by the stability hub HTML."""
    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df_train  = df[df["split"] == "train"]
    df_cal    = df[df["split"] == "cal"]
    df_oot    = df[df["split"] == "oot"]
    df_recent = df[df["split"] == "recent"]

    # Feature importance + drift PSI per feature (train -> oot)
    sample_n = min(importance_sample, len(df_train))
    sample = df_train.sample(sample_n, random_state=0)
    X_sample = sample[list(feature_names)]
    y_sample = sample["label"].to_numpy()

    feats = feature_importance(model, feature_names, feature_meta, X_sample, y_sample)
    for f in feats:
        f["drift_psi"] = float(population_stability_index(
            df_train[f["name"]].to_numpy(),
            df_oot[f["name"]].to_numpy(),
        ))

    cum, s = [], 0.0
    for i, f in enumerate(feats):
        s += f["gain"]; cum.append({"k": i + 1, "share": s})
    top10share = cum[9]["share"] if len(cum) >= 10 else (cum[-1]["share"] if cum else 0)
    top20share = cum[19]["share"] if len(cum) >= 20 else (cum[-1]["share"] if cum else 0)
    stable_share   = sum(f["gain"] for f in feats if f["stable_target"])
    tactical_share = sum(f["gain"] for f in feats if f["bucket"] == "tactical")

    # Calibration set bin edges + top-decile threshold
    cal_edges = np.array(quantile_edges(df_cal["score"].to_numpy(), 10))
    top_decile_threshold = float(cal_edges[-2])  # 90th percentile on cal

    # Daily series across all OOT-eligible windows (everything after train)
    df_post = df[df["split"].isin(["cal", "oot", "recent"])]
    daily = daily_series(df_post.assign(date=df_post["date"]), top_decile_threshold)
    dates = [r["date"] for r in daily]

    # Adversarial validation: train vs OOT
    adv_auc, adv_feats = adversarial_validation(
        df_train[list(feature_names)], df_oot[list(feature_names)],
        feature_names, feature_meta,
    )

    # Decile drift over time, on cal-set edges
    deciles = decile_drift(df_post, cal_edges)

    # Score distributions (3 worlds)
    hist_train  = score_histogram(df_train["score"].to_numpy())
    hist_cal    = score_histogram(df_cal["score"].to_numpy())
    hist_recent = score_histogram(df_recent["score"].to_numpy())

    edges_train  = quantile_edges(df_train["score"].to_numpy(), 10)
    edges_cal    = quantile_edges(df_cal["score"].to_numpy(), 10)
    edges_recent = quantile_edges(df_recent["score"].to_numpy(), 10)

    # Bin volume drift after deploy: evaluate against OOT distribution
    eval_scores = df_oot["score"].to_numpy()
    vol_train_edges  = apply_edges(eval_scores, edges_train)
    vol_cal_edges    = apply_edges(eval_scores, edges_cal)
    vol_recent_edges = apply_edges(eval_scores, edges_recent)

    # Isotonic-then-bin weekly stability on recent weeks
    raw_bins, iso_bins = isotonic_then_bin_stability(df_cal, df_recent if len(df_recent) else df_oot)

    # Cohorts
    cohorts = cohort_table(df_train, df_oot, cohort_col, fcr_threshold)

    # Maturation curve
    mature_cdf = maturation_curve(days_to_chargeback)
    mature_days = list(range(len(mature_cdf)))
    idx95 = next((i for i, v in enumerate(mature_cdf) if v >= 0.95), len(mature_cdf) - 1)
    true_fr = float(df_oot["label"].mean()) if len(df_oot) else 0.004
    lookback = [{"d": d, "true_fr": true_fr, "apparent": true_fr * mature_cdf[d]} for d in mature_days]

    # Conditional vs marginal PSI (top features only)
    psi_feats = []
    for f in feats[:15]:
        marg = f["drift_psi"]
        cond = conditional_psi(df_train, df_oot, f["name"], cohort_col)
        verdict = (
            "stable" if marg < 0.10 and cond < 0.10 else
            "minor" if marg < 0.20 else
            "real shift" if cond > marg * 0.7 else
            "mix change" if cond < marg * 0.4 else
            "partial"
        )
        psi_feats.append({
            "f": f["name"], "marg": float(marg), "cond": float(cond),
            "bucket": f["bucket"], "verdict": verdict,
        })

    # Post-deploy monitor on OOT + recent
    baseline_rates = np.array([
        df_cal[df_cal["score"].between(cal_edges[d], cal_edges[d + 1])]["label"].mean()
        for d in range(10)
    ])
    baseline_rates = np.nan_to_num(baseline_rates, nan=0.001)
    monitor = drift_monitor(df_post, cal_edges, baseline_rates)

    return {
        "dates": dates,
        "feats": feats,
        "cum": cum,
        "top10share": float(top10share),
        "top20share": float(top20share),
        "stable_share": float(stable_share),
        "tactical_share": float(tactical_share),
        "daily": daily,
        "adv_auc": float(adv_auc),
        "adv_feats": adv_feats,
        "deciles": deciles,
        "hist_train": hist_train,
        "hist_cal": hist_cal,
        "hist_recent": hist_recent,
        "edges_train": edges_train,
        "edges_cal": edges_cal,
        "edges_recent": edges_recent,
        "vol_train_edges": vol_train_edges,
        "vol_cal_edges": vol_cal_edges,
        "vol_recent_edges": vol_recent_edges,
        "raw_bins": raw_bins,
        "iso_bins": iso_bins,
        "cohorts": cohorts,
        "mature_days": mature_days,
        "mature_cdf": mature_cdf,
        "idx95": int(idx95),
        "lookback": lookback,
        "psi_feats": psi_feats,
        "monitor": monitor,
        "shift_date": shift_date,
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
    feature_meta: Mapping[str, FeatureMeta],
    feature_names: Sequence[str],
    model,
    cohort_col: str,
    days_to_chargeback: np.ndarray,
    output_path: str | Path,
    **kwargs,
) -> Path:
    payload = build_payload(
        scored_df, feature_meta, feature_names, model,
        cohort_col, days_to_chargeback, **kwargs,
    )
    return render(payload, output_path)
