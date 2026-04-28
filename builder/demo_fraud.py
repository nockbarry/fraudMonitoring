"""End-to-end demo: sophisticated synthetic fraud dataset → trained models →
all three hub reports (model performance, feature analysis, stability).

Run::
    python demo_fraud.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_dataset
from stability_hub_builder import FeatureMeta as StabFM, build_report as build_stability
from model_perf_builder import build_report as build_model_perf, feature_importance
from feature_hub_builder import build_report as build_feature_hub


OUT_DIR = Path(__file__).parent.parent


def _fmt_count(n): return f"{n:,}"


def _fmt_pct(p): return f"{p * 100:.3f}%"


# ---------------------------------------------------------------------------
# Synthetic chargeback maturation sample
# ---------------------------------------------------------------------------

def synth_chargeback_days(n: int = 4000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.weibull(1.5, n) * 22   # ~95% by ~40 days


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_models(df: pd.DataFrame, feature_names: list[str]) -> dict:
    Xtr = df.loc[df["split"] == "train", feature_names]
    ytr = df.loc[df["split"] == "train", "label"].to_numpy()

    print(f"  training GBM on {len(Xtr):,} rows × {len(feature_names)} features")
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=80, random_state=0,
    )
    t0 = time.time()
    gbm.fit(Xtr, ytr)
    print(f"    GBM trained in {time.time()-t0:.1f}s")

    print(f"  training challenger LR")
    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(C=0.5, max_iter=300, n_jobs=-1, random_state=0)),
    ])
    t0 = time.time()
    lr.fit(Xtr, ytr)
    print(f"    LR  trained in {time.time()-t0:.1f}s")

    return {"gbm": gbm, "lr": lr}


def score_all(df: pd.DataFrame, feature_names: list[str], models: dict) -> dict:
    """Returns {model_id: scored_df_copy}."""
    out = {}
    X = df[feature_names]
    for mid, m in models.items():
        scored = df.copy()
        scored["score"] = m.predict_proba(X)[:, 1]
        out[mid] = scored
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cache_path = Path(__file__).parent / ".cache_dataset.parquet"
    meta_cache = Path(__file__).parent / ".cache_meta.json"

    print("[1/5] generating dataset")
    t0 = time.time()
    if cache_path.exists() and meta_cache.exists() and "--regen" not in sys.argv:
        import json as _json
        df = pd.read_parquet(cache_path)
        cache = _json.loads(meta_cache.read_text())
        feature_names = cache["feature_names"]
        cohort_dims = cache["cohort_dims"]
        from dataset import FeatureMeta as DSFM
        meta = {n: DSFM(**v) for n, v in cache["meta"].items()}
        print(f"      loaded cache: {_fmt_count(len(df))} rows × {len(feature_names)} features in {time.time()-t0:.1f}s")
    else:
        df, feature_names, meta, cohort_dims = make_dataset(seed=42)
        df.to_parquet(cache_path, index=False)
        import json as _json
        meta_cache.write_text(_json.dumps({
            "feature_names": list(feature_names),
            "cohort_dims": list(cohort_dims),
            "meta": {n: {"bucket": m.bucket, "stable_target": m.stable_target} for n, m in meta.items()},
        }))
        print(f"      generated and cached: {_fmt_count(len(df))} rows × {len(feature_names)} features in {time.time()-t0:.1f}s")
    print(f"      buckets: invariant={sum(1 for v in meta.values() if v.bucket=='invariant')}"
          f"  tactical={sum(1 for v in meta.values() if v.bucket=='tactical')}"
          f"  compositional={sum(1 for v in meta.values() if v.bucket=='compositional')}")
    for s in ("train", "cal", "oot", "recent"):
        sub = df[df["split"] == s]
        print(f"      {s:7s} n={_fmt_count(len(sub))}  fraud={_fmt_pct(sub['label'].mean())}")

    print("[2/5] training models")
    models = train_models(df, feature_names)
    scored = score_all(df, feature_names, models)

    # ------------------------------------------------------------------
    # Stability hub (uses GBM, all four splits)
    # ------------------------------------------------------------------
    print("[3/5] building stability hub")
    t0 = time.time()
    stab_meta = {n: StabFM(m.bucket, m.stable_target) for n, m in meta.items()}
    shift_date = (df["date"].min() + pd.Timedelta(days=49)).date().isoformat()
    chargeback_days = synth_chargeback_days()
    out_stab = OUT_DIR / "stability_monitoring_hub_real.html"
    build_stability(
        scored_df=scored["gbm"][["date", "score", "label", "split"] + feature_names + ["channel", "merchant_mcc", "bin_country", "amt_band", "tenure_bucket"]],
        feature_meta=stab_meta,
        feature_names=feature_names,
        model=models["gbm"],
        cohort_col="channel",
        days_to_chargeback=chargeback_days,
        output_path=out_stab,
        shift_date=shift_date,
    )
    print(f"      wrote {out_stab.name}  ({out_stab.stat().st_size:,} bytes, {time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Model performance hub (multiple models × splits)
    # ------------------------------------------------------------------
    print("[4/5] building model performance hub")
    t0 = time.time()
    model_perf_cohorts = ["channel", "merchant_mcc", "bin_country", "amt_band", "tenure_bucket"]
    scored_models_payload = {
        "gbm": {
            "name": "GBM (champion)",
            "df_by_split": {
                s: scored["gbm"][scored["gbm"]["split"] == s][["date", "label", "score"] + model_perf_cohorts]
                for s in ("train", "cal", "oot", "recent")
            },
        },
        "lr": {
            "name": "LR (challenger)",
            "df_by_split": {
                s: scored["lr"][scored["lr"]["split"] == s][["date", "label", "score"] + model_perf_cohorts]
                for s in ("train", "cal", "oot", "recent")
            },
        },
    }
    # Compute feature importance on a sample using GBM
    sample = scored["gbm"][scored["gbm"]["split"] == "train"].sample(20_000, random_state=0)
    fi = feature_importance(models["gbm"], feature_names,
                            sample[feature_names], sample["label"].to_numpy())
    out_mp = OUT_DIR / "model_performance_hub_real.html"
    build_model_perf(
        scored_models=scored_models_payload,
        cohort_dims=model_perf_cohorts,
        feature_importance_data=fi[:30],   # top 30 for the report
        output_path=out_mp,
        model_name="Fraud GBM v1",
        champion_id="gbm",
    )
    print(f"      wrote {out_mp.name}  ({out_mp.stat().st_size:,} bytes, {time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Feature analysis hub (uses GBM)
    # ------------------------------------------------------------------
    print("[5/5] building feature analysis hub")
    t0 = time.time()
    out_fh = OUT_DIR / "feature_analysis_hub_real.html"
    build_feature_hub(
        scored_df=scored["gbm"][["date", "label", "score", "split"] + feature_names + ["channel", "merchant_mcc", "bin_country"]],
        feature_names=feature_names,
        feature_meta=meta,
        model=models["gbm"],
        cat_cols=["channel", "merchant_mcc", "bin_country"],
        output_path=out_fh,
    )
    print(f"      wrote {out_fh.name}  ({out_fh.stat().st_size:,} bytes, {time.time()-t0:.1f}s)")

    print("\nAll three hubs generated. Open in a browser to inspect:")
    for p in (out_stab, out_mp, out_fh):
        print(f"  file://{p}")


if __name__ == "__main__":
    main()
