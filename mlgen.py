"""
World Trade — End‑to‑End Regression Pipeline
===========================================

Goal
----
Train strong regression models to predict `export_value_usd_mln` from the
`world_trade_synth_fast.csv` dataset, evaluate them with solid metrics, export
plots + artifacts, and produce BI‑friendly files you can use in Power BI,
Tableau, Amazon QuickSight, and Azure.

How to run
----------
python trade_pipeline.py \
  --data ./world_trade_synth_fast.csv \
  --outdir ./artifacts_trade \
  --max-rows 20000 \
  --models rf hgb \
  --time-aware-split

Arguments (key ones)
--------------------
--data                Path to CSV (required)
--outdir              Output directory (default: ./artifacts_trade)
--max-rows            Row cap for fast training (default: 20000)
--models              Which models: [rf hgb ridge sgd] (default: rf hgb)
--time-aware-split    Use last 2 years for test (default: False -> random split)
--test-size           Random split fraction when time-aware is off (default 0.2)

Artifacts produced
------------------
- metrics.csv                      → RMSE/MAE/R² for train & test per model
- best_model_<name>.joblib         → serialized best model
- summary_report.json              → quick run manifest
- plots/
    scatter_actual_vs_pred_<model>.png
    residuals_hist_<model>.png
    feature_importance_<model>.png (if available)
- bi/
    bi_sample_trade.csv            → sampled rows for BI (fast)
    bi_aggregated_trade.csv        → optional aggregated file (slower; enable flag)

BI setup cheatsheet (high level)
--------------------------------
Power BI:  Get Data → Text/CSV → pick bi_sample_trade.csv → build visuals like
           Year slicer, bar chart by reporter_iso3, scatter (unit_price vs value).
Tableau:   Connect to Text file → pick bi_sample_trade.csv → Drag fields to Rows/Columns.
QuickSight: New dataset → Upload file → SPICE import → Build analyses.
Azure:     Upload CSV to Azure Blob or Synapse → connect via Power BI or Azure ML.

"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt


TARGET = "export_value_usd_mln"
CATEGORICAL = ["reporter_iso3", "partner_iso3", "product_code", "fta_active"]
NUMERIC = [
    "year",
    "distance_km",
    "adval_tariff_pct",
    "reporter_gdp_bln",
    "partner_gdp_bln",
    "reporter_pop_m",
    "partner_pop_m",
    "reporter_cpi",
    "partner_cpi",
    "quantity_tonnes",
    "unit_price_usd_per_tonne",
]


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def stratified_sample_by_year(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return shuffle(df, random_state=seed)
    frac = max_rows / len(df)
    out = (
        df.groupby("year", group_keys=False)
        .apply(lambda x: x.sample(frac=min(1.0, frac), random_state=seed))
    )
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return shuffle(out, random_state=seed)


def time_aware_split(df: pd.DataFrame):
    years = sorted(df["year"].unique())
    if len(years) < 3:
        # fallback to random split
        return train_test_split(df, test_size=0.2, random_state=42)
    test_years = years[-2:]
    train_df = df[~df["year"].isin(test_years)]
    test_df = df[df["year"].isin(test_years)]
    return train_df, test_df


def random_split(df: pd.DataFrame, test_size: float):
    return train_test_split(df, test_size=test_size, random_state=42)


def build_pipelines(selected: List[str]) -> Dict[str, Pipeline]:
    """Return a dict of sklearn pipelines keyed by short model names.
    Models available:
      - ridge: Ridge with OneHot (sparse-safe) + StandardScaler
      - sgd:   SGDRegressor + OneHot
      - rf:    RandomForestRegressor (numeric only)
      - hgb:   HistGradientBoostingRegressor (numeric only)
    """
    pipes = {}

    if any(m in selected for m in ["ridge", "sgd"]):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        scaler = StandardScaler(with_mean=False)
        pre_linear = ColumnTransformer(
            transformers=[("cat", ohe, CATEGORICAL), ("num", scaler, NUMERIC)],
            sparse_threshold=0.3,
        )
        if "ridge" in selected:
            pipes["ridge_ohe"] = Pipeline([
                ("prep", pre_linear),
                ("model", Ridge(alpha=1.0, solver="sag", random_state=42)),
            ])
        if "sgd" in selected:
            pipes["sgd_ohe"] = Pipeline([
                ("prep", pre_linear),
                ("model", SGDRegressor(random_state=42, max_iter=1500, tol=1e-3)),
            ])

    if "rf" in selected:
        pre_num = ColumnTransformer([("num", "passthrough", NUMERIC)])
        pipes["rf_numeric"] = Pipeline([
            ("prep", pre_num),
            ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
        ])

    if "hgb" in selected:
        pre_num = ColumnTransformer([("num", "passthrough", NUMERIC)])
        pipes["hgb_numeric"] = Pipeline([
            ("prep", pre_num),
            ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)),
        ])

    return pipes

def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }



def plot_scatter(y_true, y_pred, title, outpath):
    plt.figure()
    plt.scatter(y_true, y_pred, s=4, alpha=0.5)
    plt.xlabel("Actual export_value_usd_mln")
    plt.ylabel("Predicted export_value_usd_mln")
    plt.title(title)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, title, outpath):
    resid = y_true - y_pred
    plt.figure()
    plt.hist(resid, bins=60)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(title)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def maybe_plot_importance(model_name: str, pipe: Pipeline, outdir: str):
    # RandomForest: feature_importances_
    if model_name.startswith("rf_"):
        try:
            imp = pipe.named_steps["model"].feature_importances_
            names = NUMERIC
            plt.figure()
            plt.bar(range(len(names)), imp)
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.ylabel("Feature Importance")
            plt.title("Feature Importance (RandomForest)")
            op = Path(outdir, "plots", "feature_importance_rf.png")
            op.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(op, bbox_inches="tight")
            plt.close()
        except Exception:
            pass


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_data(args.data)
    schema = {c: str(df[c].dtype) for c in df.columns}
    rows_total = int(len(df))
    year_min, year_max = int(df['year'].min()), int(df['year'].max())

    # Sample for speed
    df_s = stratified_sample_by_year(df, args.max_rows, seed=42)

    # Split
    if args.time_aware_split:
        train_df, test_df = time_aware_split(df_s)
    else:
        train_df, test_df = random_split(df_s, args.test_size)

    X_train = train_df[CATEGORICAL + NUMERIC] if any(m in args.models for m in ["ridge","sgd"]) else train_df[NUMERIC]
    X_test  = test_df[CATEGORICAL + NUMERIC] if any(m in args.models for m in ["ridge","sgd"]) else test_df[NUMERIC]
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    # Build and train
    pipes = build_pipelines(args.models)

    rows = []
    best = None

    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        pred_tr = pipe.predict(X_train)
        pred_te = pipe.predict(X_test)
        m_tr = eval_metrics(y_train, pred_tr)
        m_te = eval_metrics(y_test, pred_te)

        rows.append({"model": name, "split": "train", **m_tr})
        rows.append({"model": name, "split": "test", **m_te})

        # Plots
        plot_scatter(y_test, pred_te, f"Actual vs Predicted — {name}", outdir/"plots"/f"scatter_actual_vs_pred_{name}.png")
        plot_residuals(y_test, pred_te, f"Residuals — {name}", outdir/"plots"/f"residuals_hist_{name}.png")
        maybe_plot_importance(name, pipe, str(outdir))

        # Track best by RMSE on test
        if best is None or m_te["rmse"] < best["rmse"]:
            best = {"name": name, **m_te}
            joblib.dump(pipe, outdir/f"best_model_{name}.joblib")

    # Save metrics
    mdf = pd.DataFrame(rows)
    mdf.to_csv(outdir/"metrics.csv", index=False)

    # Small report
    report = {
        "rows_total": rows_total,
        "rows_used_for_modeling": int(len(df_s)),
        "years": [year_min, year_max],
        "schema": schema,
        "target": TARGET,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "models_trained": list(pipes.keys()),
        "best_model": best["name"] if best else None,
        "test_scores": {"rmse": best.get("rmse"), "mae": best.get("mae"), "r2": best.get("r2")} if best else None,
    }
    (outdir/"summary_report.json").write_text(json.dumps(report, indent=2))

    # BI files
    bi_dir = outdir/"bi"
    bi_dir.mkdir(parents=True, exist_ok=True)
    # fast sample (no heavy groupby)
    bi_cols = [
        "year","reporter_iso3","partner_iso3","product_code",
        "export_value_usd_mln","quantity_tonnes","unit_price_usd_per_tonne"
    ]
    df[bi_cols].sample(n=min(100_000, len(df)), random_state=42).to_csv(bi_dir/"bi_sample_trade.csv", index=False)

    if args.make_aggregate:
        agg = (
            df[bi_cols]
            .groupby(["year","reporter_iso3","partner_iso3","product_code"], as_index=False)
            .agg(
                export_value_usd_mln_total=("export_value_usd_mln","sum"),
                quantity_tonnes_total=("quantity_tonnes","sum"),
                unit_price_usd_per_tonne_avg=("unit_price_usd_per_tonne","mean"),
            )
        )
        agg.to_csv(bi_dir/"bi_aggregated_trade.csv", index=False)

    print("Run complete. Artifacts in:", outdir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="./artifacts_trade")
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--models", nargs="+", default=["rf","hgb"], choices=["rf","hgb","ridge","sgd"])  # combine as you like
    p.add_argument("--time-aware-split", action="store_true")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--make-aggregate", action="store_true", help="also create aggregated BI file (slower)")
    args = p.parse_args()
    run(args)
