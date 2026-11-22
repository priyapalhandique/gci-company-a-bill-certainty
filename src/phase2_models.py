"""Phase 2: five-fold model comparison, calibration, gains, and revenue-at-risk capture.

Deployed model: LightGBM with the fair feature set (sensitive demographic fields removed).
Run from ``work/src``. Writes all metrics and figures under ``work/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import guardrails
from config import (
    CLIENT_CSV,
    FIGURES,
    OUTPUT,
    RECORD_CSV,
    SEED,
    SENSITIVE_FIELDS,
)

N_SPLITS = 5
TOP_FRAC = 0.10
BASE_MARGIN = 0.60
PROGRAM_COST = 30.0
REL_REDUCTION = 0.15
HORIZON_MONTHS = 12

FIGURES.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

guardrails.set_seeds(SEED)
guardrails.verify_sources()
guardrails.verify_copies()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    client = pd.read_csv(CLIENT_CSV)
    record = pd.read_csv(RECORD_CSV)
    df = record.merge(client, on="Customer_ID", how="inner", validate="one_to_one")
    guardrails.assert_dataset(df)

    df["rev_imputed"] = df["rev_Mean"].fillna(df["rev_Mean"].median())
    df["annual_rev"] = df["rev_imputed"].clip(lower=0) * HORIZON_MONTHS
    df["ovrrev_imputed"] = df["ovrrev_Mean"].fillna(0)

    # Derived, business-interpretable features. All are computed from pre-outcome fields.
    eps = 1e-6
    df["overage_share"] = (df["ovrrev_imputed"] / df["rev_imputed"].clip(lower=eps)).clip(0, 1)
    df["overage_minute_share"] = (
        df["ovrmou_Mean"].fillna(0) / df["mou_Mean"].clip(lower=eps)
    ).clip(0, 1)
    df["revenue_per_minute"] = df["rev_imputed"] / df["mou_Mean"].clip(lower=eps)
    df["care_calls_per_month"] = df["custcare_Mean"].fillna(0) / df["months"].clip(lower=1)
    df["equipment_age_ratio"] = df["eqpdays"].fillna(0) / (df["months"].clip(lower=1) * 30.44)
    df["change_rev_pct"] = df["change_rev"].fillna(0) / df["rev_imputed"].clip(lower=eps)
    df["change_mou_pct"] = df["change_mou"].fillna(0) / df["mou_Mean"].clip(lower=eps)
    df["is_overage_payer"] = (df["ovrrev_imputed"] > 0).astype(int)
    df["device_gt_12m"] = (df["eqpdays"] > 365).astype(int)
    df["handset_price_per_month"] = df["hnd_price"].fillna(0) / df["months"].clip(lower=1)
    return df


def feature_columns(df: pd.DataFrame, fair: bool) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c not in {"Customer_ID", "churn", "rev_imputed", "annual_rev", "ovrrev_imputed"}
    ]
    if fair:
        cols = [c for c in cols if c not in SENSITIVE_FIELDS]
    return cols


def is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def tree_matrix(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = df[features].copy()
    cat_cols = [c for c in features if is_categorical(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype("category").cat.codes.astype(float)
        X.loc[X[c] < 0, c] = np.nan
    # Drop constant columns; they add noise and no signal.
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep].astype(float), [c for c in cat_cols if c in keep]


def logistic_pipeline(features: list[str], df: pd.DataFrame) -> Pipeline:
    cat_cols = [c for c in features if is_categorical(df[c])]
    num_cols = [c for c in features if c not in cat_cols]
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20),
            ),
        ]
    )
    pre = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=400, tol=1e-3, random_state=SEED)),
        ]
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def cross_fitted_isotonic(p: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Out-of-fold isotonic calibration, so the calibrated score is not fit on itself."""
    out = np.zeros_like(p)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(p, y):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p[tr], y[tr])
        out[te] = iso.predict(p[te])
    return out


def top_slice_metrics(
    y: np.ndarray,
    score: np.ndarray,
    revenue: np.ndarray,
    frac: float = TOP_FRAC,
    probability: np.ndarray | None = None,
) -> dict:
    n = len(y)
    k = max(1, int(round(n * frac)))
    order = np.argsort(-score, kind="mergesort")[:k]
    yk = y[order]
    revk = revenue[order]
    prob = score if probability is None else probability
    rar_total = float((y * revenue).sum())
    return {
        "customers": int(k),
        "churn_rate": float(yk.mean()),
        "lift": float(yk.mean() / y.mean()),
        "avg_monthly_rev": float((revk / HORIZON_MONTHS).mean()),
        "annual_rev": float(revk.sum()),
        "actual_rar": float((yk * revk).sum()),
        "rar_capture": float((yk * revk).sum() / rar_total),
        "expected_rar": float((prob[order] * revk).sum()),
        "threshold": float(score[order][-1]),
    }


def gains_table(y: np.ndarray, scores: dict[str, np.ndarray], revenue: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(y)
    total_churn = float(y.sum())
    total_rar = float((y * revenue).sum())
    for name, p in scores.items():
        order = np.argsort(-p, kind="mergesort")
        ysorted = y[order]
        revsorted = revenue[order]
        cum_churn = np.cumsum(ysorted)
        cum_rar = np.cumsum(ysorted * revsorted)
        for frac in np.arange(0.05, 1.0001, 0.05):
            k = min(n, max(1, int(round(n * frac))))
            rows.append(
                {
                    "score": name,
                    "fraction": round(float(frac), 2),
                    "churn_capture": float(cum_churn[k - 1] / total_churn),
                    "rar_capture": float(cum_rar[k - 1] / total_rar),
                    "churn_rate": float(ysorted[:k].mean()),
                }
            )
    return pd.DataFrame(rows)


def group_audit(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, col: str, frac: float = TOP_FRAC) -> dict:
    k = max(1, int(round(len(y) * frac)))
    cutoff = np.sort(p)[-k]
    selected = p >= cutoff
    overall_tpr = float(y[selected].mean())
    # Group with boolean masks rather than groupby: pandas raises
    # "Categorical categories cannot be null" for NaN group keys in some
    # versions, including the Colab runtime. Missing values get their own group.
    labels = df[col].astype(object).where(df[col].notna(), "missing").to_numpy()
    groups = []
    for value in pd.unique(labels):
        idx = np.flatnonzero(labels == value)
        sel = selected[idx]
        groups.append(
            {
                "group": str(value),
                "n": int(len(idx)),
                "churn_rate": float(y[idx].mean()),
                "selection_rate": float(sel.mean()),
                "tpr": float(y[idx][sel].mean()) if sel.sum() else None,
            }
        )
    sel_rates = [g["selection_rate"] for g in groups if g["n"] >= 500]
    tprs = [g["tpr"] for g in groups if g["n"] >= 500 and g["tpr"] is not None]
    return {
        "overall_tpr": overall_tpr,
        "selection_rate_gap": float(max(sel_rates) - min(sel_rates)) if sel_rates else None,
        "tpr_gap": float(max(tprs) - min(tprs)) if tprs else None,
        "groups": groups,
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def run_model(name: str, X, y, df, features, model_factory, use_logistic: bool):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        if use_logistic:
            model = model_factory(features, df)
            model.fit(df.iloc[tr][features], y.iloc[tr])
            p = model.predict_proba(df.iloc[te][features])[:, 1]
        else:
            model = model_factory()
            model.fit(X.iloc[tr], y.iloc[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        fold_aucs.append(float(roc_auc_score(y.iloc[te], p)))
        print(f"  {name} fold {fold}/{N_SPLITS}: AUC {fold_aucs[-1]:.4f} ({time.time()-t0:.1f}s)")
    return oof, fold_aucs, model


def main() -> None:
    df = load_data()
    y = df["churn"].astype(int)
    revenue = df["annual_rev"].to_numpy()

    fair_features = feature_columns(df, fair=True)
    full_features = feature_columns(df, fair=False)
    guardrails.assert_no_target_leak(fair_features)
    guardrails.assert_fair_features(fair_features)

    X_fair, cat_fair = tree_matrix(df, fair_features)
    X_full, cat_full = tree_matrix(df, full_features)
    print(f"Fair features: {X_fair.shape[1]} ({len(cat_fair)} categorical)")
    print(f"Full features: {X_full.shape[1]} ({len(cat_full)} categorical)")

    factories = {
        "Logistic": (
            lambda: None,
            lambda features, frame: logistic_pipeline(features, frame),
            True,
        ),
        "XGBoost (full)": (
            lambda: XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="auc",
                random_state=SEED,
                n_jobs=-1,
            ),
            None,
            False,
        ),
        "LightGBM (full)": (
            lambda: LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            None,
            False,
        ),
        "LightGBM (fair)": (
            lambda: LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            None,
            False,
        ),
    }

    oof = {}
    fold_auc = {}
    fitted = {}
    for name, (factory, logistic_factory, use_logistic) in factories.items():
        print(f"Training {name} ...")
        X = X_fair if name in {"Logistic", "LightGBM (fair)"} else X_full
        features = fair_features if name in {"Logistic", "LightGBM (fair)"} else full_features
        oof[name], fold_auc[name], fitted[name] = run_model(
            name, X, y, df, features, factory if not use_logistic else logistic_factory, use_logistic
        )

    # Comparison table -----------------------------------------------------------------
    rows = []
    calibrated = {}
    for name, p in oof.items():
        p_cal = cross_fitted_isotonic(p, y.to_numpy())
        calibrated[name] = p_cal
        top_p = top_slice_metrics(y.to_numpy(), p, revenue)
        top_v = top_slice_metrics(y.to_numpy(), p * revenue, revenue, probability=p)
        rows.append(
            {
                "model": name,
                "auc": float(roc_auc_score(y, p)),
                "auc_fold_std": float(np.std(fold_auc[name])),
                "pr_auc": float(average_precision_score(y, p)),
                "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))),
                "brier": float(brier_score_loss(y, p)),
                "ece_raw": expected_calibration_error(y.to_numpy(), p),
                "ece_isotonic": expected_calibration_error(y.to_numpy(), p_cal),
                "top10_churn_p": top_p["churn_rate"],
                "top10_lift_p": top_p["lift"],
                "top10_avg_bill_p": top_p["avg_monthly_rev"],
                "top10_rar_capture_p": top_p["rar_capture"],
                "top10_churn_value": top_v["churn_rate"],
                "top10_avg_bill_value": top_v["avg_monthly_rev"],
                "top10_rar_capture_value": top_v["rar_capture"],
                "top10_expected_rar_value": top_v["expected_rar"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values("auc", ascending=False)
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string(index=False))

    # Baselines and gains --------------------------------------------------------------
    scores = {name: oof[name] for name in oof}
    scores["Revenue only"] = revenue
    scores["LightGBM (fair, value)"] = oof["LightGBM (fair)"] * revenue
    gains = gains_table(y.to_numpy(), scores, revenue)
    gains.to_csv(OUTPUT / "gains_curve.csv", index=False)

    # Deployed model: fair LightGBM ranked by expected revenue at risk -----------------
    deployed = oof["LightGBM (fair)"]
    deployed_cal = calibrated["LightGBM (fair)"]
    deployed_top = top_slice_metrics(y.to_numpy(), deployed * revenue, revenue, probability=deployed)
    deployed_top_p = top_slice_metrics(y.to_numpy(), deployed, revenue)

    # Probability-only comparison (value-blind) ---------------------------------------
    prob_top = deployed_top_p

    # Economics at the model's target list --------------------------------------------
    def economics(seg: dict, margin=BASE_MARGIN, cost=PROGRAM_COST, rel=REL_REDUCTION) -> dict:
        retained = seg["customers"] * seg["churn_rate"] * rel
        margin_recovered = retained * seg["avg_monthly_rev"] * HORIZON_MONTHS * margin
        spend = seg["customers"] * cost
        net = margin_recovered - spend
        breakeven_rel = spend / max(seg["customers"] * seg["churn_rate"] * seg["avg_monthly_rev"] * HORIZON_MONTHS * margin, 1e-9)
        return {
            "customers": seg["customers"],
            "churn_rate": seg["churn_rate"],
            "avg_monthly_rev": seg["avg_monthly_rev"],
            "retained_customers": retained,
            "margin_recovered": margin_recovered,
            "program_cost": spend,
            "net": net,
            "roi": net / spend if spend else None,
            "breakeven_relative_reduction": breakeven_rel,
            "breakeven_absolute_pp": breakeven_rel * seg["churn_rate"] * 100,
        }

    econ_value = economics(deployed_top)
    econ_prob = economics(prob_top)

    # Sensitivity table ---------------------------------------------------------------
    sens = []
    for margin in (0.40, 0.50, 0.60, 0.70):
        for cost in (20, 30, 60):
            for rel in (0.05, 0.10, 0.15, 0.20, 0.25):
                e = economics(deployed_top, margin=margin, cost=cost, rel=rel)
                sens.append(
                    {
                        "margin": margin,
                        "cost": cost,
                        "relative_reduction": rel,
                        "net": e["net"],
                        "roi": e["roi"],
                    }
                )
    pd.DataFrame(sens).to_csv(OUTPUT / "economics_sensitivity.csv", index=False)

    # Feature importance (fair LightGBM, last fold) -----------------------------------
    importances = pd.Series(
        fitted["LightGBM (fair)"].feature_importances_, index=X_fair.columns
    ).sort_values(ascending=False)
    importances.to_csv(OUTPUT / "feature_importance_fair.csv", header=["importance"])

    # Fairness audit ------------------------------------------------------------------
    fairness = {
        "area": group_audit(df, y.to_numpy(), deployed, "area"),
        "creditcd": group_audit(df, y.to_numpy(), deployed, "creditcd"),
    }
    (OUTPUT / "fairness_audit.json").write_text(json.dumps(fairness, indent=2), encoding="utf-8")

    # Save predictions ----------------------------------------------------------------
    pred = pd.DataFrame(
        {
            "Customer_ID": df["Customer_ID"],
            "churn": y,
            "annual_rev": revenue,
            "p_logistic": oof["Logistic"],
            "p_xgb_full": oof["XGBoost (full)"],
            "p_lgbm_full": oof["LightGBM (full)"],
            "p_lgbm_fair": deployed,
            "p_lgbm_fair_cal": deployed_cal,
        }
    )
    pred.to_csv(OUTPUT / "oof_predictions.csv", index=False)

    # Metrics file --------------------------------------------------------------------
    metrics = {
        "n": int(len(df)),
        "churn_rate": float(y.mean()),
        "total_annual_rev": float(revenue.sum()),
        "total_annual_rar": float((y.to_numpy() * revenue).sum()),
        "model_comparison": comparison.round(6).to_dict(orient="records"),
        "deployed_model": "LightGBM (fair)",
        "deployed_auc": float(roc_auc_score(y, deployed)),
        "deployed_pr_auc": float(average_precision_score(y, deployed)),
        "deployed_ece_raw": expected_calibration_error(y.to_numpy(), deployed),
        "deployed_ece_isotonic": expected_calibration_error(y.to_numpy(), deployed_cal),
        "top10_probability_only": deployed_top_p,
        "top10_value_ranked": deployed_top,
        "economics_value": econ_value,
        "economics_probability_only": econ_prob,
        "top_features": importances.head(12).round(2).to_dict(),
        "fairness": fairness,
    }
    (OUTPUT / "metrics_models.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Figures -------------------------------------------------------------------------
    # 1. Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ordered = comparison.sort_values("auc")
    axes[0].barh(ordered["model"], ordered["auc"], color="#0E7C86")
    axes[0].set(title="OOF ROC-AUC (5-fold)", xlim=(0.5, 0.75), xlabel="AUC")
    axes[0].bar_label(axes[0].containers[0], fmt="%.3f", padding=2, fontsize=9)
    axes[1].barh(ordered["model"], ordered["top10_rar_capture_value"] * 100, color="#F2A541")
    axes[1].set(title="Share of annual RaR captured in top decile", xlabel="% of RaR")
    axes[1].bar_label(axes[1].containers[0], fmt="%.1f%%", padding=2, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Calibration
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for name, color in [("LightGBM (fair)", "#0E7C86"), ("Logistic", "#F2A541")]:
        p = oof[name]
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum():
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, marker="o", lw=2, label=name, color=color)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set(title="Calibration (10 bins)", xlabel="Predicted churn", ylabel="Observed churn")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Gains: churners and RaR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, color in [
        ("LightGBM (fair)", "#0E7C86"),
        ("LightGBM (fair, value)", "#F2A541"),
        ("Revenue only", "#7F8C8D"),
    ]:
        g = gains[gains["score"] == name]
        axes[0].plot(g["fraction"] * 100, g["churn_capture"] * 100, marker="o", ms=3, label=name, color=color)
        axes[1].plot(g["fraction"] * 100, g["rar_capture"] * 100, marker="o", ms=3, label=name, color=color)
    for ax, title in zip(axes, ["Churners captured", "Revenue at risk captured"]):
        ax.plot([0, 100], [0, 100], ls="--", color="grey", lw=1)
        ax.set(title=title, xlabel="% of customers targeted", ylabel="% captured")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_gains.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Feature importance
    top_imp = importances.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(top_imp.index, top_imp.values, color="#0B1F3A")
    ax.set(title="LightGBM (fair): top drivers (gain)", xlabel="Importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nDEPLOYED MODEL: LightGBM (fair)")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in deployed_top.items()}, indent=2))
    print("\nECONOMICS (value-ranked top decile)")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in econ_value.items()}, indent=2))
    print("\nECONOMICS (probability-only top decile)")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in econ_prob.items()}, indent=2))
    print("\nTOP FEATURES")
    print(importances.head(12).round(1).to_string())
    print("\nWrote metrics, CSVs, and figures.")


if __name__ == "__main__":
    main()
