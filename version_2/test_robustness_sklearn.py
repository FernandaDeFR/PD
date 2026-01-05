import numpy as np
import pandas as pd
form copy import deepcopy
from pd_model_sklearn import (
    generate_synthetic_data,
    train_test_split_pd,
    fit_pd_logit,
    predict_pd_logit
)
from pd_metrics import ks_score
from sklearn.metrics import roc_auc_score, brier_score_loss

def run_single_experiment(
    n: int,
    data_seed: int,
    split_random_state: int
):
    df = generate_synthetic_data(n=n, seed=data_seed)
    X_train, X_test, y_train, y_test = train_test_split_pd(
        df,
        target_col="default_12m",
        test_size=0.3,
        random_state=split_random_state
    )
    base_model, calibrated_model = fit_pd_logit(
        X_train,
        y_train,
        calibrate=True
    )
    pd_test = predict_pd_logit(
        base_model,
        calibrated_model,
        X_test,
        use_calibrated=True
    )
    
    if len(np.unique(y_test)) < 2:
        auc = np.nan
        ks = np.nan
    else:
        auc = roc_auc_score(y_test, pd_test)
        ks = ks_score(y_test, pd_test)
    
    brier = brier_score_loss(y_test, pd_test)

    return {
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "ks": float(ks) if not np.isnan(ks) else np.nan,
        "brier": float(brier)
    }

def performance_across_seeds(
    n_runs: int = 5,
    n: int = 20000,
    base_data_seed: int = 42
):
    """
    Run the experiment multiple times with different split seeds.
    """
    metrics = []
    for i in range(n_runs):
        split_seed = base_data_seed + i
        result = run_single_experiment(
            n=n,
            data_seed=base_data_seed,
            split_random_state=split_seed
        )
        result["run"] = i
        metrics.append(result)
    
    df = pd.DataFrame(metrics)
    summary = {
        "auc_mean": float(df["auc"].mean()),
        "auc_std": float(df["auc"].std()),
        "ks_mean": float(df["ks"].mean()),
        "ks_std": float(df["ks"].std()),
        "brier_mean": float(df["brier"].mean()),
        "brier_std": float(df["brier"].std())
    }
    return df, summary

def input_stress_test(
    base_model,
    calibrated_model,
    X: pd.DataFrame,
    deltas: dict
):
    """"
    Apply simple input perturbations and measure socre changes.
    """

    X_base = X.copy()
    pd_base = predict_pd_logit(
        base_model,
        calibrated_model,
        X_base,
        use_calibrated=True
    )
    results = []

    for col, delta in deltas.items():
        X_perturbed = X_base.copy()
        if col not in X_perturbed.columns:
            continue
        
        if isinstance(delta, tuple) and len(delta) == 2 and delta[0] == "relative":
            factor = delta[1]
            X_perturbed[col] = X_perturbed[col] * (1 + factor)
        else:
            X_perturbed[col] = X_perturbed[col] + delta

        pd_perturbed = predict_pd_logit(
            base_model,
            calibrated_model,
            X_perturbed,
            use_calibrated=True
        )

        diff = pd_perturbed - pd_base
        results.append(
            {
                "feature": col,
                "delta": delta,
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "min_diff": float(diff.min()),
                "max_diff": float(diff.max())
            }
        )
    return pd.DataFrame(results)

def basic_data_checks(
    df: pd.DataFrame,
    target_col: str = "default_12m"
):
    """"
    Simple data sanity checks: types, missing, basic ranges.
    """
    checks = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        n = len(df[col])
        missing_pct = n_missing / n if n > 0 else 0.0

        col_min = df[col].min() if df[col].dtype != "O" else None
        col_max = df[col].max() if df[col].dtype != "O" else None

        checks.append(
            {
                "column": col,
                "dtype": col_type,
                "n_missing": n_missing,
                "missing_pct": float(missing_pct),
                "min": col_min,
                "max": col_max
            }
        )
    return pd.DataFrame(checks)

def consistency_between_datasets(
    df_model: pd.DataFrame,
    df_prod: pd.DataFrame,
    cols: list[str]
):
    """
    Compare basic stats between model and production datasets.
    """
    rows = []
    for col in cols:
        if col not in df_model.columns or col not in df_prod.columns:
            continue

        s_model = df_model[col]
        s_prod = df_prod[col]

        if np.issubdtype(s_model.dtype, np.number):
            stats_model = {
                "mean": float(s_model.mean()),
                "std": float(s_model.std()),
                "p1": float(s_model.quantile(0.01)),
                "p99": float(s_model.quantile(0.99))
            }
            stats_prod = {
                "mean": float(s_prod.mean()),
                "std": float(s_prod.std()),
                "p1": float(s_prod.quantile(0.01)),
                "p99": float(s_prod.quantile(0.99))
            }
        else:
            top_model = s_model.value_counts(normalize=True).head(3)
            top_prod = s_prod.value_counts(normalize=True).head(3)
            stats_model = {"top_categories": top_model.to_dict()}
            stats_prod = {"top_categories": top_prod.to_dict()}

        rows.append(
            {
                "column": col,
                "model_stats": stats_model,
                "prod_stats": stats_prod
            }
        )
    return rows

def leakage_checks(
    df: pd.DataFrame,
    target_col: str,
    candidate_leak_cols: list[str]
):
    """
    Simple leakage check: correlation / AUC of canditate columns vs target.
    """
    y = df[target_col].values
    results = []

    for col in candidate_leak_cols:
        if col not in df.columns:
            continue
        x = df[col].values
        if np.issubdtype(x.dtype, np.number):
            if np.std(x) == 0:
                corr = np.nan
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
        else:
            corr = np.nan

        if np.issubdtype(x.dtype, np.number) and len(np.unique(x)) > 1:
            try:
                auc = roc_auc_score(y, x)
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan

        results.append(
            {
                "column": col,
                "corr_with_taget": corr,
                "auc_as_predictor": float(auc) if not np.isnan(auc) else np.nan
            }
        )
    return pd.DataFrame(results)

if __name__ == "__main__":
    #Robustness across seeds
    df_runs, summary = performance_across_seeds(
        n_runs=5,
        n=20000,
        base_data_seed=42
    )
    print("=== Robustness across seeds ===")
    print("Per-run metrics:")
    print(df_runs)
    print("\nSummary:")
    print(summary)

    # Input stress test
    df = generate_synthetic_data(n=20000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split_pd(
        df,
        target_col="default_12m",
        test_size=0.3,
        random_state=7
    )
    base_model, calibrated_model = fit_pd_logit(
        X_train,
        y_train,
        calibrate=True
    )
    deltas = {
        "qtd_delations_12m": 1,
        "renda": ("relative", 0.1),
        "rel_debit_renda": ("relative", 0.2)
    }
    stress_results = input_stress_test(
        base_model,
        calibrated_model,
        X_test,
        deltas=deltas
    )
    print("\n=== Input Stress Test Results ===")
    print(stress_results)
    
    # Basic data checks
    checks = basic_data_checks(df)
    print("\n=== Basic Data Checks ===")
    print(checks.head())

    # Consistency between datasets
    df_model = df.sample(frac=0.5, random_state=1)
    df_prod = df.sample(frac=0.5, random_state=2)
    cols_to_compare = ["age", "renda", "debit", "rel_debit_renda", "chanel"]
    consistency = consistency_between_datasets(df_model, df_prod, cols_to_compare)

    print("\n=== Consistency Between Datasets ===")
    for row in consistency:
        print(row)

    # Leakage checks
    leak_cols = ["total_debit", "rel_debit_renda", "qtd_delations_12m"]
    leak_results = leakage_checks(
        df=df,
        target_col="default_12m",
        candidate_leak_cols=leak_cols
    )
    print("\n=== Leakage Checks ===")
    print(leak_results)