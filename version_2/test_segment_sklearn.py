import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from pd_model_sklearn import (
    generate_synthetic_data,
    train_test_split_pd,
    fit_pd_logit,
    predict_pd_logit
)
from pd_metrics import ks_score, compute_deciles_table

def run_base_experiment_for_segments(
    n: int = 20000,
    data_seed: int = 42,
    split_random_state: int = 7,
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
        calibrate=True,
    )
    pd_test = predict_pd_logit(
        base_model,
        calibrated_model,
        X_test,
        use_calibrated=True
    )
    df_test = X_test.copy()
    df_test["default_12m"] = y_test
    df_test["pd_score"] = pd_test

    return df_test

def segment_performance(
    y_true,
    pd_scores,
    segment
):
    """
    Compute AUC, KS, Brier, default rate by segment value.
    """
    df = pd.DataFrame(
        {
            "y": y_true,
            "p": pd_scores,
            "seg": segment
        }
    )
    results = []

    for val, grp in df.groupby("seg"):
        y = grp["y"].values
        p = grp["p"].values

        if len(np.unique(y)) < 2:
            auc = np.nan
            ks = np.nan
        else:
            auc = roc_auc_score(y, p)
            ks = ks_score(y, p)

        brier = brier_score_loss(y, p)
        default_rate = float(y.mean())
        results.append(
            {
                "segment_value": val,
                "n": int(len(grp)),
                "auc": float(auc) if not np.isnan(auc) else np.nan,
                "ks": float(ks) if not np.isnan(ks) else np.nan,
                "brier": float(brier),
                "default_rate": default_rate
            }
        )
    return pd.DataFrame(results).sort_values("segment_value ").reset_index(drop=True)

def segment_calibration(
    y_true,
    pd_scores,
    segment,
    n_bins: int = 5
):
    """
    Calibration by segment: for each segment, bin scores and compare
    mean PD vs. observed default rate.
    """
    df = pd.DataFrame(
        {
            "y": y_true,
            "p": pd_scores,
            "seg": segment
        }
    )
    rows = []

    for val, grp in df.groupby("seg"):
        if len(grp) < n_bins:
            continue

        grp = grp.sort_values("p", ascending=False)
        grp["bin"] = pd.qcut(
            grp["p"].rank(method="first"),
            q=n_bins,
            labels=False
        )
        agg = grp.groupby("bin").agg(
            n=("y", "size"),
            pd_mean=("p", "mean"),
            default_rate=("y", "mean")
        )
        agg["segment_value"] = val
        agg["bin"] = agg.index
        rows.append(agg.reset_index(drop=True))

        if not rows:
            return pd.DataFrame(
                columns=[
                    "segment_value",
                    "bin",
                    "n",
                    "pd_mean",
                    "default_rate"
                ]
            )
        out = pd.concat(rows, ignore_index=True)
        return out[["segment_value", "bin", "n", "pd_mean", "default_rate"]]
    
def goodness_of_hit_table(
    y_true,
    pd_scores,
    n_bins: int = 10
):
    """
    Simple GH-style table: expected vs observed defaults by score bin.
    """
    dec = compute_deciles_table(
        y_true=y_true,
        pd_scores=pd_scores,
        n_bins=n_bins
    ).copy()
    dec["diff"] = dec["observed"] - dec["expected"]
    dec["diff_pct"] = np.where(
        dec["expected"] > 0,
        dec["diff"] / dec["expected"],
        0.0
    )
    return dec[
        [
            "decile",
            "n",
            "p_min",
            "p_max",
            "p_avg",
            "rate",
            "expected",
            "observed",
            "diff",
            "diff_pct"
            "lift"
        ]
    ]

def hit_rate_by_band(
    y_true,
    pd_scores,
    thresholds: list[float] 
):
    """
    Hit rate by score band defined by thresholds.
    Thresholds must be sorted ascending.
    Bands: (-inf, t1], (t1, t2], ..., (tn, inf)
    """
    df = pd.DataFrame(
        {
            "y": y_true,
            "p": pd_scores
        }
    )
    thr = sorted(thresholds)
    bands = []
    prev = -np.inf
    for t in thr:
        bands.append((prev, t))
        prev = t
    bands.append((prev, np.inf))

    rows = []
    for i, (low, high) in enumerate(bands):
        mask = (df["p"] > low) & (df["p"] <= high)
        grp = df[mask]
        if len(grp) == 0:
            rows.append(
                {
                    "band_id": i,
                    "low": float(low),
                    "high": float(high),
                    "n": 0,
                    "defaults": 0,
                    "default_rate": np.nan
                }
            )
            continue

        y = grp["y"].values
        n = len(y)
        d = int(y.sum())
        rate = float(y.mean())
        rows.append(
            {
                "band_id": i,
                "low": float(low),
                "high": float(high),
                "n": n,
                "defaults": d,
                "default_rate": rate
            }
        )
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_test = run_base_experiment_for_segments()
    y_test = df_test["default_12m"].values
    pd_test = df_test["pd_score"].values
    segment = df_test["chanel"].values
    seg_perf = segment_performance(y_test, pd_test, segment)
    seg_calib = segment_calibration(y_test, pd_test, segment, n_bins=5)
    gh_table = goodness_of_hit_table(y_test, pd_test, n_bins=10)

    thr_list = [0.02, 0.05,, 0.1]
    bands = hit_rate_by_band(y_test, pd_test, thresholds=thr_list)

    print("=== Segment Performance (by channel) ===")
    print(seg_perf)

    print("\n=== Segment Calibration (by 'channel', head) ===")
    print(seg_calib.head())

    print("\n=== Goodness of Hit Table (head) ===")
    print(gh_table.head())

    print("\n=== Hit Rate by Score Bands ===")
    print(bands)