import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def compute_deciles_table(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute decile table for PD scores.
    Returns a DataFrame with decile, n, defaults, p_avg, rate, expected, observed, lift.
    """
    df_aux = pd.DataFrame({'y_true': y_true, 'pd_scores': pd_scores}).sort_values(
        "pd_scores",
        ascending=False
    )
    df_aux["decile"] = pd.qcut(
        df_aux["pd_scores"].rank(method="first"),
        q=n_bins,
        labels=False
    )
    grp = (
        df_aux.groupby("decile")
        .agg(
            n=("y", "size"),
            defaults=("y", "sum"),
            p_min=("pd_scores", "min"),
            p_max=("pd_scores", "max"),
            p_avg=("pd_scores", "mean"),
            rate=("y", "mean"),
        )
        .reset_index()
        .sort_values("decile", ascending=True)
    )
    overall_rate = df_aux["y"].mean()
    grp["expected"] = grp["p_avg"] * grp["n"]
    grp["observed"] = grp["defaults"]
    grp["lift"] = grp["rate"] / overall_rate
    return grp

def ks_score(
    y_true: np.ndarray,
    pd_scores: np.ndarray
) -> float:
    """
    Compute the KS statistic between responders and non-responders based on PD scores.
    """
    fpr, tpr, _ = roc_curve(y_true, pd_scores)
    ks = float(np.max(tpr - fpr))
    return ks

def psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps = float(1e-6)
) -> float:
    """
    Compute Population Stability Index (PSI) between two score distributions.
    """
    ref = np.array(reference).ravel()
    cur = np.array(current).ravel()

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    # bin edges based on reference quantiles
    bins = np.quantile(ref, q=np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 3:
        bins = np.linspace(ref.min(), ref.max(), n_bins + 1)

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_perc = ref_counts / max(ref_counts.sum(), eps)
    cur_perc = cur_counts / max(cur_counts.sum(), eps)

    psi_vals = (ref_perc - cur_perc) * np.log((ref_perc + eps) / (cur_perc + eps))
    return float(np.sum(psi_vals))