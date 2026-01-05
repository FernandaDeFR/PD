import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from pd_model_sklearn import (
    generate_synthetic_data,
    train_test_split_pd,
    fit_pd_logit,
    predict_pd_logit,
    evaluate_pd_logit
)
from pd_metrics import compute_deciles_table, ks_score

def run_base_experiment(
    n: int = 20000,
    data_seed: int = 42,
    split_random_state: int = 7
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
    pd_train = predict_pd_logit(
        base_model,
        calibrated_model,
        X_train,
        use_calibrated=True
    )
    pd_test = predict_pd_logit(
        base_model,
        calibrated_model,
        X_test,
        use_calibrated=True
    )
    metrics_test = evaluate_pd_logit(y_test, pd_test)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pd_train": pd_train,
        "pd_test": pd_test,
        "metrics_test": metrics_test
    }

def compute_additional_performance_metrics(
    y_true,
    pd_scores
):
    auc = roc_auc_score(y_true, pd_scores)
    gini = 2 * auc - 1.0
    ks = ks_score(y_true, pd_scores)
    ap = average_precision_score(y_true, pd_scores)
    thr = float(np.median(pd_scores))
    y_pred = (pd_scores >= thr).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "auc": float(auc),
        "gini": float(gini),
        "ks": float(ks),
        "average_precision": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "threshold_median": thr
    }

def calibration_summary(
    y_true,
    pd_scores,
    n_bins: int = 10
):
    """
    Simple calibration table: for each bin, compare average PD vs observed default rate.
    """
    deciles = compute_deciles_table(
        y_true=y_true, 
        pd_scores=pd_scores, 
        n_bins=n_bins
    )
    calib = deciles[["decile", "p_avg", "rate"]].copy()
    calib = calib.rename(
        columns={
            "p_avg": "pd_mean",
            "rate": "default_rate"
        }
    )
    return calib

def summarize_pr_curve(
    y_true,
    pd_scores
):
    """
    Summarize precision-recall curve with a few points.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, pd_scores)
    idx = np.linspace(0, len(thresholds) - 1, num=min(5, len(thresholds)), dtype=int)
    summary = []
    for i in idx:
        summary.append({
            "threshold": float(thresholds[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i])
        }
    )
    return summary

if __name__ == "__main__":
    result = run_base_experiment()
    y_test = result["y_test"]
    pd_test = result["pd_test"]
    base_metrics = result["metrics_test"]

    add_metrics = compute_additional_performance_metrics(y_test, pd_test)
    calib = calibration_summary(y_test, pd_test, n_bins=10)
    pr_summary = summarize_pr_curve(y_test, pd_test)

    print("=== Global Performance (test set) ===")
    print(f"AUC: {base_metrics['auc']:.4f}")
    print(f"Brier Score: {base_metrics['brier']:.4f}")
    print(f"Default Rate: {base_metrics['default_rate']:.3%}")
    print(f"Gini: {add_metrics['gini']:.4f}")
    print(f"KS: {add_metrics['ks']:.4f}")
    print(f"Average Precision (PR AUC): {add_metrics['average_precision']:.4f}")
    print(
        "Precision / Recall / F1 at median threshold "
        f"({add_metrics['threshold_median']:.4f}): "
        f"{add_metrics['precision_at_median']:.4f} / "
        f"{add_metrics['recall_at_median']:.4f} / "
        f"{add_metrics['f1_at_median']:.4f}"
    )
    print("\n=== Calibration Summary (HEAD) ===")
    print(calib.head())
    print("\n=== Precision-Recall Curve Summary ===")
    for row in pr_summary:
        print(
            f"Threshold: {row['threshold']:.4f} | "
            f"Precision: {row['precision']:.4f} | "
            f"Recall: {row['recall']:.4f}"
        )