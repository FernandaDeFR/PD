import numpy as np
from pd_model_sklearn import (
    generate_synthetic_data,
    train_test_split_pd,
    fit_pd_logit,
    predict_pd_logit,
    evaluate_pd_logit
)

def run_pd_sklearn_once(
    data_seed: int = 42,
    split_random_state: int = 7
):
    """
    Run a full PD modeling experiment using sklearn pipeline once and return all relevant outputs.
    """
    # Data
    df = generate_synthetic_data(n=20000, seed=data_seed)
    X_train, X_test, y_train, y_test = train_test_split_pd(
        df,
        target_col="default_12m",
        test_size=0.3,
        random_state=split_random_state
    )

    # Fit model
    base_model, calibrated_model = fit_pd_logit(
        X_train,
        y_train,
        calibrate=True
    )

    # Predict PD on test
    pd_scores = predict_pd_logit(
        base_model,
        calibrated_model,
        X_test,
        use_calibrated=True
    )

    # Evaluate
    metrics = evaluate_pd_logit(y_test, pd_scores)
    return {
        "y_test": y_test,
        "pd_scores": pd_scores,
        "metrics": metrics,
    }

def test_replicability_sklearn(
    data_seed: int = 42,
    split_random_state: int = 7,
    score_tolerance: float = 1e-12,
    metric_tolerance: float = 1e-12
):
    """
    Run the sklearn PD experiment twice and check if the results are replicable.
    """
    # First run
    result1 = run_pd_sklearn_once(
        data_seed=data_seed,
        split_random_state=split_random_state
    )
    

    # Second run
    result2 = run_pd_sklearn_once(
        data_seed=data_seed,
        split_random_state=split_random_state
    )

    # Labels
    same_labels = np.array_equal(result1["y_test"], result2["y_test"])

    # PD scores
    score_diff = np.abs(result1["pd_scores"] - result2["pd_scores"])
    max_score_diff = float(score_diff.max())
    scores_ok = max_score_diff < score_tolerance

    # Metrics
    auc_diff = abs(result1["metrics"]["auc"] - result2["metrics"]["auc"])
    brier_diff = abs(result1["metrics"]["brier"] - result2["metrics"]["brier"])
    metrics_pk = (auc_diff < metric_tolerance) and (brier_diff < metric_tolerance)

    print("=== Sklearn PD Replicability Test ===")
    print(f"Same labels: {same_labels}")
    print(f"Max PD score diference: {max_score_diff:.2e}")
    print(f"Scores within tolerance: ({score_tolerance})? {scores_ok}")
    print(f"AUC difference: {auc_diff:.2e} | Brier difference: {brier_diff:.2e}")
    print(f"Metrics within tolerance: ({metric_tolerance})? {metrics_pk}")

    all_ok = same_labels and scores_ok and metrics_pk
    print(f"Final result: {'Replicable' if all_ok else 'Not replicable'}")

    return {
        "same_labels": same_labels,
        "max_score_diff": max_score_diff,
        "scores_ok": scores_ok,
        "auc_diff": float(auc_diff),
        "brier_diff": float(brier_diff),
        "metrics_ok": metrics_pk,
        "replicable": all_ok
    }

if __name__ == "__main__":
    _ = test_replicability_sklearn()