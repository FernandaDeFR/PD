from pd_model_sklearn import (
    generate_synthetic_data,
    train_test_split_pd,
    fit_pd_logit,
    predict_pd_logit,
    evaluate_pd_logit,
    save_pd_logit_model
)
from pd_metrics import compute_deciles_table, ks_score, psi

def main():
    # Data
    df = generate_synthetic_data(n=20000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split_pd(
        df,
        target_col="default_12m",
        test_size=0.3,
        random_state=7
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

    # KS
    ks = ks_score(y_test, pd_scores)

    # Deciles
    deciles = compute_deciles_table(y_test, pd_scores, n_bins=10)

    # PSI (train vs test)
    pd_scores_train = predict_pd_logit(
        base_model,
        calibrated_model,
        X_train,
        use_calibrated=True
    )
    psi_value = psi(pd_scores_train, pd_scores, n_bins=10)  

    print("=== Sklearn PD Experiment ===")
    print(f"AUC: {metrics['auc']:.3f}")
    print(f"Brier: {metrics['brier']:.4f}")
    print(f"Default Rate: {metrics['default_rate']:.3%}")
    print(f"KS: {ks:.3f}")
    print(f"PSI: (tran vs test): {psi_value:.3f}")
    print("\nDeciles table (head):")
    print(deciles[["decile", "n", "p_avg", "rate", "expected", "observed", "lift"]].head())

    # Save model
    save_pd_logit_model(
        base_model,
        calibrated_model,
        base_path="models/pd_logit"
    )

if __name__ == "__main__":
    main()