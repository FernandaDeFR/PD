import numpy as np
import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

def generate_synthetic_data(
    n: int = 20000,
    seed: int = 42
)-> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Features
    age = rng.integers(18, 75, size=n)
    renda = np.exp(rng.normal(9.8, 0.6, size=n))
    debit = renda*rng.uniform(0.0, 2.5, size=n)
    rel_debit_renda = debit/np.maximum(renda, 1)
    qty_delayed_12m = rng.poisson(0.3, size=n)
    chanel = rng.choice(["store", "app", "correspondent"], size=n, p=[0.4, 0.4, 0.2])
    time_job = rng.integers(0, 360, size=n) #months
    garantor = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    # Adding some noise and non-linearities
    logit = (
       -3.0
       +0.8*rel_debit_renda                    # more leverage => higher risk
       +0.4*(qty_delayed_12m > 0)              
       -0.01*(age-40)                          # higher age => lower risk
       -0.001*time_job                         # longer job time => lower risk
       -0.7*garantor
       +(chanel == "correspondent")*0.25       # correspondent channel is slightly higher risk
    )

    # Converting logit to logistic probability
    pd12 = 1/(1+np.exp(-logit))

    # Generating binary target
    y = rng.binomial(1, p=np.clip(pd12, 1e-4, 1-1e-4))

    # Creating DataFrame
    df = pd.DataFrame(
        {
            "age": age,
            "renda": renda,
            "debit": debit,
            "rel_debit_renda": rel_debit_renda,
            "qty_delayed_12m": qty_delayed_12m,
            "chanel": chanel,
            "time_job_m": time_job,
            "garantor": garantor,
            "default_12m": y
        }
    )
    return df

def train_test_split_pd(
    df: pd.DataFrame,
    target_col: str = "default_12m",
    test_size: float = 0.3,
    random_state: int = 7
):
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df[target_col]
    )
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    return X_train, X_test, y_train, y_test

# Model definition

def create_pd_logit_pipeline()-> Pipeline:
    num_cols = [
        "age", 
        "renda", 
        "debit", 
        "rel_debit_renda", 
        "qty_delayed_12m", 
        "time_job_m", 
        "garantor"
    ]
    cat_cols = ["chanel"]

    num_pipe = Pipeline([("scaler", StandardScaler())])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), cat_cols)
        ]
    )

    logit = LogisticRegression(
        max_iter=5000,
        solver="lbfgs", 
        random_state=42
    )

    clf = Pipeline(
        steps=[
            ("pre", pre),
            ("logit", logit)
        ]
    )
    
    return clf

# PD API: fit, predict, evaluate

def fit_pd_logit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    calibrate: bool = True
):
    """
    Fits a PD logistic regression model with optional calibration.
    """
    base_model = create_pd_logit_pipeline()
    base_model.fit(X_train, y_train)

    if calibrate:
        return base_model, None
    
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, 
        cv=3, 
        method='isotonic'
    )
    calibrated_model.fit(X_train, y_train)

    return base_model, calibrated_model

def predict_pd_logit(
    base_model: Pipeline,
    calibrated_model: CalibratedClassifierCV | None,
    X: pd.DataFrame,
    use_calibrated: bool = True
)-> np.ndarray:
    """
    Return PD estimates for each row in X.
    """
    if use_calibrated and calibrated_model is not None:
        proba = calibrated_model.predict_proba(X)[:, 1]
    else:
        proba = base_model.predict_proba(X)[:, 1]
    return proba

def evaluate_pd_logit(
    y_true: np.ndarray,
    pd_scores: np.ndarray
):
    """
    Evaluate PD metrics (AUC, Brier, default rate).
    Return a dict.
    """
    auc = roc_auc_score(y_true, pd_scores)
    brier = brier_score_loss(y_true, pd_scores)

    return {
        "auc": float(auc),
        "brier": float(brier),
        "default_rate": float(y_true.mean())
    }

def save_pd_logit_model(
    base_model: Pipeline,
    calibrated_model: CalibratedClassifierCV | None,
    base_path: str
):
    joblib.dump(base_model, f"{base_path}_base.joblib")
    if calibrated_model is not None:
        joblib.dump(calibrated_model, f"{base_path}_calibrated.joblib")

def load_pd_logit_model(
    base_path: str
):
    base_model = joblib.load(f"{base_path}_base.joblib")
    try:
        calibrated_model = joblib.load(f"{base_path}_calibrated.joblib")
    except FileNotFoundError:
        calibrated_model = None
    return base_model, calibrated_model