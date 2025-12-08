import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import CalibratedClassifierCV

#Sintetic dada generation
rng = np.random.default_rng(42)
n = 20000

#Features
age = rng.integers(18, 75, size=n)
renda = np.exp(rng.normal(9.8, 0.6, size=n))
debit = renda*rng.uniform(0.0, 2.5, size=n)
rel_debit_renda = debit/np.maximum(renda, 1)
qty_delayed_12m = rng.poisson(0.3, size=n)
chanel = rng.choice(["store", "app", "correspondent"], size=n, p=[0.4, 0.4, 0.2])
time_job = rng.integers(0, 360, size=n) #months
garantor = rng.choice([0, 1], size=n, p=[0.85, 0.15])

#Adding some noise and non-linearities
logit = (
    -3.0
    +0.8*rel_debit_renda                    # more leverage => higher risk
    +0.4*(qty_delayed_12m > 0)              
    -0.01*(age-40)                          # higher age => lower risk
    -0.001*time_job                         # longer job time => lower risk
    -0.7*garantor
    +(chanel == "correspondent")*0.25       # correspondent channel is slightly higher risk
)

#Converting logit to logistic probability
pd12 = 1/(1+np.exp(-logit))

#Generating binary target
y = rng.binomial(1, p=np.clip(pd12, 1e-4, 1-1e-4))

#Creating DataFrame
df = pd.DataFrame({
    "age": age,
    "renda": renda,
    "debit": debit,
    "rel_debit_renda": rel_debit_renda,
    "qty_delayed_12m": qty_delayed_12m,
    "chanel": chanel,
    "time_job_m": time_job,
    "garantor": garantor,
    "default_12m": y
})

#Splitting data
train_df, test_df = train_test_split(df, test_size=0.3, random_state=7, stratify=df["default_12m"])
X_train = train_df.drop(columns=["default_12m"])
y_train = train_df["default_12m"].values
X_test = test_df.drop(columns=["default_12m"])
y_test = test_df["default_12m"].values

#Pipeline: one-hot for categorical + logistic regression
num_cols = ["age", "renda", "debit", "rel_debit_renda", "qty_delayed_12m", "time_job_m", "garantor"]
cat_cols = ["chanel"]
num_pipeline = Pipeline([("scaler", StandardScaler())])
pre = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), cat_cols)
    ]
)
clf = Pipeline(steps=[
    ("pre", pre),
    ("logit", LogisticRegression(max_iter=5000 ,solver="lbfgs", random_state=42))
])
clf.fit(X_train, y_train)

#Discrimination and calibration
proba_test = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)
brier = brier_score_loss(y_test, proba_test)

print(f"AUC: {auc:.3f} | Brier: {brier:.4f} | default_test_rate: {y_test.mean():.3%}")

#Calibration
calibrated = CalibratedClassifierCV(estimator=clf, cv=3, method='isotonic')
calibrated.fit(X_train, y_train)
proba_cal = calibrated.predict_proba(X_test)[:, 1]
auc_cal = roc_auc_score(y_test, proba_cal)
brier_cal = brier_score_loss(y_test, proba_cal)

print(f"Calibrated AUC: {auc_cal:.3f} | Brier: {brier_cal:.4f}")

#Decis (Expected x Observed)
def decile_table(y_true, y_score, n_bins=10):
    df_aux = pd.DataFrame({"y": y_true, "p": y_score}).sort_values("p", ascending=False)
    df_aux["decile"] = pd.qcut(df_aux["p"].rank(method="first"), q=n_bins, labels=False)
    grp = df_aux.groupby("decile").agg(
        n=("y", "size"),
        defaults=("y", "sum"),
        p_min=("p", "min"),
        p_max=("p", "max"),
        p_mean=("p", "mean"),
        rate=("y", "mean")
    ).reset_index().sort_values("decile", ascending=False)
    grp["expected"] = grp["p_mean"] * grp["n"]
    grp["observed"] = grp["defaults"]
    grp["lift"] = grp["rate"] / (df_aux["y"].mean())
    return grp

deciles = decile_table(y_test, proba_cal, n_bins=10)

print(deciles[["decile", "n", "p_mean", "rate", "expected", "observed", "lift"]].to_string(index=False))

# KS 
def ks_score(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return np.max(tpr - fpr)

print(f"KS: {ks_score(y_test, proba_cal):.3f}")

def psi(reference, current, n_bins=10, eps=1e-6):
    ref = np.asarray(reference).ravel()
    cur = np.asarray(current).ravel()

    ref = ref[np.isfinite(ref)] #remove NaN and inf
    cur = cur[np.isfinite(cur)]

    bins = np.quantile(ref, q=np.linspace(0, 1, n_bins + 1)) #bin quantis based on reference
    bins = np.unique(bins)
    if len(bins) < 3:
        bins = np.linspace(ref.min(), ref.max(), n_bins + 1)

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)
    ref_perc = ref_counts / max(ref_counts.sum(), eps)
    cur_perc = cur_counts / max(cur_counts.sum(), eps)
    psi_vals = (ref_perc - cur_perc) * np.log((ref_perc + eps) / (cur_perc + eps))
    return float(np.sum(psi_vals))

proba_train = calibrated.predict_proba(X_train)[:, 1]
print(f"PSI score (train vs test) : {psi(proba_train, proba_cal):.3f}")    