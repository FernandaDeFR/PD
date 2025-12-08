import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss

import tensorflow as tf
import keras
from keras import layers

#Sintetic dada generation
rng = np.random.default_rng(42)    # Random number generator
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

X_train_num = X_train[num_cols].copy()
X_test_num = X_test[num_cols].copy()
X_train_cat = X_train[cat_cols].copy()
X_test_cat = X_test[cat_cols].copy()

scaler = StandardScaler()
X_train_num_scaler = scaler.fit_transform(X_train_num)
X_test_num_scaler = scaler.transform(X_test_num)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat_ohe = ohe.fit_transform(X_train_cat)
X_test_cat_ohe = ohe.transform(X_test_cat)

X_train_final = np.hstack([X_train_num_scaler, X_train_cat_ohe])   #horizontal stack (put together the two arrays)
X_test_final = np.hstack([X_test_num_scaler, X_test_cat_ohe])

input_dim = X_train_final.shape[1]      #number of features after preprocessing

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),       #input layer with number of features
    layers.Dense(32, activation='relu'),    
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')   #output layer (number between 0 and 1) = probability of default
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  #ajust learning rate during training
    loss='binary_crossentropy',                            #loss function for binary classification
    metrics=[keras.metrics.AUC(name='auc')]                #metric to evaluate model performance
)

model.summary()  #print model architecture

history = model.fit(
    X_train_final,
    y_train,
    validation_split=0.2,  
    epochs=20,
    batch_size=256,
    verbose=1             #print progress
)

proba_test_nn = model.predict(X_test_final).ravel()   #ravel to convert (flatten) from 2D to 1D array
auc_nn = roc_auc_score(y_test, proba_test_nn)
brier_nn = brier_score_loss(y_test, proba_test_nn)

print(f"Neural Network AUC: {auc_nn:.3f} | Brier: {brier_nn:.4f} | default_test_rate: {y_test.mean():.3%}")