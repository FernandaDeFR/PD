# Credit Risk Modeling with Synthetic Data

This repository contains two Python scripts that generate a synthetic credit-risk dataset and train predictive models to estimate 12-month default probability. The dataset is created artificially to support modeling, validation, and algorithm comparison in a controlled environment.

## Files

### 1. `pd.py`

Baseline model using Logistic Regression with a full preprocessing and modeling pipeline.

Includes:

* synthetic data generation
* train/test split
* numerical scaling and one-hot encoding
* logistic regression training
* performance metrics (AUC, Brier Score)
* isotonic calibration
* decile table (expected vs. observed)
* KS computation
* PSI computation (stability between train and test)

This script implements a complete credit-risk modeling and validation workflow.

### 2. `pd_nn.py`

Alternative model using a neural network built with Keras/TensorFlow.

Includes:

* the same synthetic dataset
* manual preprocessing (scaler + one-hot encoding)
* dense neural network with two hidden layers
* model training with validation split
* performance evaluation with AUC and Brier Score

Useful for comparing linear vs. non-linear models.

## Purpose

To provide a controlled environment for studying, testing, and comparing credit-risk models in terms of performance, calibration, stability, and general behavior.

## Requirements

Main libraries:

* numpy
* pandas
* scikit-learn
* tensorflow / keras
