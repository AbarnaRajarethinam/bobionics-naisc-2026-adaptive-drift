import pandas as pd
import numpy as np


# ==========================================================
# CATEGORICAL DRIFT MITIGATION
# ==========================================================

def reweight_categories(train_series, prod_series):

    train_dist = train_series.value_counts(normalize=True)
    prod_dist = prod_series.value_counts(normalize=True)

    weights = {}

    for category in train_dist.index:

        train_p = train_dist.get(category, 0)
        prod_p = max(prod_dist.get(category, 0), 1e-6)

        weights[category] = train_p / prod_p

    return weights


def apply_reweighting(prod_series, weights):

    return prod_series.map(weights).fillna(1.0)


def adjust_distribution(train_series, prod_series):

    train_categories = set(train_series.dropna().unique())

    adjusted = prod_series.apply(
        lambda x: x if x in train_categories else "Other"
    )

    return adjusted


def recalibrate_encoding(train_series, prod_series):

    combined = pd.concat([train_series, prod_series]).astype(str)

    categories = combined.unique()

    mapping = {cat: i for i, cat in enumerate(categories)}

    train_encoded = train_series.map(mapping)
    prod_encoded = prod_series.map(mapping)

    return train_encoded, prod_encoded, mapping


def mitigate_categorical_drift(train_df, prod_df, drift_table, categorical_cols):

    prod_df = prod_df.copy()
    actions = {}

    drifted = set(drift_table[drift_table["Drift_Detected"] == True]["Feature"])

    for feature in categorical_cols:

        if feature not in drifted:
            continue

        train_vals = set(train_df[feature].dropna().unique())

        # collapse unseen categories
        prod_df[feature] = prod_df[feature].apply(
            lambda x: x if x in train_vals else "Other"
        )

        actions[feature] = {
            "method": "rare category collapse only"
        }

    return prod_df, actions
# ==========================================================
# NUMERICAL DRIFT MITIGATION
# ==========================================================

def compute_sample_weights(train_series, prod_series):

    train_mean = train_series.mean()
    prod_mean = prod_series.mean()

    train_std = train_series.std() + 1e-6
    prod_std = prod_series.std() + 1e-6

    weights = (train_std / prod_std)

    return weights


def apply_numeric_reweighting(prod_series, weight):

    return prod_series * weight


def normalize_to_training_distribution(train_series, prod_series):

    train_mean = train_series.mean()
    train_std = train_series.std() + 1e-6

    prod_mean = prod_series.mean()
    prod_std = prod_series.std() + 1e-6

    standardized = (prod_series - prod_mean) / prod_std

    normalized = standardized * train_std + train_mean

    return normalized


def recalibrate_feature_scale(train_series, prod_series):

    train_min = train_series.min()
    train_max = train_series.max()

    prod_min = prod_series.min()
    prod_max = prod_series.max()

    scaled = (prod_series - prod_min) / (prod_max - prod_min + 1e-6)

    recalibrated = scaled * (train_max - train_min) + train_min

    return recalibrated

def mitigate_numerical_drift(train_df, prod_df, drift_table, numerical_cols):

    prod_df = prod_df.copy()
    mitigation_actions = {}

    drifted = set(drift_table[drift_table["Drift_Detected"] == True]["Feature"])

    for feature in numerical_cols:

        if feature not in drifted:
            continue

        train = train_df[feature]
        prod = prod_df[feature]

        # --------------------------------------------------
        # STEP 1: measure drift severity
        # --------------------------------------------------
        drift_score = abs(train.mean() - prod.mean()) / (train.std() + 1e-6)

        # --------------------------------------------------
        # STEP 2: choose ONLY ONE correction strategy
        # --------------------------------------------------

        if drift_score < 0.5:
            # light correction (safe)
            adjusted = prod + (train.mean() - prod.mean()) * 0.2
            method = "soft mean shift"

        elif drift_score < 1.5:
            # moderate correction (alignment)
            adjusted = (prod - prod.mean()) / (prod.std() + 1e-6)
            adjusted = adjusted * train.std() + train.mean()
            method = "standardization alignment"

        else:
            # heavy drift → clipping only (VERY IMPORTANT)
            lower = train.quantile(0.01)
            upper = train.quantile(0.99)
            adjusted = prod.clip(lower, upper)
            method = "quantile clipping"

        prod_df[feature] = adjusted

        mitigation_actions[feature] = {
            "method": method,
            "drift_score": drift_score
        }

    return prod_df, mitigation_actions

    