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
    mitigation_actions = {}

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    for feature in categorical_cols:

        if feature not in drifted["Feature"].values:
            continue

        train_series = train_df[feature]
        prod_series = prod_df[feature]

        adjusted_series = adjust_distribution(train_series, prod_series)

        weights = reweight_categories(train_series, adjusted_series)
        weight_series = apply_reweighting(adjusted_series, weights)

        train_encoded, prod_encoded, mapping = recalibrate_encoding(
            train_series, adjusted_series
        )

        prod_df[feature] = adjusted_series
        prod_df[f"{feature}_weight"] = weight_series
        prod_df[f"{feature}_encoded"] = prod_encoded

        mitigation_actions[feature] = {
            "method": "adjustment + weight exposure + encoding recalibration",
            "num_categories": len(mapping)
        }

    return prod_df, mitigation_actions


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

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    for feature in numerical_cols:

        if feature not in drifted["Feature"].values:
            continue

        train_series = train_df[feature]
        prod_series = prod_df[feature]

        weight = compute_sample_weights(train_series, prod_series)

        weighted_series = apply_numeric_reweighting(prod_series, weight)
        normalized_series = normalize_to_training_distribution(train_series, weighted_series)
        recalibrated_series = recalibrate_feature_scale(train_series, normalized_series)

        prod_df[feature] = recalibrated_series
        prod_df[f"{feature}_weight"] = np.full(len(prod_df), weight)

        mitigation_actions[feature] = {
            "method": "sample_reweighting + normalization + feature_recalibration",
            "weight_applied": weight
        }

    return prod_df, mitigation_actions