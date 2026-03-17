import pandas as pd
import numpy as np


def reweight_categories(train_series, prod_series):
    """
    Compute weights to align production distribution to training distribution
    """

    train_dist = train_series.value_counts(normalize=True)
    prod_dist = prod_series.value_counts(normalize=True)

    weights = {}

    for category in train_dist.index:

        train_p = train_dist.get(category, 0)
        prod_p = max(prod_dist.get(category, 0), 1e-6)

        weights[category] = train_p / prod_p

    return weights


def apply_reweighting(prod_series, weights):
    """
    Apply weights to production data
    """

    return prod_series.map(weights).fillna(1.0)


def adjust_distribution(train_series, prod_series):
    """
    Align production categories to training categories
    """

    train_categories = set(train_series.dropna().unique())

    # Replace unseen categories with "Other"
    adjusted = prod_series.apply(
        lambda x: x if x in train_categories else "Other"
    )

    return adjusted


def recalibrate_encoding(train_series, prod_series):
    """
    Ensure consistent encoding between train and production
    """

    combined = pd.concat([train_series, prod_series]).astype(str)

    categories = combined.unique()

    mapping = {cat: i for i, cat in enumerate(categories)}

    train_encoded = train_series.map(mapping)
    prod_encoded = prod_series.map(mapping)

    return train_encoded, prod_encoded, mapping


def mitigate_categorical_drift(train_df, prod_df, drift_table, categorical_cols):
    """
    Apply mitigation strategies for drifted categorical features
    """

    prod_df = prod_df.copy()
    mitigation_actions = {}

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    for feature in categorical_cols:

        if feature not in drifted["Feature"].values:
            continue

        train_series = train_df[feature]
        prod_series = prod_df[feature]

        # --- Step 1: Distribution Adjustment ---
        adjusted_series = adjust_distribution(train_series, prod_series)

        # --- Step 2: Category Reweighting ---
        weights = reweight_categories(train_series, adjusted_series)
        weighted_series = apply_reweighting(adjusted_series, weights)

        # --- Step 3: Encoding Recalibration ---
        _, recalibrated_series, mapping = recalibrate_encoding(
            train_series,
            adjusted_series
        )

        # Apply ONLY safe mitigation to data
        prod_df[feature] = adjusted_series

        # Expose weights (DO NOT force into data)
        prod_df[f"{feature}_weight"] = weighted_series

        # (Optional but good) expose encoded version for model use
        prod_df[f"{feature}_encoded"] = recalibrated_series

        mitigation_actions[feature] = {
            "method": "adjustment + weight exposure + encoding recalibration",
            "num_categories": len(mapping),
            "weights_applied_to_data": False
        }

    return prod_df, mitigation_actions