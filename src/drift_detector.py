# drift_detector.py
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

EPS = 1e-6


# ==========================================================
# CATEGORICAL DRIFT FUNCTIONS
# ==========================================================

def fill_missing_categorical(df, categorical_cols, placeholder="Missing"):
    df = df.copy()
    df[categorical_cols] = df[categorical_cols].fillna(placeholder)
    return df


def calculate_categorical_psi(train_series, test_series):

    train_dist = train_series.value_counts(normalize=True)
    test_dist = test_series.value_counts(normalize=True)

    categories = set(train_dist.index).union(set(test_dist.index))

    psi = 0

    for cat in categories:
        train_pct = train_dist.get(cat, EPS)
        test_pct = test_dist.get(cat, EPS)

        psi += (train_pct - test_pct) * np.log(train_pct / test_pct)

    return psi


def chi_square_test(train_series, test_series):

    train_counts = train_series.value_counts()
    test_counts = test_series.value_counts()

    categories = set(train_counts.index).union(set(test_counts.index))

    train_freq = [train_counts.get(cat, 0) for cat in categories]
    test_freq = [test_counts.get(cat, 0) for cat in categories]

    contingency_table = [train_freq, test_freq]

    try:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
    except:
        chi2, p_value = np.nan, 1.0

    return chi2, p_value


def classify_psi(psi):

    if psi < 0.1:
        return "No Drift"
    elif psi < 0.25:
        return "Moderate Drift"
    else:
        return "Significant Drift"


def detect_categorical_drift(train_df, prod_df, categorical_cols):

    train_df = fill_missing_categorical(train_df, categorical_cols)
    prod_df = fill_missing_categorical(prod_df, categorical_cols)

    results = []

    for col in categorical_cols:

        psi = calculate_categorical_psi(train_df[col], prod_df[col])
        chi2, p_value = chi_square_test(train_df[col], prod_df[col])

        severity = classify_psi(psi)

        drift_flag = (psi >= 0.25) or (p_value <= 0.05)

        results.append({
            "Feature": col,
            "Type": "Categorical",
            "PSI": round(psi, 4),
            "PSI_Severity": severity,
            "Chi2_Statistic": round(chi2, 4) if not np.isnan(chi2) else np.nan,
            "p_value": round(p_value, 4),
            "Drift_Detected": drift_flag
        })

    drift_df = pd.DataFrame(results).sort_values("PSI", ascending=False)

    drifted_features = drift_df[drift_df["Drift_Detected"]]["Feature"].tolist()

    return drift_df, drifted_features


# ==========================================================
# NUMERICAL DRIFT FUNCTIONS
# ==========================================================

def calculate_numeric_psi(train_series, test_series, bins=10):

    train_series = train_series.dropna()
    test_series = test_series.dropna()

    breakpoints = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(train_series, breakpoints)

    train_bins = np.histogram(train_series, bins=breakpoints)[0] / len(train_series)
    test_bins = np.histogram(test_series, bins=breakpoints)[0] / len(test_series)

    psi = np.sum(
        (train_bins - test_bins) *
        np.log((train_bins + EPS) / (test_bins + EPS))
    )

    return psi


def ks_test(train_series, test_series):

    statistic, p_value = ks_2samp(
        train_series.dropna(),
        test_series.dropna()
    )

    return statistic, p_value


def detect_numerical_drift(train_df, prod_df, numerical_cols):

    results = []

    for col in numerical_cols:

        psi = calculate_numeric_psi(train_df[col], prod_df[col])

        ks_stat, p_value = ks_test(train_df[col], prod_df[col])

        wasserstein = wasserstein_distance(
            train_df[col].dropna(),
            prod_df[col].dropna()
        )

        severity = classify_psi(psi)

        drift_flag = (psi >= 0.25) or (p_value <= 0.05)

        results.append({
            "Feature": col,
            "Type": "Numerical",
            "PSI": round(psi, 4),
            "PSI_Severity": severity,
            "KS_Statistic": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "Wasserstein_Distance": round(wasserstein, 4),
            "Drift_Detected": drift_flag
        })

    drift_df = pd.DataFrame(results).sort_values("PSI", ascending=False)

    drifted_features = drift_df[drift_df["Drift_Detected"]]["Feature"].tolist()

    return drift_df, drifted_features


# ==========================================================
# COMBINED DRIFT DETECTION
# ==========================================================

def detect_dataset_drift(train_df, prod_df, categorical_cols, numerical_cols):

    cat_drift_df, cat_features = detect_categorical_drift(
        train_df, prod_df, categorical_cols
    )

    num_drift_df, num_features = detect_numerical_drift(
        train_df, prod_df, numerical_cols
    )

    drift_table = pd.concat([cat_drift_df, num_drift_df])

    drift_table = drift_table.sort_values("PSI", ascending=False)

    drifted_features = cat_features + num_features

    return drift_table, drifted_features