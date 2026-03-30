import argparse
import pandas as pd
import numpy as np
import os
import time

from drift_detector import detect_dataset_drift
from visualization import launch_dashboard, save_static_dashboard
from mitigation import mitigate_categorical_drift, mitigate_numerical_drift

from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
import joblib

# =========================
# TABLE FORMATTING HELPERS
# =========================

def format_grid_table(df, max_width=35):
    import textwrap
    import math
    import re

    df = df.copy()

    # -------------------------
    # CLEAN VALUES
    # -------------------------
    def clean(val):
        if val is None:
            return ""
        if isinstance(val, float) and math.isnan(val):
            return ""
        return str(val)

    df = df.map(clean)

    # -------------------------
    # EMOJI-AWARE WIDTH ESTIMATOR
    # -------------------------
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )

    def display_width(text):
        text = str(text)
        emoji_count = len(emoji_pattern.findall(text))
        base_len = len(text)
        return base_len + emoji_count  # emojis ≈ width 2

    # -------------------------
    # COLUMN WIDTHS
    # -------------------------
    col_widths = {
        col: min(
            max(display_width(col), df[col].map(display_width).max()),
            max_width
        )
        for col in df.columns
    }

    # -------------------------
    # WRAP CELLS
    # -------------------------
    wrapped = {
        col: df[col].apply(
            lambda x: textwrap.wrap(x, col_widths[col]) or [""]
        )
        for col in df.columns
    }

    # -------------------------
    # ROW HEIGHTS
    # -------------------------
    row_heights = [
        max(len(wrapped[col].iloc[i]) for col in df.columns)
        for i in range(len(df))
    ]

    # -------------------------
    # BORDER LINE
    # -------------------------
    def make_line(left, mid, right):
        return left + mid.join(
            "─" * (col_widths[c] + 2) for c in df.columns
        ) + right

    lines = []

    # TOP BORDER
    lines.append(make_line("┌", "┬", "┐"))

    # HEADER
    header = "│ " + " │ ".join(
        f"{col:<{col_widths[col]}}" for col in df.columns
    ) + " │"
    lines.append(header)

    # HEADER SEPARATOR
    lines.append(make_line("├", "┼", "┤"))

    # -------------------------
    # ROWS
    # -------------------------
    for i in range(len(df)):
        for h in range(row_heights[i]):
            row = []
            for col in df.columns:
                cell_lines = wrapped[col].iloc[i]
                value = cell_lines[h] if h < len(cell_lines) else ""
                pad = col_widths[col] - display_width(value)
                row.append(value + " " * pad)

            lines.append("│ " + " │ ".join(row) + " │")

        lines.append(make_line("├", "┼", "┤"))

    # -------------------------
    # FINAL BORDER
    # -------------------------
    if len(lines) > 2:
        lines[-1] = make_line("└", "┴", "┘")

    return "\n".join(lines)


def classify_severity(psi):

    if psi < 0.1:
        return "LOW"
    elif psi < 0.25:
        return "MODERATE"
    else:
        return "HIGH"


def severity_icon(level):

    icons = {
        "LOW": "🟢",
        "MODERATE": "🟡",
        "HIGH": "🔴"
    }

    return icons.get(level, "")


def calculate_dataset_score(drift_table):

    return drift_table["PSI"].clip(upper=1).mean()


def print_system_header():

    print("\n========================================================")
    print(" NAISC ADAPTIVE DRIFT MONITORING SYSTEM")
    print("========================================================\n")


def print_initialization(train_df, test_df, categorical_cols, numerical_cols):

    print("System Initialization")
    print("--------------------------------------------------------")

    print(f"Train Dataset Size        : {len(train_df):,} rows")
    print(f"Production Dataset Size   : {len(test_df):,} rows")
    print(f"Total Features Analysed   : {len(categorical_cols) + len(numerical_cols)}")
    print(f"Categorical Features      : {len(categorical_cols)}")
    print(f"Numerical Features        : {len(numerical_cols)}")
    print("Comparing Train vs Production Distributions")

    print("\nRunning Drift Detection Engine...")
    print("--------------------------------------------------------\n")


def print_drift_report(drift_table, categorical_cols, numerical_cols, runtime):

    drift_table = drift_table.copy()

    drift_table["Severity"] = drift_table["PSI"].apply(classify_severity)

    def get_feature_type(feature):
        if feature in categorical_cols:
            return "Categorical"
        elif feature in numerical_cols:
            return "Numerical"
        else:
            return "Unknown"

    drift_table["Feature_Type"] = drift_table["Feature"].apply(get_feature_type)

    drifted = drift_table[drift_table["Drift_Detected"] == True]
    drifted_num = drifted[drifted["Feature_Type"] == "Numerical"].shape[0]
    drifted_cat = drifted[drifted["Feature_Type"] == "Categorical"].shape[0]

    score = calculate_dataset_score(drift_table)

    high = (drift_table["Severity"] == "HIGH").sum()
    mod = (drift_table["Severity"] == "MODERATE").sum()
    low = (drift_table["Severity"] == "LOW").sum()

    if score > 0.25:
        alert = "🔴 HIGH DRIFT DETECTED"
    elif score > 0.1:
        alert = "🟡 MODERATE DRIFT DETECTED"
    else:
        alert = "🟢 LOW DRIFT"

    print("\nDrift Detection Completed\n")

    print("========================================================")
    print(f" DATA DRIFT ALERT : {alert}")
    print("========================================================\n")

    print(f"Dataset Drift Score      : {score:.3f}")
    print(f"Drifted Features         : {len(drifted)} / {len(drift_table)}")

    coverage = (len(drift_table) / len(drift_table)) * 100
    print(f"Monitoring Coverage      : {coverage:.0f}%")

    print(f"Drifted Numerical Features   : {drifted_num}")
    print(f"Drifted Categorical Features : {drifted_cat}\n")

    print("Severity Breakdown")
    print("--------------------------------------------------------")

    print(f"HIGH DRIFT FEATURES      : {high}")
    print(f"MODERATE DRIFT FEATURES  : {mod}")
    print(f"LOW DRIFT FEATURES       : {low}\n")

    print("\nDrift Table (Top Features)")
    print("--------------------------------------------------------")

    display_df = drift_table.copy()

    display_df["Severity"] = display_df["PSI"].apply(classify_severity)
    display_df["Type"] = display_df["Feature_Type"]

    display_df["Stat"] = display_df["Stat_Drift"].apply(lambda x: "YES" if x else "NO")

    cols = ["Feature", "Type", "PSI", "p_value", "Stat", "Severity", "Drift_Detected"]

    display_df = display_df[cols].sort_values("PSI", ascending=False).head(15)

    print(format_grid_table(display_df))

    print("\nExtra Statistical Details (Advanced)")
    print("--------------------------------------------------------")

    stats_cols = [
        "Feature",
        "Chi2_Statistic",
        "KS_Statistic",
        "Wasserstein_Distance",
        "p_value"
    ]

    existing_cols = [c for c in stats_cols if c in drift_table.columns]

    stats_df = drift_table[existing_cols].copy().head(15)

    print(format_grid_table(stats_df)) 

    print("\nSystem Recommendations")
    print("--------------------------------------------------------")

    if score > 0.25:
        print("⚠ Significant dataset drift detected.")
        print("⚠ Model retraining is recommended.")
        print("⚠ Investigate feature distribution changes.")
    elif score > 0.1:
        print("⚠ Moderate drift detected.")
        print("⚠ Monitor feature distributions closely.")
    else:
        print("✓ Dataset distributions remain stable.")

    print("\nArtifacts Generated")
    print("--------------------------------------------------------")

    print("outputs/drift_table.csv")
    print("outputs/drift_summary.txt")
    print("outputs/drift_dashboard.html")

    print(f"\nAnalysis Runtime : {runtime:.2f} seconds\n")


def save_summary_file(drift_table):

    os.makedirs("outputs", exist_ok=True)

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    with open("outputs/drift_summary.txt", "w") as f:

        f.write("NAISC Drift Monitoring Summary\n\n")

        f.write(f"Total Features: {len(drift_table)}\n")
        f.write(f"Drifted Features: {len(drifted)}\n\n")

        f.write("Drift Leaderboard\n\n")

        ranked = drifted.sort_values("PSI", ascending=False)

        for _, row in ranked.iterrows():

            f.write(f"{row['Feature']} | PSI={row['PSI']:.3f}\n")

def train_and_evaluate(train_df, test_df, weight_source_df=None, sample_weights=None):
    target = "ChurnStatus"

    # -------------------------------
    # Split features + target
    # -------------------------------
    X_train = train_df.drop(columns=[target, "CustomerID"], errors="ignore")
    y_train = train_df[target].map({"Yes": 1, "No": 0})

    X_test = test_df.drop(columns=[target, "CustomerID"], errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # -------------------------------
    # Handle categorical features
    # -------------------------------
    cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

        # align categories safely
        X_test[col] = X_test[col].cat.set_categories(X_train[col].cat.categories)

    # keep incoming sample_weights if provided

    if weight_source_df is not None:
        weight_cols = [c for c in weight_source_df.columns if "_weight" in c]

        if len(weight_cols) > 0:
            sample_weights = weight_source_df[weight_cols].prod(axis=1)

            # stabilize
            sample_weights = np.clip(sample_weights, 0.1, 10)
            sample_weights = sample_weights / np.mean(sample_weights)

            # IMPORTANT FIX: only use if SAME length
            if len(sample_weights) != len(X_train):
                print("[WARNING] Sample weights ignored due to length mismatch")
                sample_weights = None
            else:
                sample_weights = sample_weights.values

    # -------------------------------
    # Model
    # -------------------------------
    model = LGBMClassifier(
        verbosity=-1,
        objective="binary",
        is_unbalance=True,
        random_state=42,
        importance_type="gain"
    )

    # -------------------------------
    # Train
    # -------------------------------
    model.fit(X_train, y_train, sample_weight=sample_weights)

    train_probs = model.predict_proba(X_train)[:, 1]
    train_auprc = average_precision_score(y_train, train_probs)

    test_probs = model.predict_proba(X_test)[:, 1]

    return model, train_auprc, test_probs

def compute_sample_weights(train_df, test_df, drift_table):

    weights = np.ones(len(train_df))

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    for _, row in drifted.iterrows():
        feature = row["Feature"]
        psi = row["PSI"]

        if feature not in train_df.columns:
            continue

        col_values = train_df[feature]

        if col_values.dtype == "object":
            weights *= col_values.map(lambda x: 1 / (1 + psi)).fillna(1)
        else:
            weights *= 1 / (
                1 + psi * np.abs(col_values - col_values.mean()) / (col_values.std() + 1e-6)
            )

    weights = weights / np.mean(weights)

    return weights



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_filepath", required=True)
    parser.add_argument("--test_data_filepath", required=True)

    args = parser.parse_args()

    print("\nLaunching NAISC Drift Monitoring System")

    start = time.time()

    train_df = pd.read_csv(args.train_data_filepath)
    test_df = pd.read_csv(args.test_data_filepath)

    categorical_cols = train_df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    exclude_cols = ["CustomerID", "ChurnStatus"]

    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]
    numerical_cols = [c for c in numerical_cols if c not in exclude_cols]

    print_system_header()

    print_initialization(train_df, test_df, categorical_cols, numerical_cols)

    drift_table, drifted_features = detect_dataset_drift(
        train_df,
        test_df,
        categorical_cols,
        numerical_cols
    )

    os.makedirs("outputs", exist_ok=True)

    drift_table.to_csv("outputs/drift_table.csv", index=False)

    runtime = time.time() - start

    print_drift_report(
        drift_table,
        categorical_cols,
        numerical_cols,
        runtime
    )

    save_summary_file(drift_table)

    save_static_dashboard(drift_table)

    # ==========================================================
    # BASELINE MODEL (BEFORE MITIGATION)
    # ==========================================================

    print("\nTraining Baseline Model (Before Mitigation)...\n")

    baseline_model, baseline_train_auprc, baseline_test_probs = train_and_evaluate(
        train_df,
        test_df,
        weight_source_df=None
    )

    print(f"Baseline Train AU-PRC : {baseline_train_auprc:.4f}")

    if "ChurnStatus" in test_df.columns:
        y_test = test_df["ChurnStatus"].map({"Yes": 1, "No": 0})
        baseline_test_auprc = average_precision_score(y_test, baseline_test_probs)
        print(f"Baseline Test AU-PRC  : {baseline_test_auprc:.4f}")

    # ==========================================================
    # DRIFT MITIGATION
    # ==========================================================

    prod_df, cat_actions = mitigate_categorical_drift(
            train_df, test_df, drift_table, categorical_cols
        )

    prod_df, num_actions = mitigate_numerical_drift(
        train_df, prod_df, drift_table, numerical_cols
    )

    mitigated_train_df, _ = mitigate_categorical_drift(
            train_df, train_df.copy(), drift_table, categorical_cols
        )

    mitigated_train_df, _ = mitigate_numerical_drift(
        train_df, mitigated_train_df, drift_table, numerical_cols
    )

    print("\nMitigation Table")
    print("--------------------------------------------------------")

    rows = []

    for f, info in cat_actions.items():
        rows.append({
            "Feature": f,
            "Type": "Categorical",
            "Mitigation": info["method"]
        })

    for f, info in num_actions.items():
        rows.append({
            "Feature": f,
            "Type": "Numerical",
            "Mitigation": info["method"]
        })

    mitigation_df = pd.DataFrame(rows)

    print(format_grid_table(mitigation_df))
    print(f"\nTotal mitigated features: {len(mitigation_df)}")

    # ==========================================================
    # MODEL AFTER MITIGATION
    # ==========================================================

    print("\nTraining Model After Mitigation...\n")

    model, train_auprc, test_probs = train_and_evaluate(
        mitigated_train_df,   
        prod_df,
        weight_source_df=None,
        sample_weights = compute_sample_weights(mitigated_train_df, prod_df, drift_table)
    )

    print("========================================================")
    print(" MODEL PERFORMANCE COMPARISON")
    print("========================================================\n")

    print(f"Baseline Train AU-PRC  : {baseline_train_auprc:.4f}")
    print(f"Mitigated Train AU-PRC : {train_auprc:.4f}")

    improvement = train_auprc - baseline_train_auprc
    print(f"Train Improvement      : {improvement:+.4f}")

    if "ChurnStatus" in test_df.columns:

        y_test = test_df["ChurnStatus"].map({"Yes": 1, "No": 0})

        mitigated_test_auprc = average_precision_score(y_test, test_probs)

        print(f"\nBaseline Test AU-PRC   : {baseline_test_auprc:.4f}")
        print(f"Mitigated Test AU-PRC  : {mitigated_test_auprc:.4f}")

        improvement_test = mitigated_test_auprc - baseline_test_auprc
        print(f"Test Improvement       : {improvement_test:+.4f}")

    # Save model
    joblib.dump(model, "model.joblib")

    pred_df = pd.DataFrame({
        "CustomerID": test_df.get("CustomerID", range(len(test_df))),
        "probability_score": test_probs
    })

    pred_df.to_csv("prediction.csv", index=False)

    print("\nArtifacts Generated")
    print("--------------------------------------------------------")
    print("model.joblib")
    print("prediction.csv")

    print("Launching Interactive Monitoring Dashboard...")
    print("http://127.0.0.1:8050\n")

    launch_dashboard(
        train_df,
        prod_df,
        drift_table,
        categorical_cols,
        numerical_cols
    )


if __name__ == "__main__":
    main()