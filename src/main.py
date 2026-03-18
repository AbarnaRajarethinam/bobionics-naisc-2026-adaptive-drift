import argparse
import pandas as pd
import os
import time

from drift_detector import detect_dataset_drift
from visualization import launch_dashboard, save_static_dashboard
from mitigation import mitigate_categorical_drift, mitigate_numerical_drift

from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
import joblib


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

    print("Top Drifted Features")
    print("--------------------------------------------------------")

    ranked = drifted.sort_values("PSI", ascending=False)

    for i, (_, row) in enumerate(ranked.iterrows(), start=1):

        icon = severity_icon(row["Severity"])

        feature_label = f"{row['Feature']} ({row['Feature_Type']})"

        print(
            f"{i:2d}. {feature_label:40} "
            f"PSI={row['PSI']:.3f}   "
            f"{icon} {row['Severity']}"
        )

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


def train_and_evaluate(train_df, test_df, weight_source_df=None):
    target = "ChurnStatus"

    X_train = train_df.drop(columns=[target, "CustomerID"], errors="ignore")
    y_train = train_df[target].map({"Yes": 1, "No": 0})

    X_test = test_df.drop(columns=[target, "CustomerID"], errors="ignore")

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    for col in X_train.select_dtypes(include="object").columns:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    for col in X_train.select_dtypes(include="category").columns:
        X_test[col] = X_test[col].cat.set_categories(X_train[col].cat.categories)

    model = LGBMClassifier(
        verbosity=-1,
        objective="binary",
        is_unbalance=True,
        random_state=42,
        importance_type="gain"
    )

    if weight_source_df is not None:
        weight_cols = [col for col in weight_source_df.columns if "_weight" in col]
        sample_weights = weight_source_df[weight_cols].mean(axis=1) if weight_cols else None
    else:
        sample_weights = None

    model.fit(X_train, y_train, sample_weight=sample_weights)

    train_probs = model.predict_proba(X_train)[:, 1]
    train_auprc = average_precision_score(y_train, train_probs)

    test_probs = model.predict_proba(X_test)[:, 1]

    return model, train_auprc, test_probs


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
        train_df,
        test_df,
        drift_table,
        categorical_cols
    )

    prod_df, num_actions = mitigate_numerical_drift(
        train_df,
        prod_df,
        drift_table,
        numerical_cols
    )

    print("\nMitigation Summary")
    print("--------------------------------------------------------")

    for f, info in cat_actions.items():
        print(f"{f} → {info['method']}")

    for f, info in num_actions.items():
        print(f"{f} → {info['method']}")

    print(f"\nTotal mitigated features: {len(cat_actions) + len(num_actions)}")

    # ==========================================================
    # MODEL AFTER MITIGATION
    # ==========================================================

    print("\nTraining Model After Mitigation...\n")

    model, train_auprc, test_probs = train_and_evaluate(
        train_df,
        prod_df,
        weight_source_df=None
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