import argparse
import pandas as pd
import os
import time

from drift_detector import detect_dataset_drift
from visualization import launch_dashboard, save_static_dashboard


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

    return drift_table["PSI"].mean()


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

    print("\nRunning Drift Detection Engine...")
    print("--------------------------------------------------------\n")


def print_drift_report(drift_table, runtime):

    drift_table["Severity"] = drift_table["PSI"].apply(classify_severity)

    drifted = drift_table[drift_table["Drift_Detected"] == True]

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
    print(f"Drifted Features         : {len(drifted)} / {len(drift_table)}\n")

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

        print(
            f"{i:2d}. {row['Feature']:30} "
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

    print_drift_report(drift_table, runtime)

    save_summary_file(drift_table)

    save_static_dashboard(drift_table)

    print("Launching Interactive Monitoring Dashboard...")
    print("http://127.0.0.1:8050\n")

    launch_dashboard(
        train_df,
        test_df,
        drift_table,
        categorical_cols,
        numerical_cols
    )


if __name__ == "__main__":
    main()