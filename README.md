# bobionics-naisc-2026-adaptive-drift

Official private repository for **Team Bobionics’ submission to the NAISC Singtel 2026 Adaptive Drift Intelligence Challenge**.

This project implements an **Adaptive Data Drift Monitoring System** that detects, quantifies, visualizes, and mitigates distribution shifts between training and production datasets in a **binary classification environment**.

The system compares feature distributions between training and incoming production data, generates drift alerts, and applies mitigation strategies to maintain model stability.

---

# Project Overview

Machine learning models deployed in production often experience **data drift**, where the statistical properties of input data change over time. This can significantly degrade model performance.

Our system provides an **end-to-end monitoring pipeline** that:

1. Detects drift across categorical and numerical features
2. Quantifies drift severity using statistical metrics
3. Generates dashboards and reports for monitoring
4. Applies automated mitigation techniques to stabilize feature distributions

The system is designed to be **modular, interpretable, and scalable**, enabling teams to continuously monitor model health in production environments.

---

# Key Features Implemented

## 1. Dataset Drift Detection

The system compares **training data vs production data** using multiple statistical tests.

### Categorical Drift Detection

* **Population Stability Index (PSI)**
* **Chi-Square Test**
* Missing value handling

### Numerical Drift Detection

* **Population Stability Index (PSI)**
* **Kolmogorov–Smirnov Test**
* **Wasserstein Distance**

Each feature is automatically classified as:

* 🟢 Low Drift
* 🟡 Moderate Drift
* 🔴 Significant Drift

A dataset-level **drift score** is also computed.

---

# 2. Numerical Drift Mitigation

To address distribution shifts in numerical features, the system implements several mitigation strategies:

### Sample Reweighting

Adjusts feature influence based on variance differences between training and production data.

### Normalization Adjustment

Aligns production data statistics with training distribution.

### Feature Recalibration

Rescales production features to match the training data range.

These techniques help stabilize inputs before model inference.

---

# 3. Categorical Drift Mitigation

Categorical features are stabilized using:

* Category distribution alignment
* Category reweighting
* Encoding recalibration
* Handling unseen categories

Additional metadata such as **feature weights and encoded representations** are generated to support downstream models.

---

# 4. Drift Monitoring Dashboard

The system generates an **interactive monitoring dashboard** using Dash and Plotly.

The dashboard includes:

* Feature drift leaderboard
* Drift severity distribution
* Numerical drift visualizations
* Feature-level distribution comparisons
* Interactive feature exploration

This enables users to easily investigate which features are drifting.

---

# 5. Drift Reports and Artifacts

The pipeline automatically generates monitoring outputs:

```
outputs/
├── drift_table.csv
├── drift_summary.txt
└── drift_dashboard.html
```

These artifacts provide both **machine-readable and human-readable summaries** of drift status.

---

# Project Structure

```
src/
├── main.py                # Main pipeline entry point
├── drift_detector.py      # Statistical drift detection engine
├── mitigation.py          # Drift mitigation methods
└── visualization.py       # Dashboard and plots

public_data/
├── train.csv
└── test.csv

outputs/
├── drift_table.csv
├── drift_summary.txt
└── drift_dashboard.html
```

---

# How to Run the System

Run the drift monitoring pipeline using:

```bash
python src/main.py --train_data_filepath public_data/train.csv --test_data_filepath public_data/test.csv
```

This will:

1. Load datasets
2. Detect drift across all features
3. Generate reports and visualizations
4. Apply mitigation strategies
5. Launch the monitoring dashboard

Dashboard will be available at:

```
http://127.0.0.1:8050
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Plotly
* Dash

---

# Team Bobionics

NAISC Singtel 2026 Adaptive Drift Intelligence Challenge

