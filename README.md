# bobionics-naisc-2026-adaptive-drift

Official repository for Team Bobionics’ submission to the NAISC Singtel 2026 Adaptive Drift Intelligence Challenge.

This project implements an end-to-end Adaptive Data Drift Monitoring and Mitigation System for a binary classification use case (customer churn prediction). It detects distribution shifts between training and production data, quantifies drift severity, applies automated mitigation strategies, retrains models, and provides both terminal and interactive dashboard reporting.

---

# System Overview

Machine learning models degrade in production due to data drift, where input feature distributions change over time. This system provides a complete pipeline to detect, analyze, and correct drift while maintaining model performance stability.

The system performs:

* Feature-wise drift detection (categorical and numerical)
* Statistical significance testing and PSI scoring
* Automated drift severity classification
* Data transformation and mitigation
* Model retraining with drift-aware weighting
* Performance comparison before and after mitigation
* Interactive monitoring dashboard generation

---

# End-to-End Workflow

## 1. Data Ingestion

The system loads training and production datasets and automatically identifies:

* Categorical features
* Numerical features
* Target variable (ChurnStatus)

---

## 2. Drift Detection Engine

Implemented in `drift_detector.py`, the system compares training vs production distributions.

### Categorical Features

* Population Stability Index (PSI)
* Chi-square test
* Missing value handling

### Numerical Features

* Population Stability Index (PSI)
* Kolmogorov–Smirnov test
* Wasserstein distance

Each feature is assigned:

* PSI score
* Statistical significance (p-value)
* Drift flag
* Severity level (LOW, MODERATE, HIGH)

All results are consolidated into a unified drift table.

---

## 3. Drift Output Artifact

outputs/drift_table.csv

This file is the central system artifact containing:

* Feature names and types
* PSI values
* Statistical test results
* Drift detection flags
* Severity classification

It is used across mitigation, reporting, and visualization.

---

## 4. Terminal Reporting System

The pipeline generates a structured CLI report that includes:

* Dataset sizes and feature breakdown
* Overall drift score
* Percentage of drifted features
* Severity distribution across features
* Ranked list of highest-drift features
* Detailed statistical breakdowns per feature
* System-level recommendations

### Recommendations logic:

* High drift → retraining required
* Moderate drift → monitoring required
* Low drift → stable system

All generated artifacts are printed for traceability.

---

## 5. Drift Mitigation Engine

Implemented in `mitigation.py`.

### Categorical Mitigation

* Category alignment (handles unseen values)
* Distribution reweighting
* Encoding recalibration
* Feature-level weight generation

### Numerical Mitigation

* Sample reweighting based on drift magnitude
* Distribution normalization
* Feature scaling recalibration

This produces:

* Mitigated training dataset
* Mitigated production dataset
* Feature weights for drift correction

---

## 6. Model Training and Evaluation

A LightGBM classifier is trained in two phases.

### Baseline Model

Trained on original training data.

Outputs:

* Train AU-PRC
* Test AU-PRC (if labels are available)

### Mitigated Model

Trained on:

* Drift-mitigated training data
* Drift-mitigated production data
* Sample weights derived from drift severity

### Final Comparison

The system reports:

* Improvement in AU-PRC (train and test)
* Performance delta between baseline and mitigated models

---

## 7. Generated Artifacts

The pipeline produces:

### outputs/drift_table.csv

Central structured dataset containing all drift metrics and classifications.

### outputs/drift_summary.txt

Human-readable summary including:

* Total features analyzed
* Number of drifted features
* Ranked PSI feature list

### outputs/drift_dashboard.html

This file contains a single consolidated drift analysis graph, which provides an overview of feature drift distribution.

Note: This is a static visualization output and not a fully interactive multi-page dashboard.

### model.joblib

Final LightGBM model trained after mitigation.

### prediction.csv

Inference output containing:

* CustomerID
* Churn probability score

---

## 8. Interactive Monitoring Dashboard

Built using Dash and Plotly, launched at:

[http://127.0.0.1:8050](http://127.0.0.1:8050)

### Dashboard Components

### Summary Metrics

* Total features
* Drifted features
* Numerical vs categorical breakdown
* Average PSI
* Maximum PSI

### Drift Severity Distribution

Shows proportion of features across severity levels.

### Drift Risk Leaderboard

Ranks features by PSI to highlight highest-risk inputs.

### Numerical Drift Analysis

Visualizes PSI values with thresholds:

* 0.1 (moderate drift)
* 0.25 (high drift)

### Drift Heatmap

Shows relationship between feature type and drift severity.

### Feature Drift Explorer

Interactive feature-level inspection:

* Categorical: train vs production distribution comparison
* Numerical: histogram overlay comparison

---

## 9. Execution

Run the full pipeline:

```bash
python src/main.py --train_data_filepath public_data/train.csv --test_data_filepath public_data/test.csv
```

---

## Final System Output

After execution, the system delivers:

* Full drift analysis across all features
* Statistical validation of distribution shifts
* Automated mitigation transformations
* Model performance comparison (before vs after)
* Production-ready trained model
* Prediction outputs
* Interactive monitoring dashboard

---

## Team Bobionics

NAISC Singtel 2026 Adaptive Drift Intelligence Challenge
