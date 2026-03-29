import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output
import numpy as np


# ---------------------------
# SUMMARY CARDS
# ---------------------------
def create_summary_cards(metrics):

    card_style = {
        "background": "#ffffff",
        "borderRadius": "10px",
        "padding": "20px",
        "margin": "10px",
        "textAlign": "center",
        "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)",
        "width": "15%",
        "display": "inline-block"
    }

    return html.Div([
        html.Div([html.H4("Total Features"), html.H2(metrics["total_features"])], style=card_style),
        html.Div([html.H4("Drifted Features"), html.H2(metrics["drifted_features"])], style=card_style),
        html.Div([html.H4("Numerical Features"), html.H2(metrics["numerical_features"])], style=card_style),
        html.Div([html.H4("Categorical Features"), html.H2(metrics["categorical_features"])], style=card_style),
        html.Div([html.H4("Average PSI"), html.H2(metrics["avg_psi"])], style=card_style),
        html.Div([html.H4("Max PSI"), html.H2(metrics["max_psi"])], style=card_style)
    ], style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"})


# ---------------------------
# METRICS
# ---------------------------
def compute_summary_metrics(train_df, prod_df, drift_table, categorical_cols, numerical_cols):

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    return {
        "total_features": len(drift_table),
        "drifted_features": len(drifted),
        "numerical_features": len(numerical_cols),
        "categorical_features": len(categorical_cols),
        "avg_psi": round(drift_table["PSI"].mean(), 3),
        "max_psi": round(drift_table["PSI"].max(), 3)
    }


# ---------------------------
# SEVERITY
# ---------------------------
def classify_severity(psi):

    if psi < 0.1:
        return "LOW"
    elif psi < 0.25:
        return "MODERATE"
    else:
        return "HIGH"


SEVERITY_ORDER = ["LOW", "MODERATE", "HIGH"]


# ---------------------------
# STATIC EXPORT
# ---------------------------
def save_static_dashboard(drift_table):

    drift_table = drift_table.copy()
    drift_table["Severity"] = drift_table["PSI"].apply(classify_severity)

    fig = px.bar(
        drift_table.sort_values("PSI", ascending=False),
        x="PSI",
        y="Feature",
        orientation="h",
        color="Severity",
        title="Feature Drift Severity",
        category_orders={"Severity": SEVERITY_ORDER}
    )

    fig.write_html("outputs/drift_dashboard.html")


# ---------------------------
# NUMERICAL DRIFT
# ---------------------------
def create_numerical_drift_plot(drift_table):

    numeric = drift_table[drift_table["Type"] == "Numerical"].copy()

    if numeric.empty:
        return px.bar(title="No Numerical Drift Detected")

    fig = px.bar(
        numeric.sort_values("PSI", ascending=False),
        x="PSI",
        y="Feature",
        orientation="h",
        title="Numerical Feature Drift (PSI)",
        color="PSI"
    )

    fig.add_vline(x=0.1, line_dash="dash", line_color="orange",
                  annotation_text="Moderate Drift (0.1)")

    fig.add_vline(x=0.25, line_dash="dash", line_color="red",
                  annotation_text="Severe Drift (0.25)")

    fig.add_annotation(
        text="PSI measures distribution shift between train vs production",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False
    )

    fig.update_layout(height=500)
    return fig


# ---------------------------
# 🔥 FIXED HEATMAP (NO category_orders)
# ---------------------------
def create_drift_heatmap(drift_table):

    df = drift_table.copy()
    df = df.dropna(subset=["Type", "PSI"])
    df["Severity"] = df["PSI"].apply(classify_severity)

    df = df.reset_index(drop=True)

    heatmap_data = pd.crosstab(df["Type"], df["Severity"])

    # ✅ FORCE ORDER MANUALLY (IMPORTANT FIX)
    for col in SEVERITY_ORDER:
        if col not in heatmap_data.columns:
            heatmap_data[col] = 0

    heatmap_data = heatmap_data[SEVERITY_ORDER]

    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        title="Drift Heatmap: Feature Type vs Severity",
        color_continuous_scale="Reds"
    )

    fig.add_annotation(
        text="Shows where drift is concentrated across feature types",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False
    )

    fig.update_layout(height=400)

    return fig


# ---------------------------
# MAIN DASHBOARD
# ---------------------------
def launch_dashboard(train_df, prod_df, drift_table, categorical_cols, numerical_cols):

    drift_table = drift_table.copy()
    drift_table["Severity"] = drift_table["PSI"].apply(classify_severity)

    drifted = drift_table[drift_table["Drift_Detected"] == True]
    drifted_features = drifted["Feature"].tolist()

    app = Dash(__name__)

    metrics = compute_summary_metrics(
        train_df, prod_df, drift_table,
        categorical_cols, numerical_cols
    )

    summary_cards = create_summary_cards(metrics)

    numerical_fig = create_numerical_drift_plot(drift_table)

    # ---------------------------
    # SEVERITY ORDER FIX
    # ---------------------------
    severity_counts = drift_table["Severity"].value_counts().reset_index()
    severity_counts.columns = ["Severity", "Count"]

    severity_counts["Severity"] = pd.Categorical(
        severity_counts["Severity"],
        categories=SEVERITY_ORDER,
        ordered=True
    )

    severity_counts = severity_counts.sort_values("Severity")

    severity_fig = px.bar(
        severity_counts,
        x="Severity",
        y="Count",
        color="Severity",
        title="Drift Severity Distribution"
    )

    severity_fig.add_annotation(
        text="LOW → MODERATE → HIGH drift distribution",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False
    )

    # ---------------------------
    # LEADERBOARD
    # ---------------------------
    if drifted.empty:
        leaderboard_fig = px.bar(
            drift_table.sort_values("PSI", ascending=False).head(10),
            x="PSI",
            y="Feature",
            orientation="h",
            title="Top Features by PSI"
        )
    else:
        leaderboard_fig = px.bar(
            drifted.sort_values("PSI", ascending=False),
            x="PSI",
            y="Feature",
            orientation="h",
            color="Severity",
            title="Drifted Features Leaderboard"
        )

    leaderboard_fig.add_annotation(
        text="Higher PSI = higher risk features",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False
    )

    # ---------------------------
    # HEATMAP 
    # ---------------------------
    heatmap_fig = create_drift_heatmap(drift_table)
    

    # ---------------------------
    # DASHBOARD
    # ---------------------------
    app.layout = html.Div([

        html.H1("Adaptive Drift Monitoring System", style={"textAlign": "center"}),

        summary_cards,

        dcc.Graph(figure=severity_fig),
        dcc.Graph(figure=leaderboard_fig),
        dcc.Graph(figure=numerical_fig),
        dcc.Graph(figure=heatmap_fig),
        

        html.H3("Feature Drift Explorer", style={"textAlign": "center"}),

        dcc.Dropdown(
            id="feature_dropdown",
            options=[{"label": f, "value": f} for f in drifted_features],
            value=drifted_features[0] if drifted_features else None,
            style={"width": "60%", "margin": "auto"}
        ),

        dcc.Graph(id="feature_plot")
    ])

    @app.callback(
        Output("feature_plot", "figure"),
        Input("feature_dropdown", "value")
    )
    def update_plot(feature):

        if feature is None:
            return px.scatter(title="No feature selected")

        if feature in categorical_cols:

            train_dist = train_df[feature].value_counts(normalize=True).head(10)
            prod_dist = prod_df[feature].value_counts(normalize=True).head(10)

            df = pd.concat([train_dist, prod_dist], axis=1, keys=["Train", "Production"]).fillna(0)
            df = df.reset_index()
            df.columns = ["Category", "Train", "Production"]

            return px.bar(
                df,
                x="Category",
                y=["Train", "Production"],
                barmode="group",
                title=f"{feature} Distribution Drift"
            )

        else:

            train_data = train_df[[feature]].copy()
            prod_data = prod_df[[feature]].copy()

            train_data["dataset"] = "Train"
            prod_data["dataset"] = "Production"

            combined = pd.concat([train_data, prod_data])

            return px.histogram(
                combined,
                x=feature,
                color="dataset",
                barmode="overlay",
                opacity=0.6,
                title=f"{feature} Distribution Comparison"
            )

    app.run(debug=False)