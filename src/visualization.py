import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output

import numpy as np

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

def compute_summary_metrics(train_df, prod_df, drift_table, categorical_cols, numerical_cols):
    drifted = drift_table[drift_table["Drift_Detected"] == True]

    metrics = {
        "total_features": len(drift_table),
        "drifted_features": len(drifted),
        "numerical_features": len(numerical_cols),
        "categorical_features": len(categorical_cols),
        "avg_psi": round(drift_table["PSI"].mean(), 3),
        "max_psi": round(drift_table["PSI"].max(), 3)
    }

    return metrics


def classify_severity(psi):

    if psi < 0.1:
        return "LOW"
    elif psi < 0.25:
        return "MODERATE"
    else:
        return "HIGH"


def save_static_dashboard(drift_table):

    drift_table = drift_table.copy()
    drift_table["Severity"] = drift_table["PSI"].apply(classify_severity)

    fig = px.bar(
        drift_table.sort_values("PSI", ascending=False),
        x="PSI",
        y="Feature",
        orientation="h",
        color="Severity",
        title="Feature Drift Severity"
    )

    fig.write_html("outputs/drift_dashboard.html")

def create_numerical_drift_plot(drift_table):

    numeric = drift_table[drift_table["Type"] == "Numerical"]

    if numeric.empty:
        return {}

    fig = px.bar(
        numeric.sort_values("PSI", ascending=False),
        x="PSI",
        y="Feature",
        orientation="h",
        title="Numerical Feature Drift (PSI)",
        color="PSI"
    )

    # PSI threshold lines
    fig.add_vline(x=0.1, line_dash="dash", line_color="orange",
                  annotation_text="Moderate Drift (0.1)",
                  annotation_position="top")

    fig.add_vline(x=0.25, line_dash="dash", line_color="red",
                  annotation_text="Significant Drift (0.25)",
                  annotation_position="top")

    fig.update_layout(height=500)

    return fig


def launch_dashboard(train_df, prod_df, drift_table, categorical_cols, numerical_cols):

    drift_table["Severity"] = drift_table["PSI"].apply(classify_severity)

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    drifted_features = drifted["Feature"].tolist()

    app = Dash(__name__)

    metrics = compute_summary_metrics(
        train_df,
        prod_df,
        drift_table,
        categorical_cols,
        numerical_cols
    )

    summary_cards = create_summary_cards(metrics)

    numerical_fig = create_numerical_drift_plot(drift_table)

    severity_counts = drift_table["Severity"].value_counts().reset_index()
    severity_counts.columns = ["Severity", "Count"]

    severity_fig = px.bar(
        severity_counts,
        x="Severity",
        y="Count",
        color="Severity",
        title="Drift Severity Distribution"
    )

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

        # PSI threshold lines
        leaderboard_fig.add_vline(
            x=0.1,
            line_dash="dash",
            line_color="orange",
            annotation_text="Moderate Drift",
            annotation_position="top"
        )

        leaderboard_fig.add_vline(
            x=0.25,
            line_dash="dash",
            line_color="red",
            annotation_text="Significant Drift",
            annotation_position="bottom"
        )

    app.layout = html.Div([

        html.H1("Adaptive Drift Monitoring System", style={"textAlign": "center"}),

        summary_cards,

        dcc.Graph(figure=severity_fig),
        dcc.Graph(figure=leaderboard_fig),
        dcc.Graph(figure=numerical_fig),

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
            return {}

        if feature in categorical_cols:

            train_dist = train_df[feature].value_counts(normalize=True).head(10)
            prod_dist = prod_df[feature].value_counts(normalize=True).head(10)

            df = pd.concat(
                [train_dist, prod_dist],
                axis=1,
                keys=["Train", "Production"]
            ).fillna(0)

            df = df.reset_index()
            df.columns = ["Category", "Train", "Production"]

            fig = px.bar(
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

            fig = px.histogram(
                combined,
                x=feature,
                color="dataset",
                barmode="overlay",
                opacity=0.6,
                title=f"{feature} Distribution Comparison"
            )

        return fig

    app.run(debug=False)