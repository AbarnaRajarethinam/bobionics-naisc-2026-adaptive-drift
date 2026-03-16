import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output


def classify_severity(psi):

    if psi < 0.1:
        return "LOW"
    elif psi < 0.25:
        return "MODERATE"
    else:
        return "HIGH"


def save_static_dashboard(drift_table):

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


def launch_dashboard(
    train_df,
    prod_df,
    drift_table,
    categorical_cols,
    numerical_cols
):

    drift_table["Severity"] = drift_table["PSI"].apply(classify_severity)

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    drifted_features = drifted["Feature"].tolist()

    app = Dash(__name__)

    severity_counts = drift_table["Severity"].value_counts()

    severity_fig = px.bar(
        severity_counts,
        title="Drift Severity Distribution",
        color=severity_counts.index
    )

    leaderboard_fig = px.bar(
        drifted.sort_values("PSI", ascending=False),
        x="PSI",
        y="Feature",
        orientation="h",
        color="Severity",
        title="Drifted Features Leaderboard"
    )

    app.layout = html.Div([

        html.H1("Adaptive Drift Monitoring System"),

        html.Div([
            html.Div([
                html.H3("Features Analysed"),
                html.H2(len(drift_table))
            ], style={"width": "30%", "display": "inline-block"}),

            html.Div([
                html.H3("Drifted Features"),
                html.H2(len(drifted))
            ], style={"width": "30%", "display": "inline-block"}),

        ]),

        dcc.Graph(figure=severity_fig),

        dcc.Graph(figure=leaderboard_fig),

        html.H3("Feature Drift Explorer"),

        dcc.Dropdown(
            id="feature_dropdown",
            options=[{"label": f, "value": f} for f in drifted_features],
            value=drifted_features[0] if drifted_features else None
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