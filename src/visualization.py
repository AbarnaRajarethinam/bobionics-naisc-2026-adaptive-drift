import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.io as pio

pio.templates.default = "plotly_dark"


# =========================
#  PROFESSIONAL THEME 
# =========================
THEME = {
    "bg": "#0b1220",        # deep dashboard background
    "panel": "#111827",     # card surface
    "card": "#1f2937",      # inner KPI cards
    "text": "#e5e7eb",      # primary text
    "muted": "#9ca3af",     # secondary text
    "accent": "#3b82f6"     # modern blue accent
}


# =========================
#  UI SYSTEM 
# =========================
UI = {
    "radius": "16px",
    "gap": "16px",
    "padding": "16px",
    "shadow": "0px 6px 18px rgba(0,0,0,0.35)",
    "border": "1px solid rgba(255,255,255,0.08)"
}


def chart_card(fig):
    return html.Div([
        dcc.Graph(figure=fig)
    ], style={
        "backgroundColor": THEME["panel"],
        "borderRadius": UI["radius"],
        "padding": UI["padding"],
        "margin": "12px",
        "boxShadow": UI["shadow"],
        "border": UI["border"]
    })


# =========================
# KPI CARD HEADER ACCENT
# =========================
def accent_bar():
    return html.Div(style={
        "height": "4px",
        "width": "40px",
        "background": THEME["accent"],
        "borderRadius": "4px",
        "marginBottom": "10px"
    })


# ---------------------------
# SUMMARY CARDS 
# ---------------------------
def create_summary_cards(metrics):

    card_style = {
        "flex": "1",
        "minWidth": "180px",
        "padding": "16px",
        "borderRadius": "16px",
        "background": "linear-gradient(135deg, #1f2937, #111827)",
        "boxShadow": UI["shadow"],
        "border": UI["border"],
        "color": THEME["text"]
    }

    return html.Div([
        html.Div([
            accent_bar(),
            html.H4("Total Features", style={"color": THEME["muted"]}),
            html.H2(metrics["total_features"])
        ], style=card_style),

        html.Div([
            accent_bar(),
            html.H4("Drifted Features", style={"color": THEME["muted"]}),
            html.H2(metrics["drifted_features"])
        ], style=card_style),

        html.Div([
            accent_bar(),
            html.H4("Numerical Features", style={"color": THEME["muted"]}),
            html.H2(metrics["numerical_features"])
        ], style=card_style),

        html.Div([
            accent_bar(),
            html.H4("Categorical Features", style={"color": THEME["muted"]}),
            html.H2(metrics["categorical_features"])
        ], style=card_style),

        html.Div([
            accent_bar(),
            html.H4("Average PSI", style={"color": THEME["muted"]}),
            html.H2(metrics["avg_psi"])
        ], style=card_style),

        html.Div([
            accent_bar(),
            html.H4("Max PSI", style={"color": THEME["muted"]}),
            html.H2(metrics["max_psi"])
        ], style=card_style)

    ], style={
        "display": "flex",
        "gap": "14px",
        "flexWrap": "wrap",
        "marginBottom": "20px",
        "padding": "10px"
    })


# ---------------------------
# METRICS (
# ---------------------------
def compute_summary_metrics(train_df, prod_df, drift_table, categorical_cols, numerical_cols):

    drifted = drift_table[drift_table["Drift_Detected"] == True]

    return {
        "total_features": len(drift_table),
        "drifted_features": len(drifted),
        "numerical_features": len(numerical_cols),
        "categorical_features": len(categorical_cols),
        "avg_psi": round(drift_table["PSI"].fillna(0).mean(), 3),
        "max_psi": round(drift_table["PSI"].fillna(0).max(), 3)
    }


# ---------------------------
# SEVERITY (UNCHANGED)
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
# STATIC EXPORT (STYLE ONLY)
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

    fig.update_layout(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        title_font=dict(size=18),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.add_annotation(
        text="PSI Guide: <0.1 Low | 0.1–0.25 Moderate | >0.25 High Drift",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(color=THEME["muted"], size=12)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.write_html("outputs/drift_dashboard.html")


# ---------------------------
# NUMERICAL DRIFT (STYLE ONLY)
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

    fig.add_vline(x=0.1, line_dash="dash", line_color="orange")
    fig.add_vline(x=0.25, line_dash="dash", line_color="red")

    fig.add_annotation(
        text="Orange = Moderate threshold | Red = High drift threshold",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(color=THEME["muted"], size=12)
    )

    fig.update_layout(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


# -----------
# HEATMAP 
# ----------
def create_drift_heatmap(drift_table):

    df = drift_table.copy()
    df = df.reset_index(drop=True)
    df = df.dropna(subset=["Type", "PSI"])

    df["Severity"] = df["PSI"].apply(classify_severity)

    heatmap_data = pd.crosstab(
        df["Type"].values,
        df["Severity"].values
    )

    for col in SEVERITY_ORDER:
        if col not in heatmap_data.columns:
            heatmap_data[col] = 0

    heatmap_data = heatmap_data[SEVERITY_ORDER]

    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        title="Drift Heatmap: Feature Type vs Severity",
        color_continuous_scale="Blues"
    )

    fig.add_annotation(
        text="Shows how drift severity is distributed across feature types",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(color=THEME["muted"], size=12)
    )

    fig.update_layout(
        paper_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


# ---------------------------
# MAIN DASHBOARD (UI UPGRADE ONLY)
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
    heatmap_fig = create_drift_heatmap(drift_table)

    leaderboard_fig = px.bar(
        drifted.sort_values("PSI", ascending=False) if not drifted.empty else drift_table.head(10),
        x="PSI",
        y="Feature",
        orientation="h",
        color="Severity",
        title="Drift Risk Leaderboard"
    )

    leaderboard_fig.add_annotation(
        text="Higher PSI = higher drift risk",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(color=THEME["muted"], size=12)
    )

    leaderboard_fig.update_layout(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    leaderboard_fig.update_xaxes(showgrid=False)
    leaderboard_fig.update_yaxes(showgrid=False)

    severity_counts = drift_table["Severity"].value_counts().reset_index()
    severity_counts.columns = ["Severity", "Count"]

    severity_fig = px.bar(
        severity_counts,
        x="Severity",
        y="Count",
        color="Severity",
        title="System Drift Health"
    )

    severity_fig.add_annotation(
        text="Overall distribution of drift severity across all features",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(color=THEME["muted"], size=12)
    )

    severity_fig.update_layout(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"])
    )

    app.layout = html.Div([

        html.Div(
            "Adaptive Drift Monitoring System",
            style={
                "textAlign": "center",
                "fontSize": "30px",
                "fontWeight": "bold",
                "color": THEME["text"],
                "padding": "15px"
            }
        ),
         html.Div(
        "Real-time monitoring of feature distribution drift between training and production data.",
        style={
            "textAlign": "center",
            "fontSize": "14px",
            "color": THEME["muted"],
            "marginBottom": "15px"
        }
    ),

        summary_cards,

        html.Div([
            chart_card(severity_fig),
            chart_card(leaderboard_fig),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "16px"
        }),

        html.Div([
            chart_card(numerical_fig),
            chart_card(heatmap_fig),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "16px"
        }),

        html.Hr(style={"border": "1px solid rgba(255,255,255,0.08)"}),

        html.H3(
            "Feature Drift Explorer",
            style={"textAlign": "center", "color": THEME["text"]}
        ),

        dcc.Dropdown(
            id="feature_dropdown",
            options=[{"label": f, "value": f} for f in drifted_features],
            value=drifted_features[0] if drifted_features else None,
            style={"width": "60%", "margin": "auto"}
        ),

        dcc.Graph(id="feature_plot")

    ], style={
        "backgroundColor": THEME["bg"],
        "minHeight": "100vh",
        "paddingBottom": "40px"
    })

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

            fig = px.bar(
                df,
                x="Category",
                y=["Train", "Production"],
                barmode="group",
                title=f"{feature} Distribution Drift"
            )

            fig.add_annotation(
                text="Compares category distribution between Train vs Production",
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(color=THEME["muted"], size=12)
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

            fig.add_annotation(
                text="Overlaid histogram comparing feature distributions",
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(color=THEME["muted"], size=12)
            )

        fig.update_layout(
            paper_bgcolor=THEME["bg"],
            plot_bgcolor=THEME["bg"],
            font=dict(color=THEME["text"])
        )

        return fig

    app.run(debug=False)