import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from src.utils import load_master_dataframe
from src.modeling.sleep_predictor import run_sleep_model

# Load data
df = load_master_dataframe()
available_metrics = [col for col in df.columns if col not in ["day", "date"]]

# Prepare numeric subset for correlation
numeric_df = df.select_dtypes(include=np.number).dropna(axis=1, how='any')
corr = numeric_df.corr().round(2)

# Build app
app = dash.Dash(__name__)
app.title = "Garmin Health Dashboard"

app.layout = html.Div([
    html.H1("📊 Garmin Dashboard", style={"textAlign": "center"}),
    dcc.Tabs([
        dcc.Tab(label='📈 Metric Trends', children=[
            html.Div([
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[{"label": m, "value": m} for m in available_metrics],
                    value="stress_avg",
                    clearable=False
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=df['day'].min(),
                    max_date_allowed=df['day'].max(),
                    start_date=df['day'].min(),
                    end_date=df['day'].max()
                ),
                dcc.Graph(id='metric-trend')
            ])
        ]),
        dcc.Tab(label='📊 Correlation Heatmap', children=[
            dcc.Graph(
                id='correlation-heatmap',
                figure=ff.create_annotated_heatmap(
                    z=corr.values,
                    x=list(corr.columns),
                    y=list(corr.index),
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ]),
        dcc.Tab(label='😴 Sleep Model Summary', children=[
            html.Div(id='model-output')
        ])
    ])
])

@app.callback(
    Output('metric-trend', 'figure'),
    Input('metric-dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_plot(metric, start, end):
    filtered = df[(df['day'] >= start) & (df['day'] <= end)]
    fig = px.line(filtered, x="day", y=metric, title=f"{metric} over Time")
    return fig

@app.callback(
    Output('model-output', 'children'),
    Input('model-output', 'id')
)
def update_model_output(_):
    result = run_sleep_model()
    if result["r2"] is None:
        return html.Div("❌ Sleep model could not be run (missing data).")
    return html.Div([
        html.H4("🧠 Sleep Score Predictor Results"),
        html.P(f"R² Score: {result['r2']:.4f}"),
        html.P(f"MSE: {result['mse']:.2f}"),
        html.Img(src=f"/{result['plot_path']}", style={"maxWidth": "100%"})
    ])

if __name__ == "__main__":
    app.run(debug=True)
