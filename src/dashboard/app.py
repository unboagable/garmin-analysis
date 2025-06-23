import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from src.utils import load_master_dataframe

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
    html.H1("\U0001F4CA Garmin Dashboard", style={"textAlign": "center"}),
    dcc.Tabs([
        dcc.Tab(label='\U0001F4C8 Metric Trends', children=[
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
        dcc.Tab(label='\U0001F4CA Correlation Heatmap', children=[
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
    filtered = df[(df['day'] >= start) & (df['day'] <= end)].sort_values(by="day")
    fig = px.line(filtered, x="day", y=metric, title=f"{metric} over Time")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
