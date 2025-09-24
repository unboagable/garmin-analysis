import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from garmin_analysis.utils import load_master_dataframe
from garmin_analysis.features.coverage import filter_by_24h_coverage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Build app
app = dash.Dash(__name__)
app.title = "Garmin Health Dashboard"

def create_layout(df):
    """Create the dashboard layout with the provided data"""
    available_metrics = [col for col in df.columns if col not in ["day", "date"]]
    
    # Prepare numeric subset for correlation
    numeric_df = df.select_dtypes(include=np.number).dropna(axis=1, how='any')
    corr = numeric_df.corr().round(2)
    
    return html.Div([
        html.H1("\U0001F4CA Garmin Dashboard", style={"textAlign": "center"}),
        dcc.Tabs([
            dcc.Tab(label='\U0001F4C8 Metric Trends', children=[
                html.Div([
                    html.Div([
                        html.Label("Filter by 24h Coverage:"),
                        dcc.Checklist(
                            id='coverage-filter',
                            options=[{'label': ' Only days with 24-hour continuous coverage', 'value': 'filter'}],
                            value=[]
                        )
                    ], style={'margin': '10px'}),
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
                html.Div([
                    html.Div([
                        html.Label("Filter by 24h Coverage:"),
                        dcc.Checklist(
                            id='coverage-filter-heatmap',
                            options=[{'label': ' Only days with 24-hour continuous coverage', 'value': 'filter'}],
                            value=[]
                        )
                    ], style={'margin': '10px'}),
                    dcc.Graph(id='correlation-heatmap')
                ])
            ])
        ])
    ])

@app.callback(
    Output('metric-trend', 'figure'),
    Input('metric-dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('coverage-filter', 'value')
)
def update_plot(metric, start, end, coverage_filter):
    try:
        df = load_master_dataframe()
        
        # Apply 24-hour coverage filtering if requested
        if 'filter' in coverage_filter:
            logging.info("Applying 24-hour coverage filter...")
            df = filter_by_24h_coverage(df)
            logging.info(f"After 24h coverage filtering: {len(df)} days remaining")
        
        filtered = df[(df['day'] >= start) & (df['day'] <= end)].sort_values(by="day")
        title = f"{metric} over Time"
        if 'filter' in coverage_filter:
            title += " (24h Coverage Only)"
        
        fig = px.line(filtered, x="day", y=metric, title=title)
        return fig
    except Exception as e:
        logging.error(f"Error updating plot: {e}")
        # Return empty figure on error
        return px.line(title="Error loading data")

@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('coverage-filter-heatmap', 'value')
)
def update_heatmap(coverage_filter):
    try:
        df = load_master_dataframe()
        
        # Apply 24-hour coverage filtering if requested
        if 'filter' in coverage_filter:
            logging.info("Applying 24-hour coverage filter to heatmap...")
            df = filter_by_24h_coverage(df)
            logging.info(f"After 24h coverage filtering: {len(df)} days remaining")
        
        # Prepare numeric subset for correlation
        numeric_df = df.select_dtypes(include=np.number).dropna(axis=1, how='any')
        corr = numeric_df.corr().round(2)
        
        title = "Correlation Heatmap"
        if 'filter' in coverage_filter:
            title += " (24h Coverage Only)"
        
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale='Viridis',
            showscale=True,
            title=title
        )
        return fig
    except Exception as e:
        logging.error(f"Error updating heatmap: {e}")
        # Return empty figure on error
        return ff.create_annotated_heatmap(
            z=[[0]], x=['Error'], y=['Error'], 
            colorscale='Viridis', showscale=True, title="Error loading data"
        )

if __name__ == "__main__":
    try:
        # Load data
        df = load_master_dataframe()
        app.layout = create_layout(df)
        logging.info("Dashboard initialized successfully")
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Failed to initialize dashboard: {e}")
        # Create error layout
        app.layout = html.Div([
            html.H1("❌ Dashboard Error", style={"textAlign": "center", "color": "red"}),
            html.P(f"Failed to load data: {str(e)}", style={"textAlign": "center"}),
            html.P("Please ensure the data ingestion script has been run first.", style={"textAlign": "center"})
        ])
        app.run(debug=True)
