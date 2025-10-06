import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..utils import load_master_dataframe
from ..features.coverage import filter_by_24h_coverage
from ..features.day_of_week_analysis import calculate_day_of_week_averages, get_day_order
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Build app
app = dash.Dash(__name__)
app.title = "Garmin Health Dashboard"

def create_layout(df):
    """Create the dashboard layout with the provided data"""
    available_metrics = [col for col in df.columns if col not in ["day", "date"]]
    
    return html.Div([
        html.H1("\U0001F4CA Garmin Dashboard", style={"textAlign": "center"}),
        dcc.Tabs([
            dcc.Tab(label='📅 Day of Week Analysis', children=[
                html.Div([
                    html.Div([
                        html.Label("Filter by 24h Coverage:"),
                        dcc.Checklist(
                            id='coverage-filter-dow',
                            options=[{'label': ' Only days with 24-hour continuous coverage', 'value': 'filter'}],
                            value=[]
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        html.Label("Select Metrics:"),
                        dcc.Checklist(
                            id='dow-metrics',
                            options=[
                                {'label': 'Sleep Score', 'value': 'sleep_score'},
                                {'label': 'Body Battery Max', 'value': 'body_battery_max'},
                                {'label': 'Body Battery Min', 'value': 'body_battery_min'},
                                {'label': 'Water Intake', 'value': 'water_intake'}
                            ],
                            value=['sleep_score', 'body_battery_max', 'body_battery_min']
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        dcc.Graph(id='dow-bar-chart'),
                        dcc.Graph(id='dow-combined-chart')
                    ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '20px'})
                ])
            ]),
            dcc.Tab(label='📊 30-Day Health Overview', children=[
                html.Div([
                    html.Div([
                        html.Label("Filter by 24h Coverage:"),
                        dcc.Checklist(
                            id='coverage-filter-30day',
                            options=[{'label': ' Only days with 24-hour continuous coverage', 'value': 'filter'}],
                            value=[]
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        html.Label("Select 30-day window:"),
                        dcc.DatePickerRange(
                            id='30day-date-picker',
                            min_date_allowed=df['day'].min(),
                            max_date_allowed=df['day'].max(),
                            start_date=max(df['day'].max() - pd.Timedelta(days=30), df['day'].min()),
                            end_date=df['day'].max(),
                            display_format='YYYY-MM-DD'
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        html.Label("Select Metrics to Display:"),
                        dcc.Checklist(
                            id='30day-metrics',
                            options=[
                                {'label': 'Stress (Average)', 'value': 'stress_avg'},
                                {'label': 'Resting Heart Rate', 'value': 'rhr'},
                                {'label': 'Body Battery Max', 'value': 'bb_max'},
                                {'label': 'Body Battery Min', 'value': 'bb_min'},
                                {'label': 'Sleep Score', 'value': 'score'}
                            ],
                            value=['stress_avg', 'rhr', 'bb_max', 'bb_min', 'score']
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        dcc.Graph(id='30day-combined-chart'),
                        dcc.Graph(id='30day-individual-charts')
                    ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '20px'})
                ])
            ]),
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
    [Output('dow-bar-chart', 'figure'),
     Output('dow-combined-chart', 'figure')],
    [Input('dow-metrics', 'value'),
     Input('coverage-filter-dow', 'value')]
)
def update_day_of_week_charts(selected_metrics, coverage_filter):
    """Update day-of-week analysis charts"""
    try:
        df = load_master_dataframe()
        
        # Apply 24-hour coverage filtering if requested
        if 'filter' in coverage_filter:
            logging.info("Applying 24-hour coverage filter to day-of-week analysis...")
            df = filter_by_24h_coverage(df)
            logging.info(f"After 24h coverage filtering: {len(df)} days remaining")
        
        # Calculate day-of-week averages
        day_averages = calculate_day_of_week_averages(df)
        
        if day_averages.empty:
            # Return empty figures if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available for day-of-week analysis")
            return empty_fig, empty_fig
        
        # Filter to selected metrics
        if selected_metrics:
            day_averages = day_averages[day_averages['metric'].isin(selected_metrics)]
        
        if day_averages.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available for selected metrics")
            return empty_fig, empty_fig
        
        # Create individual bar charts for each metric
        bar_fig = go.Figure()
        
        # Define colors for each metric
        colors = {
            'sleep_score': '#1f77b4',
            'body_battery_max': '#ff7f0e', 
            'body_battery_min': '#2ca02c',
            'water_intake': '#d62728'
        }
        
        # Add bars for each metric
        for metric in day_averages['metric'].unique():
            metric_data = day_averages[day_averages['metric'] == metric]
            bar_fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=metric_data['day_of_week'],
                y=metric_data['mean'],
                error_y=dict(type='data', array=metric_data['std']),
                marker_color=colors.get(metric, '#888888'),
                text=[f"{val:.1f}" for val in metric_data['mean']],
                textposition='auto'
            ))
        
        # Set proper day ordering for x-axis
        day_order = get_day_order()
        
        bar_fig.update_layout(
            title="Day-of-Week Averages (Mean ± Std Dev)",
            xaxis_title="Day of Week",
            yaxis_title="Average Value",
            barmode='group',
            height=500,
            xaxis=dict(categoryorder='array', categoryarray=day_order)
        )
        
        # Create combined chart showing all metrics together
        combined_fig = go.Figure()
        
        # Create a pivot table for the combined view
        pivot_data = day_averages.pivot(index='day_of_week', columns='metric', values='mean')
        
        # Add traces for each metric
        for metric in pivot_data.columns:
            combined_fig.add_trace(go.Scatter(
                name=metric.replace('_', ' ').title(),
                x=pivot_data.index,
                y=pivot_data[metric],
                mode='lines+markers',
                line=dict(color=colors.get(metric, '#888888'), width=3),
                marker=dict(size=8)
            ))
        
        combined_fig.update_layout(
            title="Day-of-Week Trends Comparison",
            xaxis_title="Day of Week",
            yaxis_title="Average Value",
            height=500,
            hovermode='x unified',
            xaxis=dict(categoryorder='array', categoryarray=day_order)
        )
        
        return bar_fig, combined_fig
        
    except Exception as e:
        logging.error(f"Error updating day-of-week charts: {e}")
        # Return error figures
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error loading data: {str(e)}")
        return error_fig, error_fig

@app.callback(
    [Output('30day-combined-chart', 'figure'),
     Output('30day-individual-charts', 'figure')],
    [Input('30day-date-picker', 'start_date'),
     Input('30day-date-picker', 'end_date'),
     Input('30day-metrics', 'value'),
     Input('coverage-filter-30day', 'value')]
)
def update_30day_charts(start_date, end_date, selected_metrics, coverage_filter):
    """Update 30-day health overview charts"""
    try:
        df = load_master_dataframe()
        
        # Apply 24-hour coverage filtering if requested
        if 'filter' in coverage_filter:
            logging.info("Applying 24-hour coverage filter to 30-day analysis...")
            df = filter_by_24h_coverage(df)
            logging.info(f"After 24h coverage filtering: {len(df)} days remaining")
        
        # Filter to date range
        if start_date and end_date:
            df = df[(df['day'] >= start_date) & (df['day'] <= end_date)]
        
        if df.empty:
            # Return empty figures if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available for selected date range")
            return empty_fig, empty_fig
        
        # Filter to selected metrics only
        available_metrics = []
        for metric in selected_metrics:
            if metric in df.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No selected metrics available in data")
            return empty_fig, empty_fig
        
        # Create combined chart showing all metrics on one plot
        combined_fig = go.Figure()
        
        # Define colors for each metric
        colors = {
            'stress_avg': '#d62728',
            'rhr': '#ff7f0e', 
            'bb_max': '#2ca02c',
            'bb_min': '#9467bd',
            'score': '#1f77b4'
        }
        
        # Add traces for each metric
        for metric in available_metrics:
            metric_data = df[df[metric].notna()]
            if not metric_data.empty:
                combined_fig.add_trace(go.Scatter(
                    name=metric.replace('_', ' ').title(),
                    x=metric_data['day'],
                    y=metric_data[metric],
                    mode='lines+markers',
                    line=dict(color=colors.get(metric, '#888888'), width=2),
                    marker=dict(size=6)
                ))
        
        combined_fig.update_layout(
            title=f"30-Day Health Overview ({start_date} to {end_date})",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        # Create individual subplot charts
        if len(available_metrics) > 1:
            # Create subplots
            individual_fig = make_subplots(
                rows=len(available_metrics), 
                cols=1,
                subplot_titles=[metric.replace('_', ' ').title() for metric in available_metrics],
                vertical_spacing=0.1
            )
            
            for i, metric in enumerate(available_metrics, 1):
                metric_data = df[df[metric].notna()]
                if not metric_data.empty:
                    individual_fig.add_trace(
                        go.Scatter(
                            name=metric.replace('_', ' ').title(),
                            x=metric_data['day'],
                            y=metric_data[metric],
                            mode='lines+markers',
                            line=dict(color=colors.get(metric, '#888888'), width=2),
                            marker=dict(size=6),
                            showlegend=False
                        ),
                        row=i, col=1
                    )
            
            individual_fig.update_layout(
                title=f"Individual Health Metrics ({start_date} to {end_date})",
                height=200 * len(available_metrics),
                showlegend=False
            )
            
            # Update x-axis labels
            individual_fig.update_xaxes(title_text="Date", row=len(available_metrics), col=1)
        else:
            # Single metric - just show the combined chart again
            individual_fig = combined_fig
        
        return combined_fig, individual_fig
        
    except Exception as e:
        logging.error(f"Error updating 30-day charts: {e}")
        # Return error figures
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error loading data: {str(e)}")
        return error_fig, error_fig

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
