import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from garmin_analysis.utils.data_loading import load_master_dataframe
from garmin_analysis.logging_config import get_logger
from garmin_analysis.features.coverage import filter_by_24h_coverage
from garmin_analysis.features.day_of_week_analysis import calculate_day_of_week_averages, get_day_order
from garmin_analysis.features.time_of_day_stress_analysis import (
    load_stress_data,
    calculate_hourly_stress_averages,
    calculate_hourly_stress_by_weekday
)
from garmin_analysis.features.daily_data_quality import (
    load_daily_data_quality,
    compute_and_persist_daily_data_quality,
)
from garmin_analysis.config import DAILY_DATA_QUALITY_CSV

# Get logger
logger = get_logger(__name__)

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
                        ),
                        html.Div([
                            html.Label("Max gap (minutes):"),
                            dcc.Input(
                                id='coverage-gap-minutes-dow',
                                type='number',
                                min=1,
                                step=1,
                                value=2,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'}),
                        html.Div([
                            html.Label("Day edge tolerance (minutes):"),
                            dcc.Input(
                                id='coverage-edge-minutes-dow',
                                type='number',
                                min=0,
                                step=1,
                                value=2,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'}),
                        html.Div([
                            html.Label("Coverage allowance (minutes):"),
                            dcc.Input(
                                id='coverage-allowance-minutes-dow',
                                type='number',
                                min=0,
                                step=1,
                                value=0,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'})
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
                        ),
                        html.Div([
                            html.Label("Max gap (minutes):"),
                            dcc.Input(
                                id='coverage-gap-minutes-30day',
                                type='number',
                                min=1,
                                step=1,
                                value=2,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'}),
                        html.Div([
                            html.Label("Day edge tolerance (minutes):"),
                            dcc.Input(
                                id='coverage-edge-minutes-30day',
                                type='number',
                                min=0,
                                step=1,
                                value=2,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'}),
                        html.Div([
                            html.Label("Coverage allowance (minutes):"),
                            dcc.Input(
                                id='coverage-allowance-minutes-30day',
                                type='number',
                                min=0,
                                step=1,
                                value=0,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'})
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
            dcc.Tab(label='📊 24-Hour Coverage Analysis', children=[
                html.Div([
                    html.Div([
                        html.H3("Watch Wear Time & 24-Hour Coverage Analysis"),
                        html.P("Analyze which days have complete 24-hour heart rate monitoring coverage (indicates watch was worn continuously)"),
                    ], style={'margin': '10px'}),
                    html.Div([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='coverage-date-picker',
                            min_date_allowed=df['day'].min(),
                            max_date_allowed=df['day'].max(),
                            start_date=df['day'].min(),
                            end_date=df['day'].max(),
                            display_format='YYYY-MM-DD'
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        dcc.Graph(id='coverage-timeline'),
                        dcc.Graph(id='coverage-percentage-chart'),
                        dcc.Graph(id='coverage-heatmap')
                    ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '20px'}),
                    html.Div([
                        html.H4("Coverage Statistics"),
                        html.Div(id='coverage-stats')
                    ], style={'margin': '20px'})
                ])
            ]),
            dcc.Tab(label='📈 Data Quality', children=[
                html.Div([
                    html.Div([
                        html.H3("Daily Data Quality Score"),
                        html.P("Composite score (0-100) combining 24h coverage and metric completeness. Higher = better data."),
                        html.Button("Refresh", id="dq-refresh-btn", n_clicks=0),
                    ], style={'margin': '10px'}),
                    html.Div([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='dq-date-picker',
                            min_date_allowed=df['day'].min(),
                            max_date_allowed=df['day'].max(),
                            start_date=max(df['day'].min(), df['day'].max() - pd.Timedelta(days=90)),
                            end_date=df['day'].max(),
                            display_format='YYYY-MM-DD'
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        dcc.Graph(id='dq-score-timeline'),
                        dcc.Graph(id='dq-score-distribution'),
                        dcc.Graph(id='dq-coverage-vs-completeness')
                    ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '20px'}),
                    html.Div(id='dq-stats', style={'margin': '20px'})
                ])
            ]),
            dcc.Tab(label='😰 Stress by Time of Day', children=[
                html.Div([
                    html.Div([
                        html.H3("Stress Patterns Throughout the Day"),
                        html.P("Analyze how your stress levels vary by hour of day and day of week"),
                    ], style={'margin': '10px'}),
                    html.Div([
                        html.Label("Analysis Options:"),
                        dcc.Checklist(
                            id='stress-show-weekday',
                            options=[{'label': ' Show day-of-week breakdown', 'value': 'weekday'}],
                            value=['weekday']
                        )
                    ], style={'margin': '10px'}),
                    html.Div([
                        dcc.Graph(id='stress-hourly-line'),
                        dcc.Graph(id='stress-hourly-bar'),
                        dcc.Graph(id='stress-heatmap'),
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
                        ),
                        html.Div([
                            html.Label("Max gap (minutes):"),
                            dcc.Input(
                                id='coverage-gap-minutes',
                                type='number',
                                min=1,
                                step=1,
                                value=2,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'}),
                        html.Div([
                            html.Label("Day edge tolerance (minutes):"),
                            dcc.Input(
                                id='coverage-edge-minutes',
                                type='number',
                                min=0,
                                step=1,
                                value=2,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'}),
                        html.Div([
                            html.Label("Coverage allowance (minutes):"),
                            dcc.Input(
                                id='coverage-allowance-minutes',
                                type='number',
                                min=0,
                                step=1,
                                value=0,
                                style={'width': '120px'}
                            )
                        ], style={'marginTop': '6px'})
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
    Input('coverage-filter', 'value'),
    Input('coverage-gap-minutes', 'value'),
    Input('coverage-edge-minutes', 'value'),
    Input('coverage-allowance-minutes', 'value')
)
def update_plot(metric, start, end, coverage_filter, coverage_gap_minutes=None, coverage_edge_minutes=None, coverage_allowance_minutes=None):
    try:
        df = load_master_dataframe()
        
        # Apply 24-hour coverage filtering if requested
        if 'filter' in coverage_filter:
            logger.info("Applying 24-hour coverage filter...")
            try:
                max_gap_minutes = int(coverage_gap_minutes) if coverage_gap_minutes is not None else 2
                if max_gap_minutes < 1:
                    max_gap_minutes = 2
            except (TypeError, ValueError):
                max_gap_minutes = 2
            try:
                edge_minutes = int(coverage_edge_minutes) if coverage_edge_minutes is not None else 2
                if edge_minutes < 0:
                    edge_minutes = 2
            except (TypeError, ValueError):
                edge_minutes = 2
            try:
                allowance_minutes = int(coverage_allowance_minutes) if coverage_allowance_minutes is not None else 0
                if allowance_minutes < 0:
                    allowance_minutes = 0
            except (TypeError, ValueError):
                allowance_minutes = 0
            df = filter_by_24h_coverage(
                df,
                max_gap=pd.Timedelta(minutes=max_gap_minutes),
                day_edge_tolerance=pd.Timedelta(minutes=edge_minutes),
                total_missing_allowance=pd.Timedelta(minutes=allowance_minutes),
            )
            logger.info(f"After 24h coverage filtering: {len(df)} days remaining")
        
        filtered = df[(df['day'] >= start) & (df['day'] <= end)].sort_values(by="day")
        title = f"{metric} over Time"
        if 'filter' in coverage_filter:
            title += " (24h Coverage Only)"
        
        # Check if coverage data is available to distinguish watch wear status
        has_coverage_data = 'coverage_pct' in filtered.columns or 'has_24h_coverage' in filtered.columns
        
        if has_coverage_data and metric in filtered.columns:
            # Create figure with go.Scatter to have more control over colors
            fig = go.Figure()
            
            # Determine watch wear status
            if 'has_24h_coverage' in filtered.columns:
                # Use has_24h_coverage if available (more strict)
                watch_worn = filtered[filtered['has_24h_coverage'] == True]
                watch_not_worn = filtered[filtered['has_24h_coverage'] != True]
            elif 'coverage_pct' in filtered.columns:
                # Use coverage_pct threshold (e.g., >= 80% means watch was worn)
                watch_worn = filtered[filtered['coverage_pct'] >= 80]
                watch_not_worn = filtered[filtered['coverage_pct'] < 80]
            else:
                watch_worn = filtered
                watch_not_worn = pd.DataFrame()
            
            # Add trace for data when watch was worn (normal color)
            if not watch_worn.empty and watch_worn[metric].notna().any():
                fig.add_trace(go.Scatter(
                    x=watch_worn['day'],
                    y=watch_worn[metric],
                    mode='lines+markers',
                    name=f'{metric} (Watch Worn)',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6, color='#1f77b4'),
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f}<br>Watch: Worn<extra></extra>'
                ))
            
            # Add trace for data when watch was not worn (lighter/grayed color)
            if not watch_not_worn.empty and watch_not_worn[metric].notna().any():
                fig.add_trace(go.Scatter(
                    x=watch_not_worn['day'],
                    y=watch_not_worn[metric],
                    mode='lines+markers',
                    name=f'{metric} (Watch Not Worn)',
                    line=dict(color='#cccccc', width=1.5, dash='dot'),
                    marker=dict(size=5, color='#cccccc', opacity=0.6),
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f}<br>Watch: Not Worn<extra></extra>'
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title=metric.replace('_', ' ').title(),
                hovermode='x unified',
                showlegend=True
            )
        else:
            # Fallback to simple line chart if no coverage data
            fig = px.line(filtered, x="day", y=metric, title=title)
        
        return fig
    except Exception as e:
        logger.error(f"Error updating plot: {e}")
        # Return empty figure on error
        return px.line(title="Error loading data")

@app.callback(
    [Output('dow-bar-chart', 'figure'),
     Output('dow-combined-chart', 'figure')],
    [Input('dow-metrics', 'value'),
     Input('coverage-filter-dow', 'value'),
     Input('coverage-gap-minutes-dow', 'value'),
     Input('coverage-edge-minutes-dow', 'value'),
     Input('coverage-allowance-minutes-dow', 'value')]
)
def update_day_of_week_charts(selected_metrics, coverage_filter, coverage_gap_minutes_dow=None, coverage_edge_minutes_dow=None, coverage_allowance_minutes_dow=None):
    """Update day-of-week analysis charts"""
    try:
        df = load_master_dataframe()
        
        # Apply 24-hour coverage filtering if requested
        if 'filter' in coverage_filter:
            logger.info("Applying 24-hour coverage filter to day-of-week analysis...")
            try:
                max_gap_minutes = int(coverage_gap_minutes_dow) if coverage_gap_minutes_dow is not None else 2
                if max_gap_minutes < 1:
                    max_gap_minutes = 2
            except (TypeError, ValueError):
                max_gap_minutes = 2
            try:
                edge_minutes = int(coverage_edge_minutes_dow) if coverage_edge_minutes_dow is not None else 2
                if edge_minutes < 0:
                    edge_minutes = 2
            except (TypeError, ValueError):
                edge_minutes = 2
            try:
                allowance_minutes = int(coverage_allowance_minutes_dow) if coverage_allowance_minutes_dow is not None else 0
                if allowance_minutes < 0:
                    allowance_minutes = 0
            except (TypeError, ValueError):
                allowance_minutes = 0
            df = filter_by_24h_coverage(
                df,
                max_gap=pd.Timedelta(minutes=max_gap_minutes),
                day_edge_tolerance=pd.Timedelta(minutes=edge_minutes),
                total_missing_allowance=pd.Timedelta(minutes=allowance_minutes),
            )
            logger.info(f"After 24h coverage filtering: {len(df)} days remaining")
        
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
        logger.error(f"Error updating day-of-week charts: {e}")
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
     Input('coverage-filter-30day', 'value'),
     Input('coverage-gap-minutes-30day', 'value'),
     Input('coverage-edge-minutes-30day', 'value'),
     Input('coverage-allowance-minutes-30day', 'value')]
)
def update_30day_charts(start_date, end_date, selected_metrics, coverage_filter, coverage_gap_minutes_30day=None, coverage_edge_minutes_30day=None, coverage_allowance_minutes_30day=None):
    """Update 30-day health overview charts"""
    try:
        df = load_master_dataframe()
        
        # Apply 24-hour coverage filtering if requested
        if 'filter' in coverage_filter:
            logger.info("Applying 24-hour coverage filter to 30-day analysis...")
            try:
                max_gap_minutes = int(coverage_gap_minutes_30day) if coverage_gap_minutes_30day is not None else 2
                if max_gap_minutes < 1:
                    max_gap_minutes = 2
            except (TypeError, ValueError):
                max_gap_minutes = 2
            try:
                edge_minutes = int(coverage_edge_minutes_30day) if coverage_edge_minutes_30day is not None else 2
                if edge_minutes < 0:
                    edge_minutes = 2
            except (TypeError, ValueError):
                edge_minutes = 2
            try:
                allowance_minutes = int(coverage_allowance_minutes_30day) if coverage_allowance_minutes_30day is not None else 0
                if allowance_minutes < 0:
                    allowance_minutes = 0
            except (TypeError, ValueError):
                allowance_minutes = 0
            df = filter_by_24h_coverage(
                df,
                max_gap=pd.Timedelta(minutes=max_gap_minutes),
                day_edge_tolerance=pd.Timedelta(minutes=edge_minutes),
                total_missing_allowance=pd.Timedelta(minutes=allowance_minutes),
            )
            logger.info(f"After 24h coverage filtering: {len(df)} days remaining")
        
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
        
        # Check if coverage data is available
        has_coverage_data = 'coverage_pct' in df.columns or 'has_24h_coverage' in df.columns
        
        # Add traces for each metric
        for metric in available_metrics:
            metric_data = df[df[metric].notna()]
            if not metric_data.empty:
                if has_coverage_data:
                    # Split data by watch wear status
                    if 'has_24h_coverage' in df.columns:
                        worn_data = metric_data[metric_data['has_24h_coverage'] == True]
                        not_worn_data = metric_data[metric_data['has_24h_coverage'] != True]
                    else:
                        worn_data = metric_data[metric_data['coverage_pct'] >= 80]
                        not_worn_data = metric_data[metric_data['coverage_pct'] < 80]
                    
                    # Add trace for watch worn (normal color)
                    if not worn_data.empty:
                        combined_fig.add_trace(go.Scatter(
                            name=metric.replace('_', ' ').title() + ' (Worn)',
                            x=worn_data['day'],
                            y=worn_data[metric],
                            mode='lines+markers',
                            line=dict(color=colors.get(metric, '#888888'), width=2),
                            marker=dict(size=6),
                            hovertemplate=f'{metric.replace("_", " ").title()}: %{{y:.2f}}<br>Watch: Worn<extra></extra>'
                        ))
                    
                    # Add trace for watch not worn (lighter color)
                    if not not_worn_data.empty:
                        # Use lighter version of the color
                        base_color = colors.get(metric, '#888888')
                        combined_fig.add_trace(go.Scatter(
                            name=metric.replace('_', ' ').title() + ' (Not Worn)',
                            x=not_worn_data['day'],
                            y=not_worn_data[metric],
                            mode='lines+markers',
                            line=dict(color='#cccccc', width=1.5, dash='dot'),
                            marker=dict(size=5, color='#cccccc', opacity=0.6),
                            hovertemplate=f'{metric.replace("_", " ").title()}: %{{y:.2f}}<br>Watch: Not Worn<extra></extra>'
                        ))
                else:
                    # No coverage data - use original behavior
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
            
            # Check if coverage data is available
            has_coverage_data = 'coverage_pct' in df.columns or 'has_24h_coverage' in df.columns
            
            for i, metric in enumerate(available_metrics, 1):
                metric_data = df[df[metric].notna()]
                if not metric_data.empty:
                    if has_coverage_data:
                        # Split data by watch wear status
                        if 'has_24h_coverage' in df.columns:
                            worn_data = metric_data[metric_data['has_24h_coverage'] == True]
                            not_worn_data = metric_data[metric_data['has_24h_coverage'] != True]
                        else:
                            worn_data = metric_data[metric_data['coverage_pct'] >= 80]
                            not_worn_data = metric_data[metric_data['coverage_pct'] < 80]
                        
                        # Add trace for watch worn (normal color)
                        if not worn_data.empty:
                            individual_fig.add_trace(
                                go.Scatter(
                                    name=metric.replace('_', ' ').title() + ' (Worn)',
                                    x=worn_data['day'],
                                    y=worn_data[metric],
                                    mode='lines+markers',
                                    line=dict(color=colors.get(metric, '#888888'), width=2),
                                    marker=dict(size=6),
                                    showlegend=False,
                                    hovertemplate=f'{metric.replace("_", " ").title()}: %{{y:.2f}}<br>Watch: Worn<extra></extra>'
                                ),
                                row=i, col=1
                            )
                        
                        # Add trace for watch not worn (lighter color)
                        if not not_worn_data.empty:
                            individual_fig.add_trace(
                                go.Scatter(
                                    name=metric.replace('_', ' ').title() + ' (Not Worn)',
                                    x=not_worn_data['day'],
                                    y=not_worn_data[metric],
                                    mode='lines+markers',
                                    line=dict(color='#cccccc', width=1.5, dash='dot'),
                                    marker=dict(size=5, color='#cccccc', opacity=0.6),
                                    showlegend=False,
                                    hovertemplate=f'{metric.replace("_", " ").title()}: %{{y:.2f}}<br>Watch: Not Worn<extra></extra>'
                                ),
                                row=i, col=1
                            )
                    else:
                        # No coverage data - use original behavior
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
        logger.error(f"Error updating 30-day charts: {e}")
        # Return error figures
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error loading data: {str(e)}")
        return error_fig, error_fig

@app.callback(
    [Output('coverage-timeline', 'figure'),
     Output('coverage-percentage-chart', 'figure'),
     Output('coverage-heatmap', 'figure'),
     Output('coverage-stats', 'children')],
    [Input('coverage-date-picker', 'start_date'),
     Input('coverage-date-picker', 'end_date')]
)
def update_coverage_charts(start_date, end_date):
    """Update 24-hour coverage analysis charts"""
    try:
        df = load_master_dataframe()
        
        # Check if coverage columns exist
        coverage_cols = ['coverage_pct', 'coverage_hours', 'has_24h_coverage', 'gap_count', 'total_missing_minutes']
        has_coverage_data = any(col in df.columns for col in coverage_cols)
        
        if not has_coverage_data:
            error_fig = go.Figure()
            error_fig.update_layout(
                title="Coverage data not available. Please run data ingestion to calculate coverage metrics.",
                height=400
            )
            empty_stats = html.P("Coverage metrics not available in dataset. Run data ingestion to calculate.")
            return error_fig, error_fig, error_fig, empty_stats
        
        # Filter to date range
        if start_date and end_date:
            df = df[(df['day'] >= start_date) & (df['day'] <= end_date)].copy()
        
        if df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available for selected date range", height=400)
            return empty_fig, empty_fig, empty_fig, html.P("No data available")
        
        # Timeline chart showing coverage hours over time
        timeline_fig = go.Figure()
        
        if 'coverage_hours' in df.columns:
            timeline_fig.add_trace(go.Scatter(
                name='Coverage Hours',
                x=df['day'],
                y=df['coverage_hours'],
                mode='lines+markers',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
        
        timeline_fig.add_hline(y=24, line_dash="dash", line_color="green", 
                              annotation_text="24 hours (full coverage)")
        
        timeline_fig.update_layout(
            title='Daily Coverage Hours Over Time',
            xaxis_title='Date',
            yaxis_title='Coverage Hours',
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        # Coverage percentage chart
        pct_fig = go.Figure()
        
        if 'coverage_pct' in df.columns:
            valid_pct = df[df['coverage_pct'].notna()]
            if not valid_pct.empty:
                # Color by coverage level
                colors = []
                for pct in valid_pct['coverage_pct']:
                    if pct >= 95:
                        colors.append('#2ca02c')  # Green - excellent
                    elif pct >= 80:
                        colors.append('#ff7f0e')  # Orange - good
                    elif pct >= 50:
                        colors.append('#ffbb78')  # Light orange - fair
                    else:
                        colors.append('#d62728')  # Red - poor
                
                pct_fig.add_trace(go.Bar(
                    x=valid_pct['day'],
                    y=valid_pct['coverage_pct'],
                    marker_color=colors,
                    text=[f"{val:.1f}%" for val in valid_pct['coverage_pct']],
                    textposition='outside',
                    hovertemplate='Date: %{x}<br>Coverage: %{y:.1f}%<extra></extra>'
                ))
        
        pct_fig.add_hline(y=95, line_dash="dash", line_color="green", 
                         annotation_text="95% (excellent)")
        pct_fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                         annotation_text="80% (good)")
        
        pct_fig.update_layout(
            title='Daily Coverage Percentage',
            xaxis_title='Date',
            yaxis_title='Coverage Percentage (%)',
            height=400,
            yaxis=dict(range=[0, 105])
        )
        
        # Monthly heatmap showing coverage by month
        heatmap_fig = go.Figure()
        
        if 'coverage_pct' in df.columns and 'has_24h_coverage' in df.columns:
            df['year'] = pd.to_datetime(df['day']).dt.year
            df['month'] = pd.to_datetime(df['day']).dt.month
            df['month_name'] = pd.to_datetime(df['day']).dt.strftime('%B')
            
            # Create monthly aggregates
            monthly_coverage = df.groupby(['year', 'month', 'month_name']).agg({
                'coverage_pct': 'mean',
                'has_24h_coverage': 'sum',
                'day': 'count'
            }).reset_index()
            monthly_coverage.columns = ['year', 'month', 'month_name', 'avg_coverage_pct', 'days_24h', 'total_days']
            monthly_coverage['pct_24h'] = (monthly_coverage['days_24h'] / monthly_coverage['total_days'] * 100).round(1)
            
            if not monthly_coverage.empty:
                # Create label for heatmap
                monthly_coverage['label'] = monthly_coverage.apply(
                    lambda row: f"{row['month_name']} {int(row['year'])}", axis=1
                )
                
                # Create pivot for heatmap
                heatmap_data = monthly_coverage.pivot(
                    index='year',
                    columns='month',
                    values='avg_coverage_pct'
                )
                
                month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                if not heatmap_data.empty:
                    heatmap_fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data.values,
                        x=[month_labels[i-1] for i in heatmap_data.columns],
                        y=[str(int(y)) for y in heatmap_data.index],
                        colorscale='RdYlGn',
                        colorbar=dict(title='Avg Coverage %'),
                        hovertemplate='Year: %{y}<br>Month: %{x}<br>Avg Coverage: %{z:.1f}%<extra></extra>'
                    ))
                    
                    heatmap_fig.update_layout(
                        title='Average Coverage Percentage by Month',
                        xaxis_title='Month',
                        yaxis_title='Year',
                        height=400
                    )
        
        # Coverage statistics summary
        stats_html = []
        if 'coverage_pct' in df.columns and 'has_24h_coverage' in df.columns:
            valid_coverage = df[df['coverage_pct'].notna()]
            if not valid_coverage.empty:
                avg_coverage = valid_coverage['coverage_pct'].mean()
                days_with_24h = valid_coverage['has_24h_coverage'].sum() if 'has_24h_coverage' in valid_coverage.columns else 0
                total_days = len(valid_coverage)
                pct_24h = (days_with_24h / total_days * 100) if total_days > 0 else 0
                
                stats_html = [
                    html.Div([
                        html.Div([
                            html.H5(f"{avg_coverage:.1f}%", style={'margin': '0', 'fontSize': '24px', 'color': '#1f77b4'}),
                            html.P("Average Daily Coverage", style={'margin': '0', 'fontSize': '12px'})
                        ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
                        html.Div([
                            html.H5(f"{days_with_24h}", style={'margin': '0', 'fontSize': '24px', 'color': '#2ca02c'}),
                            html.P(f"Days with 24h Coverage ({pct_24h:.1f}%)", style={'margin': '0', 'fontSize': '12px'})
                        ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
                        html.Div([
                            html.H5(f"{total_days}", style={'margin': '0', 'fontSize': '24px', 'color': '#888888'}),
                            html.P("Total Days Analyzed", style={'margin': '0', 'fontSize': '12px'})
                        ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                    ])
                ]
        
        if not stats_html:
            stats_html = [html.P("Coverage statistics not available")]
        
        return timeline_fig, pct_fig, heatmap_fig, stats_html
        
    except Exception as e:
        logger.error(f"Error updating coverage charts: {e}")
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error loading coverage data: {str(e)}", height=400)
        return error_fig, error_fig, error_fig, html.P(f"Error: {str(e)}")

@app.callback(
    [Output('dq-score-timeline', 'figure'),
     Output('dq-score-distribution', 'figure'),
     Output('dq-coverage-vs-completeness', 'figure'),
     Output('dq-stats', 'children')],
    [Input('dq-date-picker', 'start_date'),
     Input('dq-date-picker', 'end_date'),
     Input('dq-refresh-btn', 'n_clicks')]
)
def update_data_quality_charts(start_date, end_date, n_clicks):
    """Update Data Quality tab charts"""
    try:
        # On refresh click, recompute from master
        if n_clicks and n_clicks > 0:
            compute_and_persist_daily_data_quality()

        dq_df = load_daily_data_quality()
        if dq_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data quality data. Run data ingestion to generate.",
                height=400
            )
            return empty_fig, empty_fig, empty_fig, html.P(
                f"Daily data quality CSV not found at {DAILY_DATA_QUALITY_CSV}. "
                "Run: poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs"
            )

        dq_df['day'] = pd.to_datetime(dq_df['day'])
        if start_date and end_date:
            dq_df = dq_df[(dq_df['day'] >= start_date) & (dq_df['day'] <= end_date)]

        if dq_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data in selected date range", height=400)
            return empty_fig, empty_fig, empty_fig, html.P("No data in selected range")

        # Timeline
        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(
            x=dq_df['day'],
            y=dq_df['data_quality_score'],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='Data Quality Score'
        ))
        timeline_fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80 (good)")
        timeline_fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="50 (fair)")
        timeline_fig.update_layout(
            title='Daily Data Quality Score Over Time',
            xaxis_title='Date',
            yaxis_title='Score (0-100)',
            height=400,
            yaxis=dict(range=[0, 105])
        )

        # Distribution
        dist_fig = px.histogram(
            dq_df, x='data_quality_score', nbins=20,
            title='Distribution of Daily Data Quality Scores',
            labels={'data_quality_score': 'Score', 'count': 'Days'}
        )
        dist_fig.update_layout(height=400)

        # Coverage vs Completeness scatter
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=dq_df['coverage_score'],
            y=dq_df['completeness_score'],
            mode='markers',
            marker=dict(
                size=8,
                color=dq_df['data_quality_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Overall')
            ),
            text=dq_df['day'].dt.strftime('%Y-%m-%d'),
            hovertemplate='Date: %{text}<br>Coverage: %{x:.1f}<br>Completeness: %{y:.1f}<extra></extra>'
        ))
        scatter_fig.add_hline(y=80, line_dash="dash", line_color="gray", opacity=0.5)
        scatter_fig.add_vline(x=80, line_dash="dash", line_color="gray", opacity=0.5)
        scatter_fig.update_layout(
            title='Coverage vs Completeness (color = overall score)',
            xaxis_title='Coverage Score',
            yaxis_title='Completeness Score',
            height=400
        )

        # Stats
        avg_score = dq_df['data_quality_score'].mean()
        high_quality = (dq_df['data_quality_score'] >= 80).sum()
        total = len(dq_df)
        pct_high = (high_quality / total * 100) if total > 0 else 0
        stats_html = html.Div([
            html.Div([
                html.H5(f"{avg_score:.1f}", style={'margin': '0', 'fontSize': '24px', 'color': '#1f77b4'}),
                html.P("Average Data Quality Score", style={'margin': '0', 'fontSize': '12px'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            html.Div([
                html.H5(f"{high_quality} ({pct_high:.1f}%)", style={'margin': '0', 'fontSize': '24px', 'color': '#2ca02c'}),
                html.P("Days with Score ≥ 80", style={'margin': '0', 'fontSize': '12px'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            html.Div([
                html.H5(f"{total}", style={'margin': '0', 'fontSize': '24px', 'color': '#888888'}),
                html.P("Total Days", style={'margin': '0', 'fontSize': '12px'})
            ], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
        ])

        return timeline_fig, dist_fig, scatter_fig, stats_html

    except Exception as e:
        logger.error(f"Error updating data quality charts: {e}")
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}", height=400)
        return error_fig, error_fig, error_fig, html.P(f"Error: {str(e)}")

@app.callback(
    [Output('stress-hourly-line', 'figure'),
     Output('stress-hourly-bar', 'figure'),
     Output('stress-heatmap', 'figure')],
    [Input('stress-show-weekday', 'value')]
)
def update_stress_charts(show_weekday):
    """Update stress by time-of-day charts"""
    try:
        # Load stress data
        stress_df = load_stress_data()
        
        if stress_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No stress data available")
            return empty_fig, empty_fig, empty_fig
        
        # Calculate hourly averages
        hourly_stats = calculate_hourly_stress_averages(stress_df)
        
        if hourly_stats.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Unable to calculate hourly averages")
            return empty_fig, empty_fig, empty_fig
        
        # Create hourly line chart with confidence interval
        line_fig = go.Figure()
        
        hours = hourly_stats['hour']
        means = hourly_stats['mean']
        ci_lower = hourly_stats['ci_lower']
        ci_upper = hourly_stats['ci_upper']
        
        # Add confidence interval as filled area
        line_fig.add_trace(go.Scatter(
            name='95% CI Upper',
            x=hours,
            y=ci_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        line_fig.add_trace(go.Scatter(
            name='95% CI Lower',
            x=hours,
            y=ci_lower,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.2)',
            line=dict(width=0),
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add mean line
        line_fig.add_trace(go.Scatter(
            name='Mean Stress',
            x=hours,
            y=means,
            mode='lines+markers',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8)
        ))
        
        line_fig.update_layout(
            title='Average Stress Levels by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Average Stress Level',
            height=500,
            hovermode='x unified',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(24)),
                ticktext=[f'{h:02d}:00' for h in range(24)]
            )
        )
        
        # Create hourly bar chart
        bar_fig = go.Figure()
        
        # Color bars by stress level
        colors = []
        for stress_val in means:
            if stress_val < 30:
                colors.append('#2ca02c')  # Green - low stress
            elif stress_val < 50:
                colors.append('#ff7f0e')  # Orange - medium stress
            else:
                colors.append('#d62728')  # Red - high stress
        
        bar_fig.add_trace(go.Bar(
            x=hours,
            y=means,
            error_y=dict(type='data', array=hourly_stats['std'], visible=True),
            marker_color=colors,
            text=[f'{val:.1f}' for val in means],
            textposition='outside',
            hovertemplate='Hour: %{x}:00<br>Mean Stress: %{y:.1f}<extra></extra>'
        ))
        
        bar_fig.update_layout(
            title='Stress Distribution by Hour (Mean ± Std Dev)',
            xaxis_title='Hour of Day',
            yaxis_title='Average Stress Level',
            height=500,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(24)),
                ticktext=[f'{h:02d}:00' for h in range(24)]
            )
        )
        
        # Create heatmap if weekday breakdown is requested
        heatmap_fig = go.Figure()
        
        if 'weekday' in show_weekday:
            hourly_weekday_stats = calculate_hourly_stress_by_weekday(stress_df)
            
            if not hourly_weekday_stats.empty:
                # Pivot data for heatmap
                heatmap_data = hourly_weekday_stats.pivot(
                    index='day_of_week',
                    columns='hour',
                    values='mean'
                )
                
                # Ensure proper day ordering (Sunday first)
                day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                heatmap_data = heatmap_data.reindex(day_order)
                
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=[f'{h:02d}:00' for h in range(24)],
                    y=heatmap_data.index,
                    colorscale='RdYlGn_r',
                    colorbar=dict(title='Stress Level'),
                    hovertemplate='Day: %{y}<br>Hour: %{x}<br>Avg Stress: %{z:.1f}<extra></extra>'
                ))
                
                heatmap_fig.update_layout(
                    title='Stress Levels Heatmap: Hour of Day × Day of Week',
                    xaxis_title='Hour of Day',
                    yaxis_title='Day of Week',
                    height=500
                )
            else:
                heatmap_fig.update_layout(
                    title='Weekday breakdown not available'
                )
        else:
            heatmap_fig.update_layout(
                title='Enable "Show day-of-week breakdown" to see this chart'
            )
        
        return line_fig, bar_fig, heatmap_fig
        
    except Exception as e:
        logger.error(f"Error updating stress charts: {e}")
        # Return error figures
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error loading stress data: {str(e)}")
        return error_fig, error_fig, error_fig

# Initialize layout at module level
try:
    # Load data
    df = load_master_dataframe()
    app.layout = create_layout(df)
    logger.info("Dashboard initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize dashboard: {e}")
    # Create error layout
    app.layout = html.Div([
        html.H1("❌ Dashboard Error", style={"textAlign": "center", "color": "red"}),
        html.P(f"Failed to load data: {str(e)}", style={"textAlign": "center"}),
        html.P("Please ensure the data ingestion script has been run first.", style={"textAlign": "center"})
    ])

if __name__ == "__main__":
    import os
    # Use environment variable for debug mode (default: False for security)
    # Set DASH_DEBUG=1 or DASH_DEBUG=true to enable debug mode
    debug_mode = os.getenv('DASH_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    if debug_mode:
        logger.warning("Running dashboard in DEBUG mode. Do not use in production!")
    app.run(debug=debug_mode)
