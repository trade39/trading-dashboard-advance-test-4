"""
plotting.py

Contains functions to generate various interactive Plotly visualizations
for the Trading Performance Dashboard.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any, Union

from config import (
    COLORS, PLOTLY_THEME_DARK, PLOTLY_THEME_LIGHT,
    PLOT_BG_COLOR_DARK, PLOT_PAPER_BG_COLOR_DARK, PLOT_FONT_COLOR_DARK,
    PLOT_BG_COLOR_LIGHT, PLOT_PAPER_BG_COLOR_LIGHT, PLOT_FONT_COLOR_LIGHT,
    PLOT_LINE_COLOR, PLOT_MARKER_PROFIT_COLOR, PLOT_MARKER_LOSS_COLOR,
    PLOT_BENCHMARK_LINE_COLOR,
    EXPECTED_COLUMNS, APP_TITLE
)
from utils.common_utils import format_currency, format_percentage # For tooltips

import logging
logger = logging.getLogger(APP_TITLE)


def _apply_custom_theme(fig: go.Figure, theme: str = 'dark') -> go.Figure:
    plotly_theme_template = PLOTLY_THEME_DARK if theme == 'dark' else PLOTLY_THEME_LIGHT
    bg_color = PLOT_BG_COLOR_DARK if theme == 'dark' else PLOT_BG_COLOR_LIGHT
    paper_bg_color = PLOT_PAPER_BG_COLOR_DARK if theme == 'dark' else PLOT_PAPER_BG_COLOR_LIGHT
    font_color = PLOT_FONT_COLOR_DARK if theme == 'dark' else PLOT_FONT_COLOR_LIGHT
    grid_color = COLORS.get('gray', '#808080') if theme == 'dark' else '#e0e0e0'

    fig.update_layout(
        template=plotly_theme_template,
        plot_bgcolor=bg_color, paper_bgcolor=paper_bg_color, font_color=font_color,
        margin=dict(l=50, r=50, t=60, b=50), # Adjusted top margin for title
        xaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False),
        hoverlabel=dict(
            bgcolor=COLORS.get('card_background_dark', '#273334') if theme == 'dark' else COLORS.get('card_background_light', '#F0F2F6'),
            font_size=12, font_family="Inter, sans-serif", bordercolor=COLORS.get('royal_blue')
        )
    )
    return fig

# --- Existing Plotting Functions (plot_equity_curve_and_drawdown, etc.) ---
# ... (Keep all existing plotting functions as they are) ...
def plot_equity_curve_and_drawdown(
    df: pd.DataFrame,
    date_col: str = EXPECTED_COLUMNS['date'],
    cumulative_pnl_col: str = 'cumulative_pnl',
    drawdown_pct_col: Optional[str] = 'drawdown_pct',
    theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or date_col not in df.columns or cumulative_pnl_col not in df.columns:
        logger.warning("Equity curve plot: Data is insufficient (df empty, or date/cumulative_pnl_col missing).")
        return None
    
    has_drawdown_data = False
    if drawdown_pct_col and drawdown_pct_col in df.columns and not df[drawdown_pct_col].dropna().empty:
        has_drawdown_data = True
    elif drawdown_pct_col: 
        logger.warning(f"Drawdown column '{drawdown_pct_col}' not found or empty. Plotting equity curve only.")

    fig_rows = 2 if has_drawdown_data else 1
    row_heights = [0.7, 0.3] if has_drawdown_data else [1.0]
    subplot_titles = ("Equity Curve", "Drawdown (%)" if has_drawdown_data else None)

    fig = make_subplots(
        rows=fig_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=[s for s in subplot_titles if s] 
    )

    fig.add_trace(
        go.Scatter(x=df[date_col], y=df[cumulative_pnl_col], mode='lines', name='Strategy Equity', line=dict(color=PLOT_LINE_COLOR, width=2)),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=1)


    if has_drawdown_data:
        fig.add_trace(
            go.Scatter(x=df[date_col], y=df[drawdown_pct_col], mode='lines', name='Drawdown', line=dict(color=COLORS.get('red', '#FF0000'), width=1.5), fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, tickformat=".2f")
        min_dd_val = df[drawdown_pct_col].min() 
        max_dd_val = df[drawdown_pct_col].max() 
        if pd.isna(min_dd_val) or pd.isna(max_dd_val) or (min_dd_val == 0 and max_dd_val == 0) :
             fig.update_yaxes(range=[-1, 1], row=2, col=1) 

    fig.update_layout(title_text='Strategy Equity and Drawdown', hovermode='x unified')
    return _apply_custom_theme(fig, theme)


def plot_equity_vs_benchmark(
    strategy_equity: pd.Series, 
    benchmark_cumulative_returns: pd.Series, 
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    if strategy_equity.empty and benchmark_cumulative_returns.empty:
        logger.warning("Equity vs Benchmark plot: Both strategy and benchmark series are empty.")
        return None

    fig = go.Figure()

    if not strategy_equity.empty:
        fig.add_trace(go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity,
            mode='lines',
            name=strategy_name,
            line=dict(color=PLOT_LINE_COLOR, width=2)
        ))
    
    if not benchmark_cumulative_returns.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative_returns.index,
            y=benchmark_cumulative_returns,
            mode='lines',
            name=benchmark_name,
            line=dict(color=PLOT_BENCHMARK_LINE_COLOR, width=2, dash='dash')
        ))

    fig.update_layout(
        title_text=f'{strategy_name} vs. {benchmark_name} Performance',
        xaxis_title="Date",
        yaxis_title="Normalized Value / Cumulative Return (%)", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)

def plot_pnl_distribution(
    df: pd.DataFrame, pnl_col: str = EXPECTED_COLUMNS['pnl'],
    title: str = "PnL Distribution (per Trade)", theme: str = 'dark',
    nbins: int = 50
) -> Optional[go.Figure]:
    if df is None or df.empty or pnl_col not in df.columns or df[pnl_col].dropna().empty:
        logger.warning("PnL distribution plot: Data is insufficient.")
        return None
    fig = px.histogram(df, x=pnl_col, nbins=nbins, title=title, marginal="box", color_discrete_sequence=[PLOT_LINE_COLOR])
    fig.update_layout(xaxis_title="PnL per Trade", yaxis_title="Frequency")
    return _apply_custom_theme(fig, theme)

def plot_time_series_decomposition(
    decomposition_result: Any, title: str = "Time Series Decomposition", theme: str = 'dark'
) -> Optional[go.Figure]:
    if decomposition_result is None:
        logger.warning("Time series decomposition plot: No decomposition result provided.")
        return None
    try:
        observed = getattr(decomposition_result, 'observed', pd.Series(dtype=float)) 
        trend = getattr(decomposition_result, 'trend', pd.Series(dtype=float))
        seasonal = getattr(decomposition_result, 'seasonal', pd.Series(dtype=float))
        resid = getattr(decomposition_result, 'resid', pd.Series(dtype=float))
        
        if observed.empty:
            logger.warning("Time series decomposition plot: Observed series is empty.")
            return None
            
        x_axis = observed.index
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
        fig.add_trace(go.Scatter(x=x_axis, y=observed, mode='lines', name='Observed', line=dict(color=PLOT_LINE_COLOR)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=trend, mode='lines', name='Trend', line=dict(color=COLORS.get('green', '#00FF00'))), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=seasonal, mode='lines', name='Seasonal', line=dict(color=COLORS.get('royal_blue', '#4169E1'))), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=resid, mode='lines+markers', name='Residual', line=dict(color=COLORS.get('gray', '#808080')), marker=dict(size=3)), row=4, col=1)
        fig.update_layout(title_text=title, height=700, showlegend=False)
        return _apply_custom_theme(fig, theme)
    except AttributeError as e:
        logger.error(f"Decomposition result missing attributes: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error plotting decomposition: {e}", exc_info=True)
        return None

def plot_value_over_time(
    series: pd.Series, series_name: str, title: Optional[str] = None,
    x_axis_title: str = "Date / Time", y_axis_title: Optional[str] = None,
    theme: str = 'dark', line_color: str = PLOT_LINE_COLOR
) -> Optional[go.Figure]:
    if series is None or series.empty:
        logger.warning(f"Plot value over time for '{series_name}': Series is empty.")
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=series_name, line=dict(color=line_color)))
    fig.update_layout(title_text=title if title else series_name, xaxis_title=x_axis_title, yaxis_title=y_axis_title if y_axis_title else series_name)
    return _apply_custom_theme(fig, theme)

def plot_pnl_by_category(
    df: pd.DataFrame, category_col: str, pnl_col: str = EXPECTED_COLUMNS['pnl'],
    title_prefix: str = "Total PnL by", theme: str = 'dark',
    aggregation_func: str = 'sum' # 'sum' or 'mean'
) -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns or pnl_col not in df.columns:
        logger.warning(f"PnL by category plot: Data insufficient or missing columns ('{category_col}', '{pnl_col}').")
        return None
    
    if aggregation_func == 'mean':
        grouped_pnl = df.groupby(category_col)[pnl_col].mean().reset_index().sort_values(by=pnl_col, ascending=False)
        yaxis_title = "Average PnL"
        title = f"{title_prefix.replace('Total', 'Average')} {category_col.replace('_', ' ').title()}"
    else: # Default to sum
        grouped_pnl = df.groupby(category_col)[pnl_col].sum().reset_index().sort_values(by=pnl_col, ascending=False)
        yaxis_title = "Total PnL"
        title = f"{title_prefix} {category_col.replace('_', ' ').title()}"

    fig = px.bar(grouped_pnl, x=category_col, y=pnl_col, title=title,
                 color=pnl_col, color_continuous_scale=[COLORS.get('red', '#FF0000'), COLORS.get('gray', '#808080'), COLORS.get('green', '#00FF00')])
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title=yaxis_title)
    return _apply_custom_theme(fig, theme)


def plot_win_rate_analysis(
    df: pd.DataFrame, category_col: str, win_col: str = 'win', # 'win' is boolean
    title_prefix: str = "Win Rate by", theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns or win_col not in df.columns:
        logger.warning(f"Win rate analysis plot: Data insufficient or missing columns ('{category_col}', '{win_col}').")
        return None
    if not pd.api.types.is_bool_dtype(df[win_col]) and not pd.api.types.is_numeric_dtype(df[win_col]): # Check if win_col is boolean or 0/1
        logger.error(f"Win column '{win_col}' must be boolean or numeric (0/1) for win rate analysis.")
        return None

    category_counts = df.groupby(category_col).size().rename('total_trades_in_cat')
    # Ensure win_col is treated as boolean for sum (True becomes 1)
    category_wins = df.groupby(category_col)[win_col].sum().rename('wins_in_cat')
    
    win_rate_df = pd.concat([category_counts, category_wins], axis=1).fillna(0)
    win_rate_df['win_rate_pct'] = (win_rate_df['wins_in_cat'] / win_rate_df['total_trades_in_cat'] * 100).fillna(0)
    win_rate_df = win_rate_df.reset_index().sort_values(by='win_rate_pct', ascending=False)
    
    fig = px.bar(win_rate_df, x=category_col, y='win_rate_pct', title=f"{title_prefix} {category_col.replace('_', ' ').title()}",
                 color='win_rate_pct', color_continuous_scale=px.colors.sequential.Greens)
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title="Win Rate (%)", yaxis_ticksuffix="%")
    return _apply_custom_theme(fig, theme)

def plot_rolling_performance(
    df: pd.DataFrame, date_col: str, metric_series: pd.Series, metric_name: str,
    title: Optional[str] = None, theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty or date_col not in df.columns or metric_series.empty:
        logger.warning(f"Rolling performance plot for '{metric_name}': Data insufficient.")
        return None
    plot_x_data = df[date_col] if len(df[date_col]) == len(metric_series) else metric_series.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_x_data, y=metric_series, mode='lines', name=metric_name, line=dict(color=PLOT_LINE_COLOR)))
    fig.update_layout(title_text=title if title else f"Rolling {metric_name}",
                      xaxis_title="Date" if date_col in df.columns and len(df[date_col]) == len(metric_series) else "Trade Number / Period",
                      yaxis_title=metric_name)
    return _apply_custom_theme(fig, theme)

def plot_correlation_matrix(
    df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
    title: str = "Correlation Matrix of Numeric Features", theme: str = 'dark'
) -> Optional[go.Figure]:
    if df is None or df.empty:
        logger.warning("Correlation matrix plot: DataFrame is empty.")
        return None
    df_numeric = df[numeric_cols].copy() if numeric_cols else df.select_dtypes(include=np.number)
    if df_numeric.empty or df_numeric.shape[1] < 2:
        logger.warning("Correlation matrix plot: Not enough numeric columns (need at least 2).")
        return None
    corr_matrix = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale='RdBu', zmin=-1, zmax=1, text=corr_matrix.round(2).astype(str),
        texttemplate="%{text}", hoverongaps=False ))
    fig.update_layout(title_text=title)
    return _apply_custom_theme(fig, theme)

def plot_bootstrap_distribution_and_ci(
    bootstrap_statistics: List[float],
    observed_statistic: float,
    lower_bound: float,
    upper_bound: float,
    statistic_name: str,
    theme: str = 'dark'
) -> Optional[go.Figure]:
    if not bootstrap_statistics:
        logger.warning(f"Bootstrap distribution plot for '{statistic_name}': No bootstrap statistics provided.")
        return None
    if pd.isna(observed_statistic) or pd.isna(lower_bound) or pd.isna(upper_bound):
        logger.warning(f"Bootstrap distribution plot for '{statistic_name}': Observed stat or CI bounds are NaN.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=bootstrap_statistics, name='Bootstrap<br>Distribution',
        marker_color=COLORS.get('royal_blue', '#4169E1'), opacity=0.75, histnorm='probability density'
    ))
    fig.add_vline(
        x=observed_statistic, line_width=2, line_dash="dash", line_color=COLORS.get('green', '#00FF00'),
        name=f'Observed<br>{statistic_name}<br>({observed_statistic:.4f})'
    )
    fig.add_vline(
        x=lower_bound, line_width=2, line_dash="dot", line_color=COLORS.get('orange', '#FFA500'),
        name=f'Lower 95% CI<br>({lower_bound:.4f})'
    )
    fig.add_vline(
        x=upper_bound, line_width=2, line_dash="dot", line_color=COLORS.get('orange', '#FFA500'),
        name=f'Upper 95% CI<br>({upper_bound:.4f})'
    )
    fig.update_layout(
        title_text=f'Bootstrap Distribution for {statistic_name}', xaxis_title=statistic_name, yaxis_title='Density',
        bargap=0.1, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)


# --- New Plotting Functions for Categorical Analysis ---

def plot_stacked_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    stack_col: str,
    value_col: Optional[str] = None, # If None, counts occurrences
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a stacked bar chart.
    If value_col is None, it counts occurrences for stacking.
    If value_col is provided, it sums that column for stacking.
    """
    if df is None or df.empty or category_col not in df.columns or stack_col not in df.columns:
        logger.warning(f"Stacked bar chart: Data insufficient or missing columns ('{category_col}', '{stack_col}').")
        return None
    if value_col and value_col not in df.columns:
        logger.warning(f"Stacked bar chart: Value column '{value_col}' not found.")
        return None

    if value_col:
        grouped_df = df.groupby([category_col, stack_col])[value_col].sum().reset_index()
        y_values = value_col
        y_axis_title = f"Sum of {value_col.replace('_', ' ').title()}"
    else:
        grouped_df = df.groupby([category_col, stack_col]).size().reset_index(name='count')
        y_values = 'count'
        y_axis_title = "Count"
    
    if grouped_df.empty:
        logger.info(f"No data to plot for stacked bar chart ({category_col} by {stack_col}).")
        return None

    fig_title = title if title else f"{stack_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"
    
    fig = px.bar(
        grouped_df,
        x=category_col,
        y=y_values,
        color=stack_col,
        title=fig_title,
        barmode='stack',
        color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title,
        legend_title_text=stack_col.replace('_', ' ').title()
    )
    return _apply_custom_theme(fig, theme)


def plot_heatmap(
    df_pivot: pd.DataFrame, # Expects a pre-pivoted DataFrame
    title: str = "Heatmap",
    x_axis_title: Optional[str] = None,
    y_axis_title: Optional[str] = None,
    color_scale: str = "RdBu", # e.g., 'RdBu', 'Viridis'
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    show_text: bool = True,
    text_format: str = ".2f", # Format for text on cells
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Generates a heatmap from a pivot table.
    """
    if df_pivot is None or df_pivot.empty:
        logger.warning("Heatmap: Input pivot DataFrame is empty.")
        return None

    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale=color_scale,
        zmin=z_min,
        zmax=z_max,
        text=df_pivot.applymap(lambda x: f"{x:{text_format}}" if pd.notna(x) else "").values if show_text else None,
        texttemplate="%{text}" if show_text else None,
        hoverongaps=False
    ))
    fig.update_layout(
        title_text=title,
        xaxis_title=x_axis_title if x_axis_title else df_pivot.columns.name,
        yaxis_title=y_axis_title if y_axis_title else df_pivot.index.name
    )
    return _apply_custom_theme(fig, theme)


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    group_col: str,
    title: Optional[str] = None,
    aggregation_func: str = 'mean', # 'mean', 'sum', 'count'
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a grouped bar chart.
    """
    if df is None or df.empty or category_col not in df.columns or value_col not in df.columns or group_col not in df.columns:
        logger.warning(f"Grouped bar chart: Data insufficient or missing columns ('{category_col}', '{value_col}', '{group_col}').")
        return None

    if aggregation_func == 'mean':
        grouped_df = df.groupby([category_col, group_col])[value_col].mean().reset_index()
        y_axis_title = f"Average {value_col.replace('_', ' ').title()}"
    elif aggregation_func == 'sum':
        grouped_df = df.groupby([category_col, group_col])[value_col].sum().reset_index()
        y_axis_title = f"Total {value_col.replace('_', ' ').title()}"
    elif aggregation_func == 'count':
        grouped_df = df.groupby([category_col, group_col]).size().reset_index(name='count')
        value_col = 'count' # Use 'count' as the value column for plotting
        y_axis_title = "Count"
    else:
        logger.error(f"Unsupported aggregation function '{aggregation_func}' for grouped bar chart.")
        return None
    
    if grouped_df.empty:
        logger.info(f"No data to plot for grouped bar chart ({category_col}, {value_col}, {group_col}).")
        return None

    fig_title = title if title else f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}, Grouped by {group_col.replace('_', ' ').title()}"

    fig = px.bar(
        grouped_df,
        x=category_col,
        y=value_col,
        color=group_col,
        title=fig_title,
        barmode='group',
        color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title,
        legend_title_text=group_col.replace('_', ' ').title()
    )
    return _apply_custom_theme(fig, theme)


def plot_box_plot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a box plot.
    """
    if df is None or df.empty or category_col not in df.columns or value_col not in df.columns:
        logger.warning(f"Box plot: Data insufficient or missing columns ('{category_col}', '{value_col}').")
        return None
    
    fig_title = title if title else f"{value_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"
    
    fig = px.box(
        df,
        x=category_col,
        y=value_col,
        color=category_col if color_discrete_map else None, # Color by category if map provided
        title=fig_title,
        points="outliers", # Show outliers
        color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=value_col.replace('_', ' ').title()
    )
    return _apply_custom_theme(fig, theme)


def plot_donut_chart(
    df: pd.DataFrame,
    category_col: str,
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a donut chart (pie chart with a hole).
    """
    if df is None or df.empty or category_col not in df.columns:
        logger.warning(f"Donut chart: Data insufficient or missing column ('{category_col}').")
        return None

    counts = df[category_col].value_counts().reset_index()
    counts.columns = [category_col, 'count']

    if counts.empty:
        logger.info(f"No data to plot for donut chart ({category_col}).")
        return None

    fig_title = title if title else f"Distribution of {category_col.replace('_', ' ').title()}"
    
    fig = px.pie(
        counts,
        names=category_col,
        values='count',
        title=fig_title,
        hole=0.4, # This makes it a donut chart
        color_discrete_map=color_discrete_map
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return _apply_custom_theme(fig, theme)


def plot_radar_chart(
    df_radar: pd.DataFrame, # Expects rows as categories, columns as groups (e.g., Win/Loss)
    categories_col: str, # Column name in df_radar that contains the category labels for theta
    value_cols: List[str], # List of columns in df_radar to plot as different traces
    title: Optional[str] = None,
    fill: str = 'toself', # 'toself' or 'tonext'
    theme: str = 'dark',
    color_discrete_sequence: Optional[List[str]] = None # e.g., [COLORS['green'], COLORS['red']]
) -> Optional[go.Figure]:
    """
    Generates a radar chart.
    df_radar: DataFrame where each row is a category (e.g., psychological factor)
              and columns are groups (e.g., 'AvgScore_Wins', 'AvgScore_Losses').
    categories_col: The name of the column in df_radar that holds the category labels.
    value_cols: List of column names in df_radar whose values will be plotted.
    """
    if df_radar is None or df_radar.empty or categories_col not in df_radar.columns:
        logger.warning("Radar chart: Data insufficient or categories column missing.")
        return None
    if not value_cols or not all(col in df_radar.columns for col in value_cols):
        logger.warning("Radar chart: Value columns list is empty or contains missing columns.")
        return None

    fig = go.Figure()
    
    category_labels = df_radar[categories_col].tolist()
    if not category_labels:
        logger.warning("Radar chart: No category labels found.")
        return None

    for i, val_col in enumerate(value_cols):
        trace_color = color_discrete_sequence[i % len(color_discrete_sequence)] if color_discrete_sequence else None
        fig.add_trace(go.Scatterpolar(
            r=df_radar[val_col].tolist(),
            theta=category_labels,
            fill=fill,
            name=val_col.replace('_', ' ').title(),
            line_color=trace_color
        ))

    fig_title = title if title else "Radar Chart Comparison"
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                # range=[0, df_radar[value_cols].max().max()] # Optional: set range
            )),
        showlegend=True,
        title=fig_title
    )
    return _apply_custom_theme(fig, theme)


def plot_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a scatter plot.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Scatter plot: Data insufficient or missing columns ('{x_col}', '{y_col}').")
        return None
    if color_col and color_col not in df.columns:
        logger.warning(f"Scatter plot: Color column '{color_col}' not found.")
        color_col = None # Disable coloring
    if size_col and size_col not in df.columns:
        logger.warning(f"Scatter plot: Size column '{size_col}' not found.")
        size_col = None # Disable sizing

    fig_title = title if title else f"{y_col.replace('_', ' ').title()} vs. {x_col.replace('_', ' ').title()}"
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=fig_title,
        color_discrete_map=color_discrete_map,
        # trendline="ols" # Optional: add a trendline
    )
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        legend_title_text=color_col.replace('_', ' ').title() if color_col else None
    )
    return _apply_custom_theme(fig, theme)

