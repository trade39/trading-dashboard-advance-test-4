# pages/3_🎯_Categorical_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple
import plotly.express as px

# --- Configuration and Utility Imports ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, COLORS, PLOTLY_THEME_DARK, PLOTLY_THEME_LIGHT
    from utils.common_utils import display_custom_message, format_currency, format_percentage
except ImportError as e:
    st.error(f"Categorical Analysis Page Error: Critical config/utils import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error"
    EXPECTED_COLUMNS = {"pnl": "pnl_fallback", "date": "date_fallback", "strategy": "strategy_fallback", "market_conditions_str": "market_conditions_fallback", "r_r_csv_num": "r_r_fallback", "direction_str": "direction_fallback"}
    COLORS = {"green": "#00FF00", "red": "#FF0000", "gray": "#808080"}
    PLOTLY_THEME_DARK = "plotly_dark"
    PLOTLY_THEME_LIGHT = "plotly_white"
    def display_custom_message(msg, type="error"): st.error(msg)
    def format_currency(val): return f"${val:,.2f}"
    def format_percentage(val): return f"{val:.2%}"
    logger = logging.getLogger("CategoricalAnalysisPage_Fallback")
    logger.error(f"CRITICAL IMPORT ERROR (Config/Utils) in Categorical Analysis Page: {e}", exc_info=True)
    st.stop()

# --- Plotting and Component Imports ---
try:
    from plotting import (
        _apply_custom_theme,
        plot_pnl_by_category,
        plot_stacked_bar_chart,
        plot_heatmap,
        plot_value_over_time,
        plot_grouped_bar_chart,
        plot_box_plot,
        plot_donut_chart,
        plot_radar_chart,
        plot_scatter_plot,
        plot_pnl_distribution,
        plot_win_rate_analysis
    )
    from components.calendar_view import PnLCalendarComponent
except ImportError as e:
    st.error(f"Categorical Analysis Page Error: Critical plotting/component import failed: {e}.")
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR (Plotting/Components) in Categorical Analysis Page: {e}", exc_info=True)
    def _apply_custom_theme(fig, theme): return fig
    def plot_pnl_by_category(*args, **kwargs): return None
    def plot_stacked_bar_chart(*args, **kwargs): return None
    def plot_heatmap(*args, **kwargs): return None # Ensure this is defined for fallback
    st.stop()


logger = logging.getLogger(APP_TITLE)

def get_column_name(conceptual_key: str) -> Optional[str]:
    """Helper to get actual column name from EXPECTED_COLUMNS mapping."""
    return EXPECTED_COLUMNS.get(conceptual_key)


def show_categorical_analysis_page():
    st.title("🎯 Categorical Performance Analysis")
    logger.info("Rendering Categorical Analysis Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view categorical analysis.", "info")
        return

    df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    pnl_col = get_column_name('pnl')
    trade_result_col = 'trade_result_processed' # This is an engineered column from data_processing.py
    win_col = 'win' # This is an engineered column

    logger.debug(f"Categorical Analysis Page: DataFrame columns available at page start: {df.columns.tolist()}")
    logger.debug(f"Categorical Analysis Page: DataFrame shape at page start: {df.shape}")


    if df.empty:
        display_custom_message("No data matches the current filters. Cannot perform categorical analysis.", "info")
        return
    if not pnl_col or pnl_col not in df.columns:
        display_custom_message(f"Essential PnL column ('{pnl_col}') not found. Analysis cannot proceed.", "error")
        return
    if trade_result_col not in df.columns:
        logger.warning(f"Engineered Trade Result column ('{trade_result_col}') not found. Some analyses may be affected.")
    if win_col not in df.columns:
        logger.warning(f"Engineered Win column ('{win_col}') not found. Some analyses may be affected.")

    # --- 1. Strategy Performance ---
    st.header("1. Strategy Performance Insights")
    with st.expander("Strategy Metrics", expanded=True):
        col1a, col1b = st.columns(2)
        with col1a:
            strategy_col = get_column_name('strategy')
            if strategy_col and pnl_col and strategy_col in df.columns and pnl_col in df.columns:
                fig_avg_pnl_strategy = plot_pnl_by_category(
                    df, category_col=strategy_col, pnl_col=pnl_col,
                    title_prefix="Average PnL by", aggregation_func='mean', theme=plot_theme
                )
                if fig_avg_pnl_strategy: st.plotly_chart(fig_avg_pnl_strategy, use_container_width=True)
                else: display_custom_message(f"Could not generate Avg PnL by {strategy_col} chart.", "warning")
            else:
                display_custom_message(f"Missing columns for Avg PnL by Strategy. Needed: '{strategy_col}' (for strategy) and '{pnl_col}' (for PnL).", "warning")

        with col1b:
            trade_plan_col = get_column_name('trade_plan_str')
            if trade_plan_col and trade_result_col and trade_plan_col in df.columns and trade_result_col in df.columns:
                fig_result_by_plan = plot_stacked_bar_chart(
                    df, category_col=trade_plan_col, stack_col=trade_result_col,
                    title=f"{trade_result_col.replace('_',' ').title()} by {trade_plan_col.replace('_',' ').title()}",
                    theme=plot_theme,
                    color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
                )
                if fig_result_by_plan: st.plotly_chart(fig_result_by_plan, use_container_width=True)
                else: display_custom_message(f"Could not generate Trade Result by {trade_plan_col} chart.", "warning")
            else:
                display_custom_message(f"Missing columns for Trade Result by Trade Plan. Needed: '{trade_plan_col}' and '{trade_result_col}'.", "warning")

        # R:R Heatmap Section
        rr_col = get_column_name('r_r_csv_num')
        direction_col = get_column_name('direction_str')
        # strategy_col is already defined from col1a

        st.markdown("---") # Visual separator for R:R Heatmap section
        logger.info(f"R:R Heatmap - Attempting generation. Resolved columns: Strategy='{strategy_col}', R:R='{rr_col}', Direction='{direction_col}'")

        prereq_cols_resolved = all([strategy_col, rr_col, direction_col])
        prereq_cols_exist_in_df = False
        if prereq_cols_resolved:
            prereq_cols_exist_in_df = all(c in df.columns for c in [strategy_col, rr_col, direction_col])

        heatmap_generated_successfully = False

        if prereq_cols_resolved and prereq_cols_exist_in_df:
            logger.debug(f"R:R Heatmap - All prerequisite columns ('{strategy_col}', '{rr_col}', '{direction_col}') found in DataFrame.")
            pivot_table_created_successfully = False
            try:
                df_rr_heatmap_prep = df[[strategy_col, rr_col, direction_col]].copy()
                logger.debug(f"R:R Heatmap - Initial shape for heatmap prep: {df_rr_heatmap_prep.shape}")

                original_rr_nan_count = df_rr_heatmap_prep[rr_col].isnull().sum()
                df_rr_heatmap_prep[rr_col] = pd.to_numeric(df_rr_heatmap_prep[rr_col], errors='coerce')
                coerced_rr_nan_count = df_rr_heatmap_prep[rr_col].isnull().sum()
                logger.debug(f"R:R Heatmap - Column '{rr_col}' dtype after to_numeric: {df_rr_heatmap_prep[rr_col].dtype}. NaNs before: {original_rr_nan_count}, NaNs after coerce: {coerced_rr_nan_count}.")

                shape_before_dropna = df_rr_heatmap_prep.shape
                df_rr_heatmap_cleaned = df_rr_heatmap_prep.dropna(subset=[rr_col, strategy_col, direction_col])
                logger.debug(f"R:R Heatmap - Shape after dropna: {df_rr_heatmap_cleaned.shape} (was {shape_before_dropna})")

                if not df_rr_heatmap_cleaned.empty:
                    if df_rr_heatmap_cleaned[strategy_col].nunique() < 1 or df_rr_heatmap_cleaned[direction_col].nunique() < 1:
                        logger.warning("R:R Heatmap - Not enough unique categories for pivot.")
                    else:
                        pivot_rr = pd.pivot_table(df_rr_heatmap_cleaned, values=rr_col,
                                                  index=[strategy_col, direction_col],
                                                  aggfunc='mean').unstack(level=-1)
                        if isinstance(pivot_rr.columns, pd.MultiIndex):
                             pivot_rr.columns = pivot_rr.columns.droplevel(0)

                        if not pivot_rr.empty:
                            logger.info(f"R:R Heatmap - Pivot table created successfully. Shape: {pivot_rr.shape}")
                            pivot_table_created_successfully = True

                            try: # Displaying pivot table (optional, can be removed for cleaner UI)
                                # st.dataframe(pivot_rr.style.format("{:.2f}").background_gradient(cmap='viridis', axis=None))
                                pass # Removed direct display of pivot table
                            except Exception as e_pivot_display:
                                logger.error(f"Error during (optional) pivot table display: {e_pivot_display}", exc_info=True)


                            try:
                                logger.debug("R:R Heatmap - Attempting to plot heatmap from pivot table.")
                                fig_rr_heatmap = plot_heatmap(
                                    pivot_rr, title=f"Average R:R by {strategy_col.replace('_',' ').title()} and {direction_col.replace('_',' ').title()}",
                                    color_scale="Viridis", theme=plot_theme, text_format=".2f"
                                )
                                if fig_rr_heatmap:
                                    st.plotly_chart(fig_rr_heatmap, use_container_width=True)
                                    logger.info("R:R Heatmap - Successfully plotted.")
                                    heatmap_generated_successfully = True
                                else:
                                    display_custom_message("Could not generate R:R heatmap plot (plot_heatmap function returned None).", "error")
                                    logger.error("R:R Heatmap - plot_heatmap function returned None.")
                            except Exception as e_plot:
                                logger.error(f"Error during plot_heatmap or st.plotly_chart: {e_plot}", exc_info=True)
                                display_custom_message(f"Error displaying R:R heatmap: {str(e_plot)}", "error")
                        else:
                            logger.warning("R:R Heatmap - Pivot table is empty.")
                else:
                    logger.warning("R:R Heatmap - df_rr_heatmap_cleaned is empty.")
            except Exception as e_rr_heatmap_outer:
                logger.error(f"Outer error during R:R heatmap generation: {e_rr_heatmap_outer}", exc_info=True)
                display_custom_message(f"An unexpected error occurred while preparing the R:R heatmap: {str(e_rr_heatmap_outer)}", "error")

        if not heatmap_generated_successfully:
            final_warning_parts = [
                "Could not generate R:R Heatmap."
            ]
            if not prereq_cols_resolved:
                final_warning_parts.append("One or more conceptual keys (strategy, r_r_csv_num, direction_str) could not be resolved from config.")
            elif not prereq_cols_exist_in_df:
                missing_in_df_list = [c for c in [strategy_col, rr_col, direction_col] if c and c not in df.columns]
                final_warning_parts.append(f"Target columns missing from DataFrame: {missing_in_df_list if missing_in_df_list else 'None identified, check logs.'}.")
            elif not pivot_table_created_successfully: # Implies columns existed but data prep failed
                 final_warning_parts.append("Data preparation for pivot table failed (e.g., no valid numeric R:R data after cleaning, or no overlapping categories).")


            final_warning_parts.append(f"Please ensure target columns (Strategy='{strategy_col}', R:R='{rr_col}', Direction='{direction_col}') exist in your CSV, are correctly mapped in config, and R:R column has numeric data.")
            final_warning_message = " ".join(final_warning_parts)
            logger.warning(final_warning_message + f" Available DF columns: {df.columns.tolist()}")
            display_custom_message(final_warning_message, "warning")
        # Removed the st.markdown("---") that was here to avoid double separator if heatmap fails.
        # If heatmap succeeds, the plot itself provides separation. If it fails, the warning is the main content.

    # --- 2. Temporal Analysis ---
    st.header("2. Temporal Analysis")
    with st.expander("Time-Based Performance", expanded=True):
        col2a, col2b = st.columns(2)
        with col2a:
            month_num_col = 'trade_month_num'
            month_name_col = 'trade_month_name'

            if month_num_col in df.columns and month_name_col in df.columns and win_col in df.columns:
                try:
                    monthly_win_rate_data = df.groupby(month_num_col)[win_col].mean() * 100
                    month_map_df = df[[month_num_col, month_name_col]].drop_duplicates().sort_values(month_num_col)
                    month_mapping = pd.Series(month_map_df[month_name_col].values, index=month_map_df[month_num_col]).to_dict()
                    monthly_win_rate = monthly_win_rate_data.rename(index=month_mapping).sort_index()

                    if not monthly_win_rate.empty:
                        fig_monthly_wr = plot_value_over_time(
                            monthly_win_rate, series_name="Monthly Win Rate",
                            title="Win Rate by Month", x_axis_title="Month", y_axis_title="Win Rate (%)",
                            theme=plot_theme
                        )
                        if fig_monthly_wr: st.plotly_chart(fig_monthly_wr, use_container_width=True)
                        else: display_custom_message("Could not generate Monthly Win Rate chart.", "warning")
                    else: display_custom_message("No data for Monthly Win Rate chart.", "info")
                except Exception as e_mwr:
                    logger.error(f"Error in Monthly Win Rate: {e_mwr}", exc_info=True)
                    display_custom_message(f"Error generating Monthly Win Rate: {str(e_mwr)}", "error")
            else:
                display_custom_message(f"Missing columns for Monthly Win Rate. Needed: '{month_num_col}', '{month_name_col}', '{win_col}'.", "warning")

        with col2b:
            session_col = get_column_name('session_str')
            time_frame_col = get_column_name('time_frame_str')
            if session_col and time_frame_col and trade_result_col and \
               all(c in df.columns for c in [session_col, time_frame_col, trade_result_col]):
                try:
                    count_df = df.groupby([session_col, time_frame_col, trade_result_col]).size().reset_index(name='count')
                    pivot_session_tf = count_df.pivot_table(index=session_col, columns=time_frame_col, values='count', fill_value=0, aggfunc='sum')

                    if not pivot_session_tf.empty:
                        fig_session_tf_heatmap = plot_heatmap(
                            pivot_session_tf, title=f"Trade Count by {session_col.replace('_',' ').title()} and {time_frame_col.replace('_',' ').title()}",
                            color_scale="Blues", theme=plot_theme, text_format=".0f"
                        )
                        if fig_session_tf_heatmap: st.plotly_chart(fig_session_tf_heatmap, use_container_width=True)
                        else: display_custom_message("Could not generate Session/Time Frame heatmap.", "warning")
                    else: display_custom_message("Not enough data for Session/Time Frame heatmap.", "info")
                except Exception as e_sess_tf_heatmap:
                    logger.error(f"Error generating Session/Time Frame heatmap: {e_sess_tf_heatmap}", exc_info=True)
                    display_custom_message(f"Error creating Session/Time Frame heatmap: {str(e_sess_tf_heatmap)}", "error")
            else:
                display_custom_message(f"Missing columns for Session/Time Frame Heatmap. Needed: '{session_col}', '{time_frame_col}', '{trade_result_col}'.", "warning")

        date_col_cal = get_column_name('date')
        if date_col_cal and pnl_col and date_col_cal in df.columns and pnl_col in df.columns:
            try:
                daily_pnl_df_agg = df.groupby(
                    df[date_col_cal].dt.normalize()
                )[pnl_col].sum().reset_index()
                daily_pnl_df_agg = daily_pnl_df_agg.rename(columns={date_col_cal: 'date', pnl_col: 'pnl'})

                available_years = sorted(daily_pnl_df_agg['date'].dt.year.unique(), reverse=True)
                if available_years:
                    selected_year = st.selectbox(
                        "Select Year for P&L Calendar:", options=available_years, index=0,
                        key="cat_analysis_calendar_year_select"
                    )
                    if selected_year:
                        calendar_component = PnLCalendarComponent(
                            daily_pnl_df=daily_pnl_df_agg, year=selected_year, plot_theme=plot_theme
                        )
                        calendar_component.render()
                else: display_custom_message("No yearly data for P&L calendar.", "info")
            except Exception as e_cal:
                logger.error(f"Error rendering P&L Calendar: {e_cal}", exc_info=True)
                display_custom_message(f"Could not generate P&L Calendar: {str(e_cal)}", "error")
        else:
            display_custom_message(f"Missing columns for P&L Calendar. Needed: '{date_col_cal}', '{pnl_col}'.", "warning")


    # --- 3. Market Context Impact ---
    st.header("3. Market Context Impact")
    with st.expander("Market Condition Effects", expanded=True):
        col3a, col3b = st.columns(2)
        with col3a:
            event_type_col = get_column_name('event_type_str')
            if event_type_col and trade_result_col and event_type_col in df.columns and trade_result_col in df.columns:
                fig_result_by_event = plot_grouped_bar_chart(
                    df, category_col=event_type_col, value_col=trade_result_col, 
                    group_col=trade_result_col, aggregation_func='count', 
                    title=f"{trade_result_col.replace('_',' ').title()} Count by {event_type_col.replace('_',' ').title()}",
                    theme=plot_theme,
                    color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
                )
                if fig_result_by_event: st.plotly_chart(fig_result_by_event, use_container_width=True)
                else: display_custom_message(f"Could not generate Trade Result by {event_type_col} chart.", "warning")
            else:
                display_custom_message(f"Missing columns for Trade Result by Event Type. Needed: '{event_type_col}', '{trade_result_col}'.", "warning")

        with col3b:
            market_cond_col = get_column_name('market_conditions_str') 
            if market_cond_col and pnl_col and market_cond_col in df.columns and pnl_col in df.columns:
                fig_pnl_by_market = plot_box_plot(
                    df, category_col=market_cond_col, value_col=pnl_col,
                    title=f"PnL Distribution by {market_cond_col.replace('_',' ').title()}", theme=plot_theme
                )
                if fig_pnl_by_market: st.plotly_chart(fig_pnl_by_market, use_container_width=True)
                else: display_custom_message(f"Could not generate PnL by {market_cond_col} boxplot.", "warning")
            else:
                display_custom_message(f"Missing columns for PnL by Market Conditions. Needed: '{market_cond_col}', '{pnl_col}'.", "warning")
        
        market_sent_col = get_column_name('market_sentiment_str') 
        if market_sent_col and win_col and market_sent_col in df.columns and win_col in df.columns:
            try:
                sentiment_win_rate_df = df.groupby(market_sent_col, observed=False)[win_col].mean().reset_index() 
                sentiment_win_rate_df[win_col] *= 100 
                if not sentiment_win_rate_df.empty:
                    fig_sent_wr = px.bar(sentiment_win_rate_df, x=market_sent_col, y=win_col,
                                         title=f"Win Rate by {market_sent_col.replace('_',' ').title()}",
                                         labels={win_col: "Win Rate (%)", market_sent_col: market_sent_col.replace('_',' ').title()},
                                         color=win_col, color_continuous_scale="Greens")
                    fig_sent_wr.update_yaxes(ticksuffix="%")
                    st.plotly_chart(_apply_custom_theme(fig_sent_wr, plot_theme), use_container_width=True)
                else: display_custom_message(f"No data for Market Sentiment vs Win Rate.", "info")

            except Exception as e_sent_wr:
                logger.error(f"Error generating Market Sentiment vs Win Rate: {e_sent_wr}", exc_info=True)
                display_custom_message(f"Error creating Market Sentiment vs Win Rate: {str(e_sent_wr)}", "error")
        else:
            display_custom_message(f"Missing columns for Market Sentiment vs Win Rate. Needed: '{market_sent_col}', '{win_col}'.", "warning")


    # --- 4. Behavioral Factors ---
    st.header("4. Behavioral Factors")
    with st.expander("Trader Psychology & Compliance", expanded=True):
        col4a, col4b = st.columns(2)
        with col4a:
            psych_col = get_column_name('psychological_factors_str') 
            if psych_col and trade_result_col and psych_col in df.columns and trade_result_col in df.columns:
                df_psych = df.copy()
                if df_psych[psych_col].dtype == 'object': 
                    df_psych[psych_col] = df_psych[psych_col].astype(str).str.split(',').str[0].str.strip().fillna('N/A')
                
                fig_psych_result = plot_stacked_bar_chart(
                    df_psych, category_col=psych_col, stack_col=trade_result_col,
                    title=f"{trade_result_col.replace('_',' ').title()} by Dominant {psych_col.replace('_',' ').title()}",
                    theme=plot_theme,
                    color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
                )
                if fig_psych_result: st.plotly_chart(fig_psych_result, use_container_width=True)
                else: display_custom_message(f"Could not generate Psychological Factors vs Trade Result chart.", "warning")
            else:
                display_custom_message(f"Missing columns for Psychological Factors vs Trade Result. Needed: '{psych_col}', '{trade_result_col}'.", "warning")

        with col4b:
            compliance_col = get_column_name('compliance_check_str') 
            if compliance_col and compliance_col in df.columns:
                fig_compliance = plot_donut_chart(
                    df, category_col=compliance_col,
                    title=f"{compliance_col.replace('_',' ').title()} Outcomes", theme=plot_theme
                )
                if fig_compliance: st.plotly_chart(fig_compliance, use_container_width=True)
                else: display_custom_message(f"Could not generate Compliance Check donut chart.", "warning")
            else:
                display_custom_message(f"Missing column for Compliance Check ('{compliance_col}').", "warning")
        
    # --- 5. Capital & Risk Analysis ---
    st.header("5. Capital & Risk Insights")
    with st.expander("Capital Management and Drawdown", expanded=True):
        col5a, col5b = st.columns(2)
        with col5a:
            initial_bal_col = get_column_name('initial_balance_num') 
            drawdown_csv_col = get_column_name('drawdown_value_csv') 
            
            if initial_bal_col and drawdown_csv_col and trade_result_col and \
               all(c in df.columns for c in [initial_bal_col, drawdown_csv_col, trade_result_col]):
                fig_bal_dd = plot_scatter_plot(
                    df, x_col=initial_bal_col, y_col=drawdown_csv_col, color_col=trade_result_col,
                    title=f"{drawdown_csv_col.replace('_',' ').title()} vs. {initial_bal_col.replace('_',' ').title()}",
                    theme=plot_theme,
                    color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
                )
                if fig_bal_dd: st.plotly_chart(fig_bal_dd, use_container_width=True)
                else: display_custom_message("Could not generate Initial Balance vs Drawdown scatter plot.", "warning")
            else:
                display_custom_message(f"Missing columns for Balance vs Drawdown. Needed: '{initial_bal_col}', '{drawdown_csv_col}', '{trade_result_col}'.", "warning")

        with col5b:
            trade_plan_col = get_column_name('trade_plan_str') 
            if trade_plan_col and drawdown_csv_col and trade_plan_col in df.columns and drawdown_csv_col in df.columns:
                fig_avg_dd_plan = plot_pnl_by_category( 
                    df, category_col=trade_plan_col, pnl_col=drawdown_csv_col, 
                    title_prefix="Average Drawdown by", aggregation_func='mean', theme=plot_theme
                )
                if fig_avg_dd_plan: st.plotly_chart(fig_avg_dd_plan, use_container_width=True)
                else: display_custom_message(f"Could not generate Avg Drawdown by {trade_plan_col} chart.", "warning")
            else:
                display_custom_message(f"Missing columns for Avg Drawdown by Trade Plan. Needed: '{trade_plan_col}', '{drawdown_csv_col}'.", "warning")

        if drawdown_csv_col and drawdown_csv_col in df.columns:
            df_dd_hist = df.copy()
            df_dd_hist[drawdown_csv_col] = pd.to_numeric(df_dd_hist[drawdown_csv_col], errors='coerce')
            df_dd_hist.dropna(subset=[drawdown_csv_col], inplace=True)

            if not df_dd_hist.empty:
                fig_dd_hist = plot_pnl_distribution( 
                    df_dd_hist, pnl_col=drawdown_csv_col, title=f"{drawdown_csv_col.replace('_',' ').title()} Distribution",
                    theme=plot_theme, nbins=30
                )
                if fig_dd_hist: st.plotly_chart(fig_dd_hist, use_container_width=True)
                else: display_custom_message("Could not generate Drawdown histogram.", "warning")
            else:
                display_custom_message(f"No numeric data in '{drawdown_csv_col}' for histogram.", "info")
        else:
            display_custom_message(f"Missing column for Drawdown Histogram ('{drawdown_csv_col}').", "warning")


    # --- 6. Exit & Directional Insights ---
    st.header("6. Exit & Directional Insights")
    with st.expander("Trade Exits and Directional Bias", expanded=True):
        col6a, col6b = st.columns(2)
        with col6a:
            exit_type_col = get_column_name('exit_type_csv_str') 
            if exit_type_col and exit_type_col in df.columns:
                fig_exit_type = plot_donut_chart(
                    df, category_col=exit_type_col,
                    title=f"{exit_type_col.replace('_',' ').title()} Distribution", theme=plot_theme
                )
                if fig_exit_type: st.plotly_chart(fig_exit_type, use_container_width=True)
                else: display_custom_message(f"Could not generate Exit Type donut chart.", "warning")
            else:
                display_custom_message(f"Missing column for Exit Type ('{exit_type_col}').", "warning")

        with col6b:
            direction_col = get_column_name('direction_str') 
            if direction_col and win_col and direction_col in df.columns and win_col in df.columns:
                fig_dir_wr = plot_win_rate_analysis(
                    df, category_col=direction_col, win_col=win_col,
                    title_prefix="Win Rate by", theme=plot_theme
                )
                if fig_dir_wr: st.plotly_chart(fig_dir_wr, use_container_width=True)
                else: display_custom_message(f"Could not generate Win Rate by {direction_col} chart.", "warning")
            else:
                display_custom_message(f"Missing columns for Win Rate by Direction. Needed: '{direction_col}', '{win_col}'.", "warning")

        time_frame_col = get_column_name('time_frame_str') 
        if direction_col and time_frame_col and trade_result_col and \
           all(c in df.columns for c in [direction_col, time_frame_col, trade_result_col]):
            try:
                df_grouped_facet = df.groupby([direction_col, time_frame_col, trade_result_col], observed=False).size().reset_index(name='count')
                if not df_grouped_facet.empty:
                    fig_result_dir_tf = px.bar(df_grouped_facet, x=direction_col, y='count', color=trade_result_col,
                                            facet_col=time_frame_col, facet_col_wrap=3, 
                                            title=f"{trade_result_col.replace('_',' ').title()} by {direction_col.replace('_',' ').title()} and {time_frame_col.replace('_',' ').title()}",
                                            labels={'count': "Number of Trades"}, barmode='group',
                                            color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')})
                    st.plotly_chart(_apply_custom_theme(fig_result_dir_tf, plot_theme), use_container_width=True)
                else: display_custom_message("No data for Trade Result by Direction and Time Frame.", "info")
            except Exception as e_gbtf:
                logger.error(f"Error in Trade Result by Direction and Time Frame: {e_gbtf}", exc_info=True)
                display_custom_message(f"Error creating faceted chart: {str(e_gbtf)}", "error")
        else:
            display_custom_message(f"Missing columns for Trade Result by Direction & Time Frame. Needed: '{direction_col}', '{time_frame_col}', '{trade_result_col}'.", "warning")


if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_categorical_analysis_page()
