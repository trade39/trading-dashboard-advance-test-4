# data_processing.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
import re

from config import EXPECTED_COLUMNS, APP_TITLE

logger = logging.getLogger(APP_TITLE)

def _calculate_drawdown_series_for_df(cumulative_pnl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Helper to calculate absolute and percentage drawdown series.
    """
    if cumulative_pnl.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    high_water_mark = cumulative_pnl.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl
    hwm_for_pct = high_water_mark.replace(0, np.nan)
    drawdown_pct_series = (drawdown_abs_series / hwm_for_pct).fillna(0) * 100
    if (drawdown_abs_series > 0).any() and (drawdown_pct_series[drawdown_abs_series > 0] == 0).all() and (high_water_mark == 0).all():
        pass
    return drawdown_abs_series, drawdown_pct_series

def clean_text_column(text_series: pd.Series) -> pd.Series:
    """
    Cleans a text series by removing URLs, stripping whitespace, and handling NaNs.
    """
    if not isinstance(text_series, pd.Series):
        return pd.Series(text_series, dtype=str)

    processed_series = text_series.astype(str).fillna('').str.strip()
    url_pattern = r"\(?https?://[^\s\)\"]+\)?|www\.[^\s\)\"]+"
    notion_link_pattern = r"\(https://www\.notion\.so/[^)]+\)"
    empty_parens_pattern = r"^\(''\)$"

    def clean_element(text: str) -> Any:
        if pd.isna(text) or text.lower() == 'nan': return pd.NA
        cleaned_text = text
        cleaned_text = re.sub(notion_link_pattern, '', cleaned_text)
        cleaned_text = re.sub(url_pattern, '', cleaned_text)
        cleaned_text = re.sub(empty_parens_pattern, '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text if cleaned_text else pd.NA

    return processed_series.apply(clean_element)


@st.cache_data(ttl=3600, show_spinner="Loading and processing trade data...")
def load_and_process_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """
    Loads data from CSV, cleans headers, processes column types based on EXPECTED_COLUMNS,
    and performs feature engineering.
    """
    if uploaded_file is None:
        logger.info("No file uploaded.")
        return None
    try:
        df = pd.read_csv(uploaded_file)
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}", exc_info=True)
        st.error(f"Error reading CSV: {e}")
        return None

    original_columns = df.columns.tolist()
    # Standardize column names
    df.columns = [
        str(col).strip().lower().replace(' ', '_').replace('\t', '')
        .replace('(', '').replace(')', '').replace('%', 'pct')
        .replace(':', 'rr') # Handles R:R -> r_r
        for col in original_columns
    ]
    cleaned_columns = df.columns.tolist()
    logger.info(f"Original CSV headers: {original_columns}")
    logger.info(f"Cleaned DataFrame headers in data_processing (df.columns): {cleaned_columns}") # ADDED LOGGING

    critical_expected_keys = ['date', 'pnl']
    missing_critical_info = []
    for key in critical_expected_keys:
        actual_col_name_in_df = EXPECTED_COLUMNS.get(key)
        if not actual_col_name_in_df:
            missing_critical_info.append(f"Configuration for critical key '{key}' is missing in EXPECTED_COLUMNS.")
        elif actual_col_name_in_df not in df.columns: # Check against cleaned_columns
            missing_critical_info.append(f"Critical column for '{key}' (expected as '{actual_col_name_in_df}') not found after cleaning.")

    if missing_critical_info:
        error_message = (f"Critical columns missing/misconfigured: {', '.join(missing_critical_info)}. "
                         f"Available columns after cleaning: {df.columns.tolist()}. "
                         f"Please check your CSV file or `config.EXPECTED_COLUMNS`.")
        logger.error(error_message)
        st.error(error_message)
        return None

    try:
        for conceptual_key, actual_col_name in EXPECTED_COLUMNS.items():
            if actual_col_name not in df.columns:
                logger.warning(f"Column for conceptual key '{conceptual_key}' (expected as '{actual_col_name}') not found in DataFrame. Skipping specific processing for this column.")
                if conceptual_key == 'risk': df['risk_numeric_internal'] = 0.0
                elif conceptual_key == 'duration_minutes': df['duration_minutes_numeric'] = pd.NA
                continue

            logger.debug(f"Processing conceptual key '{conceptual_key}' (actual column: '{actual_col_name}')")
            series = df[actual_col_name]

            if conceptual_key == 'date':
                df[actual_col_name] = pd.to_datetime(series, errors='coerce')
                if df[actual_col_name].isnull().sum() > 0:
                    logger.warning(f"{df[actual_col_name].isnull().sum()} invalid date formats in '{actual_col_name}'. Rows with invalid dates will be dropped.")
                    df.dropna(subset=[actual_col_name], inplace=True)
                if df.empty: logger.error("DataFrame empty after dropping invalid dates."); return None

            elif conceptual_key == 'pnl' or conceptual_key.endswith('_num') or conceptual_key in [
                'entry_price', 'exit_price', 'risk', 'signal_confidence', 'duration_minutes',
                'trade_size_num', 'r_r_csv_num',
                'initial_balance_num', 'current_balance_num',
                'drawdown_value_csv', 'stop_distance_num', 'candle_count_num',
                'loss_indicator_num', 'win_indicator_num', 'cumulative_equity_csv',
                'absolute_daily_pnl_csv', 'profit_value_csv', 'loss_value_csv', 'duration_hrs_csv',
                'peak_value_csv'
            ]:
                df[actual_col_name] = pd.to_numeric(series, errors='coerce')
                if df[actual_col_name].isnull().sum() > 0:
                     logger.debug(f"Column '{actual_col_name}' for key '{conceptual_key}' has {df[actual_col_name].isnull().sum()} NaNs after to_numeric.")
                if conceptual_key == 'pnl' and df[actual_col_name].isnull().all():
                     logger.error(f"PnL column ('{actual_col_name}') is all NaN after conversion. Cannot proceed.")
                     st.error(f"The PnL column ('{actual_col_name}') contains no valid numeric data.")
                     return None


            elif conceptual_key.endswith('_str') or conceptual_key in [
                'trade_id', 'symbol', 'notes', 'strategy', 'entry_time_str', 'trade_month_str',
                'trade_day_str', 'trade_plan_str', 'bias_str', 'tag_str', 'time_frame_str',
                'direction_str',
                'session_str', 'market_conditions_str', 'event_type_str',
                'events_details_str', 'market_sentiment_str', 'psychological_factors_str',
                'compliance_check_str', 'account_str', 'trade_outcome_csv_str', 'exit_type_csv_str',
                'error_exit_type_related_str'
            ]:
                df[actual_col_name] = clean_text_column(series).fillna('N/A')

            else: # Default for unhandled keys, treat as string
                logger.debug(f"Conceptual key '{conceptual_key}' (column '{actual_col_name}') not explicitly typed, treating as string.")
                df[actual_col_name] = series.astype(str).fillna('N/A')

        # Ensure internal numeric columns are present for feature engineering
        risk_col_mapped = EXPECTED_COLUMNS.get('risk')
        if risk_col_mapped and risk_col_mapped in df.columns and pd.api.types.is_numeric_dtype(df[risk_col_mapped]):
            df['risk_numeric_internal'] = df[risk_col_mapped].fillna(0.0)
        elif 'risk_numeric_internal' not in df.columns: # If 'risk' wasn't mapped or found
            df['risk_numeric_internal'] = 0.0
            logger.warning("Risk column for 'risk_numeric_internal' not found or not numeric. Defaulted to 0.0.")

        duration_col_mapped = EXPECTED_COLUMNS.get('duration_minutes')
        if duration_col_mapped and duration_col_mapped in df.columns and pd.api.types.is_numeric_dtype(df[duration_col_mapped]):
            df['duration_minutes_numeric'] = df[duration_col_mapped].copy().fillna(pd.NA)
        elif 'duration_minutes_numeric' not in df.columns:
            df['duration_minutes_numeric'] = pd.NA # Use pandas NA for consistency
            logger.warning("Duration column for 'duration_minutes_numeric' not found or not numeric. Defaulted to NA.")

    except Exception as e:
        logger.error(f"Error during type conversion/cleaning: {e}", exc_info=True)
        st.error(f"Type conversion error: {e}")
        return None

    # Sort by date before feature engineering that relies on order (like cumsum)
    date_col_for_sort = EXPECTED_COLUMNS.get('date')
    if date_col_for_sort and date_col_for_sort in df.columns:
        df.sort_values(by=date_col_for_sort, inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        logger.warning(f"Date column '{date_col_for_sort}' not found for sorting. Cumulative calculations might be incorrect if data is not pre-sorted.")


    # --- Feature Engineering ---
    try:
        pnl_col_fe = EXPECTED_COLUMNS['pnl'] # Assumes pnl column exists due to critical check
        df['cumulative_pnl'] = df[pnl_col_fe].cumsum()
        df['win'] = df[pnl_col_fe] > 0

        trade_outcome_col = EXPECTED_COLUMNS.get('trade_outcome_csv_str')
        if trade_outcome_col and trade_outcome_col in df.columns:
            df['trade_result_processed'] = df[trade_outcome_col].astype(str).str.upper()
            valid_outcomes = ['WIN', 'LOSS', 'BREAKEVEN', 'BE']
            df.loc[~df['trade_result_processed'].isin(valid_outcomes), 'trade_result_processed'] = 'UNKNOWN'
        else:
            df['trade_result_processed'] = np.select(
                [df[pnl_col_fe] > 0, df[pnl_col_fe] < 0],
                ['WIN', 'LOSS'],
                default='BREAKEVEN'
            )
        logger.info(f"Processed 'trade_result_processed' column. Example values: {df['trade_result_processed'].unique()[:5]}")

        date_col_fe = EXPECTED_COLUMNS['date'] # Assumes date column exists
        df['trade_hour'] = df[date_col_fe].dt.hour
        df['trade_day_of_week'] = df[date_col_fe].dt.day_name()
        df['trade_month_num'] = df[date_col_fe].dt.month
        df['trade_month_name'] = df[date_col_fe].dt.strftime('%B')
        df['trade_year'] = df[date_col_fe].dt.year
        df['trade_date_only'] = df[date_col_fe].dt.date

        if 'cumulative_pnl' in df.columns and not df['cumulative_pnl'].empty:
            df['drawdown_abs'], df['drawdown_pct'] = _calculate_drawdown_series_for_df(df['cumulative_pnl'])
        else:
            df['drawdown_abs'] = pd.Series(dtype=float)
            df['drawdown_pct'] = pd.Series(dtype=float)

        risk_col_internal_fe = 'risk_numeric_internal'
        if pd.api.types.is_numeric_dtype(df.get(risk_col_internal_fe)):
            df['reward_risk_ratio_calculated'] = df.apply(
                lambda row: row[pnl_col_fe] / abs(row[risk_col_internal_fe])
                            if pd.notna(row[pnl_col_fe]) and pd.notna(row[risk_col_internal_fe]) and abs(row[risk_col_internal_fe]) > 1e-9
                            else pd.NA,
                axis=1
            )
        else:
            df['reward_risk_ratio_calculated'] = pd.NA

        df['trade_number'] = range(1, len(df) + 1)
        logger.info("Feature engineering complete.")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}", exc_info=True)
        st.error(f"Feature engineering error: {e}")
        return df

    if df.empty:
        st.warning("No valid trade data found after processing."); return None

    logger.info(f"Data processing complete. Final DataFrame shape: {df.shape}. Final columns: {df.columns.tolist()}")
    return df
