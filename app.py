"""
app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime
import base64 # For Base64 encoding of the logo

# --- Utility Modules ---
try:
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message
except ImportError as e:
    st.error(f"Fatal Error: Could not import utility modules. App cannot start. Details: {e}")
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Fatal Error importing utils: {e}", exc_info=True)
    st.stop()

# --- Component Modules ---
try:
    from components.sidebar_manager import SidebarManager
except ImportError as e:
    st.error(f"Fatal Error: Could not import SidebarManager. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing SidebarManager: {e}", exc_info=True)
    st.stop()

# --- Service Modules ---
try:
    from services.data_service import DataService
    from services.analysis_service import AnalysisService, get_benchmark_data_static
except ImportError as e:
    st.error(f"Fatal Error: Could not import service modules. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing services: {e}", exc_info=True)
    st.stop()

# --- Core Application Modules (Configs) ---
try:
    from config import (
        APP_TITLE, EXPECTED_COLUMNS, RISK_FREE_RATE,
        LOG_FILE, LOG_LEVEL, LOG_FORMAT, DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS
    )
except ImportError as e:
    st.error(f"Fatal Error: Could not import configuration (config.py). App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing config: {e}", exc_info=True)
    APP_TITLE = "TradingAppError"
    LOG_FILE = "logs/error_app.log"; LOG_LEVEL = "ERROR"; LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    RISK_FREE_RATE = 0.02; EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}; DEFAULT_BENCHMARK_TICKER = "SPY"
    AVAILABLE_BENCHMARKS = {"S&P 500 (SPY)": "SPY", "None": ""}
    st.stop()


# --- Initialize Centralized Logger ---
logger = setup_logger(
    logger_name=APP_TITLE,
    log_file=LOG_FILE,
    level=LOG_LEVEL,
    log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Absolute path to this app.py: {os.path.abspath(__file__)}")


# --- Page Configuration (Global for all pages) ---
# Corrected logo path for browser tab icon
LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=LOGO_PATH_FOR_BROWSER_TAB, # Use the .png extension
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/trading-mastery-hub',
        'Report a bug': "https://github.com/your-repo/trading-mastery-hub/issues",
        'About': f"## {APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)
logger.debug("Streamlit page_config set.")

# --- Load Custom CSS ---
try:
    css_file_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_file_path):
        load_css(css_file_path)
        logger.info(f"Custom CSS loaded from {css_file_path}")
    else:
        logger.warning(f"style.css not found at {css_file_path}. Attempting to load from root 'style.css'.")
        if os.path.exists("style.css"):
            load_css("style.css")
            logger.info("Custom CSS loaded from root 'style.css'.")
        else:
            logger.error("style.css not found in expected locations. Custom styles may not apply.")
            st.warning("style.css not found. Custom styles may not apply.")
except Exception as e:
    logger.error(f"Failed to load style.css: {e}", exc_info=True)
    st.warning("Could not load custom styles. The app may not appear as intended.")


# --- Initialize Session State ---
default_session_state = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None,
    'kpi_results': None, 'kpi_confidence_intervals': {},
    'risk_free_rate': RISK_FREE_RATE, 'current_theme': "dark",
    'uploaded_file_name': None, 'last_processed_file_id': None,
    'last_filtered_data_shape': None, 'sidebar_filters': None,
    'active_tab': "📈 Overview",
    'selected_benchmark_ticker': DEFAULT_BENCHMARK_TICKER,
    'selected_benchmark_display_name': next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == DEFAULT_BENCHMARK_TICKER), "None"),
    'benchmark_daily_returns': None,
    'initial_capital': 100000.0,
    'last_applied_filters': None,
    'last_fetched_benchmark_ticker': None,
    'last_benchmark_data_filter_shape': None,
    'last_kpi_calc_state_id': None
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value
logger.debug("Global session state initialized/checked.")


# --- Instantiate Services ---
data_service = DataService()
analysis_service_instance = AnalysisService()
logger.debug("DataService and AnalysisService (instance) instantiated.")


# --- Sidebar Rendering and Filter Management ---

# --- Logo Handling ---
# Corrected logo path to use .png
LOGO_PATH_SIDEBAR = "assets/Trading_Mastery_Hub_600x600.png"
module_dir = os.path.dirname(__file__)
absolute_logo_path = os.path.join(module_dir, LOGO_PATH_SIDEBAR)
logger.info(f"Attempting to load logo from absolute path: {absolute_logo_path}")

logo_to_display = None

if os.path.exists(absolute_logo_path):
    try:
        with open(absolute_logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # Corrected MIME type for PNG
        logo_to_display = f"data:image/png;base64,{encoded_string}"
        logger.info(f"Successfully encoded logo from {absolute_logo_path} to base64.")
    except Exception as e:
        logger.error(f"Error encoding logo from {absolute_logo_path} to base64: {e}", exc_info=True)
        logo_to_display = LOGO_PATH_SIDEBAR
        logger.warning(f"Base64 encoding failed. Will try st.logo with relative path: {LOGO_PATH_SIDEBAR}")
else:
    logger.error(f"Logo file NOT FOUND at absolute path: {absolute_logo_path}. Cannot display logo.")
    logo_to_display = LOGO_PATH_SIDEBAR
    st.sidebar.warning(f"Logo image not found at expected path: {LOGO_PATH_SIDEBAR}. Please check file location and case.")

if logo_to_display:
    try:
        # For st.logo, if using base64, it's directly passed.
        # If using a path, st.logo handles it.
        # The icon_image for collapsed state also needs to be correct.
        st.logo(logo_to_display, icon_image=logo_to_display if "base64" in logo_to_display else LOGO_PATH_SIDEBAR)
        logger.info(f"Sidebar logo set using st.logo. Source type: {'Base64' if 'base64' in logo_to_display else 'Path'}")
    except Exception as e:
        logger.error(f"Error setting st.logo with '{logo_to_display[:30]}...': {e}", exc_info=True)
        st.sidebar.error("Could not load logo image for sidebar via st.logo.")
else:
    logger.error("Logo could not be prepared for display (file not found or encoding failed).")


st.sidebar.header(APP_TITLE)
st.sidebar.markdown("---")


uploaded_file = st.sidebar.file_uploader(
    "Upload Trading Journal (CSV)", type=["csv"],
    help=f"Expected columns include: {', '.join(EXPECTED_COLUMNS.values())}",
    key="app_wide_file_uploader"
)

sidebar_manager = SidebarManager(st.session_state.processed_data)
current_sidebar_filters = sidebar_manager.render_sidebar_controls()

st.session_state.sidebar_filters = current_sidebar_filters

if current_sidebar_filters:
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar:
        st.session_state.risk_free_rate = rfr_from_sidebar
        logger.info(f"Global risk-free rate updated to: {st.session_state.risk_free_rate:.4f}")
        st.session_state.kpi_results = None

    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', "")
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next(
            (name for name, ticker in AVAILABLE_BENCHMARKS.items()
             if ticker == st.session_state.selected_benchmark_ticker), "None"
        )
        logger.info(f"Benchmark ticker updated to: {st.session_state.selected_benchmark_ticker}")
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None

    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar:
        st.session_state.initial_capital = initial_capital_from_sidebar
        logger.info(f"Initial capital updated to: {st.session_state.initial_capital:.2f}")
        st.session_state.kpi_results = None


# --- Data Loading and Processing --- (remains unchanged)
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
    if st.session_state.last_processed_file_id != current_file_id or st.session_state.processed_data is None:
        logger.info(f"File '{uploaded_file.name}' (ID: {current_file_id}) selected. Initiating processing.")
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            st.session_state.processed_data = data_service.get_processed_trading_data(uploaded_file)

        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.last_processed_file_id = current_file_id
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}
        st.session_state.filtered_data = st.session_state.processed_data
        st.session_state.benchmark_daily_returns = None

        if st.session_state.processed_data is not None:
            logger.info(f"DataService processed '{uploaded_file.name}'. Shape: {st.session_state.processed_data.shape}")
            display_custom_message(f"Successfully processed '{uploaded_file.name}'. Navigate pages to see analysis.", "success", icon="✅")
        else:
            logger.error(f"DataService failed to process file: {uploaded_file.name}.")
            display_custom_message(f"Failed to process '{uploaded_file.name}'. Check logs and file format.", "error")
            st.session_state.processed_data = None
            st.session_state.filtered_data = None
            st.session_state.kpi_results = None
            st.session_state.kpi_confidence_intervals = {}
            st.session_state.benchmark_daily_returns = None


# --- Data Filtering --- (remains unchanged)
if st.session_state.processed_data is not None and st.session_state.sidebar_filters:
    if st.session_state.filtered_data is None or \
       st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        logger.info("Applying global filters via DataService...")
        st.session_state.filtered_data = data_service.filter_data(
            st.session_state.processed_data,
            st.session_state.sidebar_filters,
            column_map=EXPECTED_COLUMNS
        )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        logger.debug(f"Data filtered. New shape: {st.session_state.filtered_data.shape if st.session_state.filtered_data is not None else 'None'}")
        st.session_state.kpi_results = None
        st.session_state.kpi_confidence_intervals = {}
        st.session_state.benchmark_daily_returns = None


# --- Benchmark Data Fetching --- (remains unchanged)
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "":
        refetch_benchmark = False
        if st.session_state.benchmark_daily_returns is None:
            refetch_benchmark = True
        elif st.session_state.last_fetched_benchmark_ticker != selected_ticker:
            refetch_benchmark = True
        elif st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape:
            refetch_benchmark = True

        if refetch_benchmark:
            date_col_name = EXPECTED_COLUMNS.get('date')
            if date_col_name and date_col_name in st.session_state.filtered_data.columns:
                min_date_ts = pd.to_datetime(st.session_state.filtered_data[date_col_name], errors='coerce').min()
                max_date_ts = pd.to_datetime(st.session_state.filtered_data[date_col_name], errors='coerce').max()
                s_ticker = str(selected_ticker)
                s_min_date = min_date_ts.strftime('%Y-%m-%d') if pd.notna(min_date_ts) else None
                s_max_date = max_date_ts.strftime('%Y-%m-%d') if pd.notna(max_date_ts) else None
                
                if s_min_date and s_max_date:
                    logger.info(f"Fetching benchmark data for {s_ticker} from {s_min_date} to {s_max_date}.")
                    with st.spinner(f"Fetching benchmark data for {s_ticker}..."):
                        st.session_state.benchmark_daily_returns = get_benchmark_data_static(
                            s_ticker, s_min_date, s_max_date
                        )
                    st.session_state.last_fetched_benchmark_ticker = selected_ticker
                    st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape
                    if st.session_state.benchmark_daily_returns is None:
                        display_custom_message(f"Could not fetch benchmark data for {selected_ticker}. Proceeding without benchmark comparison.", "warning")
                    else:
                        logger.info(f"Benchmark data for {selected_ticker} fetched successfully. Shape: {st.session_state.benchmark_daily_returns.shape}")
                else:
                    logger.warning("Min/max dates from filtered_data are NaT or invalid, cannot fetch benchmark data.")
                    st.session_state.benchmark_daily_returns = None
            else:
                logger.warning(f"Date column ('{date_col_name}') not found in filtered_data for benchmark date range.")
                st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None
    elif st.session_state.benchmark_daily_returns is not None :
        logger.info("No benchmark selected or deselected. Clearing benchmark data.")
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None


# --- KPI Calculation --- (remains unchanged)
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    current_kpi_calc_state_id = (
        st.session_state.filtered_data.shape,
        st.session_state.risk_free_rate,
        st.session_state.initial_capital,
        st.session_state.selected_benchmark_ticker,
        pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns, index=True).sum() if st.session_state.benchmark_daily_returns is not None and not st.session_state.benchmark_daily_returns.empty else None
    )
    if st.session_state.kpi_results is None or \
       st.session_state.last_kpi_calc_state_id != current_kpi_calc_state_id:
        logger.info("Recalculating global KPIs and CIs...")
        with st.spinner("Calculating performance metrics & CIs..."):
            kpi_service_result = analysis_service_instance.get_core_kpis(
                st.session_state.filtered_data,
                st.session_state.risk_free_rate,
                benchmark_daily_returns=st.session_state.get('benchmark_daily_returns'),
                initial_capital=st.session_state.get('initial_capital')
            )
            if kpi_service_result and 'error' not in kpi_service_result:
                st.session_state.kpi_results = kpi_service_result
                st.session_state.last_kpi_calc_state_id = current_kpi_calc_state_id
                logger.info("Global KPIs calculated.")
                ci_service_result = analysis_service_instance.get_bootstrapped_kpi_cis(
                    st.session_state.filtered_data,
                    kpis_to_bootstrap=['avg_trade_pnl', 'win_rate', 'sharpe_ratio']
                )
                if ci_service_result and 'error' not in ci_service_result:
                    st.session_state.kpi_confidence_intervals = ci_service_result
                    logger.info(f"Global KPI CIs calculated: {list(ci_service_result.keys())}")
                else:
                    error_msg_ci = ci_service_result.get('error', 'Unknown error') if ci_service_result else "CI calculation failed"
                    display_custom_message(f"Warning: Confidence Interval calculation error: {error_msg_ci}", "warning")
                    st.session_state.kpi_confidence_intervals = {}
            else:
                error_msg_kpi = kpi_service_result.get('error', 'Unknown error') if kpi_service_result else "KPI calculation failed"
                display_custom_message(f"Error calculating KPIs: {error_msg_kpi}", "error")
                st.session_state.kpi_results = None
                st.session_state.kpi_confidence_intervals = {}
elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
        display_custom_message("No data matches current filters. Adjust filters or check data.", "info")
    st.session_state.kpi_results = None
    st.session_state.kpi_confidence_intervals = {}


# --- Initial Welcome Message or No Data Message --- (remains unchanged)
if st.session_state.processed_data is None and not uploaded_file:
    st.markdown(f"### Welcome to {APP_TITLE}!")
    st.markdown("Use the sidebar to upload your trading journal (CSV file) and select analysis options to get started.")
    logger.info("Displaying welcome message as no data is loaded.")
elif st.session_state.processed_data is not None and \
     (st.session_state.filtered_data is None or st.session_state.filtered_data.empty) and \
     not (st.session_state.kpi_results and 'error' not in st.session_state.kpi_results):
     if st.session_state.sidebar_filters and uploaded_file:
        display_custom_message(
            "No data matches the current filter selection. Please adjust your filters in the sidebar or verify the uploaded data content.",
            "info"
        )

logger.info(f"Application '{APP_TITLE}' main script execution finished for this run cycle.")
