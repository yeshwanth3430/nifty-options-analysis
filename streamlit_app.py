import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
# from rolling_straddle import calculate_rolling_straddle, get_straddle_signals, calculate_straddle_returns  # removed
from streddle_vol_view import show_streddle_vol_view
# from rolling_straddle_single import get_rolling_straddle_ohlc, get_above_below_volumes  # removed
from rolling_straddle_view import show_rolling_straddle_view
# from straddle_view import show_straddle_view  # removed
import numpy as np

# Set page config
st.set_page_config(page_title="NIFTY Options Analysis", layout="wide")

# Connect to the database
db = duckdb.connect('nifty_data.duckdb')

# Title and description
st.title("📈 NIFTY Options Analysis Dashboard")
st.markdown("Interactive dashboard for analyzing NIFTY spot and options data")

# Get available dates
date_info = db.execute("SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(DISTINCT date) as num_dates FROM spot_data").fetchdf()
min_date = date_info['min_date'].iloc[0]
max_date = date_info['max_date'].iloc[0]
num_dates = date_info['num_dates'].iloc[0]

st.info(f"Available data from **{min_date}** to **{max_date}** (Total trading days: {num_dates})")

# Date picker on main page
selected_date = st.date_input(
    "Select Date for Analysis",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key="main_date_picker"
)

# --- Creative Sidebar Navigation ---
with st.sidebar:
    st.markdown("# 🚀 Navigation")
    if st.button('🎨 Theme', key='theme_btn'):
        st.info('Theme feature coming soon! Stay tuned for more colors and styles!')
    st.button('➕ Straddle and Vol', key='straddle_vol_btn')
    st.markdown('---')
    nav_options = [
        #'📊 Straddle',  # removed
        '🧪 Backtest',
        '🔄 Rolling Straddle',
        '➕ Straddle and Vol'
    ]
    nav_choice = st.radio(
        'Go to:',
        nav_options,
        index=[
            #'Straddle',  # removed
            'Backtest',
            'Rolling Straddle',
            'Straddle and Vol'
        ].index(st.session_state.get('nav_selection', 'Rolling Straddle')) if st.session_state.get('nav_selection', 'Rolling Straddle') in ['Backtest', 'Rolling Straddle', 'Straddle and Vol'] else 0,
        key='nav_radio'
    )
    # Map emoji nav to session state
    nav_map = {
        #'📊 Straddle': 'Straddle',  # removed
        '🧪 Backtest': 'Backtest',
        '🔄 Rolling Straddle': 'Rolling Straddle',
        '➕ Straddle and Vol': 'Straddle and Vol'
    }
    st.session_state['nav_selection'] = nav_map[nav_choice]

# --- Main Content based on Navigation ---
#if st.session_state['nav_selection'] == 'Straddle':
#    show_straddle_view(db, selected_date)
if st.session_state['nav_selection'] == 'Straddle and Vol':
    show_streddle_vol_view(db)
elif st.session_state['nav_selection'] == 'Rolling Straddle':
    show_rolling_straddle_view(db, selected_date)
else:
    st.markdown("<div style='text-align:center; margin-top: 100px;'><h2>Will be available soon</h2></div>", unsafe_allow_html=True)

# Close the database connection
db.close() 