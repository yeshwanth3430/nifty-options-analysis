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
import os
import requests

# Set page config
st.set_page_config(page_title="NIFTY Options Analysis", layout="wide")

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Connect to the database
DB_FILE = 'nifty_data.duckdb'
DROPBOX_URL = 'https://www.dropbox.com/scl/fi/fdxfzz19mltip6b5yt420/nifty_data.duckdb?rlkey=1o4jpip00fq7789x99sw1gfs0&st=vknhv0qn&dl=1'

if not os.path.exists(DB_FILE):
    with st.spinner('Downloading database from Dropbox...'):
        r = requests.get(DROPBOX_URL)
        with open(DB_FILE, 'wb') as f:
            f.write(r.content)
        st.success('Database downloaded!')

db = duckdb.connect(DB_FILE)

# --- PIN Authentication ---
PIN = "4142"  # Set your desired PIN here

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Enter PIN to Access Dashboard")
    pin_input = st.text_input("Enter PIN", type="password")
    if st.button("Submit"):
        if pin_input == PIN:
            st.session_state["authenticated"] = True
            st.experimental_rerun()
        else:
            st.error("Incorrect PIN. Please try again.")
    st.stop()  # Stop the app here if not authenticated

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