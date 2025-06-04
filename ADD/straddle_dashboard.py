import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

# Constants
TRADE_LOG_FILE = "trades/straddle_live_log.csv"
SPOT_LOG_FILE = "trades/nifty_spot_log.csv"

# Create trades directory if it doesn't exist
os.makedirs("trades", exist_ok=True)

# Create empty files with headers if they don't exist
if not os.path.exists(TRADE_LOG_FILE):
    pd.DataFrame(columns=[
        'entry_time', 'exit_time', 'strike', 'ce_entry', 'pe_entry',
        'ce_exit', 'pe_exit', 'pnl', 'exit_reason', 'status'
    ]).to_csv(TRADE_LOG_FILE, index=False)

if not os.path.exists(SPOT_LOG_FILE):
    pd.DataFrame(columns=['timestamp', 'spot_price']).to_csv(SPOT_LOG_FILE, index=False)

st.title("📊 Live Rolling Straddle Dashboard")

# --- Load Data ---
@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_data():
    try:
        trade_df = pd.read_csv(TRADE_LOG_FILE, parse_dates=['entry_time', 'exit_time'])
        spot_df = pd.read_csv(SPOT_LOG_FILE, parse_dates=['timestamp'])
        return trade_df, spot_df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(), pd.DataFrame()

trade_df, spot_df = load_data()

# --- Market Overview ---
st.subheader("Market Overview")
col1, col2, col3 = st.columns(3)

with col1:
    if not spot_df.empty:
        latest_spot = spot_df['spot_price'].iloc[-1]
        spot_change = spot_df['spot_price'].iloc[-1] - spot_df['spot_price'].iloc[-2] if len(spot_df) > 1 else 0
        st.metric("NIFTY Spot", f"₹{latest_spot:,.2f}", f"{spot_change:,.2f}")
    else:
        st.metric("NIFTY Spot", "Waiting for data...")

with col2:
    if not trade_df.empty:
        total_trades = len(trade_df)
        st.metric("Total Trades", total_trades)
    else:
        st.metric("Total Trades", 0)

with col3:
    if not trade_df.empty:
        closed_trades = trade_df[trade_df['status'] == 'CLOSED']
        total_pnl = closed_trades['pnl'].sum() if not closed_trades.empty else 0
        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
    else:
        st.metric("Total P&L", "₹0.00")

# --- NIFTY Spot Chart ---
st.subheader("NIFTY Spot Price")
if not spot_df.empty:
    # Get last 1 hour of data
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_spot = spot_df[spot_df['timestamp'] > one_hour_ago]
    
    if not recent_spot.empty:
        fig_spot = go.Figure()
        fig_spot.add_trace(go.Scatter(
            x=recent_spot['timestamp'],
            y=recent_spot['spot_price'],
            mode='lines',
            name='NIFTY Spot',
            line=dict(color='green')
        ))
        fig_spot.update_layout(
            height=300,
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_spot, use_container_width=True)
    else:
        st.info("Waiting for spot price data...")
else:
    st.info("Waiting for spot price data...")

# --- Straddle Premium Chart ---
st.subheader("ATM Straddle Premium vs VWAP")
if not trade_df.empty:
    trade_df['entry_premium'] = trade_df['entry_premium'].astype(float)
    trade_df.sort_values(by='entry_time', inplace=True)
    trade_df['vwap'] = trade_df['entry_premium'].expanding().mean()

    fig_premium = go.Figure()
    fig_premium.add_trace(go.Scatter(
        x=trade_df['entry_time'],
        y=trade_df['entry_premium'],
        mode='lines+markers',
        name='Straddle Premium',
        line=dict(color='blue')
    ))
    fig_premium.add_trace(go.Scatter(
        x=trade_df['entry_time'],
        y=trade_df['vwap'],
        mode='lines',
        name='VWAP',
        line=dict(color='orange', dash='dot')
    ))
    fig_premium.update_layout(
        height=300,
        xaxis_title="Time",
        yaxis_title="Premium",
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_premium, use_container_width=True)
else:
    st.info("Waiting for trade data...")

# --- Trade Statistics ---
st.subheader("Trade Statistics")
col1, col2 = st.columns(2)

with col1:
    if not trade_df.empty:
        closed_trades = trade_df[trade_df['status'] == 'CLOSED']
        open_trades = trade_df[trade_df['status'] == 'OPEN']
        
        st.metric("Open Trades", len(open_trades))
        st.metric("Closed Trades", len(closed_trades))
        
        if not closed_trades.empty:
            win_rate = (closed_trades['pnl'] > 0).mean() * 100
            avg_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if len(closed_trades[closed_trades['pnl'] > 0]) > 0 else 0
            avg_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].mean() if len(closed_trades[closed_trades['pnl'] < 0]) > 0 else 0
            
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Avg Profit", f"₹{avg_profit:,.2f}")
            st.metric("Avg Loss", f"₹{avg_loss:,.2f}")
    else:
        st.metric("Open Trades", 0)
        st.metric("Closed Trades", 0)
        st.metric("Win Rate", "0.0%")
        st.metric("Avg Profit", "₹0.00")
        st.metric("Avg Loss", "₹0.00")

with col2:
    if not trade_df.empty:
        closed_trades = trade_df[trade_df['status'] == 'CLOSED']
        total_pnl = closed_trades['pnl'].sum() if not closed_trades.empty else 0
        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
        if not closed_trades.empty:
            max_profit = closed_trades['pnl'].max()
            max_loss = closed_trades['pnl'].min()
            st.metric("Max Profit", f"₹{max_profit:,.2f}")
            st.metric("Max Loss", f"₹{max_loss:,.2f}")
    else:
        st.metric("Total P&L", "₹0.00")
        st.metric("Max Profit", "₹0.00")
        st.metric("Max Loss", "₹0.00")

# --- Trade Logs ---
st.subheader("Trade Logs")
if not trade_df.empty:
    # Format the dataframe for display
    display_df = trade_df[['entry_time', 'exit_time', 'strike', 'ce_entry', 'pe_entry', 
                          'ce_exit', 'pe_exit', 'pnl', 'exit_reason', 'status']].copy()
    display_df['pnl'] = display_df['pnl'].map('₹{:,.2f}'.format)
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No trades recorded yet.")

# --- Download Section ---
st.subheader("Download Data")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "📥 Download Trade Log",
        data=trade_df.to_csv(index=False) if not trade_df.empty else "",
        file_name="straddle_live_log.csv",
        mime="text/csv",
        disabled=trade_df.empty
    )
with col2:
    st.download_button(
        "📥 Download Spot Data",
        data=spot_df.to_csv(index=False) if not spot_df.empty else "",
        file_name="nifty_spot_log.csv",
        mime="text/csv",
        disabled=spot_df.empty
    ) 