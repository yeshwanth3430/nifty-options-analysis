import pandas as pd
import time
from datetime import datetime
from breeze_connect import BreezeConnect
import configparser
import os
import threading
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# --- Setup Breeze API ---
config = configparser.ConfigParser()
config.read_dict({
    'credentials': {
        'api_key': "3437N18UK04d3jJ883fS03*A16h61654",
        'api_secret': "q^A5956!48195N5)28S50F76)vO99s41",
        'session_token': "51699952"
    }
})

api = BreezeConnect(api_key=config['credentials']['api_key'])
api.generate_session(
    api_secret=config['credentials']['api_secret'],
    session_token=config['credentials']['session_token']
)

# --- Constants ---
UNDERLYING = "NIFTY"
EXCHANGE = "NFO"
PRODUCT = "options"
STOPLOSS = 15
TARGET = 15
LOT_SIZE = 50
LOG_FILE = "trades/straddle_live_log.csv"
SPOT_LOG_FILE = "trades/nifty_spot_log.csv"

# --- Globals ---
vwap_data = []  # [(timestamp, premium)]
spot_data = []  # [(timestamp, spot_price)]
open_trades = []
trade_log = []
last_entry_time = None
latest_spot = None
latest_option_prices = {}  # key: (strike, type), value: ltp
expiry = None
websocket_connected = False  # New flag to track WebSocket connection

# Create trades directory and files
os.makedirs("trades", exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        'entry_time', 'exit_time', 'strike', 'ce_entry', 'pe_entry',
        'ce_exit', 'pe_exit', 'pnl', 'exit_reason', 'status'
    ]).to_csv(LOG_FILE, index=False)
if not os.path.exists(SPOT_LOG_FILE):
    pd.DataFrame(columns=['timestamp', 'spot_price']).to_csv(SPOT_LOG_FILE, index=False)

# --- Utility Functions ---
def get_atm_strike(spot):
    return round(spot / 50) * 50

def calculate_vwap():
    df = pd.DataFrame(vwap_data, columns=['timestamp', 'premium'])
    df['price'] = df['premium']
    df['qty'] = 1
    df['cum_price_qty'] = (df['price'] * df['qty']).cumsum()
    df['cum_qty'] = df['qty'].cumsum()
    df['vwap'] = df['cum_price_qty'] / df['cum_qty']
    return df.iloc[-1]['vwap'] if not df.empty else None

def enter_trade(atm_strike, ce_price, pe_price, premium):
    global last_entry_time
    now = datetime.now()
    trade = {
        "entry_time": now,
        "strike": atm_strike,
        "ce_entry": ce_price,
        "pe_entry": pe_price,
        "ce_sl": ce_price + STOPLOSS,
        "pe_sl": pe_price + STOPLOSS,
        "ce_target": ce_price - TARGET,
        "pe_target": pe_price - TARGET,
        "status": "OPEN",
        "entry_premium": premium
    }
    open_trades.append(trade)
    last_entry_time = now
    print(f"[{now}] Entered short straddle at {atm_strike} with premium {premium}")

def monitor_trades():
    global open_trades
    closed = []
    for trade in open_trades:
        if trade['status'] != "OPEN":
            continue
        ce_ltp = latest_option_prices.get((trade['strike'], "CALL"), None)
        pe_ltp = latest_option_prices.get((trade['strike'], "PUT"), None)
        if ce_ltp is None or pe_ltp is None:
            continue
        pnl = (trade['ce_entry'] - ce_ltp + trade['pe_entry'] - pe_ltp) * LOT_SIZE
        if ce_ltp >= trade['ce_sl'] or pe_ltp >= trade['pe_sl']:
            reason = "STOPLOSS"
        elif ce_ltp <= trade['ce_target'] and pe_ltp <= trade['pe_target']:
            reason = "TARGET"
        else:
            continue

        trade['exit_time'] = datetime.now()
        trade['ce_exit'] = ce_ltp
        trade['pe_exit'] = pe_ltp
        trade['exit_reason'] = reason
        trade['pnl'] = pnl
        trade['status'] = "CLOSED"
        trade_log.append(trade)
        closed.append(trade)
        print(f"[{trade['exit_time']}] Exited trade - Reason: {reason}, PnL: {pnl}")

    open_trades = [t for t in open_trades if t['status'] == "OPEN"]

def save_logs():
    df = pd.DataFrame(trade_log)
    df.to_csv(LOG_FILE, index=False)
    spot_df = pd.DataFrame(spot_data, columns=['timestamp', 'spot_price'])
    spot_df.to_csv(SPOT_LOG_FILE, index=False)

# --- WebSocket Handlers ---
def on_ticks(tick_data):
    global latest_spot, expiry, websocket_connected
    try:
        # Validate tick data
        if not isinstance(tick_data, dict):
            print(f"Invalid tick data format: {tick_data}")
            return
            
        # Extract required fields with validation
        instrument = tick_data.get('instrument_token')
        ltp = tick_data.get('last_traded_price')
        product_type = tick_data.get('product_type')
        
        if not all([instrument, ltp, product_type]):
            print(f"Missing required fields in tick data: {tick_data}")
            return
        
        # Handle NIFTY spot data
        if product_type == 'cash':
            latest_spot = ltp
            now = datetime.now()
            spot_data.append((now, ltp))
            print(f"[{now}] Latest NIFTY spot: {latest_spot}")
            save_logs()
            return
            
        # Handle options data
        strike = tick_data.get('strike_price')
        opt_type = tick_data.get('option_type')
        
        if not all([strike, opt_type]):
            print(f"Missing strike or option type in tick data: {tick_data}")
            return
            
        latest_option_prices[(strike, opt_type)] = ltp

        # Calculate VWAP and check for trade entry
        if latest_spot:
            atm_strike = get_atm_strike(latest_spot)
            print(f"Current ATM Strike: {atm_strike} (based on spot: {latest_spot})")

            # Subscribe to ATM CE/PE if not already tracked
            for opt_type in ["CALL", "PUT"]:
                key = (atm_strike, opt_type)
                if key not in latest_option_prices:
                    try:
                        api.subscribe_feeds(
                            stock_code=UNDERLYING,
                            exchange_code=EXCHANGE,
                            product_type=PRODUCT,
                            expiry_date=expiry,
                            strike_price=atm_strike,
                            right=opt_type
                        )
                        print(f"Subscribed to {atm_strike} {opt_type}")
                    except Exception as sub_e:
                        print(f"Subscription failed for {key}: {sub_e}")

            ce = latest_option_prices.get((atm_strike, "CALL"))
            pe = latest_option_prices.get((atm_strike, "PUT"))
            if ce and pe:
                now = datetime.now()
                premium = ce + pe
                vwap_data.append((now, premium))
                vwap = calculate_vwap()
                if vwap and premium < vwap:
                    if not last_entry_time or (now - last_entry_time).seconds > 30:
                        enter_trade(atm_strike, ce, pe, premium)
    except Exception as e:
        print(f"Tick processing error: {e}")
        print(f"Error details: {str(e)}")
        print(f"Tick data: {tick_data}")

def websocket_loop():
    global expiry, websocket_connected
    expiry = "06JUN2024"  # <-- Update manually each week
    
    while True:  # Continuous reconnection loop
        try:
            print("\n=== Attempting WebSocket Connection ===")
            
            # Step 1: Subscribe to NIFTY spot data
            print("Step 1: Subscribing to NIFTY spot data...")
            api.subscribe_feeds(
                stock_code="NIFTY",
                exchange_code="NSE",
                product_type="cash"
            )
            print("✅ Successfully subscribed to NIFTY spot data")
            
            # Step 2: Connect to WebSocket with proper headers
            print("Step 2: Connecting to WebSocket...")
            api.ws_connect()
            print("✅ Successfully connected to WebSocket")
            
            # Step 3: Set up the callback
            api.on_ticks = on_ticks
            websocket_connected = True
            print("✅ WebSocket setup complete")
            
            # Step 4: Keep the connection alive
            while websocket_connected:
                try:
                    # Send heartbeat every 30 seconds
                    api.send_heartbeat()
                    time.sleep(30)
                except Exception as hb_error:
                    print(f"❌ Heartbeat error: {str(hb_error)}")
                    websocket_connected = False
                    break
                
        except Exception as e:
            websocket_connected = False
            print(f"❌ WebSocket Error: {str(e)}")
            print("Attempting to reconnect in 5 seconds...")
            time.sleep(5)

# --- Dashboard Functions ---
def load_data():
    try:
        trade_df = pd.read_csv(LOG_FILE, parse_dates=['entry_time', 'exit_time'])
        spot_df = pd.read_csv(SPOT_LOG_FILE, parse_dates=['timestamp'])
        return trade_df, spot_df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(), pd.DataFrame()

def run_dashboard():
    st.set_page_config(layout="wide")
    st.title("📊 Live Rolling Straddle Dashboard")

    # Add connection status and debug info
    st.sidebar.title("Connection Status")
    connection_status = st.sidebar.empty()
    debug_info = st.sidebar.empty()
    
    while True:
        # Update connection status with more details
        if websocket_connected and latest_spot is not None:
            connection_status.success("✅ Connected to WebSocket")
            debug_info.info(f"Latest Spot: ₹{latest_spot:,.2f}")
        else:
            connection_status.error("❌ Waiting for WebSocket connection...")
            if not websocket_connected:
                debug_info.warning("WebSocket not connected. Check terminal for connection attempts.")
            elif latest_spot is None:
                debug_info.warning("Connected but waiting for spot data...")
            
        trade_df, spot_df = load_data()

        # Market Overview
        st.subheader("Market Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            if not spot_df.empty:
                latest_spot_price = spot_df['spot_price'].iloc[-1]
                spot_change = spot_df['spot_price'].iloc[-1] - spot_df['spot_price'].iloc[-2] if len(spot_df) > 1 else 0
                st.metric("NIFTY Spot", f"₹{latest_spot_price:,.2f}", f"{spot_change:,.2f}")
            else:
                st.metric("NIFTY Spot", "Waiting for data...")
                st.info("""
                Connection Status:
                1. Check if market is open (9:15 AM - 3:30 PM IST)
                2. Verify API credentials
                3. Check terminal for connection logs
                """)

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

        # NIFTY Spot Chart
        st.subheader("NIFTY Spot Price")
        if not spot_df.empty:
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

        # Straddle Premium Chart
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

        # Trade Statistics
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

        # Trade Logs
        st.subheader("Trade Logs")
        if not trade_df.empty:
            display_df = trade_df[['entry_time', 'exit_time', 'strike', 'ce_entry', 'pe_entry', 
                                'ce_exit', 'pe_exit', 'pnl', 'exit_reason', 'status']].copy()
            display_df['pnl'] = display_df['pnl'].map('₹{:,.2f}'.format)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No trades recorded yet.")

        # Download Section
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

        time.sleep(5)  # Update every 5 seconds

# --- Main Execution ---
if __name__ == "__main__":
    print("\n=== Starting Straddle Trader ===")
    print("1. Initializing WebSocket connection...")
    
    # Start the WebSocket connection in a separate thread
    websocket_thread = threading.Thread(target=websocket_loop)
    websocket_thread.daemon = True
    websocket_thread.start()
    
    # Wait a bit for initial connection
    time.sleep(2)
    
    print("2. Starting dashboard...")
    # Start the dashboard
    run_dashboard() 