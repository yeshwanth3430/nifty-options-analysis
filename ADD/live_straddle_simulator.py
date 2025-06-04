# live_straddle_simulator.py (WebSocket version)
import pandas as pd
import time
from datetime import datetime
from breeze_connect import BreezeConnect
import configparser
import os
import threading
import pytz
import duckdb

# --- Setup Breeze API ---
config = configparser.ConfigParser()
config.read_dict({
    'credentials': {
        'api_key': "3437N18UK04d3jJ883fS03*A16h61654",
        'api_secret': "q^A5956!48195N5)28S50F76)vO99s41",
        'session_token': "51699952"
    }
})

print("Loaded API credentials:")
print(f"  api_key: {config['credentials']['api_key'][:4]}***{config['credentials']['api_key'][-4:]}")
print(f"  api_secret: {'*' * len(config['credentials']['api_secret'])}")
print(f"  session_token: {config['credentials']['session_token']}")

api = BreezeConnect(api_key=config['credentials']['api_key'])
try:
    api.generate_session(
        api_secret=config['credentials']['api_secret'],
        session_token=config['credentials']['session_token']
    )
    print("[DEBUG] Session generated successfully.")
except Exception as e:
    print(f"[ERROR] Failed to generate session: {e}")

# --- Constants ---
UNDERLYING = "NIFTY"
EXCHANGE = "NFO"
PRODUCT = "options"
STOPLOSS = 15
TARGET = 15
LOT_SIZE = 50
LOG_FILE = "trades/straddle_live_log.csv"
SPOT_LOG_FILE = "trades/nifty_spot_log.csv"
EXPIRY = "06JUN2024"  # Update this weekly

# --- Database Setup ---
db = duckdb.connect('straddle_data.db')
db.execute("""
    CREATE TABLE IF NOT EXISTS option_prices (
        timestamp TIMESTAMP,
        strike_price DOUBLE,
        option_type VARCHAR,
        last_traded_price DOUBLE,
        volume BIGINT,
        bid_price DOUBLE,
        ask_price DOUBLE,
        bid_quantity BIGINT,
        ask_quantity BIGINT
    )
""")

db.execute("""
    CREATE TABLE IF NOT EXISTS spot_prices (
        timestamp TIMESTAMP,
        last_traded_price DOUBLE,
        volume BIGINT,
        bid_price DOUBLE,
        ask_price DOUBLE,
        bid_quantity BIGINT,
        ask_quantity BIGINT
    )
""")

# --- Globals ---
vwap_data = []  # [(timestamp, premium)]
spot_data = []  # [(timestamp, spot_price)]
open_trades = []
trade_log = []
last_entry_time = None
latest_spot = None
latest_option_prices = {}  # key: (strike, type), value: ltp

os.makedirs("trades", exist_ok=True)

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
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
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

        trade['exit_time'] = datetime.now(pytz.timezone('Asia/Kolkata'))
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

def save_spot_logs():
    df = pd.DataFrame(spot_data, columns=['timestamp', 'spot_price'])
    df.to_csv(SPOT_LOG_FILE, index=False)

def store_option_data(tick_data):
    """Store option data in DuckDB"""
    try:
        db.execute("""
            INSERT INTO option_prices (
                timestamp, strike_price, option_type,
                last_traded_price, volume, bid_price,
                ask_price, bid_quantity, ask_quantity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(pytz.timezone('Asia/Kolkata')),
            tick_data.get('strike_price', 0.0),
            tick_data.get('option_type', ''),
            tick_data.get('last_traded_price', 0.0),
            tick_data.get('volume', 0),
            tick_data.get('bid_price', 0.0),
            tick_data.get('ask_price', 0.0),
            tick_data.get('bid_quantity', 0),
            tick_data.get('ask_quantity', 0)
        ))
    except Exception as e:
        print(f"Error storing option data: {e}")

def store_spot_data(tick_data):
    """Store spot data in DuckDB"""
    try:
        db.execute("""
            INSERT INTO spot_prices (
                timestamp, last_traded_price, volume,
                bid_price, ask_price, bid_quantity, ask_quantity
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(pytz.timezone('Asia/Kolkata')),
            tick_data.get('last_traded_price', 0.0),
            tick_data.get('volume', 0),
            tick_data.get('bid_price', 0.0),
            tick_data.get('ask_price', 0.0),
            tick_data.get('bid_quantity', 0),
            tick_data.get('ask_quantity', 0)
        ))
    except Exception as e:
        print(f"Error storing spot data: {e}")

# --- WebSocket Handlers ---
def on_ticks(tick_data):
    global latest_spot, latest_option_prices
    try:
        # Handle NIFTY spot data
        if tick_data.get('product_type') == 'cash':
            latest_spot = tick_data.get('last_traded_price')
            store_spot_data(tick_data)
            now = datetime.now(pytz.timezone('Asia/Kolkata'))
            spot_data.append((now, latest_spot))
            print(f"[{now}] Latest NIFTY spot: {latest_spot}")
            save_spot_logs()
            return

        # Handle options data
        strike = tick_data.get('strike_price')
        opt_type = tick_data.get('option_type')
        if not all([strike, opt_type]):
            return

        latest_option_prices[(strike, opt_type)] = tick_data.get('last_traded_price')
        store_option_data(tick_data)

        # Calculate VWAP and check for trade entry
        if latest_spot:
            atm_strike = get_atm_strike(latest_spot)
            ce = latest_option_prices.get((atm_strike, "CALL"))
            pe = latest_option_prices.get((atm_strike, "PUT"))
            
            if ce and pe:
                now = datetime.now(pytz.timezone('Asia/Kolkata'))
                premium = ce + pe
                vwap_data.append((now, premium))
                vwap = calculate_vwap()
                
                if vwap and premium < vwap:
                    if not last_entry_time or (now - last_entry_time).seconds > 30:
                        enter_trade(atm_strike, ce, pe, premium)

    except Exception as e:
        print(f"[ERROR] Exception in on_ticks: {e}")
        print(f"[ERROR] Tick data: {tick_data}")

def websocket_loop():
    print("[DEBUG] Starting websocket_loop()")
    try:
        print("[DEBUG] Connecting WebSocket...")
        api.ws_connect()
        api.on_ticks = on_ticks
        print("[DEBUG] WebSocket connected.")

        # Subscribe to NIFTY spot data
        print("[DEBUG] Subscribing to NIFTY spot data...")
        api.subscribe_feeds(
            stock_code="NIFTY",
            exchange_code="NSE",
            product_type="cash",
            get_exchange_quotes=True,
            get_market_depth=False
        )
        print("[DEBUG] Subscribed to NIFTY spot data.")

        # Keep the WebSocket connection alive
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        time.sleep(5)
        websocket_loop()  # Retry connection

# --- Main Loop ---
if __name__ == "__main__":
    # Start WebSocket connection in a separate thread
    websocket_thread = threading.Thread(target=websocket_loop)
    websocket_thread.daemon = True
    websocket_thread.start()

    # Main monitoring loop
    while True:
        try:
            monitor_trades()
            save_logs()
            save_spot_logs()
            time.sleep(1)
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)
