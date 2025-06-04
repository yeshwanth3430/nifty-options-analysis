from breeze_connect import BreezeConnect
import urllib.parse
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
import os
import duckdb
from typing import List, Dict, Any
import streamlit as st
import threading
import queue
from streamlit_autorefresh import st_autorefresh

class DataManager:
    def __init__(self):
        # API credentials (these should be stored securely in production)
        self.api_key = "3437N18UK04d3jJ883fS03*A16h61654"
        self.api_secret = "q^A5956!48195N5)28S50F76)vO99s41"
        self.breeze = None
        self.session_token = None
        self.data_queue = queue.Queue()
        self.is_subscribed = False
        self.live_data_thread = None
        self.websocket_connected = False
        
        # Initialize DuckDB connection
        self.db = duckdb.connect('live_market_data.db')
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables in DuckDB"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS live_quotes (
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                ltp DOUBLE
            )
        """)

    def get_login_url(self) -> str:
        """Generate the login URL for session token retrieval"""
        return f"https://api.icicidirect.com/apiuser/login?api_key={urllib.parse.quote_plus(self.api_key)}"

    def connect(self, session_token: str) -> bool:
        """Connect to ICICI Direct API with the provided session token"""
        try:
            self.session_token = session_token
            self.breeze = BreezeConnect(api_key=self.api_key)
            self.breeze.generate_session(
                api_secret=self.api_secret,
                session_token=self.session_token
            )
            return True
        except Exception as e:
            print(f"Error connecting to API: {str(e)}")
            return False

    def _on_ticks(self, tick_data):
        """Callback function for WebSocket ticks"""
        try:
            print(f"Received tick data: {tick_data}")  # Debug log
            # Add timestamp to the data
            tick_data['timestamp'] = datetime.now(pytz.timezone('Asia/Kolkata'))
            
            # Only keep required fields
            processed_data = {
                'timestamp': tick_data['timestamp'],
                'open': float(tick_data.get('open', 0.0)),
                'high': float(tick_data.get('high', 0.0)),
                'low': float(tick_data.get('low', 0.0)),
                'close': float(tick_data.get('close', 0.0)),
                'ltp': float(tick_data.get('ltp', tick_data.get('last_traded_price', 0.0)))
            }
            
            print(f"Processed data: {processed_data}")  # Debug log
            self.data_queue.put(processed_data)
        except Exception as e:
            print(f"Error in tick callback: {str(e)}")
            print(f"Raw tick data: {tick_data}")  # Debug log

    def _process_live_data(self):
        """Process and store live data in DuckDB"""
        while self.is_subscribed:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get()
                    print(f"Processing data from queue: {data}")  # Debug log
                    
                    # Insert data into DuckDB
                    self.db.execute("""
                        INSERT INTO live_quotes (
                            timestamp, open, high, low, close, ltp
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        data['timestamp'],
                        data['open'],
                        data['high'],
                        data['low'],
                        data['close'],
                        data['ltp']
                    ))
                    print("Data successfully inserted into DuckDB")  # Debug log
            except Exception as e:
                print(f"Error processing live data: {str(e)}")
                print(f"Data that caused error: {data}")  # Debug log
            time.sleep(0.1)

    def connect_websocket(self) -> bool:
        """Connect to WebSocket"""
        try:
            if not self.websocket_connected:
                self.breeze.ws_connect()
                self.breeze.on_ticks = self._on_ticks
                self.websocket_connected = True
                return True
            return False
        except Exception as e:
            print(f"Error connecting to WebSocket: {str(e)}")
            return False

    def subscribe_live_data(self, exchange_code: str, stock_code: str, product_type: str = "cash", 
                          expiry_date: str = None, strike_price: str = None, right: str = None):
        """Subscribe to live market data"""
        try:
            if not self.is_subscribed:
                if not self.websocket_connected:
                    if not self.connect_websocket():
                        return False

                self.breeze.subscribe_feeds(
                    exchange_code=exchange_code,
                    stock_code=stock_code,
                    product_type=product_type,
                    expiry_date=expiry_date,
                    strike_price=strike_price,
                    right=right,
                    get_market_depth=False,
                    get_exchange_quotes=True
                )
                self.is_subscribed = True
                
                # Start processing thread
                self.live_data_thread = threading.Thread(target=self._process_live_data)
                self.live_data_thread.daemon = True
                self.live_data_thread.start()
                
                return True
            return False
        except Exception as e:
            print(f"Error subscribing to live data: {str(e)}")
            return False

    def unsubscribe_live_data(self):
        """Unsubscribe from live market data"""
        try:
            if self.is_subscribed:
                self.is_subscribed = False
                if self.live_data_thread:
                    self.live_data_thread.join(timeout=1.0)
                self.breeze.unsubscribe_feeds()
                return True
            return False
        except Exception as e:
            print(f"Error unsubscribing from live data: {str(e)}")
            return False

    def get_live_data(self, limit: int = 100) -> pd.DataFrame:
        """Get recent live data from DuckDB. If limit is None or 0, return all rows."""
        try:
            if limit is None or limit == 0:
                return self.db.execute("""
                    SELECT * FROM live_quotes 
                    ORDER BY timestamp DESC
                """).df()
            else:
                return self.db.execute("""
                    SELECT * FROM live_quotes 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, [limit]).df()
        except Exception as e:
            print(f"Error getting live data: {str(e)}")
            return pd.DataFrame()

    def fetch_historical_data(self, symbol: str, from_date: str, to_date: str, interval: str = "1day") -> pd.DataFrame:
        """Fetch historical data for a given symbol"""
        try:
            data = self.breeze.get_historical_data(
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                symbol_code=symbol,
                exchange_code="NSE"
            )
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to parquet file"""
        try:
            df.to_parquet(filename, index=False)
            print(f"Data saved successfully to {filename}")
        except Exception as e:
            print(f"Error saving data: {str(e)}")

def main():
    # Initialize Streamlit interface
    st.title("ICICI Direct Data Manager")
    
    # Create DataManager instance
    data_manager = DataManager()
    
    # Display login URL
    st.write("Login URL for session token:")
    st.code(data_manager.get_login_url())
    
    # Get session token from user
    session_token = st.text_input("Enter your session token:")
    
    if session_token:
        if data_manager.connect(session_token):
            st.success("Connected to ICICI Direct API successfully!")
            
            # Create tabs for different functionalities
            tab1, tab2 = st.tabs(["Historical Data", "Live Data"])
            
            with tab1:
                # Historical data section
                symbol = st.text_input("Enter stock symbol (e.g., RELIANCE):")
                from_date = st.date_input("From Date:")
                to_date = st.date_input("To Date:")
                
                if st.button("Fetch Historical Data"):
                    if symbol and from_date and to_date:
                        df = data_manager.fetch_historical_data(
                            symbol=symbol,
                            from_date=from_date.strftime("%Y-%m-%d"),
                            to_date=to_date.strftime("%Y-%m-%d")
                        )
                        
                        if not df.empty:
                            st.dataframe(df)
                            
                            if st.button("Save Historical Data"):
                                filename = f"{symbol}_{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}.parquet"
                                data_manager.save_to_parquet(df, filename)
                                st.success(f"Data saved to {filename}")
                        else:
                            st.error("No data found for the given parameters")
            
            with tab2:
                # Live data section
                st.subheader("Live Data Subscription")
                
                # Input fields for live data subscription
                exchange_code = st.selectbox("Exchange Code", ["NSE", "NFO"])
                stock_code = st.text_input("Stock Code (e.g., NIFTY, RELIANCE):")
                product_type = st.selectbox("Product Type", ["cash", "options", "futures"])
                
                # Additional fields for options
                expiry_date = None
                strike_price = None
                right = None
                
                if product_type == "options":
                    expiry_date = st.text_input("Expiry Date (e.g., 31-Mar-2024):")
                    strike_price = st.text_input("Strike Price:")
                    right = st.selectbox("Option Type", ["Call", "Put"])
                
                # Add auto-refresh every 5 seconds (5000 ms)
                st_autorefresh(interval=5000, key="live_data_refresh")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Start Live Data"):
                        if stock_code:
                            if data_manager.subscribe_live_data(
                                exchange_code=exchange_code,
                                stock_code=stock_code,
                                product_type=product_type,
                                expiry_date=expiry_date,
                                strike_price=strike_price,
                                right=right
                            ):
                                st.success(f"Subscribed to live data for {stock_code}")
                            else:
                                st.error("Failed to subscribe to live data")
                
                with col2:
                    if st.button("Stop Live Data"):
                        if data_manager.unsubscribe_live_data():
                            st.success("Unsubscribed from live data")
                        else:
                            st.error("Failed to unsubscribe from live data")
                
                # Always show live data table (auto-refreshed)
                live_data = data_manager.get_live_data(limit=None)
                if not live_data.empty:
                    st.dataframe(live_data)
                else:
                    st.info("No live data available")
        else:
            st.error("Failed to connect to API. Please check your session token.")

if __name__ == "__main__":
    main() 