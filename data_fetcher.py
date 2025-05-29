from breeze_connect import BreezeConnect
import urllib.parse
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
import math
import os
import duckdb
import concurrent.futures
from typing import List, Dict, Any

# Initialize SDK
api_key = "3437N18UK04d3jJ883fS03*A16h61654"
api_secret = "q^A5956!48195N5)28S50F76)vO99s41"
session_token = "51659117"

breeze = BreezeConnect(api_key=api_key)

# Print login URL for session token retrieval
print("https://api.icicidirect.com/apiuser/login?api_key=" + urllib.parse.quote_plus(api_key))

# Generate Session
breeze.generate_session(
    api_secret=api_secret,
    session_token=session_token
)

print("Connected to ICICI Direct API successfully.")

# Initialize DuckDB connection
db = duckdb.connect('nifty_data.duckdb')

# Create tables if they don't exist
db.execute("""
    CREATE TABLE IF NOT EXISTS spot_data (
        date DATE,
        datetime TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        PRIMARY KEY (date, datetime)
    )
""")

db.execute("""
    CREATE TABLE IF NOT EXISTS options_data (
        date DATE,
        datetime TIMESTAMP,
        strike_price INTEGER,
        option_type VARCHAR,
        expiry_date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        open_interest BIGINT,
        PRIMARY KEY (date, datetime, strike_price, option_type)
    )
""")

# Function to get nearest expiry date
def get_nearest_expiry(target_date_str, expiry_dates):
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

    # Convert all expiry dates to datetime objects
    expiry_dates_dt = [datetime.strptime(date, '%d-%m-%Y') for date in expiry_dates]

    # Find the nearest expiry date that's AFTER the target date
    nearest_expiry = None
    min_diff = float('inf')

    for date in expiry_dates_dt:
        # Calculate difference in days
        diff = (date - target_date).days
        # Only consider expiry dates that are after the target date
        if diff >= 0 and diff < min_diff:
            min_diff = diff
            nearest_expiry = date

    if not nearest_expiry:
        print(f"Could not find a valid expiry date after {target_date_str}")
        return None

    # Check if the expiry date is a weekend (Saturday = 5, Sunday = 6)
    if nearest_expiry.weekday() >= 5:
        # Find the previous trading day
        while nearest_expiry.weekday() >= 5:
            nearest_expiry = nearest_expiry - timedelta(days=1)
        print(f"Expiry adjusted from weekend to previous trading day: {nearest_expiry.strftime('%d-%m-%Y')}")

    print(f"Using nearest available expiry: {nearest_expiry.strftime('%d-%m-%Y')} (difference: {min_diff} days)")
    # Format as required by the API (ISO format with Z)
    expiry_iso = nearest_expiry.strftime('%Y-%m-%dT07:00:00.000Z')

    return expiry_iso

# Function to parse datetime
def parse_datetime(dt_str):
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1]
        dt = datetime.fromisoformat(dt_str.replace('.000', ''))
        utc = pytz.UTC
        return utc.localize(dt)
    return datetime.fromisoformat(dt_str)

# Function to fetch NIFTY Spot data for a day
def fetch_spot_daily_data(date_str):
    """Fetch full day spot data for NIFTY"""
    print(f"Fetching full day spot data for {date_str}")

    # Set the start and end times for the day
    start_time = "09:15:00"
    end_time = "15:30:00"

    # Create full datetime strings
    start_datetime = f"{date_str}T{start_time}.000Z"
    end_datetime = f"{date_str}T{end_time}.000Z"

    try:
        # Get 1-minute interval data for higher resolution
        data = breeze.get_historical_data_v2(
            interval="1minute",
            from_date=start_datetime,
            to_date=end_datetime,
            stock_code="NIFTY",
            exchange_code="NSE",
            product_type="cash"
        )

        if isinstance(data, dict) and 'Success' in data and isinstance(data['Success'], list):
            df = pd.DataFrame(data['Success'])
            print(f"  Retrieved {len(df)} spot data points")

            # Calculate daily OHLC
            if not df.empty:
                daily_high = df['high'].astype(float).max()
                daily_low = df['low'].astype(float).min()
                daily_open = float(df.iloc[0]['open'])
                daily_close = float(df.iloc[-1]['close'])

                print(f"  Daily OHLC: Open={daily_open}, High={daily_high}, Low={daily_low}, Close={daily_close}")
                return {
                    'high': daily_high,
                    'low': daily_low,
                    'open': daily_open,
                    'close': daily_close,
                    'raw_data': df
                }
            else:
                print("  Empty dataframe returned")
                return None
        else:
            print(f"  Error retrieving spot data:", data)
            return None

    except Exception as e:
        print(f"  An error occurred while fetching spot data: {e}")
        return None

# Function to get all strikes between high+600 and low-600
def get_strikes_from_range(high, low, strike_interval=50):
    """Get all strikes between high+600 and low-600 at given interval"""
    upper_bound = high + 600
    lower_bound = low - 600

    # Round to nearest strike interval
    upper_strike = math.ceil(upper_bound / strike_interval) * strike_interval
    lower_strike = math.floor(lower_bound / strike_interval) * strike_interval

    strikes = list(range(lower_strike, upper_strike + strike_interval, strike_interval))
    return strikes

def fetch_options_data_for_strike(args: Dict[str, Any]) -> pd.DataFrame:
    """Fetch options data for a single strike and option type"""
    date_str = args['date_str']
    strike_price = args['strike_price']
    option_type = args['option_type']
    expiry_date = args['expiry_date']
    
    print(f"Worker fetching {option_type} options data for strike {strike_price} on {date_str} with expiry {expiry_date}")

    # Set the start and end times for the day
    start_time = "09:15:00"
    end_time = "15:30:00"

    # Create full datetime strings
    start_datetime = f"{date_str}T{start_time}.000Z"
    end_datetime = f"{date_str}T{end_time}.000Z"

    try:
        data = breeze.get_historical_data_v2(
            interval="1minute",
            from_date=start_datetime,
            to_date=end_datetime,
            stock_code="NIFTY",
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_date,
            right=option_type.lower(),
            strike_price=str(strike_price)
        )

        if isinstance(data, dict):
            if 'Success' in data and isinstance(data['Success'], list):
                df = pd.DataFrame(data['Success'])
                if not df.empty:
                    df['strike_price'] = strike_price
                    df['option_type'] = option_type.upper()
                    print(f"  Retrieved {len(df)} data points for {option_type} strike {strike_price}")
                    return df
                else:
                    print(f"  No data points returned for {option_type} strike {strike_price}")
                    return None
            elif 'Error' in data:
                print(f"  API Error for {option_type} strike {strike_price}: {data['Error']}")
                return None
            else:
                print(f"  Unexpected API response for {option_type} strike {strike_price}:", data)
                return None
        else:
            print(f"  Invalid API response type for {option_type} strike {strike_price}:", type(data))
            return None

    except Exception as e:
        print(f"  An error occurred for {option_type} strike {strike_price}: {str(e)}")
        return None

def process_date(date_str, expiry_dates):
    """Process a single date to fetch spot and options data"""
    print(f"\n{'='*80}")
    print(f"PROCESSING DATE: {date_str}")
    print(f"{'='*80}")

    # Get nearest expiry date
    expiry_date = get_nearest_expiry(date_str, expiry_dates)
    if not expiry_date:
        print(f"Could not find a valid expiry date for {date_str}")
        return

    print(f"Using expiry date: {expiry_date}")

    # Fetch spot data for the day
    spot_data = fetch_spot_daily_data(date_str)
    if not spot_data:
        print(f"Could not fetch spot data for {date_str}")
        return

    # Get strikes based on high and low
    strikes = get_strikes_from_range(spot_data['high'], spot_data['low'])
    print(f"Will fetch options data for {len(strikes)} strikes: {min(strikes)} to {max(strikes)}")

    # Save spot data to DuckDB
    spot_df = spot_data['raw_data']
    spot_df['date'] = date_str
    
    # Select and rename columns to match our schema
    spot_df = spot_df[['date', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert column types to match schema
    spot_df['date'] = pd.to_datetime(spot_df['date']).dt.date
    spot_df['datetime'] = pd.to_datetime(spot_df['datetime'])
    spot_df['open'] = spot_df['open'].astype(float)
    spot_df['high'] = spot_df['high'].astype(float)
    spot_df['low'] = spot_df['low'].astype(float)
    spot_df['close'] = spot_df['close'].astype(float)
    spot_df['volume'] = spot_df['volume'].astype(int)

    db.execute("""
        INSERT OR REPLACE INTO spot_data 
        SELECT date, datetime, open, high, low, close, volume 
        FROM spot_df
    """)
    print(f"Saved {len(spot_df)} spot data points to DuckDB")

    # Prepare arguments for parallel processing
    fetch_args = []
    for strike in strikes:
        fetch_args.append({
            'date_str': date_str,
            'strike_price': strike,
            'option_type': 'call',
            'expiry_date': expiry_date
        })
        fetch_args.append({
            'date_str': date_str,
            'strike_price': strike,
            'option_type': 'put',
            'expiry_date': expiry_date
        })

    # Use ThreadPoolExecutor to fetch options data in parallel
    all_options_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_args = {executor.submit(fetch_options_data_for_strike, args): args for args in fetch_args}
        for future in concurrent.futures.as_completed(future_to_args):
            args = future_to_args[future]
            try:
                df = future.result()
                if df is not None:
                    all_options_data.append(df)
            except Exception as e:
                print(f"Error processing {args['option_type']} strike {args['strike_price']}: {e}")

    # Combine all options data
    if all_options_data:
        combined_options_df = pd.concat(all_options_data, ignore_index=True)
        combined_options_df.drop_duplicates(subset=['datetime', 'strike_price', 'option_type'], keep='first', inplace=True)
        
        # Add date column and select required columns
        combined_options_df['date'] = date_str
        combined_options_df = combined_options_df[['date', 'datetime', 'strike_price', 'option_type', 'expiry_date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
        
        # Convert column types to match schema
        combined_options_df['date'] = pd.to_datetime(combined_options_df['date']).dt.date
        combined_options_df['datetime'] = pd.to_datetime(combined_options_df['datetime'])
        combined_options_df['expiry_date'] = pd.to_datetime(combined_options_df['expiry_date']).dt.date
        combined_options_df['strike_price'] = combined_options_df['strike_price'].astype(int)
        combined_options_df['option_type'] = combined_options_df['option_type'].astype(str)
        combined_options_df['open'] = combined_options_df['open'].astype(float)
        combined_options_df['high'] = combined_options_df['high'].astype(float)
        combined_options_df['low'] = combined_options_df['low'].astype(float)
        combined_options_df['close'] = combined_options_df['close'].astype(float)
        combined_options_df['volume'] = combined_options_df['volume'].astype(int)
        combined_options_df['open_interest'] = combined_options_df['open_interest'].astype(int)
        
        # Save options data to DuckDB
        db.execute("""
            INSERT OR REPLACE INTO options_data 
            SELECT date, datetime, strike_price, option_type, expiry_date, open, high, low, close, volume, open_interest 
            FROM combined_options_df
        """)
        print(f"Saved {len(combined_options_df)} options data points to DuckDB")

    print(f"Processing completed for {date_str}")
    return True

# Main execution
if __name__ == "__main__":
    # Expiry dates provided by the user
    expiry_dates = [
        "05-01-2023", "12-01-2023", "19-01-2023", "25-01-2023","02-02-2023", "09-02-2023",
        "16-02-2023", "23-02-2023", "02-03-2023", "09-03-2023", "16-03-2023",
        "23-03-2023","29-03-2023", "06-04-2023", "13-04-2023", "20-04-2023", "27-04-2023",
        "04-05-2023", "11-05-2023", "18-05-2023", "25-05-2023", "01-06-2023",
        "08-06-2023", "15-06-2023", "22-06-2023", "29-06-2023", "06-07-2023",
        "13-07-2023", "20-07-2023", "27-07-2023", "03-08-2023", "10-08-2023",
        "17-08-2023", "24-08-2023", "31-08-2023", "07-09-2023", "14-09-2023",
        "21-09-2023", "28-09-2023", "05-10-2023", "12-10-2023", "19-10-2023",
        "26-10-2023", "02-11-2023", "09-11-2023", "16-11-2023", "23-11-2023",
        "30-11-2023", "07-12-2023", "14-12-2023", "21-12-2023", "28-12-2023",
        "04-01-2024", "11-01-2024", "18-01-2024", "25-01-2024", "01-02-2024",
        "08-02-2024", "15-02-2024", "22-02-2024", "29-02-2024", "07-03-2024",
        "14-03-2024", "21-03-2024", "28-03-2024", "04-04-2024","10-04-2024", "18-04-2024",
        "25-04-2024", "02-05-2024", "09-05-2024", "16-05-2024", "23-05-2024",
        "30-05-2024", "06-06-2024", "13-06-2024", "20-06-2024", "27-06-2024",
        "04-07-2024", "11-07-2024", "18-07-2024", "25-07-2024", "01-08-2024",
        "08-08-2024", "14-08-2024", "22-08-2024", "29-08-2024", "05-09-2024", "12-09-2024",
        "19-09-2024", "26-09-2024", "03-10-2024", "10-10-2024", "17-10-2024",
        "24-10-2024", "31-10-2024", "07-11-2024", "14-11-2024", "21-11-2024",
        "28-11-2024", "05-12-2024", "12-12-2024", "19-12-2024", "26-12-2024",
        "02-01-2025", "09-01-2025", "16-01-2025", "30-01-2025", "06-02-2025",
        "13-02-2025", "20-02-2025", "27-02-2025", "06-03-2025", "13-03-2025",
        "20-03-2025", "27-03-2025", "03-04-2025", "09-04-2025", "17-04-2025",
        "24-04-2025", "30-04-2025"
    ]

    # Get date range from user
    print("\nEnter the date range for data collection:")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # Generate list of dates between start_date and end_date
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates_to_process = []
    
    current = start
    while current <= end:
        # Process every day, including weekends and holidays
        dates_to_process.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"\nWill process {len(dates_to_process)} trading days from {start_date} to {end_date}")
    print("Dates to process:", dates_to_process)

    # Process each date
    for date_str in dates_to_process:
        success = process_date(date_str, expiry_dates)
        if success:
            print(f"Successfully processed {date_str}")
        else:
            print(f"Failed to process {date_str}")
        time.sleep(10)  # Wait between processing dates

    # Close DuckDB connection
    db.close()
    print("\nAll data collection completed and saved to DuckDB!")

# Example queries you can run after data collection:
db = duckdb.connect('nifty_data.duckdb')

# Get all spot data for a specific date
db.execute("SELECT * FROM spot_data WHERE date = '2023-01-10'").fetchdf()

# Get all call options for a specific strike
db.execute("SELECT * FROM options_data WHERE strike_price = 18000 AND option_type = 'CALL'").fetchdf()

# Get daily high and low for spot
db.execute("SELECT date, MAX(high) as daily_high, MIN(low) as daily_low FROM spot_data GROUP BY date").fetchdf()