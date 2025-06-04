import duckdb

con = duckdb.connect('live_market_data.db')

# Check if the table exists
try:
    # Print column names
    columns = con.execute("PRAGMA table_info('live_quotes')").fetchdf()
    print("Columns in live_quotes:")
    print(columns[['name', 'type']])
    # Print row count
    count = con.execute("SELECT COUNT(*) FROM live_quotes").fetchone()[0]
    print(f"\nRow count in live_quotes: {count}")
    if count > 0:
        print("\nFirst 20 rows:")
        print(con.execute("SELECT * FROM live_quotes ORDER BY timestamp DESC LIMIT 20").fetchdf())
    else:
        print("The table exists but is empty.")
except Exception as e:
    print(f"Error: {e}\nTable 'live_quotes' may not exist in live_market_data.db.") 