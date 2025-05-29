# NIFTY Options Analysis Dashboard

An interactive dashboard for analyzing NIFTY spot and options data using Streamlit.

## Features

- Interactive date selection for analysis
- Rolling Straddle analysis
- Straddle and Volatility analysis
- Backtesting capabilities

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Data

The app uses a DuckDB database (`nifty_data.duckdb`) for storing and querying the NIFTY options data.

## Requirements

- Python 3.7+
- Streamlit
- DuckDB
- Pandas
- Plotly
- NumPy 