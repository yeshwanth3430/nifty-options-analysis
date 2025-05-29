import streamlit as st
import pandas as pd
from datetime import datetime

def get_expiry_date(db, date):
    expiry = db.execute(f"""
        SELECT MIN(expiry_date) FROM options_data WHERE expiry_date >= '{date}'
    """).fetchone()
    return expiry[0] if expiry else None

def get_nearest_strike(strikes, atm_strike):
    if not strikes:
        return None
    return min(strikes, key=lambda x: abs(x - atm_strike))

def get_streddle_vol_table(db, start_date, end_date, debug_limit=10):
    times_df = db.execute(f"""
        SELECT DISTINCT strftime('%H:%M', datetime) as time
        FROM spot_data
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY time
    """).fetchdf()
    times = times_df['time'].tolist()
    dates_df = db.execute(f"""
        SELECT DISTINCT date FROM spot_data WHERE date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date
    """).fetchdf()
    dates = dates_df['date'].tolist()
    records = []
    debug_count = 0
    for date in dates:
        expiry = get_expiry_date(db, date)
        if not expiry:
            continue
        dte = (pd.to_datetime(expiry) - pd.to_datetime(date)).days
        if dte == 6:
            print(f"[DTE DEBUG] Trading Date: {date}, Expiry: {expiry}, DTE: {dte}")
        row = {'Date': date, 'DTE': dte, 'Expiry': expiry}
        for time in times:
            spot_row = db.execute(f"""
                SELECT close FROM spot_data WHERE date = '{date}' AND strftime('%H:%M', datetime) = '{time}' LIMIT 1
            """).fetchone()
            if not spot_row:
                if debug_count < debug_limit:
                    print(f"[DEBUG] No spot for {date} {time}")
                    debug_count += 1
                row[time] = None
                continue
            spot = spot_row[0]
            atm_strike = int(round(spot / 50.0) * 50)
            strikes_df = db.execute(f"""
                SELECT DISTINCT strike_price FROM options_data WHERE date = '{date}' AND strftime('%H:%M', datetime) = '{time}'
            """).fetchdf()
            strikes = strikes_df['strike_price'].tolist()
            nearest_strike = get_nearest_strike(strikes, atm_strike)
            if nearest_strike is None:
                if debug_count < debug_limit:
                    print(f"[DEBUG] No strikes for {date} {time} (ATM: {atm_strike})")
                    debug_count += 1
                row[time] = None
                continue
            ce = db.execute(f"""
                SELECT close FROM options_data WHERE date = '{date}' AND strftime('%H:%M', datetime) = '{time}' AND strike_price = {nearest_strike} AND option_type = 'CALL' LIMIT 1
            """).fetchone()
            pe = db.execute(f"""
                SELECT close FROM options_data WHERE date = '{date}' AND strftime('%H:%M', datetime) = '{time}' AND strike_price = {nearest_strike} AND option_type = 'PUT' LIMIT 1
            """).fetchone()
            if not ce or not pe:
                if debug_count < debug_limit:
                    print(f"[DEBUG] No CE or PE for {date} {time} Nearest Strike {nearest_strike} (CE: {ce}, PE: {pe})")
                    debug_count += 1
                row[time] = None
            else:
                row[time] = ce[0] + pe[0]
        records.append(row)
    df = pd.DataFrame(records)
    return df

def show_streddle_vol_view(db):
    st.header('➕ Streddle and Vol')
    date_info = db.execute("SELECT MIN(date) as min_date, MAX(date) as max_date FROM spot_data").fetchdf()
    min_date = date_info['min_date'].iloc[0]
    max_date = date_info['max_date'].iloc[0]
    st.info(f"Available data from **{min_date}** to **{max_date}**")
    col_sv1, col_sv2 = st.columns(2)
    with col_sv1:
        sv_start = st.date_input('Start Date', value=min_date, min_value=min_date, max_value=max_date, key='sv_start')
    with col_sv2:
        sv_end = st.date_input('End Date', value=max_date, min_value=sv_start, max_value=max_date, key='sv_end')
    if st.button('Show Streddle and Vol Table'):
        with st.spinner('Calculating...'):
            table = get_streddle_vol_table(db, sv_start.strftime('%Y-%m-%d'), sv_end.strftime('%Y-%m-%d'))
        if not table.empty:
            st.dataframe(table, use_container_width=True)
        else:
            st.warning('No data available for the selected range.') 