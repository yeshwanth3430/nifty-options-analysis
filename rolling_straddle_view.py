import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
# All rolling straddle logic is now in this file

def get_rolling_straddle_ohlc(db, selected_date):
    query = f'''
        WITH atm_data AS (
            SELECT 
                s.datetime,
                s.close AS spot_close,
                ROUND(s.close / 50.0) * 50 AS atm_strike
            FROM spot_data s
            WHERE s.date = '{selected_date}'
            ORDER BY s.datetime
        )
        SELECT 
            a.datetime,
            ce.open + pe.open AS open,
            ce.high + pe.high AS high,
            ce.low + pe.low AS low,
            ce.close + pe.close AS close,
            ce.volume AS ce_volume,
            pe.volume AS pe_volume
        FROM atm_data a
        LEFT JOIN options_data ce ON ce.date = '{selected_date}' AND ce.datetime = a.datetime AND ce.strike_price = a.atm_strike AND ce.option_type = 'CALL'
        LEFT JOIN options_data pe ON pe.date = '{selected_date}' AND pe.datetime = a.datetime AND pe.strike_price = a.atm_strike AND pe.option_type = 'PUT'
        WHERE ce.open IS NOT NULL AND pe.open IS NOT NULL
        ORDER BY a.datetime
    '''
    df = db.execute(query).fetchdf()
    if not df.empty:
        df['straddle_volume'] = df['ce_volume'] + df['pe_volume']
        df['straddle_volume'] = df['straddle_volume'].replace(0, np.nan)  # Avoid zero volume
        df['vwap'] = (df['close'] * df['straddle_volume']).cumsum() / (df['straddle_volume'].cumsum())
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df

def get_above_below_volumes(db, selected_date):
    times_df = db.execute(f"""
        SELECT DISTINCT strftime('%H:%M', datetime) as time FROM spot_data WHERE date = '{selected_date}' ORDER BY time
    """).fetchdf()
    times = times_df['time'].tolist()
    records = []
    for time in times:
        spot_row = db.execute(f"""
            SELECT close FROM spot_data WHERE date = '{selected_date}' AND strftime('%H:%M', datetime) = '{time}' LIMIT 1
        """).fetchone()
        if not spot_row:
            continue
        spot = spot_row[0]
        atm_strike = int(round(spot / 50.0) * 50)
        strikes_df = db.execute(f"""
            SELECT DISTINCT strike_price FROM options_data WHERE date = '{selected_date}' AND strftime('%H:%M', datetime) = '{time}' ORDER BY strike_price
        """).fetchdf()
        strikes = sorted(strikes_df['strike_price'].tolist())
        if not strikes:
            continue
        if atm_strike not in strikes:
            continue
        atm_idx = strikes.index(atm_strike)
        ce_above_strikes = strikes[atm_idx+1:atm_idx+11]
        pe_below_strikes = strikes[max(0, atm_idx-10):atm_idx]
        ce_above_vol = 0
        if ce_above_strikes:
            ce_above_vol = db.execute(f"""
                SELECT SUM(volume) FROM options_data WHERE date = '{selected_date}' AND strftime('%H:%M', datetime) = '{time}' AND strike_price IN ({','.join(map(str, ce_above_strikes))}) AND option_type = 'CALL'
            """).fetchone()[0] or 0
        pe_below_vol = 0
        if pe_below_strikes:
            pe_below_vol = db.execute(f"""
                SELECT SUM(volume) FROM options_data WHERE date = '{selected_date}' AND strftime('%H:%M', datetime) = '{time}' AND strike_price IN ({','.join(map(str, pe_below_strikes))}) AND option_type = 'PUT'
            """).fetchone()[0] or 0
        dt_row = db.execute(f"""
            SELECT datetime FROM spot_data WHERE date = '{selected_date}' AND strftime('%H:%M', datetime) = '{time}' LIMIT 1
        """).fetchone()
        records.append({
            'datetime': dt_row[0] if dt_row else None,
            'ce_above_vol': ce_above_vol,
            'pe_below_vol': pe_below_vol
        })
    return pd.DataFrame(records)

def calculate_rolling_straddle(db, start_date, end_date, lookback_days=5):
    spot_data = db.execute(f"""
        SELECT date, datetime, close
        FROM spot_data
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date, datetime
    """).fetchdf()
    straddle_results = []
    for date in pd.date_range(start=start_date, end=end_date):
        date_str = date.strftime('%Y-%m-%d')
        daily_data = spot_data[spot_data['date'] == date_str]
        if daily_data.empty:
            continue
        for _, row in daily_data.iterrows():
            spot_price = row['close']
            atm_strike = int(round(spot_price / 50.0) * 50)
            ce_data = db.execute(f"""
                SELECT close FROM options_data
                WHERE date = '{date_str}' 
                AND datetime = '{row['datetime']}'
                AND strike_price = {atm_strike}
                AND option_type = 'CALL'
                LIMIT 1
            """).fetchone()
            pe_data = db.execute(f"""
                SELECT close FROM options_data
                WHERE date = '{date_str}'
                AND datetime = '{row['datetime']}'
                AND strike_price = {atm_strike}
                AND option_type = 'PUT'
                LIMIT 1
            """).fetchone()
            if ce_data and pe_data:
                straddle_premium = ce_data[0] + pe_data[0]
                straddle_results.append({
                    'date': date_str,
                    'datetime': row['datetime'],
                    'spot_price': spot_price,
                    'atm_strike': atm_strike,
                    'ce_premium': ce_data[0],
                    'pe_premium': pe_data[0],
                    'straddle_premium': straddle_premium
                })
    straddle_df = pd.DataFrame(straddle_results)
    if not straddle_df.empty:
        straddle_df['rolling_mean'] = straddle_df['straddle_premium'].rolling(window=lookback_days).mean()
        straddle_df['rolling_std'] = straddle_df['straddle_premium'].rolling(window=lookback_days).std()
        straddle_df['z_score'] = (straddle_df['straddle_premium'] - straddle_df['rolling_mean']) / straddle_df['rolling_std']
    return straddle_df

def get_straddle_signals(straddle_df, z_score_threshold=2.0):
    signals = []
    for _, row in straddle_df.iterrows():
        if pd.isna(row['z_score']):
            continue
        if row['z_score'] > z_score_threshold:
            signals.append({
                'date': row['date'],
                'datetime': row['datetime'],
                'signal': 'SELL',
                'z_score': row['z_score'],
                'straddle_premium': row['straddle_premium']
            })
        elif row['z_score'] < -z_score_threshold:
            signals.append({
                'date': row['date'],
                'datetime': row['datetime'],
                'signal': 'BUY',
                'z_score': row['z_score'],
                'straddle_premium': row['straddle_premium']
            })
    return pd.DataFrame(signals)

def calculate_straddle_returns(signals_df, db):
    returns = []
    for _, signal in signals_df.iterrows():
        next_date = (datetime.strptime(signal['date'], '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        spot_change = db.execute(f"""
            SELECT 
                (MAX(close) - MIN(close)) / MIN(close) as price_change
            FROM spot_data
            WHERE date = '{next_date}'
        """).fetchone()
        if spot_change:
            returns.append({
                'date': signal['date'],
                'signal': signal['signal'],
                'z_score': signal['z_score'],
                'straddle_premium': signal['straddle_premium'],
                'next_day_return': spot_change[0]
            })
    return pd.DataFrame(returns)

# --- UI code below remains unchanged ---
def show_rolling_straddle_view(db, selected_date):
    # --- Straddle View (moved from Straddle tab) ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Spot Price Analysis")
        spot_data = db.execute(f"""
            SELECT datetime, open, high, low, close, volume
            FROM spot_data
            WHERE date = '{selected_date}'
            ORDER BY datetime
        """).fetchdf()
        if not spot_data.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=spot_data['datetime'],
                open=spot_data['open'],
                high=spot_data['high'],
                low=spot_data['low'],
                close=spot_data['close']
            )])
            fig.update_layout(
                title=f"NIFTY Spot Price - {selected_date}",
                yaxis_title="Price",
                xaxis_title="Time",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No spot data available for the selected date.")
    with col2:
        st.subheader("ATM Straddle Premium (CE+PE)")
        spot_5min = db.execute(f"""
            SELECT datetime, close FROM spot_data
            WHERE date = '{selected_date}'
            ORDER BY datetime
        """).fetchdf()
        straddle_premiums = []
        if not spot_5min.empty:
            for idx, row in spot_5min.iterrows():
                dt = row['datetime']
                spot = row['close']
                atm_strike = int(round(spot / 50.0) * 50)
                ce = db.execute(f"""
                    SELECT open, high, low, close FROM options_data
                    WHERE date = '{selected_date}' AND datetime = '{dt}' AND strike_price = {atm_strike} AND option_type = 'CALL'
                    LIMIT 1
                """).fetchone()
                pe = db.execute(f"""
                    SELECT open, high, low, close FROM options_data
                    WHERE date = '{selected_date}' AND datetime = '{dt}' AND strike_price = {atm_strike} AND option_type = 'PUT'
                    LIMIT 1
                """).fetchone()
                if ce and pe:
                    straddle_premiums.append({
                        'datetime': dt,
                        'atm_strike': atm_strike,
                        'ce': ce[3],
                        'pe': pe[3],
                        'straddle': ce[3] + pe[3]
                    })
            if straddle_premiums:
                straddle_df = pd.DataFrame(straddle_premiums)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=straddle_df['datetime'],
                    y=straddle_df['straddle'],
                    mode='lines+markers',
                    name='ATM Straddle Premium',
                    line=dict(color='purple')
                ))
                fig.update_layout(
                    title="ATM Straddle Premium (CE+PE) Over Time",
                    xaxis_title="Time",
                    yaxis_title="Premium",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No ATM straddle data available for this date.")
        else:
            st.warning("No spot data available for the selected date.")
    st.header('🔄 Rolling Straddle (Candlestick)')
    straddle_ohlc = get_rolling_straddle_ohlc(db, selected_date)
    if not straddle_ohlc.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=straddle_ohlc['datetime'],
            y=straddle_ohlc['close'],
            mode='lines',
            name='Rolling Straddle (CE+PE)',
            line=dict(color='mediumseagreen', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=straddle_ohlc['datetime'],
            y=straddle_ohlc['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='orange', width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=straddle_ohlc['datetime'],
            y=straddle_ohlc['ema9'],
            mode='lines',
            name='EMA 9',
            line=dict(color='green', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=straddle_ohlc['datetime'],
            y=straddle_ohlc['ema12'],
            mode='lines',
            name='EMA 12',
            line=dict(color='purple', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=straddle_ohlc['datetime'],
            y=straddle_ohlc['ema20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='blue', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f'Rolling Straddle (ATM CE+PE) - {selected_date}',
            yaxis_title='Straddle Premium',
            xaxis_title='Time',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('No rolling straddle data available for the selected date.')

    st.subheader('Rolling Straddle DTE-Matched Stats (ATM, Whole Day)')
    expiry_row = db.execute(f"""
        SELECT MIN(expiry_date) FROM options_data WHERE date = '{selected_date}'
    """).fetchone()
    if expiry_row and expiry_row[0]:
        expiry_date = pd.to_datetime(expiry_row[0])
        dte = (expiry_date - pd.to_datetime(selected_date)).days
        st.markdown(f"**DTE (Days to Expiry) for {selected_date}:** {dte} days (Expiry: {expiry_date.date()})")
        all_dates = db.execute(f"""
            SELECT DISTINCT date FROM options_data WHERE date < '{selected_date}' ORDER BY date DESC
        """).fetchdf()['date'].tolist()
        matching_dates = []
        for dt in all_dates:
            exp_row = db.execute(f"""
                SELECT MIN(expiry_date) FROM options_data WHERE date = '{dt}'
            """).fetchone()
            if exp_row and exp_row[0]:
                exp_dt = pd.to_datetime(exp_row[0])
                dt_dte = (exp_dt - pd.to_datetime(dt)).days
                if dt_dte == dte:
                    matching_dates.append(dt)
            if len(matching_dates) >= 6:
                break
        history_rows = []
        for dt in matching_dates:
            straddle_df = get_rolling_straddle_ohlc(db, dt)
            if not straddle_df.empty:
                open_val = straddle_df['open'].iloc[0]
                close_val = straddle_df['close'].iloc[-1]
                max_val = straddle_df['high'].max()
                min_val = straddle_df['low'].min()
                history_rows.append({
                    'Date': dt,
                    'Open': open_val,
                    'Close': close_val,
                    'Max': max_val,
                    'Min': min_val
                })
        straddle_today = get_rolling_straddle_ohlc(db, selected_date)
        if not straddle_today.empty:
            today_open = straddle_today['open'].iloc[0]
            today_close = straddle_today['close'].iloc[-1]
            today_max = straddle_today['high'].max()
            today_min = straddle_today['low'].min()
        else:
            today_open = today_close = today_max = today_min = None
        if history_rows:
            hist_df = pd.DataFrame(history_rows)
            mean_row = {
                'Date': 'Mean',
                'Open': np.mean(hist_df['Open']),
                'Close': np.mean(hist_df['Close']),
                'Max': np.mean(hist_df['Max']),
                'Min': np.mean(hist_df['Min'])
            }
            median_row = {
                'Date': 'Median',
                'Open': np.median(hist_df['Open']),
                'Close': np.median(hist_df['Close']),
                'Max': np.median(hist_df['Max']),
                'Min': np.median(hist_df['Min'])
            }
            hist_df = pd.concat([hist_df, pd.DataFrame([mean_row, median_row])], ignore_index=True)
            st.markdown(f"**Last 6 Dates (Same DTE) Rolling Straddle Stats (ATM, whole day) for Each Day:**")
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info('No historical data found for same DTE in last 6 dates.')
    else:
        st.info('Could not determine expiry date for selected day.')

    st.subheader('Volume: 10 Strikes Above (CE) & Below (PE) ATM')
    ab_vol_df = get_above_below_volumes(db, selected_date)
    spot_df = db.execute(f"""
        SELECT datetime, close FROM spot_data WHERE date = '{selected_date}' ORDER BY datetime
    """).fetchdf()
    if not ab_vol_df.empty and not spot_df.empty:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=ab_vol_df['datetime'],
            y=ab_vol_df['ce_above_vol'],
            mode='lines',
            name='CE Volume (10 Above ATM)',
            line=dict(color='red', width=2),
            yaxis='y1'
        ))
        fig_vol.add_trace(go.Scatter(
            x=ab_vol_df['datetime'],
            y=ab_vol_df['pe_below_vol'],
            mode='lines',
            name='PE Volume (10 Below ATM)',
            line=dict(color='blue', width=2),
            yaxis='y1'
        ))
        fig_vol.add_trace(go.Scatter(
            x=spot_df['datetime'],
            y=spot_df['close'],
            mode='lines',
            name='Spot Price',
            line=dict(color='black', width=2, dash='solid'),
            yaxis='y2'
        ))
        fig_vol.update_layout(
            title=f'Volume Profile (10 Strikes Above/Below ATM) & Spot Price - {selected_date}',
            yaxis=dict(
                title='Volume',
                side='left',
                showgrid=False
            ),
            yaxis2=dict(
                title='Spot Price',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            xaxis_title='Time',
            height=400
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    elif not ab_vol_df.empty:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=ab_vol_df['datetime'],
            y=ab_vol_df['ce_above_vol'],
            mode='lines',
            name='CE Volume (10 Above ATM)',
            line=dict(color='red', width=2)
        ))
        fig_vol.add_trace(go.Scatter(
            x=ab_vol_df['datetime'],
            y=ab_vol_df['pe_below_vol'],
            mode='lines',
            name='PE Volume (10 Below ATM)',
            line=dict(color='blue', width=2)
        ))
        fig_vol.update_layout(
            title=f'Volume Profile (10 Strikes Above/Below ATM) - {selected_date}',
            yaxis_title='Volume',
            xaxis_title='Time',
            height=400
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning('No volume data available for the selected date.') 