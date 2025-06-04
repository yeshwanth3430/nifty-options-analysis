import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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
            a.atm_strike,
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
        df['straddle_volume'] = df['straddle_volume'].replace(0, pd.NA)  # Avoid zero volume
        df['vwap'] = (df['close'] * df['straddle_volume']).cumsum() / (df['straddle_volume'].cumsum())
        # Filter for 09:16:00 to 15:30:00
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'].dt.strftime('%H:%M').between('09:16', '15:30')]
    return df

def get_dte_for_date(db, date_str):
    expiry_row = db.execute(f"SELECT MIN(expiry_date) FROM options_data WHERE date = '{date_str}'").fetchone()
    if expiry_row and expiry_row[0]:
        expiry_date = pd.to_datetime(expiry_row[0])
        dte = (expiry_date - pd.to_datetime(date_str)).days
        return dte
    return None

def count_crossings_rolling_vwap(df, debug=False):
    # Use rolling VWAP for crossings
    if df.empty or 'close' not in df or 'vwap' not in df:
        return 0, 0
    above = 0
    below = 0
    prev_side = None
    for idx, row in df.iterrows():
        if pd.isna(row['close']) or pd.isna(row['vwap']):
            continue
        side = 'above' if row['close'] >= row['vwap'] else 'below'
        if idx == 0:
            if side == 'above':
                above += 1
            else:
                below += 1
            prev_side = side
            continue
        if side != prev_side:
            if side == 'above':
                above += 1
            else:
                below += 1
            prev_side = side
    return above, below

def count_crossings_in_window(df, start_time, end_time):
    # Count crossings in a specific time window (inclusive)
    window_df = df[df['datetime'].dt.strftime('%H:%M').between(start_time, end_time)]
    return count_crossings_rolling_vwap(window_df)

def vwap_sl_strategy(df):
    # Short at 09:16 if straddle < vwap, SL is rolling vwap, enter short on cross from above to below vwap, exit on cross above vwap (next open), or at 15:30 close
    if df.empty or 'close' not in df or 'vwap' not in df or 'open' not in df:
        return 0, 0.0, []
    df = df.reset_index(drop=True)
    in_trade = False
    entry_price = 0.0
    entry_time = None
    entry_vwap = None
    trades = []
    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            # 09:16 candle
            if row['close'] < row['vwap']:
                in_trade = True
                entry_price = row['open']
                entry_time = row['datetime']
                entry_vwap = row['vwap']
            continue
        prev = df.iloc[i-1]
        # Entry: cross from above to below vwap (only if not in trade)
        if not in_trade and prev['close'] >= prev['vwap'] and row['close'] < row['vwap']:
            if i+1 < len(df):
                in_trade = True
                entry_price = df.iloc[i+1]['open']
                entry_time = df.iloc[i+1]['datetime']
                entry_vwap = df.iloc[i+1]['vwap']
            continue
        # Exit: cross above vwap (SL hit)
        if in_trade and prev['close'] < prev['vwap'] and row['close'] >= row['vwap']:
            if i+1 < len(df):
                exit_price = df.iloc[i+1]['open']
                exit_time = df.iloc[i+1]['datetime']
            else:
                exit_price = row['close']
                exit_time = row['datetime']
            trades.append({'entry_time': entry_time, 'entry_price': entry_price, 'entry_vwap': entry_vwap, 'exit_time': exit_time, 'exit_price': exit_price, 'pnl': entry_price - exit_price, 'sl_type': 'VWAP'})
            in_trade = False
            entry_price = 0.0
            entry_time = None
            entry_vwap = None
            continue
    # If still in trade at end, exit at last close
    if in_trade:
        exit_price = df.iloc[-1]['close']
        exit_time = df.iloc[-1]['datetime']
        trades.append({'entry_time': entry_time, 'entry_price': entry_price, 'entry_vwap': entry_vwap, 'exit_time': exit_time, 'exit_price': exit_price, 'pnl': entry_price - exit_price, 'sl_type': 'VWAP'})
    num_trades = len(trades)
    total_pnl = sum(t['pnl'] for t in trades)
    return num_trades, total_pnl, trades

def vwap_sl_rr_stats_from_tradelog(trade_log_df, all_minute_data, rr_list=[1,2,3,4]):
    # For each R:R, simulate exits for each trade in the trade log using the minute data
    results = []
    if trade_log_df.empty or not all_minute_data:
        return results
    all_minute_df = pd.concat(all_minute_data, ignore_index=True)
    for rr in rr_list:
        rr_pnls = []
        for idx, trade in trade_log_df.iterrows():
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            entry_price = trade['entry_price']
            entry_vwap = trade['entry_vwap']
            # Get minute data for this trade's day and after entry_time
            day_str = trade['Date']
            minute_df = all_minute_df[(all_minute_df['datetime'] >= entry_time) & (all_minute_df['datetime'] <= exit_time) & (all_minute_df['datetime'].dt.strftime('%Y-%m-%d') == day_str)]
            if minute_df.empty:
                rr_pnls.append(entry_price - trade['exit_price'])
                continue
            risk = abs(entry_price - entry_vwap)
            target = entry_price - rr * risk
            in_trade = True
            for i, row in minute_df.iterrows():
                # TP hit
                if row['low'] <= target:
                    rr_pnls.append(entry_price - target)
                    in_trade = False
                    break
                # SL hit (VWAP cross up)
                if row['close'] >= row['vwap']:
                    rr_pnls.append(entry_price - row['open'])
                    in_trade = False
                    break
            if in_trade:
                # If neither hit, exit at original exit price
                rr_pnls.append(entry_price - trade['exit_price'])
        n_trades = len(rr_pnls)
        total_pnl = sum(rr_pnls)
        max_dd = 0
        cum_pnl = 0
        peak = 0
        for pnl in rr_pnls:
            cum_pnl += pnl
            if cum_pnl > peak:
                peak = cum_pnl
            dd = peak - cum_pnl
            if dd > max_dd:
                max_dd = dd
        wins = [p for p in rr_pnls if p > 0]
        losses = [p for p in rr_pnls if p <= 0]
        n_win = len(wins)
        n_loss = len(losses)
        avg_gain = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = (n_win * avg_gain + n_loss * avg_loss) / n_trades if n_trades > 0 else 0
        sharpe = (np.mean(rr_pnls) / np.std(rr_pnls)) * np.sqrt(n_trades) if n_trades > 1 and np.std(rr_pnls) > 0 else 0
        results.append({
            'R:R': f'1:{rr}',
            'Number of Trades': n_trades,
            'Total P&L': total_pnl,
            'Max D:D': max_dd,
            'No. of Wins': n_win,
            'No. of Losses': n_loss,
            'Avg Gain': avg_gain,
            'Avg Loss': avg_loss,
            'Expectancy': expectancy,
            'Sharpe Ratio': sharpe
        })
    return results

def vwap_sl_rr_tradelog_from_tradelog(trade_log_df, all_minute_data, rr=1):
    # For a given R:R, generate a trade log DataFrame with entry/exit times, prices, and P&L
    if trade_log_df.empty or not all_minute_data:
        return pd.DataFrame()
    all_minute_df = pd.concat(all_minute_data, ignore_index=True)
    rr_trades = []
    for idx, trade in trade_log_df.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        entry_price = trade['entry_price']
        entry_vwap = trade['entry_vwap']
        day_str = trade['Date']
        minute_df = all_minute_df[(all_minute_df['datetime'] >= entry_time) & (all_minute_df['datetime'] <= exit_time) & (all_minute_df['datetime'].dt.strftime('%Y-%m-%d') == day_str)]
        if minute_df.empty:
            rr_trades.append({**trade, 'rr_exit_time': trade['exit_time'], 'rr_exit_price': trade['exit_price'], 'rr_pnl': entry_price - trade['exit_price'], 'rr_exit_type': 'Original'})
            continue
        risk = abs(entry_price - entry_vwap)
        target = entry_price - rr * risk
        in_trade = True
        for i, row in minute_df.iterrows():
            # TP hit
            if row['low'] <= target:
                rr_trades.append({**trade, 'rr_exit_time': row['datetime'], 'rr_exit_price': target, 'rr_pnl': entry_price - target, 'rr_exit_type': f'Target {rr}:1'})
                in_trade = False
                break
            # SL hit (VWAP cross up)
            if row['close'] >= row['vwap']:
                rr_trades.append({**trade, 'rr_exit_time': row['datetime'], 'rr_exit_price': row['open'], 'rr_pnl': entry_price - row['open'], 'rr_exit_type': 'VWAP SL'})
                in_trade = False
                break
        if in_trade:
            rr_trades.append({**trade, 'rr_exit_time': trade['exit_time'], 'rr_exit_price': trade['exit_price'], 'rr_pnl': entry_price - trade['exit_price'], 'rr_exit_type': 'Original'})
    rr_trades_df = pd.DataFrame(rr_trades)
    return rr_trades_df

def pls_sl_strategy(df, sl_points=15):
    # Short at 09:16 if straddle < vwap, SL is entry_price + sl_points, enter short on cross from above to below vwap, exit on SL or at 15:30 close
    if df.empty or 'close' not in df or 'vwap' not in df or 'open' not in df:
        return 0, 0.0, []
    df = df.reset_index(drop=True)
    in_trade = False
    entry_price = 0.0
    entry_time = None
    entry_vwap = None
    sl_price = None
    entry_strike = None
    straddle_shifts = 0
    strikes_list = []
    trades = []
    for i in range(len(df)):
        row = df.iloc[i]
        # Use atm_strike from DataFrame if available, else fallback to rounded spot
        atm_strike = row['atm_strike'] if 'atm_strike' in row else int(round(row['close'] / 50.0) * 50)
        if i == 0:
            # 09:16 candle
            if row['close'] < row['vwap']:
                in_trade = True
                entry_price = row['open']
                entry_time = row['datetime']
                entry_vwap = row['vwap']
                sl_price = entry_price + sl_points
                entry_strike = atm_strike
                straddle_shifts = 1
                strikes_list = [atm_strike]
            continue
        prev = df.iloc[i-1]
        prev_atm_strike = prev['atm_strike'] if 'atm_strike' in prev else int(round(prev['close'] / 50.0) * 50)
        # Entry: cross from above to below vwap (always close previous trade and open new one)
        if prev['close'] >= prev['vwap'] and row['close'] < row['vwap']:
            if in_trade:
                # Close previous trade at current open
                exit_price = row['open']
                exit_time = row['datetime']
                trades.append({'entry_time': entry_time, 'entry_price': entry_price, 'entry_vwap': entry_vwap, 'exit_time': exit_time, 'exit_price': exit_price, 'pnl': entry_price - exit_price, 'sl_type': f'{sl_points}pts-reopen', 'straddle_shifts': straddle_shifts, 'straddle_strikes': ','.join(map(str, strikes_list)), 'sl_price': sl_price})
            # Open new trade
            in_trade = True
            entry_price = row['open']
            entry_time = row['datetime']
            entry_vwap = row['vwap']
            sl_price = entry_price + sl_points
            entry_strike = atm_strike
            straddle_shifts = 1
            strikes_list = [atm_strike]
            continue
        # If in trade, check for straddle shift
        if in_trade and atm_strike != entry_strike:
            straddle_shifts += 1
            entry_strike = atm_strike
            strikes_list.append(atm_strike)
        # Exit: SL hit (price >= sl_price)
        if in_trade and row['high'] >= sl_price:
            exit_price = sl_price
            exit_time = row['datetime']
            trades.append({'entry_time': entry_time, 'entry_price': entry_price, 'entry_vwap': entry_vwap, 'exit_time': exit_time, 'exit_price': exit_price, 'pnl': entry_price - exit_price, 'sl_type': f'{sl_points}pts', 'straddle_shifts': straddle_shifts, 'straddle_strikes': ','.join(map(str, strikes_list)), 'sl_price': sl_price})
            in_trade = False
            entry_price = 0.0
            entry_time = None
            entry_vwap = None
            sl_price = None
            entry_strike = None
            straddle_shifts = 0
            strikes_list = []
            continue
    # If still in trade at end, exit at last close
    if in_trade:
        exit_price = df.iloc[-1]['close']
        exit_time = df.iloc[-1]['datetime']
        trades.append({'entry_time': entry_time, 'entry_price': entry_price, 'entry_vwap': entry_vwap, 'exit_time': exit_time, 'exit_price': exit_price, 'pnl': entry_price - exit_price, 'sl_type': f'{sl_points}pts', 'straddle_shifts': straddle_shifts, 'straddle_strikes': ','.join(map(str, strikes_list)), 'sl_price': sl_price})
    num_trades = len(trades)
    total_pnl = sum(t['pnl'] for t in trades)
    return num_trades, total_pnl, trades

def pls_sl_rr_stats_from_tradelog(trade_log_df, all_minute_data, sl_points=15, rr_list=[1,2,3,4]):
    # For each R:R, simulate exits for each trade in the trade log using the minute data
    results = []
    if trade_log_df.empty or not all_minute_data:
        return results
    all_minute_df = pd.concat(all_minute_data, ignore_index=True)
    for rr in rr_list:
        rr_pnls = []
        for idx, trade in trade_log_df.iterrows():
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            entry_price = trade['entry_price']
            day_str = trade['Date']
            minute_df = all_minute_df[(all_minute_df['datetime'] >= entry_time) & (all_minute_df['datetime'] <= exit_time) & (all_minute_df['datetime'].dt.strftime('%Y-%m-%d') == day_str)]
            if minute_df.empty:
                rr_pnls.append(entry_price - trade['exit_price'])
                continue
            risk = sl_points  # Fixed risk of sl_points
            target = entry_price - rr * risk
            in_trade = True
            for i, row in minute_df.iterrows():
                # TP hit
                if row['low'] <= target:
                    rr_pnls.append(entry_price - target)
                    in_trade = False
                    break
                # SL hit (price >= entry_price + sl_points)
                if row['high'] >= entry_price + sl_points:
                    rr_pnls.append(entry_price - (entry_price + sl_points))
                    in_trade = False
                    break
            if in_trade:
                # If neither hit, exit at original exit price
                rr_pnls.append(entry_price - trade['exit_price'])
        n_trades = len(rr_pnls)
        total_pnl = sum(rr_pnls)
        max_dd = 0
        cum_pnl = 0
        peak = 0
        for pnl in rr_pnls:
            cum_pnl += pnl
            if cum_pnl > peak:
                peak = cum_pnl
            dd = peak - cum_pnl
            if dd > max_dd:
                max_dd = dd
        wins = [p for p in rr_pnls if p > 0]
        losses = [p for p in rr_pnls if p <= 0]
        n_win = len(wins)
        n_loss = len(losses)
        avg_gain = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = (n_win * avg_gain + n_loss * avg_loss) / n_trades if n_trades > 0 else 0
        sharpe = (np.mean(rr_pnls) / np.std(rr_pnls)) * np.sqrt(n_trades) if n_trades > 1 and np.std(rr_pnls) > 0 else 0
        results.append({
            'R:R': f'1:{rr}',
            'Number of Trades': n_trades,
            'Total P&L': total_pnl,
            'Max D:D': max_dd,
            'No. of Wins': n_win,
            'No. of Losses': n_loss,
            'Avg Gain': avg_gain,
            'Avg Loss': avg_loss,
            'Expectancy': expectancy,
            'Sharpe Ratio': sharpe
        })
    return results

def pls_sl_rr_tradelog_from_tradelog(trade_log_df, all_minute_data, rr=1):
    # For a given R:R, generate a trade log DataFrame with entry/exit times, prices, and P&L for PLS-SL logic
    if trade_log_df.empty or not all_minute_data:
        return pd.DataFrame()
    all_minute_df = pd.concat(all_minute_data, ignore_index=True)
    rr_trades = []
    for idx, trade in trade_log_df.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        entry_price = trade['entry_price']
        day_str = trade['Date']
        # Extract sl_points from sl_type (e.g., '15pts' or '15pts-reopen')
        sl_type = str(trade.get('sl_type', '15pts'))
        try:
            sl_points = int(sl_type.split('pts')[0])
        except Exception:
            sl_points = 15
        target = entry_price - rr * sl_points
        stop_loss = entry_price + sl_points
        minute_df = all_minute_df[(all_minute_df['datetime'] >= entry_time) & (all_minute_df['datetime'] <= exit_time) & (all_minute_df['datetime'].dt.strftime('%Y-%m-%d') == day_str)]
        if minute_df.empty:
            rr_trades.append({**trade, 'rr_exit_time': trade['exit_time'], 'rr_exit_price': trade['exit_price'], 'rr_pnl': entry_price - trade['exit_price'], 'rr_exit_type': 'Original', 'straddle_shifts': 1, 'straddle_strikes': str(trade.get('atm_strike', ''))})
            continue
        in_trade = True
        strikes_list = []
        prev_strike = None
        for i, row in minute_df.iterrows():
            atm_strike = row['atm_strike'] if 'atm_strike' in row else int(round(row['close'] / 50.0) * 50)
            if prev_strike is None or atm_strike != prev_strike:
                strikes_list.append(atm_strike)
                prev_strike = atm_strike
            # TP hit
            if row['low'] <= target:
                rr_trades.append({**trade, 'rr_exit_time': row['datetime'], 'rr_exit_price': target, 'rr_pnl': entry_price - target, 'rr_exit_type': f'Target {rr}:1', 'straddle_shifts': len(strikes_list), 'straddle_strikes': ','.join(map(str, strikes_list))})
                in_trade = False
                break
            # SL hit
            if row['high'] >= stop_loss:
                rr_trades.append({**trade, 'rr_exit_time': row['datetime'], 'rr_exit_price': stop_loss, 'rr_pnl': entry_price - stop_loss, 'rr_exit_type': f'SL {sl_points}', 'straddle_shifts': len(strikes_list), 'straddle_strikes': ','.join(map(str, strikes_list))})
                in_trade = False
                break
        if in_trade:
            rr_trades.append({**trade, 'rr_exit_time': trade['exit_time'], 'rr_exit_price': trade['exit_price'], 'rr_pnl': entry_price - trade['exit_price'], 'rr_exit_type': 'Original', 'straddle_shifts': len(strikes_list), 'straddle_strikes': ','.join(map(str, strikes_list))})
    rr_trades_df = pd.DataFrame(rr_trades)
    return rr_trades_df

def show_backtest_view(db):
    st.header('🧪 Backtest')
    st.markdown('Select a strategy to backtest:')
    # Get available min/max dates
    date_info = db.execute("SELECT MIN(date) as min_date, MAX(date) as max_date FROM spot_data").fetchdf()
    min_date = date_info['min_date'].iloc[0]
    max_date = date_info['max_date'].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=min_date, min_value=min_date, max_value=max_date, key='bt_start')
    with col2:
        end_date = st.date_input('End Date', value=max_date, min_value=start_date, max_value=max_date, key='bt_end')
    dte_values = db.execute("SELECT DISTINCT DATEDIFF('day', date, MIN(expiry_date)) as dte FROM options_data GROUP BY date ORDER BY dte").fetchdf()['dte'].dropna().astype(int).unique().tolist()
    dte_values = [d for d in dte_values if d <= 6]
    selected_dtes = st.multiselect('Select DTE(s) (Days to Expiry)', dte_values, default=[dte_values[0]], key='dte_select')
    strategies = ['VWAP - A/B', 'VWAP - SL', 'pls -sl']
    selected_strategy = st.selectbox('Available Strategies', strategies, key='backtest_strategy_select')

    if st.button(f'Select "{selected_strategy}"'):
        st.session_state['selected_strategy'] = selected_strategy
        st.session_state['selected_start_date'] = start_date
        st.session_state['selected_end_date'] = end_date
        st.session_state['selected_dtes'] = selected_dtes
        st.success(f'Selected strategy: {selected_strategy} from {start_date} to {end_date} with DTE={selected_dtes}.')

    # Use session_state for the rest of the UI
    strategy = st.session_state.get('selected_strategy')
    start_date = st.session_state.get('selected_start_date')
    end_date = st.session_state.get('selected_end_date')
    selected_dtes = st.session_state.get('selected_dtes')

    if strategy == 'VWAP - A/B':
        day_stats = []
        for dt in pd.date_range(start=start_date, end=end_date):
            date_str = dt.strftime('%Y-%m-%d')
            dte = get_dte_for_date(db, date_str)
            if dte not in selected_dtes:
                continue
            df = get_rolling_straddle_ohlc(db, date_str)
            above, below = count_crossings_rolling_vwap(df, debug=False)
            # Time windows
            win1_above, win1_below = count_crossings_in_window(df, '09:16', '10:15')
            win2_above, win2_below = count_crossings_in_window(df, '10:16', '12:15')
            win3_above, win3_below = count_crossings_in_window(df, '12:16', '14:15')
            win4_above, win4_below = count_crossings_in_window(df, '14:16', '15:30')
            day_stats.append({
                'Date': date_str,
                'DTE': dte,
                'Crossed Above VWAP': above,
                'Crossed Below VWAP': below,
                '09:16-10:15 Above': win1_above,
                '09:16-10:15 Below': win1_below,
                '10:16-12:15 Above': win2_above,
                '10:16-12:15 Below': win2_below,
                '12:16-14:15 Above': win3_above,
                '12:16-14:15 Below': win3_below,
                '14:16-15:30 Above': win4_above,
                '14:16-15:30 Below': win4_below
            })
        stats_df = pd.DataFrame(day_stats)
        st.dataframe(stats_df, use_container_width=True)
    elif strategy == 'VWAP - SL':
        day_stats = []
        all_trades = []
        all_minute_data = []
        for dt in pd.date_range(start=start_date, end=end_date):
            date_str = dt.strftime('%Y-%m-%d')
            dte = get_dte_for_date(db, date_str)
            if dte not in selected_dtes:
                continue
            df = get_rolling_straddle_ohlc(db, date_str)
            num_trades, total_pnl, trades = vwap_sl_strategy(df)
            for t in trades:
                t['Date'] = date_str
                t['DTE'] = dte
            all_trades.extend(trades)
            day_stats.append({'Date': date_str, 'DTE': dte, 'Number of Trades': num_trades, 'Total P&L': total_pnl})
            if not df.empty:
                all_minute_data.append(df)
        stats_df = pd.DataFrame(day_stats)
        st.dataframe(stats_df, use_container_width=True)
        # Display trade log below summary
        trade_log_df = pd.DataFrame(all_trades)
        # Display R:R stats table for all minute data combined
        if all_minute_data:
            all_minute_df = pd.concat(all_minute_data, ignore_index=True)
            # Add 'Until SL or 15:30' row using actual trade log
            rr_stats = []
            if not trade_log_df.empty:
                trades = trade_log_df['pnl'].tolist()
                n_trades = len(trades)
                total_pnl = sum(trades)
                max_dd = 0
                cum_pnl = 0
                peak = 0
                for pnl in trades:
                    cum_pnl += pnl
                    if cum_pnl > peak:
                        peak = cum_pnl
                    dd = peak - cum_pnl
                    if dd > max_dd:
                        max_dd = dd
                wins = [p for p in trades if p > 0]
                losses = [p for p in trades if p <= 0]
                n_win = len(wins)
                n_loss = len(losses)
                avg_gain = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                expectancy = (n_win * avg_gain + n_loss * avg_loss) / n_trades if n_trades > 0 else 0
                sharpe = (np.mean(trades) / np.std(trades)) * np.sqrt(n_trades) if n_trades > 1 and np.std(trades) > 0 else 0
                until_sl_row = {
                    'R:R': 'Until SL or 15:30',
                    'Number of Trades': n_trades,
                    'Total P&L': total_pnl,
                    'Max D:D': max_dd,
                    'No. of Wins': n_win,
                    'No. of Losses': n_loss,
                    'Avg Gain': avg_gain,
                    'Avg Loss': avg_loss,
                    'Expectancy': expectancy,
                    'Sharpe Ratio': sharpe
                }
                rr_stats.append(until_sl_row)
                rr_stats += vwap_sl_rr_stats_from_tradelog(trade_log_df, all_minute_data, rr_list=[1,2,3,4])
            rr_stats_df = pd.DataFrame(rr_stats)
            # Round Sharpe Ratio and Expectancy to integers (no decimal points)
            if 'Sharpe Ratio' in rr_stats_df.columns:
                rr_stats_df['Sharpe Ratio'] = rr_stats_df['Sharpe Ratio'].round(0).astype(int)
            if 'Expectancy' in rr_stats_df.columns:
                rr_stats_df['Expectancy'] = rr_stats_df['Expectancy'].round(0).astype(int)
            # Formula explanation:
            # Sharpe Ratio = (Mean P&L per trade / Std Dev of P&L per trade) * sqrt(Number of Trades)
            # Expectancy = (Number of Wins * Avg Gain + Number of Losses * Avg Loss) / Number of Trades
            st.subheader('VWAP-SL R:R Stats (All Days Combined)')
            st.dataframe(rr_stats_df, use_container_width=True)
        # Always show R:R selectbox and trade log if data is available
        if not trade_log_df.empty and all_minute_data:
            rr_options = [0, 1, 2, 3, 4]
            rr_label_map = {0: 'Until SL or 15:30', 1: '1:1', 2: '1:2', 3: '1:3', 4: '1:4'}
            selected_rrs = st.multiselect('Select R:R ratios to display details (e.g., 1 for 1:1, 2 for 1:2, etc.; 0 for Until SL or 15:30)', rr_options, default=[1], format_func=lambda x: rr_label_map[x], key='rr_input_multi')
            show_details = st.button('Show Details for Selected R:R')
            if show_details:
                for selected_rr in selected_rrs:
                    if selected_rr == 0:
                        # Until SL or 15:30
                        log_df = trade_log_df.copy()
                        pnl_col = 'pnl'
                        exit_time_col = 'exit_time'
                        exit_price_col = 'exit_price'
                        exit_type_col = 'sl_type'
                    else:
                        log_df = vwap_sl_rr_tradelog_from_tradelog(trade_log_df, all_minute_data, rr=selected_rr)
                        pnl_col = 'rr_pnl'
                        exit_time_col = 'rr_exit_time'
                        exit_price_col = 'rr_exit_price'
                        exit_type_col = 'rr_exit_type'
                    if not log_df.empty:
                        st.subheader(f'Trade Log (with Straddle Shifts) for R:R = {rr_label_map[selected_rr]}')
                        st.dataframe(log_df[[
                            'Date', 'DTE', 'entry_time', 'entry_price', 'entry_vwap',
                            exit_time_col, exit_price_col, pnl_col, exit_type_col,
                            'straddle_shifts', 'straddle_strikes'
                        ]], use_container_width=True)
                        # Monthly P&L table for selected R:R
                        log_df['Month'] = pd.to_datetime(log_df['Date']).dt.to_period('M')
                        monthly_stats = []
                        for month, group in log_df.groupby('Month'):
                            month_str = str(month)
                            pnl_series = group[pnl_col].cumsum()
                            # Max Drawdown
                            roll_max = pnl_series.cummax()
                            drawdown = roll_max - pnl_series
                            max_dd = drawdown.max() if not drawdown.empty else 0
                            # Recovery Days: number of trades to recover from max DD
                            recovery_days = 0
                            if max_dd > 0:
                                dd_idx = drawdown.idxmax() if not drawdown.empty else None
                                if dd_idx is not None:
                                    after_dd = pnl_series[dd_idx:]
                                    recovery_idx = after_dd[after_dd >= roll_max[dd_idx]].first_valid_index()
                                    if recovery_idx is not None:
                                        recovery_days = recovery_idx - dd_idx
                                        if hasattr(recovery_days, 'days'):
                                            recovery_days = recovery_days.days
                                        else:
                                            recovery_days = int(recovery_days)
                            # Expectancy
                            wins = group[group[pnl_col] > 0][pnl_col]
                            losses = group[group[pnl_col] <= 0][pnl_col]
                            n_win = len(wins)
                            n_loss = len(losses)
                            n_trades = len(group)
                            avg_gain = wins.mean() if n_win > 0 else 0
                            avg_loss = losses.mean() if n_loss > 0 else 0
                            expectancy = (n_win * avg_gain + n_loss * avg_loss) / n_trades if n_trades > 0 else 0
                            # MaxDD to Profit Ratio
                            total_profit = group[pnl_col].sum()
                            maxdd_to_profit = (max_dd / total_profit) if total_profit != 0 else 0
                            monthly_stats.append({
                                'Month': month_str,
                                'Total P&L': round(total_profit, 2),
                                'Max D:D': round(max_dd, 2),
                                'Recovery Days': recovery_days,
                                'Expectancy': round(expectancy, 2),
                                'MaxDD to Profit Ratio': round(maxdd_to_profit, 2)
                            })
                        monthly_stats_df = pd.DataFrame(monthly_stats)
                        st.subheader(f'Monthly P&L for SL = {sl_points} pts, R:R = {rr_label_map[selected_rr]}')
                        st.dataframe(monthly_stats_df, use_container_width=True)
                        # --- Straddle Shifts Distribution Table ---
                        if not log_df.empty and 'straddle_shifts' in log_df.columns:
                            shift_counts = log_df['straddle_shifts'].value_counts().sort_index()
                            shift_dist_df = shift_counts.reset_index()
                            shift_dist_df.columns = ['Number of Straddle Shifts', 'Number of Trades']
                            st.subheader('Distribution of Straddle Shifts per Trade')
                            st.dataframe(shift_dist_df, use_container_width=True)
                        # Display total number of straddle shifts for this R:R
                        total_shifts = log_df['straddle_shifts'].sum()
                        st.markdown(f"**Total Straddle Shifts for R:R = {rr_label_map[selected_rr]}: {total_shifts}**")
    elif strategy == 'pls -sl':
        sl_options = [15, 20, 25, 30, 35]
        selected_sl = st.selectbox('Select SL value (pts)', sl_options, index=0, key='sl_input_single')
        run_bt = st.button('Run Backtest for Selected SL')
        if run_bt:
            sl_val = selected_sl
            day_stats = []
            all_trades = []
            all_minute_data = []
            for dt in pd.date_range(start=start_date, end=end_date):
                date_str = dt.strftime('%Y-%m-%d')
                dte = get_dte_for_date(db, date_str)
                if dte not in selected_dtes:
                    continue
                df = get_rolling_straddle_ohlc(db, date_str)
                num_trades, total_pnl, trades = pls_sl_strategy(df, sl_points=sl_val)
                for t in trades:
                    t['Date'] = date_str
                    t['DTE'] = dte
                all_trades.extend(trades)
                if not df.empty:
                    all_minute_data.append(df)
            trade_log_df = pd.DataFrame(all_trades)
            st.write('Debug: trade_log_df', trade_log_df)  # Debug output
            if all_minute_data and not trade_log_df.empty:
                rr_stats = []
                trades = trade_log_df['pnl'].tolist()
                n_trades = len(trades)
                total_pnl = sum(trades)
                max_dd = 0
                cum_pnl = 0
                peak = 0
                for pnl in trades:
                    cum_pnl += pnl
                    if cum_pnl > peak:
                        peak = cum_pnl
                    dd = peak - cum_pnl
                    if dd > max_dd:
                        max_dd = dd
                wins = [p for p in trades if p > 0]
                losses = [p for p in trades if p <= 0]
                n_win = len(wins)
                n_loss = len(losses)
                avg_gain = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                expectancy = (n_win * avg_gain + n_loss * avg_loss) / n_trades if n_trades > 0 else 0
                sharpe = (np.mean(trades) / np.std(trades)) * np.sqrt(n_trades) if n_trades > 1 and np.std(trades) > 0 else 0
                total_shifts = trade_log_df['straddle_shifts'].sum()
                avg_shifts = trade_log_df['straddle_shifts'].mean() if not trade_log_df.empty else 0
                until_sl_row = {
                    'R:R': f'Until SL={sl_val} or 15:30',
                    'Number of Trades': n_trades,
                    'Total P&L': total_pnl,
                    'Max D:D': max_dd,
                    'No. of Wins': n_win,
                    'No. of Losses': n_loss,
                    'Avg Gain': avg_gain,
                    'Avg Loss': avg_loss,
                    'Expectancy': expectancy,
                    'Sharpe Ratio': sharpe,
                    'Total Straddle Shifts': total_shifts,
                    'Avg Straddle Shifts': round(avg_shifts, 2)
                }
                rr_stats.append(until_sl_row)
                rr_list = [1,2,3,4]
                for rr in rr_list:
                    rr_tradelog_df = pls_sl_rr_tradelog_from_tradelog(trade_log_df, all_minute_data, rr=rr)
                    rr_total_shifts = rr_tradelog_df['straddle_shifts'].sum() if not rr_tradelog_df.empty else 0
                    rr_avg_shifts = rr_tradelog_df['straddle_shifts'].mean() if not rr_tradelog_df.empty else 0
                    stats = pls_sl_rr_stats_from_tradelog(trade_log_df, all_minute_data, sl_points=sl_val, rr_list=[rr])[0]
                    stats['Total Straddle Shifts'] = rr_total_shifts
                    stats['Avg Straddle Shifts'] = round(rr_avg_shifts, 2)
                    rr_stats.append(stats)
                rr_stats_df = pd.DataFrame(rr_stats)
                if 'Sharpe Ratio' in rr_stats_df.columns:
                    rr_stats_df['Sharpe Ratio'] = rr_stats_df['Sharpe Ratio'].round(0).astype(int)
                if 'Expectancy' in rr_stats_df.columns:
                    rr_stats_df['Expectancy'] = rr_stats_df['Expectancy'].round(0).astype(int)
                st.session_state['pls_sl_rr_stats_df'] = rr_stats_df
                # --- Monthly P&L Table for selected SL ---
                trade_log_df['Month'] = pd.to_datetime(trade_log_df['Date']).dt.to_period('M')
                monthly_stats = []
                for month, group in trade_log_df.groupby('Month'):
                    month_str = str(month)
                    pnl_series = group['pnl'].cumsum()
                    roll_max = pnl_series.cummax()
                    drawdown = roll_max - pnl_series
                    max_dd = drawdown.max() if not drawdown.empty else 0
                    recovery_days = 0
                    if max_dd > 0:
                        dd_idx = drawdown.idxmax() if not drawdown.empty else None
                        if dd_idx is not None:
                            after_dd = pnl_series[dd_idx:]
                            recovery_idx = after_dd[after_dd >= roll_max[dd_idx]].first_valid_index()
                            if recovery_idx is not None:
                                recovery_days = recovery_idx - dd_idx
                                if hasattr(recovery_days, 'days'):
                                    recovery_days = recovery_days.days
                                else:
                                    recovery_days = int(recovery_days)
                    wins = group[group['pnl'] > 0]['pnl']
                    losses = group[group['pnl'] <= 0]['pnl']
                    n_win = len(wins)
                    n_loss = len(losses)
                    n_trades = len(group)
                    avg_gain = wins.mean() if n_win > 0 else 0
                    avg_loss = losses.mean() if n_loss > 0 else 0
                    expectancy = (n_win * avg_gain + n_loss * avg_loss) / n_trades if n_trades > 0 else 0
                    total_profit = group['pnl'].sum()
                    maxdd_to_profit = (max_dd / total_profit) if total_profit != 0 else 0
                    monthly_stats.append({
                        'Month': month_str,
                        'Total P&L': round(total_profit, 2),
                        'Max D:D': round(max_dd, 2),
                        'Recovery Days': recovery_days,
                        'Expectancy': round(expectancy, 2),
                        'MaxDD to Profit Ratio': round(maxdd_to_profit, 2)
                    })
                monthly_stats_df = pd.DataFrame(monthly_stats)
                st.subheader(f'Monthly P&L for SL = {sl_val} pts')
                st.dataframe(monthly_stats_df, use_container_width=True)
                # --- Straddle Shifts Distribution Table ---
                if not trade_log_df.empty and 'straddle_shifts' in trade_log_df.columns:
                    shift_counts = trade_log_df['straddle_shifts'].value_counts().sort_index()
                    shift_dist_df = shift_counts.reset_index()
                    shift_dist_df.columns = ['Number of Straddle Shifts', 'Number of Trades']
                    st.subheader('Distribution of Straddle Shifts per Trade')
                    st.dataframe(shift_dist_df, use_container_width=True)
                # After R:R stats, show straddle shift distribution for 1:1 R:R (with win/loss breakdown)
                rr1_tradelog_df = pls_sl_rr_tradelog_from_tradelog(trade_log_df, all_minute_data, rr=1)
                if not rr1_tradelog_df.empty and 'straddle_shifts' in rr1_tradelog_df.columns:
                    shift_winlose = rr1_tradelog_df.groupby('straddle_shifts').agg(
                        Number_of_Win_Trades = ('rr_pnl', lambda x: (x > 0).sum()),
                        Number_of_Loss_Trades = ('rr_pnl', lambda x: (x <= 0).sum())
                    ).reset_index()
                    shift_winlose.columns = ['Number of Straddle Shifts', 'Number of Win Trades', 'Number of Loss Trades']
                    st.subheader('Distribution of Straddle Shifts per Trade (1:1 R:R)')
                    st.dataframe(shift_winlose, use_container_width=True)
            else:
                st.session_state['pls_sl_rr_stats_df'] = None
        # Always display the results if present in session_state
        if 'pls_sl_rr_stats_df' in st.session_state:
            rr_stats_df = st.session_state['pls_sl_rr_stats_df']
            if rr_stats_df is not None:
                st.subheader(f'PLS-SL R:R Stats (SL={selected_sl} pts, All Days Combined)')
                st.dataframe(rr_stats_df, use_container_width=True)
            else:
                st.warning('No trades found for the selected parameters.')
    else:
        st.error("Invalid strategy selected.") 