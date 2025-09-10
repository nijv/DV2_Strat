import numpy as np
import pandas as pd
import json
import time
from numba import jit, njit
import warnings
warnings.filterwarnings('ignore')

def load_data(sector=None):
    """Load S&P 500 tickers and data, optionally filtered by sector"""
    # Load S&P 500 tickers and sectors from JSON file
    with open('sp500_tickers.json', 'r') as f:
        tickers_data = json.load(f)
    
    # Filter by sector if specified
    if sector:
        tickers_data = [item for item in tickers_data if item['sector'].lower() == sector.lower()]
        print(f"Filtering by sector: {sector}")
        if not tickers_data:
            print(f"No stocks found for sector: {sector}")
            return {}
    
    print(f"Loading data for {len(tickers_data)} stocks...")
    
    data = {}
    for item in tickers_data:
        ticker = item['ticker']
        try:
            df = pd.read_csv(f"data/{ticker}.csv", index_col=0, parse_dates=True)
            if not df.dropna().empty:
                data[ticker] = df.dropna()
        except FileNotFoundError:
            print(f"Warning: {ticker}.csv not found")
            continue
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            continue
    
    print(f"Loaded {len(data)} stocks")
    return data

@njit
def calculate_dv2_numba(close_prices, high_prices, low_prices, length=126):
    """Numba-optimized DV2 calculation - ORIGINAL VERSION (2-day rolling mean)"""
    n = len(close_prices)
    hl_avg = (high_prices + low_prices) / 2.0
    dv_raw = np.full(n, np.nan)
    
    # Calculate DV raw with 2-day rolling mean (original strategy)
    for i in range(1, n):  # Start from index 1 for 2-day window
        dv_raw[i] = np.mean((close_prices[i-1:i+1] / hl_avg[i-1:i+1]) - 1.0)
    
    # Calculate percentile ranks - EXACT pandas behavior
    dv2_values = np.full(n, np.nan)
    for i in range(length, n):  # Start from length, not length-1
        window = dv_raw[i-length+1:i+1]
        valid_window = window[~np.isnan(window)]
        # Require exactly `length` valid values to match pandas rolling behavior
        if len(valid_window) == length:
            current_value = dv_raw[i]
            if not np.isnan(current_value):
                # Count values less than current
                less_count = np.sum(valid_window < current_value)
                # Count values equal to current  
                equal_count = np.sum(valid_window == current_value)
                
                # Pandas formula: avg_rank = less_count + (equal_count + 1) / 2.0
                # Then: rank_pct = avg_rank / count
                avg_rank = less_count + (equal_count + 1) / 2.0
                rank_pct = avg_rank / len(valid_window)
                dv2_values[i] = rank_pct * 100
    
    return dv2_values

@njit
def calculate_sma200_numba(close_prices):
    """Numba-optimized SMA200 calculation - EXACT pandas replication"""
    n = len(close_prices)
    sma200 = np.full(n, np.nan)
    
    # Pandas: close.shift().rolling(200, min_periods=200).mean()
    # This means: shift the series by 1, then calculate 200-day SMA
    # So SMA at index i uses prices from (i-200) to (i-1), not (i-199) to i
    
    for i in range(200, n):
        # Use prices from (i-200) to (i-1) to match close.shift().rolling(200)
        sma200[i] = np.mean(close_prices[i-200:i])
    
    return sma200

@njit
def calculate_momentum_6m_numba(close_prices):
    """Numba-optimized 6-month momentum calculation"""
    n = len(close_prices)
    momentum = np.full(n, np.nan)
    
    for i in range(126, n):
        momentum[i] = (close_prices[i] / close_prices[i-126]) - 1.0
    
    return momentum

def dv2(close, high, low, length=126):
    # Calculate the DV2 indicator using numba (original 2-day version)
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    
    dv2_values = calculate_dv2_numba(close_arr, high_arr, low_arr, length)
    return pd.Series(dv2_values, index=close.index)

def natr(high, low, close):
    # Use exact pandas calculation to avoid EWM complexity
    # Don't optimize this one since pandas EWM with adjust=True is very complex
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14).mean() / close * 100

def sma200(close):
    # Simple Moving Average over 200 days using numba
    close_arr = close.values
    sma_values = calculate_sma200_numba(close_arr)
    return pd.Series(sma_values, index=close.index)

def momentum_6m(close):
    # 6-month momentum using numba
    close_arr = close.values
    mom_values = calculate_momentum_6m_numba(close_arr)
    return pd.Series(mom_values, index=close.index)

INITIAL_EQUITY = 100000
MAX_POSITIONS = 10  # Original strategy uses 10 positions
ENTRY_COMMISSION = 0.001  # 0.1% entry commission
EXIT_COMMISSION = 0.001   # 0.1% exit commission

def backtest(data, initial_equity=INITIAL_EQUITY, max_positions=MAX_POSITIONS, entry_commission=ENTRY_COMMISSION, exit_commission=EXIT_COMMISSION):
    for symbol, df in data.items():
        df["dv2"] = dv2(df.close, df.high, df.low)
        df["natr"] = natr(df.high, df.low, df.close)
        df["sma200"] = sma200(df.close)
        df["mom_6m"] = momentum_6m(df.close)

    dates = sorted(set().union(*[df.index for df in data.values()]))
    equity = pd.Series(index=dates, dtype=float)
    cash = initial_equity
    positions = {}
    trades = []

    total_dates = len(dates)
    
    for i, date in enumerate(dates):
        # Skip the first 200 days to allow for SMA200 and 6-month momentum calculation
        if i < 200:
            equity[date] = cash
            continue

        # Exits - ORIGINAL STRATEGY: Momentum-based exit
        for symbol in list(positions):
            df = data[symbol]
            if (dates[i - 1] not in df.index) or (dates[i - 2] not in df.index):
                continue

            close_yesterday = df.loc[dates[i - 1], "close"]
            high_prev = df.loc[dates[i - 2], "high"]

            # Original exit condition: close > previous high
            if close_yesterday > high_prev:
                if date not in df.index or np.isnan(df.loc[date, "open"]):
                    continue

                opening_price = df.loc[date, "open"]
                position = positions.pop(symbol)
                exit_value = position["shares"] * opening_price
                exit_commission_cost = exit_value * exit_commission
                cash += exit_value - exit_commission_cost
                trades.append({
                    "symbol": symbol,
                    "entry_date": position["entry_date"],
                    "exit_date": date,
                    "entry_px": position["entry_px"],
                    "exit_px": opening_price,
                    "shares": position["shares"],
                    "pnl": position["shares"] * (opening_price - position["entry_px"]) - position.get("entry_commission", 0) - exit_commission_cost
                })

        # Entries - ORIGINAL STRATEGY conditions
        open_slots = max_positions - len(positions)
        if open_slots > 0:
            current_equity = cash + sum(
                position["shares"] * data[symbol].loc[date, "close"]
                for symbol, position in positions.items()
                if date in data[symbol].index
            )
            slot_cash = current_equity / max_positions

            candidates = []
            for symbol, df in data.items():
                if symbol in positions or dates[i - 1] not in df.index:
                    continue

                row = df.loc[dates[i - 1]]
                dv_yesterday, sma200_yesterday, price_yesterday, mom_6m_yesterday = (
                    row["dv2"], row["sma200"], row["close"], row["mom_6m"]
                )

                # Original entry conditions: mom_6m_yesterday > 0 (must be positive)
                if (dv_yesterday < 10 and 
                    price_yesterday > sma200_yesterday and 
                    mom_6m_yesterday > 0):
                    candidates.append((symbol, row["natr"]))

            for symbol, _ in sorted(candidates, key=lambda x: (x[1], x[0]), reverse=True)[:open_slots]:
                df = data[symbol]
                if date not in df.index or np.isnan(df.loc[date, "open"]):
                    continue

                opening_price = df.loc[date, "open"]
                shares = int(slot_cash // opening_price)
                if shares == 0:
                    continue

                entry_value = shares * opening_price
                entry_commission_cost = entry_value * entry_commission
                cash -= entry_value + entry_commission_cost
                positions[symbol] = {
                    "shares": shares, 
                    "entry_px": opening_price, 
                    "entry_date": date,
                    "entry_commission": entry_commission_cost
                }

        # Calculate money in the market
        open_positions_value = sum(
            position["shares"] * data[symbol].loc[date, "close"]
            for symbol, position in positions.items()
            if date in data[symbol].index and not np.isnan(data[symbol].loc[date, "close"])
        )
        equity[date] = cash + open_positions_value

    return equity.ffill(), pd.DataFrame(trades)

print("Loading OHLC data from CSV files...")

# Start timing
start_time = time.time()

# Check if numba is available
try:
    import numba
    print(f"Numba optimization enabled (version: {numba.__version__})")
except ImportError:
    print("Warning: Numba not installed. Install with: pip install numba")
    print("Running without optimization...")
    exit()

# Sector selection
print("\nSector Options:")
print("1. Run all sectors")
print("2. Run specific sector")

import sys
if len(sys.argv) > 1 and sys.argv[1] == "auto":
    choice = "1"
else:
    choice = input("Choose option (1 or 2): ").strip()

sector = None
if choice == "2":
    # Load available sectors
    with open('sp500_tickers.json', 'r') as f:
        tickers_data = json.load(f)
    
    sectors = sorted(set(item['sector'] for item in tickers_data))
    print(f"\nAvailable sectors:")
    for i, s in enumerate(sectors, 1):
        count = sum(1 for item in tickers_data if item['sector'] == s)
        print(f"{i:2d}. {s} ({count} stocks)")
    
    try:
        sector_choice = int(input(f"\nChoose sector (1-{len(sectors)}): ")) - 1
        if 0 <= sector_choice < len(sectors):
            sector = sectors[sector_choice]
            print(f"Selected sector: {sector}")
        else:
            print("Invalid choice, running all sectors")
    except ValueError:
        print("Invalid input, running all sectors")

data = load_data(sector)

if not data:
    print("No OHLC data loaded – exiting.")
    exit()

print(f"Successfully loaded data for {len(data)} stocks")

equity_curve, trades = backtest(data)

TRADING_DAYS = 252 # Average trading days in a year
total_return = equity_curve.iat[-1] / equity_curve.iat[0] - 1 
cagr = (1 + total_return) ** (TRADING_DAYS / len(equity_curve)) - 1
max_dd = (equity_curve / equity_curve.cummax() - 1).min()

daily_returns = equity_curve.pct_change().dropna()
sharpe_ratio = daily_returns.mean() / daily_returns.std() * (TRADING_DAYS ** 0.5)

print(f"\nBacktest Results:")
print(f"Total return : {total_return:,.2%}")
print(f"CAGR         : {cagr:,.2%}")
print(f"Max drawdown : {max_dd:,.2%}")
print(f"Sharpe ratio : {sharpe_ratio:.2f}")
print(f"Trades       : {len(trades):,}")

# Calculate trade statistics
if len(trades) > 0:
    profits = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]
    pct_profitable = len(profits) / len(trades) * 100
    pct_loss = len(losses) / len(trades) * 100
    print(f"Profitable trades: {len(profits):,} ({pct_profitable:.1f}%)")
    print(f"Losing trades   : {len(losses):,} ({pct_loss:.1f}%)")

# Save trades to CSV
trades.to_csv("dv2_trades_numba.csv", index=False)
print("Results saved to dv2_trades_numba.csv")

# Save equity curve to CSV
equity_curve.to_csv("dv2_equity_curve_numba.csv", header=True)
print("Equity curve saved to dv2_equity_curve_numba.csv")

# Calculate and display execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
