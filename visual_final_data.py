import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import yfinance as yf

def load_data(strategy_choice):
    # Determine file names based on strategy choice
    if strategy_choice == "1":  # Pandas strategy
        trades_file = "dv2_trades.csv"
        equity_file = "dv2_equity_curve.csv"
        strategy_name = "Pandas Strategy"
    else:  # Numba strategy
        trades_file = "dv2_trades_numba.csv"
        equity_file = "dv2_equity_curve_numba.csv"
        strategy_name = "DV2 Strategy"
    
    # Load trades and equity curve data
    trades = pd.read_csv(trades_file, parse_dates=['entry_date', 'exit_date'])
    equity_curve = pd.read_csv(equity_file, index_col=0, parse_dates=True).squeeze()
    return equity_curve, trades, strategy_name

def get_sp500_benchmark(equity_curve):
    # Get S&P 500 data for comparison
    try:
        start_date = equity_curve.index.min().strftime('%Y-%m-%d')
        end_date = equity_curve.index.max().strftime('%Y-%m-%d')
        
        spy_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        
        if spy_data.empty:
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
        
        if not spy_data.empty:
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.droplevel(1)
            
            spy_prices = spy_data['Close']
            spy_aligned = spy_prices.reindex(equity_curve.index, method='ffill')
            spy_returns = spy_aligned.pct_change().fillna(0)
            return (1 + spy_returns).cumprod() * 100000
        
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")
    return None

def create_charts(equity_curve, spy_curve, strategy_name):
    # Image creation
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'{strategy_name} vs S&P 500 Performance', fontsize=16, fontweight='bold', y=0.95)
    
    strategy_color, sp500_color = '#6A4C93', '#FF6B35'
    
    # 1st graph: Total Returns (log scale)
    strategy_pct = (equity_curve / equity_curve.iloc[0] - 1) * 100
    ax1.semilogy(strategy_pct.index, strategy_pct.values + 100, 
                 color=strategy_color, linewidth=2.5, label='Strategy')
    
    if spy_curve is not None:
        spy_pct = (spy_curve / spy_curve.iloc[0] - 1) * 100
        ax1.semilogy(spy_pct.index, spy_pct.values + 100, 
                     color=sp500_color, linewidth=2, label='S&P 500')
    
    ax1.set_title('Total Return (log scale)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Total Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x-100:,.0f}%'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    
    # 2nd graph: Drawdowns
    strategy_dd = (equity_curve / equity_curve.cummax() - 1) * 100
    ax2.fill_between(strategy_dd.index, strategy_dd.values, 0, 
                     color=strategy_color, alpha=0.3, label='Strategy')
    ax2.plot(strategy_dd.index, strategy_dd.values, color=strategy_color, linewidth=1.5)
    
    if spy_curve is not None:
        spy_dd = (spy_curve / spy_curve.cummax() - 1) * 100
        ax2.fill_between(spy_dd.index, spy_dd.values, 0,
                         color=sp500_color, alpha=0.3, label='S&P 500')
        ax2.plot(spy_dd.index, spy_dd.values, color=sp500_color, linewidth=1.5)
    
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    
    # 3rd graph: Annual Returns
    strategy_annual = equity_curve.resample('YE').last().pct_change().dropna() * 100
    
    if spy_curve is not None:
        spy_annual = spy_curve.resample('YE').last().pct_change().dropna() * 100
        common_years = strategy_annual.index.intersection(spy_annual.index)
        strategy_annual, spy_annual = strategy_annual[common_years], spy_annual[common_years]
    
    years = strategy_annual.index.year
    x_pos = np.arange(len(years))
    width = 0.35
    
    ax3.bar(x_pos - width/2, strategy_annual.values, width,
            label='Strategy', color=strategy_color, alpha=0.8)
    
    if spy_curve is not None:
        ax3.bar(x_pos + width/2, spy_annual.values, width,
                label='S&P 500', color=sp500_color, alpha=0.8)
    
    ax3.set_title('Annual Return (%)', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel('Annual Return (%)', fontsize=12)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.legend(loc='upper right', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(years, rotation=45)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.4)
    
    # Save with strategy-specific filename
    filename = f"dv2_strategy_performance_{strategy_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return filename

def main():
    # Strategy selection
    print("Select which strategy results to visualize:")
    print("1. Pandas Strategy (backtest.py)")
    print("2. Numba Strategy (backtest_numba.py)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice not in ["1", "2"]:
        print("Invalid choice. Using Pandas Strategy (1).")
        choice = "1"
    
    # Determine required files based on choice
    if choice == "1":
        required_files = ['dv2_trades.csv', 'dv2_equity_curve.csv']
    else:
        required_files = ['dv2_trades_numba.csv', 'dv2_equity_curve_numba.csv']
    
    # Check required files
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found. Please run the corresponding strategy first.")
            if choice == "1":
                print("Run: python backtest.py")
            else:
                print("Run: python backtest_numba.py")
            return
    
    # Load data
    print("Loading data...")
    equity_curve, trades, strategy_name = load_data(choice)
    print(f"Loaded equity curve with {len(equity_curve)} data points")
    print(f"Loaded {len(trades)} trades")
    
    # Get benchmark
    print("Downloading S&P 500 benchmark...")
    spy_curve = get_sp500_benchmark(equity_curve)
    
    # Create visualization
    print(f"Creating performance dashboard for {strategy_name}...")
    filename = create_charts(equity_curve, spy_curve, strategy_name)
    
    print(f"Visualization saved as '{filename}'")

main()
