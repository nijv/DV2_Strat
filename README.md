# S&P 500 Mean Reversion Strategy

I built this trading strategy as recommended by my internship advisor at [Menthor Q](https://menthorq.com/). The approach was inspired by [this Quantitativo article](https://www.quantitativo.com/p/a-different-indicator) on mean reversion indicators. The idea is simple: buy S&P 500 stocks when they're beaten down but still in uptrends, then sell when they bounce back.

## What it does

This strategy looks for stocks that have dropped significantly but are still in long-term uptrends. It uses the DV2 indicator to find oversold conditions and the 200-day moving average to confirm the trend direction.

## How it works

**Entry conditions:**
- DV2 < 10 (stock is oversold based on 2-day price action)
- Price above 200-day moving average (confirming uptrend)
- Positive 6-month momentum
- Maximum 10 positions at once

**Exit condition:**
- Close price exceeds previous day's high

**Position sizing:**
- Equal dollar amounts across all positions
- 0.1% transaction costs included

## Files

```
get_sp500_tickers.py     # Downloads current S&P 500 list from Wikipedia
get_candle_data.py       # Downloads stock price data via yfinance
backtest.py              # Main strategy using pandas
backtest_numba.py        # Same strategy, optimized with Numba (2.77x faster)
visual_final_data.py     # Creates performance charts
```

## Usage

```bash
# Install requirements
pip install pandas numpy matplotlib yfinance numba requests

# Download data
python get_sp500_tickers.py
python get_candle_data.py

# Run backtest
python backtest.py          # Takes ~94 seconds
or
python backtest_numba.py    # Takes ~34 seconds

# View results
python visual_final_data.py
```

## Performance (2001-2024)

- **Total return:** 3,573% vs S&P 500's 500%
- **Annual return:** 18.72% vs 8.5%
- **Worst drawdown:** -34.81%
- **Sharpe ratio:** 0.93
- **Trades:** 9,840 total, 6,324 winners (64.3%)

![Strategy Performance](https://github.com/nijv/DV2_Strat/blob/main/dv2_strategy_performance_numba_strategy.png)

## Technical details

The strategy processes 500+ stocks. The Numba version uses JIT compilation to speed up the DV2 calculation, which involves rolling percentile rankings.

You can backtest individual sectors or the full S&P 500. The visualization tool compares strategy performance against the benchmark and shows drawdown periods.

## Notes

Built with standard Python libraries. The pandas version is straightforward to read and modify. The Numba version replicates the same logic but runs much faster by compiling the math-heavy parts.

Data comes from Yahoo Finance via the yfinance library. S&P 500 constituents are scraped from Wikipedia to stay current.

## Final Considerations

**Overfitting Risk**  
While the strategy can be improved, there is a high risk of overfitting the model to historical data, which may cause it to fail in live trading.

**Survivorship Bias**  
The results are optimistic. This backtest uses present S&P 500's companies, excluding historical companies that have failed or been delisted over the last 20 years.

**Educational Purpose**  
This project was built for educational and training purposes to demonstrate a quantitative research workflow. The results should not be considered representative of live performance.
