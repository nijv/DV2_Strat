import yfinance as yf
import pandas as pd
import json
import os

# Choose start and end dates for data download
# Greater than 200 days of data is needed for indicators (SMA200)
start_date = '2004-01-01'
end_date = '2025-01-15'

def load_tickers():
    # Load S&P 500 tickers from JSON file
    with open('sp500_tickers.json', 'r') as f:
        tickers_data = json.load(f)
    # Extract just the ticker symbols from the structure
    tickers = [item['ticker'] for item in tickers_data]
    return tickers

def download_ohlc_data(tickers, start_date, end_date):
    """Download OHLC data for all tickers and save to CSV files."""
    
    # Create data directory 
    os.makedirs('data', exist_ok=True)
    
    successful_count = 0
    failed_count = 0

    for i, ticker in enumerate(tickers):
        # Download progress
        print(f"\rDownloading: {i+1}/{len(tickers)}", end="", flush=True)
        
        try:
            # Download data for single ticker
            df = yf.download(ticker, start=start_date, end=end_date, 
                             auto_adjust=True, progress=False)
            
            if df.empty:
                failed_count += 1
                continue
                
            
            if df.columns.nlevels > 1:
                # Avoid error while normalizing column names
                df.columns = df.columns.droplevel(1)
            
            # Take only OHLC columns and clean data
            df.columns = df.columns.str.lower() # lowercase columns
            df = df[['open', 'high', 'low', 'close']].dropna() 
            
            if not df.empty:
                # Save to CSV
                df.to_csv(f"data/{ticker}.csv")
                successful_count += 1
            else:
                failed_count += 1
        except Exception:
            failed_count += 1
            continue
    
    print(f"\nDownload complete: {successful_count} successful, {failed_count} failed")
    return successful_count

# Run the download
tickers = load_tickers()
print(f"Found {len(tickers)} tickers")
print("Downloading OHLC data...")

successful_count = download_ohlc_data(tickers, start_date, end_date)

print(f"\nCompleted! Successfully downloaded {successful_count} stocks")
print("Data saved to 'data/' directory")
print("Run 'backtest.py' next to test the strategy")
