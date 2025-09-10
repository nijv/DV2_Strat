#Scrape S&P 500 tickers from Wikipedia and save to JSON file
import requests #HTML scraping
import pandas as pd
import urllib3
import json

# Disable SSL warnings for requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_sp500_tickers():
    """Get S&P 500 tickers with sector information from Wikipedia and save to file."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Add headers to appear like a legitimate browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, verify=False) 
    response.raise_for_status() # Check if request was successful
    
    # Parse the HTML content with pandas
    # Use StringIO to avoid deprecation warning
    from io import StringIO
    tables = pd.read_html(StringIO(response.text))
    
    # The first table contains the current S&P 500 constituents
    sp500_table = tables[0] # first table from the Wiki page
    
    # Extract both ticker symbols and sectors
    tickers_with_sectors = []
    for _, row in sp500_table.iterrows():
        ticker = row['Symbol'].replace('.', '-')  # Replace dots with dashes for yfinance compatibility
        sector = row['GICS Sector']  # GICS Sector column contains the sector information
        
        tickers_with_sectors.append({
            'ticker': ticker,
            'sector': sector
        })
    
    return tickers_with_sectors

tickers_data = get_sp500_tickers()
print(f"Found {len(tickers_data)} tickers with sector information")

# Save tickers with sector data to JSON file
with open('sp500_tickers.json', 'w') as f:
    json.dump(tickers_data, f, indent=2)

print("Tickers with sectors saved to sp500_tickers.json")