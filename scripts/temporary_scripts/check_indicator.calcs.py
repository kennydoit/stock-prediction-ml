import sys
from pathlib import Path
import pandas as pd

# Add database path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'database'))

from database_manager import DatabaseManager

def calculate_obv(df):
    """Calculate On-Balance Volume (OBV) for a DataFrame with 'close' and 'volume' columns."""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    return df

def main():
    symbol = "ACN"
    with DatabaseManager() as db:
        price_data = db.get_stock_prices(symbol)
    if price_data.empty:
        print(f"No data found for {symbol}")
        return

    print("Price data sample:")
    print(price_data[['close', 'volume']].head(10))

    price_data = calculate_obv(price_data)
    print("\nOBV calculation sample:")
    print(price_data[['close', 'volume', 'obv']])

    print("\nOBV stats:")
    print(price_data['obv'].describe())

if __name__ == "__main__":
    main()