import tkinter as tk
from tkinter import filedialog
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
import seaborn as sns
import chardet


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    return chardet.detect(rawdata)['encoding']


def parse_html_report(file_path):
    # Detect file encoding
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Extract settings
    settings = {}
    for row in soup.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) >= 4 and cols[0].get_text().strip() in ['Expert:', 'Symbol:', 'Period:', 'Initial Deposit:']:
            key = cols[0].get_text().strip().replace(':', '')
            value = cols[3].get_text().strip()
            settings[key] = value

    # Extract results
    results = {}
    result_rows = soup.find_all('tr')
    for row in result_rows:
        cols = row.find_all('td')
        if len(cols) >= 4 and ':' in cols[0].get_text():
            key = cols[0].get_text().strip().replace(':', '')
            value = cols[3].get_text().strip()
            results[key] = value

    # Extract trades - more robust parsing
    trades = []
    tables = soup.find_all('table')
    for table in tables:
        if 'Orders' in str(table):
            rows = table.find_all('tr')
            # Find header row to determine column positions
            header_row = None
            for row in rows:
                if 'Open Time' in str(row):
                    header_row = row
                    break

            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                try:
                    time_idx = headers.index('Open Time')
                    order_idx = headers.index('Order')
                    symbol_idx = headers.index('Symbol')
                    type_idx = headers.index('Type')
                    volume_idx = headers.index('Volume')
                    price_idx = headers.index('Price')
                    sl_idx = headers.index('S / L') if 'S / L' in headers else -1
                    tp_idx = headers.index('T / P') if 'T / P' in headers else -1
                    close_idx = headers.index('Time') if 'Time' in headers else -1
                    state_idx = headers.index('State')
                    comment_idx = headers.index('Comment')

                    for row in rows[rows.index(header_row) + 1:]:
                        cols = row.find_all('td')
                        if len(cols) >= len(headers):
                            trade = {
                                'Open Time': cols[time_idx].get_text().strip(),
                                'Order': cols[order_idx].get_text().strip(),
                                'Symbol': cols[symbol_idx].get_text().strip(),
                                'Type': cols[type_idx].get_text().strip(),
                                'Volume': cols[volume_idx].get_text().strip().split('/')[0].strip(),
                                'Price': cols[price_idx].get_text().strip(),
                                'S/L': cols[sl_idx].get_text().strip() if sl_idx != -1 else '',
                                'T/P': cols[tp_idx].get_text().strip() if tp_idx != -1 else '',
                                'Close Time': cols[close_idx].get_text().strip() if close_idx != -1 else '',
                                'State': cols[state_idx].get_text().strip(),
                                'Comment': cols[comment_idx].get_text().strip()
                            }
                            trades.append(trade)
                except ValueError as e:
                    print(f"Header parsing error: {e}")

    return settings, results, trades


def analyze_results(settings, results, trades):
    if not trades:
        print("No trades found in the report")
        return pd.DataFrame()

    # Convert trades to DataFrame
    df = pd.DataFrame(trades)

    # Clean and convert data types with error handling
    numeric_cols = ['Volume', 'Price', 'S/L', 'T/P']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(r'[^\d.-]', '', regex=True), errors='coerce')

    # Convert date columns with error handling
    date_cols = ['Open Time', 'Close Time']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calculate trade duration if we have both open and close times
    if 'Open Time' in df.columns and 'Close Time' in df.columns:
        df['Duration'] = df['Close Time'] - df['Open Time']
        df['Duration Hours'] = df['Duration'].dt.total_seconds() / 3600

    # Determine trade direction (buy/sell)
    if 'Type' in df.columns:
        df['Direction'] = df['Type'].str.lower().str.contains('buy').map({True: 'Buy', False: 'Sell'})

    # Try to extract profit from comment
    if 'Comment' in df.columns:
        profit_pattern = r'Profit:\s*([\d.-]+)'
        df['Profit'] = df['Comment'].str.extract(profit_pattern)[0].astype(float)

    # If profit not in comment, attempt to calculate from price movement (simplified)
    if 'Profit' not in df.columns or df['Profit'].isna().all():
        if all(col in df.columns for col in ['Price', 'T/P', 'Volume', 'Direction']):
            df['Profit'] = np.where(
                df['Direction'] == 'Buy',
                (df['T/P'] - df['Price']) * df['Volume'] * 100000,  # Simplified for demo
                (df['Price'] - df['T/P']) * df['Volume'] * 100000  # Simplified for demo
            )

    # Calculate additional metrics
    if 'Profit' in df.columns:
        df['Result'] = np.where(df['Profit'] >= 0, 'Win', 'Loss')

    # Calculate risk/reward if we have stop loss and take profit
    if all(col in df.columns for col in ['S/L', 'T/P', 'Price', 'Direction']):
        df['Risk'] = np.where(
            df['Direction'] == 'Buy',
            df['Price'] - df['S/L'],
            df['S/L'] - df['Price']
        )
        df['Reward'] = np.where(
            df['Direction'] == 'Buy',
            df['T/P'] - df['Price'],
            df['Price'] - df['T/P']
        )
        df['RR'] = df['Reward'] / df['Risk']

    return df


def generate_plots(df, settings, results):
    if df.empty:
        print("No data available for plotting")
        return

    plt.figure(figsize=(18, 20))
    plt.suptitle(f"Advanced Strategy Analysis\n{settings.get('Expert', '')} on {settings.get('Symbol', '')}", y=1.02)

    # 1. Equity Curve
    if all(col in df.columns for col in ['Open Time', 'Profit']):
        plt.subplot(4, 2, 1)
        df_sorted = df.sort_values('Open Time')
        initial_deposit = float(settings.get('Initial Deposit', '100000').replace(' ', ''))
        df_sorted['Cumulative Profit'] = df_sorted['Profit'].cumsum()
        df_sorted['Equity'] = initial_deposit + df_sorted['Cumulative Profit']
        plt.plot(df_sorted['Open Time'], df_sorted['Equity'])
        plt.title('Equity Curve')
        plt.grid(True)

    # 2. Profit Distribution
    if 'Profit' in df.columns:
        plt.subplot(4, 2, 2)
        sns.histplot(df['Profit'], bins=30, kde=True)
        plt.axvline(0, color='r', linestyle='--')
        plt.title('Profit Distribution')
        plt.grid(True)

    # 3. Win/Loss by Direction
    if all(col in df.columns for col in ['Direction', 'Result']):
        plt.subplot(4, 2, 3)
        win_loss = df.groupby(['Direction', 'Result']).size().unstack()
        win_loss.plot(kind='bar', stacked=True)
        plt.title('Win/Loss by Trade Direction')
        plt.grid(True)

    # 4. Profit by Hour
    if 'Open Time' in df.columns and 'Profit' in df.columns:
        plt.subplot(4, 2, 4)
        df['Hour'] = df['Open Time'].dt.hour
        hour_profit = df.groupby('Hour')['Profit'].mean()
        hour_profit.plot(kind='bar')
        plt.title('Average Profit by Hour')
        plt.grid(True)

    # 5. Duration vs Profit
    if all(col in df.columns for col in ['Duration Hours', 'Profit']):
        plt.subplot(4, 2, 5)
        plt.scatter(df['Duration Hours'], df['Profit'], alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.title('Duration vs Profit')
        plt.grid(True)

    # 6. Drawdown
    if 'Equity' in locals().get('df_sorted', pd.DataFrame()).columns:
        plt.subplot(4, 2, 6)
        df_sorted['Peak'] = df_sorted['Equity'].cummax()
        df_sorted['Drawdown'] = (df_sorted['Equity'] - df_sorted['Peak']) / df_sorted['Peak']
        plt.plot(df_sorted['Open Time'], df_sorted['Drawdown'] * 100)
        plt.title('Drawdown Over Time')
        plt.grid(True)

    # 7. Risk/Reward
    if 'RR' in df.columns:
        plt.subplot(4, 2, 7)
        df['RR'].hist(bins=20)
        plt.title('Risk/Reward Ratio')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def print_summary_statistics(df, settings, results):
    print("\n=== Strategy Summary ===")
    print(f"Expert: {settings.get('Expert', 'N/A')}")
    print(f"Symbol: {settings.get('Symbol', 'N/A')}")
    print(f"Period: {settings.get('Period', 'N/A')}")
    print(f"Initial Deposit: {settings.get('Initial Deposit', 'N/A')}")

    print("\n=== Performance Metrics ===")
    for metric in ['Total Net Profit', 'Gross Profit', 'Gross Loss', 'Profit Factor',
                   'Expected Payoff', 'Total Trades', 'Sharpe Ratio']:
        if metric in results:
            print(f"{metric}: {results[metric]}")

    if not df.empty:
        print("\n=== Calculated Statistics ===")
        if 'Profit' in df.columns:
            print(f"Average Profit: {df['Profit'].mean():.2f}")
            print(f"Median Profit: {df['Profit'].median():.2f}")

        if 'Result' in df.columns:
            win_rate = len(df[df['Result'] == 'Win']) / len(df) * 100
            print(f"Win Rate: {win_rate:.2f}%")

        if 'RR' in df.columns:
            print(f"Average Risk/Reward: {df['RR'].mean():.2f}")

        if 'Duration Hours' in df.columns:
            print(f"Average Duration: {df['Duration Hours'].mean():.2f} hours")


def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select MT5 Strategy Tester HTML Report",
        filetypes=[("HTML Files", "*.html;*.htm"), ("All Files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Analyzing file: {file_path}")

    try:
        settings, results, trades = parse_html_report(file_path)
        df = analyze_results(settings, results, trades)

        print_summary_statistics(df, settings, results)

        if not df.empty:
            generate_plots(df, settings, results)
        else:
            print("No trade data found to analyze")
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")


if __name__ == "__main__":
    main()