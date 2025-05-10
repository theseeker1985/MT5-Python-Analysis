import tkinter as tk
from tkinter import filedialog
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time
import re
import seaborn as sns
import chardet
from scipy import stats
import calendar
from matplotlib.gridspec import GridSpec

# Trading session times (in server time)
SESSION_TIMES = {
    'Asian': (time(0, 0), time(8, 0)),
    'European': (time(8, 0), time(16, 0)),
    'US': (time(16, 0), time(23, 59, 59))
}


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

    settings = {}
    results = {}
    trades = []

    # Find all tables
    tables = soup.find_all('table')

    if not tables:
        return settings, results, trades

    # First table contains settings and results
    first_table = tables[0]

    # Parse settings and results
    current_section = None
    for row in first_table.find_all('tr'):
        cols = row.find_all(['td', 'th'])
        if not cols:
            continue

        # Check for section headers
        if len(cols) == 1 and 'Settings' in cols[0].get_text():
            current_section = 'settings'
            continue
        elif len(cols) == 1 and 'Results' in cols[0].get_text():
            current_section = 'results'
            continue

        # Parse settings
        if current_section == 'settings':
            if len(cols) >= 4 and cols[0].get_text().strip().endswith(':'):
                key = cols[0].get_text().strip()[:-1].strip()
                value = cols[3].get_text().strip()
                settings[key] = value
            elif len(cols) >= 4 and cols[0].get_text().strip() == '' and 'Inputs:' in settings:
                # Handle input parameters
                value = cols[3].get_text().strip()
                if value:
                    if 'Inputs' not in settings:
                        settings['Inputs'] = []
                    settings['Inputs'].append(value)

        # Parse results
        elif current_section == 'results':
            # Results are spread across multiple columns in this format
            if len(cols) >= 4:
                # First metric in row
                key1 = cols[0].get_text().strip().replace(':', '')
                value1 = cols[1].get_text().strip()
                if key1 and value1:
                    results[key1] = value1

                # Second metric in row (if exists)
                if len(cols) >= 7:
                    key2 = cols[4].get_text().strip().replace(':', '')
                    value2 = cols[5].get_text().strip()
                    if key2 and value2:
                        results[key2] = value2

                # Third metric in row (if exists)
                if len(cols) >= 10:
                    key3 = cols[7].get_text().strip().replace(':', '')
                    value3 = cols[8].get_text().strip()
                    if key3 and value3:
                        results[key3] = value3

    # Parse trades from orders table (second table)
    if len(tables) > 1:
        orders_table = tables[1]
        header_row = None

        # Find the orders header row
        for row in orders_table.find_all('tr'):
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

                # Process each trade row
                for row in orders_table.find_all('tr')[orders_table.find_all('tr').index(header_row) + 1:]:
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
            except Exception as e:
                print(f"Error parsing trades: {e}")

    # Clean up numeric values in settings and results
    def clean_numeric(value):
        if isinstance(value, str):
            return value.replace(' ', '').replace(',', '')
        return value

    settings = {k: clean_numeric(v) for k, v in settings.items()}
    results = {k: clean_numeric(v) for k, v in results.items()}

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
            # Try multiple datetime formats
            df[col] = pd.to_datetime(df[col],
                                     format='%Y.%m.%d %H:%M:%S',
                                     errors='coerce')
            # If first format fails, try alternative format
            if df[col].isna().any():
                df[col] = pd.to_datetime(df[col],
                                         format='%Y-%m-%d %H:%M:%S',
                                         errors='coerce')

    # Filter out rows with invalid dates
    df = df[~df['Open Time'].isna()].copy()

    if df.empty:
        print("No valid trades found after date filtering")
        return df

    # Calculate trade duration if we have both open and close times
    if 'Open Time' in df.columns and 'Close Time' in df.columns:
        df['Duration'] = df['Close Time'] - df['Open Time']
        df['Duration Hours'] = df['Duration'].dt.total_seconds() / 3600
        df['Duration Minutes'] = df['Duration'].dt.total_seconds() / 60

    # Determine trade direction (buy/sell)
    if 'Type' in df.columns:
        df['Direction'] = df['Type'].str.lower().str.contains('buy').map({True: 'Buy', False: 'Sell'})

    # Extract profit from comment
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
        df['Abs Profit'] = abs(df['Profit'])

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

    # Add trading session information
    if 'Open Time' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['Open Time']):
            df['Open Time'] = pd.to_datetime(df['Open Time'], errors='coerce')

        # Filter out NaT values again
        df = df[~df['Open Time'].isna()].copy()

        if not df.empty:
            df['Hour'] = df['Open Time'].dt.hour
            df['Day of Week'] = df['Open Time'].dt.day_name()
            df['Month'] = df['Open Time'].dt.month_name()
            df['Day of Month'] = df['Open Time'].dt.day

            # Determine trading session with proper NaT handling
            def get_session(x):
                try:
                    x_time = x.time()
                    for session, (start, end) in SESSION_TIMES.items():
                        if start <= x_time <= end:
                            return session
                    return 'Other'
                except (AttributeError, ValueError):
                    return 'Unknown'

            df['Session'] = df['Open Time'].apply(get_session)

    # Calculate time between order placement and execution
    if all(col in df.columns for col in ['Open Time', 'Close Time']):
        df['Execution Time'] = df['Close Time'] - df['Open Time']
        df['Execution Minutes'] = df['Execution Time'].dt.total_seconds() / 60

    return df


def generate_advanced_plots(df, settings, results):
    if df.empty:
        print("No data available for plotting")
        return

    # Create figure with multiple subplots
    plt.figure(figsize=(24, 30))
    plt.suptitle(f"Advanced Strategy Analysis\n{settings.get('Expert', '')} on {settings.get('Symbol', '')}", y=1.02,
                 fontsize=16)

    # Grid layout
    gs = GridSpec(6, 4, figure=plt.gcf())

    # 1. Equity Curve with Drawdown (only if we have valid datetime and profit data)
    if all(col in df.columns for col in ['Open Time', 'Profit']) and not df['Open Time'].isna().any():
        ax1 = plt.subplot(gs[0, :2])
        df_sorted = df.sort_values('Open Time')
        initial_deposit = float(settings.get('Initial Deposit', '100000').replace(' ', ''))
        df_sorted['Cumulative Profit'] = df_sorted['Profit'].cumsum()
        df_sorted['Equity'] = initial_deposit + df_sorted['Cumulative Profit']
        df_sorted['Peak'] = df_sorted['Equity'].cummax()
        df_sorted['Drawdown'] = (df_sorted['Equity'] - df_sorted['Peak']) / df_sorted['Peak']

        ax1.plot(df_sorted['Open Time'], df_sorted['Equity'], label='Equity')
        ax1.plot(df_sorted['Open Time'], df_sorted['Peak'], 'r--', label='Peak Equity')
        ax1.set_title('Equity Curve with Drawdown', pad=20)
        ax1.grid(True)
        ax1.legend()

        ax1b = ax1.twinx()
        ax1b.plot(df_sorted['Open Time'], df_sorted['Drawdown'] * 100, 'g:', alpha=0.5)
        ax1b.set_ylabel('Drawdown (%)', color='g')
        ax1b.tick_params(axis='y', labelcolor='g')

    # 2. Profit Distribution by Session
    if 'Session' in df.columns and 'Profit' in df.columns:
        ax2 = plt.subplot(gs[0, 2:])
        sns.boxplot(x='Session', y='Profit', data=df, ax=ax2)
        ax2.axhline(0, color='r', linestyle='--')
        ax2.set_title('Profit Distribution by Trading Session', pad=20)
        ax2.grid(True)

    # 3. Trade Duration Analysis
    if 'Duration Hours' in df.columns:
        ax3 = plt.subplot(gs[1, :2])
        sns.histplot(df['Duration Hours'], bins=50, kde=True, ax=ax3)
        ax3.set_title('Trade Duration Distribution (Hours)', pad=20)
        ax3.grid(True)

    # 4. Win Rate by Hour of Day
    if 'Hour' in df.columns and 'Result' in df.columns:
        ax4 = plt.subplot(gs[1, 2:])
        win_rate = df.groupby('Hour')['Result'].apply(lambda x: (x == 'Win').mean() * 100).reset_index()
        sns.barplot(x='Hour', y='Result', data=win_rate, ax=ax4)
        ax4.set_title('Win Rate by Hour of Day', pad=20)
        ax4.set_ylabel('Win Rate (%)')
        ax4.grid(True)

    # 5. Profit by Day of Week
    if 'Day of Week' in df.columns and 'Profit' in df.columns:
        ax5 = plt.subplot(gs[2, :2])
        day_order = list(calendar.day_name)
        profit_by_day = df.groupby('Day of Week')['Profit'].mean().reindex(day_order)
        profit_by_day.plot(kind='bar', ax=ax5)
        ax5.set_title('Average Profit by Day of Week', pad=20)
        ax5.grid(True)

    # 6. Largest Winning/Losing Trades
    if 'Profit' in df.columns:
        ax6 = plt.subplot(gs[2, 2:])
        top_wins = df.nlargest(5, 'Profit')
        top_losses = df.nsmallest(5, 'Profit')

        if not top_wins.empty:
            top_wins['Label'] = top_wins.apply(lambda x: f"{x['Open Time'].date()}\n{x['Profit']:.2f}", axis=1)
            ax6.barh(top_wins['Label'], top_wins['Profit'], color='g', label='Top Wins')

        if not top_losses.empty:
            top_losses['Label'] = top_losses.apply(lambda x: f"{x['Open Time'].date()}\n{x['Profit']:.2f}", axis=1)
            ax6.barh(top_losses['Label'], top_losses['Profit'], color='r', label='Top Losses')

        ax6.axvline(0, color='k')
        ax6.set_title('Top 5 Winning and Losing Trades', pad=20)
        ax6.legend()
        ax6.grid(True)

    # 7. Execution Time Analysis
    if 'Execution Minutes' in df.columns:
        ax7 = plt.subplot(gs[3, :2])
        sns.histplot(df['Execution Minutes'], bins=50, kde=True, ax=ax7)
        ax7.set_title('Order Execution Time Distribution (Minutes)', pad=20)
        ax7.grid(True)

    # 8. Risk/Reward Analysis
    if 'RR' in df.columns:
        ax8 = plt.subplot(gs[3, 2:])
        sns.histplot(df['RR'], bins=30, kde=True, ax=ax8)
        ax8.set_title('Risk/Reward Ratio Distribution', pad=20)
        ax8.grid(True)

    # 9. Profit by Month
    if 'Month' in df.columns and 'Profit' in df.columns:
        ax9 = plt.subplot(gs[4, :2])
        month_order = list(calendar.month_name)[1:]
        profit_by_month = df.groupby('Month')['Profit'].sum().reindex(month_order)
        profit_by_month.plot(kind='bar', ax=ax9)
        ax9.set_title('Profit by Month', pad=20)
        ax9.grid(True)

    # 10. Consecutive Wins/Losses
    if 'Result' in df.columns:
        ax10 = plt.subplot(gs[4, 2:])
        df['Result Binary'] = df['Result'].apply(lambda x: 1 if x == 'Win' else 0)
        df['Streak'] = df['Result Binary'].groupby(
            (df['Result Binary'] != df['Result Binary'].shift()).cumsum()).cumcount() + 1
        df['Streak'] = df['Streak'] * df['Result Binary'].replace({1: 1, 0: -1})
        streaks = df[df['Streak'] != 0].groupby((df['Streak'].shift() != df['Streak']).cumsum()).first()
        streaks['Streak'].value_counts().sort_index().plot(kind='bar', ax=ax10)
        ax10.set_title('Consecutive Wins/Losses Distribution', pad=20)
        ax10.set_xlabel('Streak Length (positive=wins, negative=losses)')
        ax10.grid(True)

    # 11. Profit vs Duration
    if all(col in df.columns for col in ['Duration Hours', 'Profit']):
        ax11 = plt.subplot(gs[5, :2])
        sns.scatterplot(x='Duration Hours', y='Profit', hue='Result', data=df, ax=ax11)
        ax11.axhline(0, color='k', linestyle='--')
        ax11.set_title('Profit vs Trade Duration', pad=20)
        ax11.grid(True)

    # 12. Session Activity
    if 'Session' in df.columns:
        ax12 = plt.subplot(gs[5, 2:])
        session_counts = df['Session'].value_counts()
        session_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax12)
        ax12.set_title('Trade Activity by Session', pad=20)
        ax12.set_ylabel('')

    plt.tight_layout()
    plt.show()


def print_advanced_statistics(df, settings, results):
    print("\n=== STRATEGY SUMMARY ===")
    print(f"Expert: {settings.get('Expert', 'N/A')}")
    print(f"Symbol: {settings.get('Symbol', 'N/A')}")
    print(f"Period: {settings.get('Period', 'N/A')}")
    print(f"Initial Deposit: {settings.get('Initial Deposit', 'N/A')}")

    print("\n=== PERFORMANCE METRICS ===")
    for metric in ['Total Net Profit', 'Gross Profit', 'Gross Loss', 'Profit Factor',
                   'Expected Payoff', 'Total Trades', 'Sharpe Ratio', 'Recovery Factor']:
        if metric in results:
            print(f"{metric}: {results[metric]}")

    if not df.empty:
        print("\n=== ADVANCED STATISTICS ===")

        # Basic stats
        if 'Profit' in df.columns:
            print(f"\nProfit Analysis:")
            print(f"Average Profit: {df['Profit'].mean():.2f}")
            print(f"Median Profit: {df['Profit'].median():.2f}")
            print(f"Standard Deviation: {df['Profit'].std():.2f}")
            print(f"Skewness: {df['Profit'].skew():.2f}")
            print(f"Kurtosis: {df['Profit'].kurtosis():.2f}")

            # Largest wins/losses
            max_win = df['Profit'].max()
            max_loss = df['Profit'].min()
            print(f"\nLargest Winning Trade: {max_win:.2f}")
            print(f"Largest Losing Trade: {max_loss:.2f}")

        if 'Result' in df.columns:
            win_rate = len(df[df['Result'] == 'Win']) / len(df) * 100
            print(f"\nWin Rate: {win_rate:.2f}%")

            # Win/loss by direction
            if 'Direction' in df.columns:
                direction_stats = df.groupby('Direction')['Result'].value_counts(normalize=True).unstack() * 100
                print("\nWin Rate by Direction:")
                print(direction_stats.to_string())

        # Duration stats
        if 'Duration Hours' in df.columns:
            print(f"\nTrade Duration Analysis:")
            print(f"Average Duration: {df['Duration Hours'].mean():.2f} hours")
            print(f"Median Duration: {df['Duration Hours'].median():.2f} hours")
            print(f"Shortest Trade: {df['Duration Hours'].min():.2f} hours")
            print(f"Longest Trade: {df['Duration Hours'].max():.2f} hours")

            # Duration by result
            if 'Result' in df.columns:
                duration_by_result = df.groupby('Result')['Duration Hours'].describe()
                print("\nDuration by Trade Result:")
                print(duration_by_result.to_string())

        # Session analysis
        if 'Session' in df.columns:
            print(f"\nSession Analysis:")
            session_stats = df.groupby('Session').agg({
                'Profit': ['count', 'mean', 'sum'],
                'Result': lambda x: (x == 'Win').mean() * 100
            })
            session_stats.columns = ['Trade Count', 'Avg Profit', 'Total Profit', 'Win Rate %']
            print(session_stats.to_string())

            # Most profitable session
            best_session = session_stats['Total Profit'].idxmax()
            print(f"\nMost Profitable Session: {best_session}")

        # Time analysis
        if 'Hour' in df.columns:
            print(f"\nTime Analysis:")
            hour_stats = df.groupby('Hour').agg({
                'Profit': ['count', 'mean', 'sum'],
                'Result': lambda x: (x == 'Win').mean() * 100
            })
            hour_stats.columns = ['Trade Count', 'Avg Profit', 'Total Profit', 'Win Rate %']
            print("\nPerformance by Hour:")
            print(hour_stats.to_string())

            # Best/worst hours
            best_hour = hour_stats['Total Profit'].idxmax()
            worst_hour = hour_stats['Total Profit'].idxmin()
            print(f"\nMost Profitable Hour: {best_hour}:00")
            print(f"Least Profitable Hour: {worst_hour}:00")

        # Risk/Reward analysis
        if 'RR' in df.columns:
            print(f"\nRisk/Reward Analysis:")
            print(f"Average Risk/Reward: {df['RR'].mean():.2f}")
            print(f"Median Risk/Reward: {df['RR'].median():.2f}")

            # RR by result
            if 'Result' in df.columns:
                rr_by_result = df.groupby('Result')['RR'].describe()
                print("\nRisk/Reward by Trade Result:")
                print(rr_by_result.to_string())

        # Execution time analysis
        if 'Execution Minutes' in df.columns:
            print(f"\nExecution Time Analysis:")
            print(f"Average Execution Time: {df['Execution Minutes'].mean():.2f} minutes")
            print(f"Median Execution Time: {df['Execution Minutes'].median():.2f} minutes")
            print(f"Fastest Execution: {df['Execution Minutes'].min():.2f} minutes")
            print(f"Slowest Execution: {df['Execution Minutes'].max():.2f} minutes")


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

        print_advanced_statistics(df, settings, results)

        if not df.empty:
            generate_advanced_plots(df, settings, results)
        else:
            print("No trade data found to analyze")
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")


if __name__ == "__main__":
    main()
