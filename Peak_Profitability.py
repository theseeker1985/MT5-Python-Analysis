import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import chardet
import os


def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    return chardet.detect(rawdata)['encoding']


def select_files():
    """Create a GUI for selecting the HTML and CSV files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask for HTML file
    messagebox.showinfo("Select Files", "Please select the MT5 Strategy Tester HTML report")
    html_file = filedialog.askopenfilename(
        title="Select HTML Report",
        filetypes=[("HTML files", "*.html"), ("HTM files", "*.htm"), ("All files", "*.*")]
    )

    if not html_file:
        return None, None

    # Ask for CSV file
    messagebox.showinfo("Select Files", "Please select the EURUSD M5 CSV data file")
    csv_file = filedialog.askopenfilename(
        title="Select CSV Data",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    return html_file, csv_file


def parse_html_report(html_file):
    """Parse the MT5 strategy tester HTML report and extract trades information."""
    # Detect file encoding
    encoding = detect_encoding(html_file)
    print(f"Detected encoding: {encoding}")

    # Try reading with detected encoding, fall back to others if needed
    for enc in [encoding, 'utf-16', 'windows-1252', 'iso-8859-1']:
        try:
            with open(html_file, 'r', encoding=enc) as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, 'html.parser')
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"Could not decode file with any supported encoding. Tried: {encoding}, utf-16, windows-1252, iso-8859-1")

    # Debug: Print the tables found
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables in HTML")

    # Find the Deals table - look for the table that has a header row with "Deals" in it
    deals_table = None
    for table in tables:
        # Check if this table has a header row with "Deals"
        for tr in table.find_all('tr'):
            # Look for header that contains "Deals"
            if tr.find('th') and 'Deals' in tr.get_text():
                deals_table = table
                print("Found deals table!")
                break
        if deals_table:
            break

    if not deals_table:
        print("Couldn't find deals table by header, trying alternative approach...")
        # Alternative approach: look for the table with Deal, Symbol, Type columns
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if 'Deal' in headers and 'Symbol' in headers and 'Type' in headers:
                deals_table = table
                print(f"Found deals table by column headers: {headers}")
                break

    if not deals_table:
        # Last attempt - try to find a table with text that looks like deals
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                for cell in cells:
                    if cell.get_text().strip().lower() == 'buy' or cell.get_text().strip().lower() == 'sell':
                        print(f"Found potential deals table (#{i}) with buy/sell text")
                        deals_table = table
                        break
                if deals_table:
                    break
            if deals_table:
                break

    if not deals_table:
        print("WARNING: Deals table still not found! Dumping HTML structure for debugging:")
        # Print table headers for debugging
        for i, table in enumerate(tables):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            print(f"Table #{i} headers: {headers}")
            # Print first row of content
            rows = table.find_all('tr')
            if len(rows) > 1:  # At least header + one data row
                cells = [td.get_text(strip=True) for td in rows[1].find_all('td')]
                print(f"First row data: {cells[:5]}...")

        raise ValueError("Deals table not found in HTML")

    # Extract column headers - find all th elements in the deals table
    header_row = deals_table.find('tr', attrs={'bgcolor': '#E5F0FC'})  # This is the blue header row in MT5 reports

    if not header_row:
        # If we can't find the styled header row, try a different approach
        for row in deals_table.find_all('tr'):
            if row.find('th'):
                header_row = row
                break

    if not header_row:
        print("Couldn't find header row, trying alternative approach...")
        # Last resort: just use the first row that has cells
        for row in deals_table.find_all('tr'):
            if row.find_all('td') or row.find_all('th'):
                header_row = row
                break

    # Try to extract headers from the header row
    headers = []
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        if not headers:
            # If no th elements, try td elements
            headers = [td.get_text(strip=True) for td in header_row.find_all('td')]

    print(f"Found headers: {headers}")

    # Extract deal rows - get all rows after the header
    deals = []
    start_processing = False

    for row in deals_table.find_all('tr'):
        # Skip rows until after the header row
        if not start_processing:
            if row == header_row:
                start_processing = True
            continue

        # Get all cells in this row
        cols = row.find_all('td')
        if not cols:
            continue

        # Create dictionary for this deal
        deal = {}
        for i, col in enumerate(cols):
            if i < len(headers):
                deal[headers[i]] = col.get_text(strip=True)
            else:
                # If we have more columns than headers, use index as key
                deal[f"Column_{i}"] = col.get_text(strip=True)

        # Debug print
        print(f"Processing row: {deal}")

        # Only consider trade deals (buy/sell)
        # Check for type field or direction field
        deal_type = None

        if 'Type' in deal:
            deal_type = deal['Type'].lower()
        elif 'Direction' in deal:
            deal_type = deal['Direction'].lower()

        # Check if this is a trade row (buy/sell)
        if deal_type in ['buy', 'sell', 'in', 'out']:
            deals.append(deal)
            print(f"Added deal: {deal_type}")

    print(f"Total deals found: {len(deals)}")
    return deals


def load_csv_data(csv_file):
    """Load the CSV data into a pandas DataFrame."""
    # Detect file encoding
    encoding = detect_encoding(csv_file)

    # Try reading with detected encoding, fall back to others if needed
    for enc in [encoding, 'utf-8', 'windows-1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(csv_file, sep='\t', encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"Could not decode CSV file with any supported encoding. Tried: {encoding}, utf-8, windows-1252, iso-8859-1")

    # Combine date and time columns into a datetime object
    df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    df = df.drop(['<DATE>', '<TIME>'], axis=1)

    # Rename columns for easier access
    df = df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'tick_volume',
        '<VOL>': 'volume',
        '<SPREAD>': 'spread'
    })

    return df


def analyze_trade_performance(deals, price_data):
    """Analyze each trade's performance including unrealized peak profits."""
    trade_performance = []
    price_data = price_data.set_index('datetime').sort_index()

    # Identify trade entries and exits
    trades = []
    current_trade = None

    for deal in deals:
        if not all(k in deal for k in ['Time', 'Price', 'Type']):
            print(f"Skipping incomplete deal: {deal}")
            continue

        try:
            # Parse deal information
            deal_time = datetime.strptime(deal['Time'], '%Y.%m.%d %H:%M:%S')
            deal_price = float(deal.get('Price', '0').replace(' ', ''))
            deal_type = deal['Type'].lower()
            deal_direction = deal.get('Direction', '').lower()

            # Handle different formats
            is_entry = (deal_direction == 'in') or (deal_type in ['buy', 'sell'] and not current_trade)
            is_exit = (deal_direction == 'out') or (deal_type in ['buy', 'sell'] and current_trade)

            if is_entry:
                current_trade = {
                    'entry_time': deal_time,
                    'entry_price': deal_price,
                    'type': deal_type,
                }
            elif is_exit and current_trade:
                current_trade['exit_time'] = deal_time
                current_trade['exit_price'] = deal_price
                current_trade['profit'] = float(deal.get('Profit', '0').replace(' ', ''))
                trades.append(current_trade)
                current_trade = None

        except Exception as e:
            print(f"Error processing deal: {str(e)}")
            print(f"Deal data: {deal}")
            continue

    # Process each complete trade
    for trade in trades:
        try:
            # Get price data during the trade's lifetime
            trade_data = price_data[(price_data.index >= trade['entry_time']) &
                                    (price_data.index <= trade['exit_time'])]

            if trade_data.empty:
                print(f"No price data found for trade from {trade['entry_time']} to {trade['exit_time']}")
                continue

            # Calculate unrealized profit at each point
            if trade['type'] == 'buy':
                unrealized_pips = (trade_data['high'] - trade['entry_price']) * 10000
                drawdown_pips = (trade['entry_price'] - trade_data['low']) * 10000
            else:  # sell
                unrealized_pips = (trade['entry_price'] - trade_data['low']) * 10000
                drawdown_pips = (trade_data['high'] - trade['entry_price']) * 10000

            # Find peak unrealized profit
            peak_unrealized = unrealized_pips.max()
            peak_time = unrealized_pips.idxmax()
            max_drawdown = drawdown_pips.max()

            # Calculate time to peak
            time_to_peak = (peak_time - trade['entry_time']).total_seconds() / 60  # in minutes

            # Calculate profit potential (peak vs realized)
            realized_pips = trade['profit']
            profit_potential = peak_unrealized - realized_pips if realized_pips < peak_unrealized else 0

            trade_info = {
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'type': trade['type'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'realized_pips': realized_pips,
                'peak_unrealized_pips': peak_unrealized,
                'peak_time': peak_time,
                'max_drawdown_pips': max_drawdown,
                'time_to_peak_min': time_to_peak,
                'profit_potential': profit_potential,
                'duration_min': (trade['exit_time'] - trade['entry_time']).total_seconds() / 60,
                'was_profitable': trade['profit'] > 0
            }

            trade_performance.append(trade_info)

        except Exception as e:
            print(f"Error analyzing trade: {str(e)}")
            continue

    return trade_performance


def generate_performance_report(trade_performance):
    """Generate a detailed report on trade performance."""
    if not trade_performance:
        print("No trades found for analysis.")
        return None

    df = pd.DataFrame(trade_performance)

    print("\n=== Trade Performance Analysis ===")
    print(f"Total trades analyzed: {len(df)}")
    print(f"Profitable trades: {df['was_profitable'].sum()} ({df['was_profitable'].mean() * 100:.1f}%)")

    print("\n=== Profit Potential Analysis ===")
    print("Average realized pips per trade:", df['realized_pips'].mean())
    print("Average peak unrealized pips per trade:", df['peak_unrealized_pips'].mean())
    print("Average unrealized profit potential per trade:", df['profit_potential'].mean())
    print("\nMedian realized pips per trade:", df['realized_pips'].median())
    print("Median peak unrealized pips per trade:", df['peak_unrealized_pips'].median())
    print("Median unrealized profit potential per trade:", df['profit_potential'].median())

    print("\n=== Trade Duration ===")
    print("Average trade duration (minutes):", df['duration_min'].mean())
    print("Average time to peak (minutes):", df['time_to_peak_min'].mean())
    print("Median time to peak (minutes):", df['time_to_peak_min'].median())

    print("\n=== By Trade Type ===")
    print(df.groupby('type').agg({
        'realized_pips': ['mean', 'median'],
        'peak_unrealized_pips': ['mean', 'median'],
        'profit_potential': ['mean', 'median'],
        'was_profitable': 'mean'
    }))

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot peak unrealized vs realized profits
    plt.subplot(1, 2, 1)
    plt.scatter(df['peak_unrealized_pips'], df['realized_pips'], c=df['was_profitable'].map({True: 'g', False: 'r'}),
                alpha=0.6)
    plt.plot([0, df['peak_unrealized_pips'].max()], [0, df['peak_unrealized_pips'].max()], 'k--')
    plt.xlabel('Peak Unrealized Pips')
    plt.ylabel('Realized Pips')
    plt.title('Peak vs Realized Profits')
    plt.grid(True)

    # Plot profit potential distribution
    plt.subplot(1, 2, 2)
    plt.hist(df['profit_potential'], bins=20, color='blue', alpha=0.7)
    plt.xlabel('Unrealized Profit Potential (pips)')
    plt.ylabel('Number of Trades')
    plt.title('Profit Potential Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return df


def suggest_exit_strategy(df):
    """Suggest potential exit strategy improvements based on the data."""
    print("\n=== Exit Strategy Suggestions ===")

    # Calculate optimal take profit based on peak unrealized profits
    profitable_peaks = df[df['peak_unrealized_pips'] > 0]['peak_unrealized_pips']

    if len(profitable_peaks) > 0:
        # Suggest TP based on percentiles of peak profits
        tp_suggestions = {
            '20th percentile': np.percentile(profitable_peaks, 20),
            'Median (50th percentile)': np.percentile(profitable_peaks, 50),
            '80th percentile': np.percentile(profitable_peaks, 80)
        }

        print("\nSuggested Take Profit Levels (based on peak unrealized profits):")
        for k, v in tp_suggestions.items():
            print(f"{k}: {v:.1f} pips")

        # Calculate potential improvement if we took profits at median peak
        median_peak = np.percentile(profitable_peaks, 50)
        potential_improvement = df[df['peak_unrealized_pips'] > median_peak].copy()
        potential_improvement['improvement'] = median_peak - potential_improvement['realized_pips']

        print(f"\nPotential improvement by taking profits at {median_peak:.1f} pips:")
        print(f"Number of trades affected: {len(potential_improvement)}")
        print(f"Average improvement per trade: {potential_improvement['improvement'].mean():.1f} pips")
        print(f"Total improvement: {potential_improvement['improvement'].sum():.1f} pips")

    # Time-based exit suggestions
    print("\nTime-based exit suggestions:")
    print(f"Median time to peak: {df['time_to_peak_min'].median():.1f} minutes")
    print(f"75th percentile time to peak: {df['time_to_peak_min'].quantile(0.75):.1f} minutes")
    print("\nConsider implementing a time-based exit for trades that reach a certain age without hitting TP.")


def main():
    print("MT5 Strategy Tester Analysis Tool")
    print("---------------------------------")

    # Check if arguments were provided (for testing)
    import sys
    if len(sys.argv) > 2:
        html_file = sys.argv[1]
        csv_file = sys.argv[2]
    else:
        # Step 1: Select files through GUI
        html_file, csv_file = select_files()

    if not html_file or not csv_file:
        print("File selection cancelled. Exiting.")
        return

    try:
        # Step 2: Parse the HTML report
        print("\nParsing MT5 strategy tester report...")
        deals = parse_html_report(html_file)
        print(f"Found {len(deals)} deals in the report.")

        # Step 3: Load the CSV market data
        print("Loading market data from CSV...")
        price_data = load_csv_data(csv_file)
        print(f"Loaded {len(price_data)} market data points.")

        # Step 4: Analyze trade performance including unrealized peaks
        print("Analyzing trade performance...")
        trade_performance = analyze_trade_performance(deals, price_data)
        print(f"Analyzed {len(trade_performance)} complete trades.")

        # Step 5: Generate performance report
        df = generate_performance_report(trade_performance)

        # Step 6: Suggest exit strategy improvements
        if df is not None and not df.empty:
            suggest_exit_strategy(df)

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check that you selected the correct files and try again.")


if __name__ == '__main__':
    main()
