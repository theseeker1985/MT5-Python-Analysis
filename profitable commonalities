# Analyses the profitable results from mt5 optimization (csv) and shows common attributes and % of frequency
# Uses standard export, so unless it DOESN'T start with 'Pass', 'Result', 'Profit', 'Expected Payoff', 'Profit Factor',
#'Recovery Factor', 'Sharpe Ratio', 'Custom', 'Equity DD %', 'Trades' you do not need to change it


import pandas as pd
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog


def select_csv_file():
    """Open a file dialog to select the CSV file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Backtest Results CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path


def analyze_profitable_parameters(csv_file):
    """Analyze the most frequent parameter values among profitable results"""
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Filter for profitable results (Profit > 0)
        profitable_df = df[df['Profit'] > 0]

        if profitable_df.empty:
            print("No profitable results found in the data.")
            return None

        # Parameters to analyze (excluding metrics and identifiers)
        exclude_params = ['Pass', 'Result', 'Profit', 'Expected Payoff', 'Profit Factor',
                          'Recovery Factor', 'Sharpe Ratio', 'Custom', 'Equity DD %', 'Trades']
        params_to_analyze = [col for col in df.columns if col not in exclude_params]

        # Dictionary to store frequency counts for each parameter
        param_frequencies = defaultdict(dict)

        # Calculate frequencies for each parameter among profitable results
        for param in params_to_analyze:
            value_counts = profitable_df[param].value_counts().to_dict()
            param_frequencies[param] = value_counts

        # Find the most frequent value for each parameter
        most_frequent_params = {}
        for param, frequencies in param_frequencies.items():
            if frequencies:  # Only if there are values
                most_frequent = max(frequencies.items(), key=lambda x: x[1])
                most_frequent_params[param] = {
                    'most_frequent_value': most_frequent[0],
                    'frequency': most_frequent[1],
                    'percentage': round(most_frequent[1] / len(profitable_df) * 100, 2)
                }

        # Sort parameters alphabetically for clean output
        sorted_params = sorted(most_frequent_params.items(), key=lambda x: x[0])

        # Display results in the requested format
        print("\n" + "=" * 80)
        print(f"Parameter Analysis for {len(profitable_df)} Profitable Tests (out of {len(df)} total)")
        print("Most frequent parameter values with occurrence percentage:")
        print("=" * 80)

        max_param_length = max(len(param) for param, _ in sorted_params)

        for param, data in sorted_params:
            value_str = str(data['most_frequent_value'])
            print(f"{param.ljust(max_param_length)}: {value_str.ljust(20)} ({data['percentage']}% of profitable tests)")

        print("\nAdditional Statistics:")
        print(f"- Total profitable tests: {len(profitable_df)}")
        print(f"- Percentage profitable: {round(len(profitable_df) / len(df) * 100, 2)}%")
        print("=" * 80)

        return most_frequent_params

    except Exception as e:
        print(f"\nError processing file: {e}")
        return None


if __name__ == "__main__":
    print("MT5 Backtest Parameter Analyzer")
    print("Please select your backtest results CSV file...")

    csv_file = select_csv_file()

    if csv_file:
        print(f"\nAnalyzing file: {csv_file}")
        results = analyze_profitable_parameters(csv_file)

        # Option to save results to file
        if results:
            save_option = input("\nWould you like to save these results to a file? (y/n): ").lower()
            if save_option == 'y':
                output_file = csv_file.replace('.csv', '_analysis.txt')
                with open(output_file, 'w') as f:
                    f.write("Parameter Analysis Results\n")
                    f.write("=" * 50 + "\n")
                    for param, data in sorted(results.items()):
                        f.write(f"{param}: {data['most_frequent_value']} ({data['percentage']}%)\n")
                print(f"\nResults saved to: {output_file}")
    else:
        print("No file selected. Exiting.")
