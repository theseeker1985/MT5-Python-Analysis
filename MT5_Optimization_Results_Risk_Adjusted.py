'''
Composite Scoring System:
Profit (25% weight)
Profit Factor (25%)
Recovery Factor (20%)
Sharpe Ratio (15%)
Equity DD % (15% negative weight)

Normalization:
Each metric is normalized against its maximum value in the dataset
Drawdown is specially handled (lower values score higher)

Comprehensive Output:
Displays the best parameter set with all key metrics
Shows all strategy parameters used

Saves two files:
Full sorted results with composite scores
Top 10 parameter sets for quick review

Risk-Adjusted Selection:
Balances profitability metrics with risk metrics
Favors parameter sets with strong performance across all dimensions

How to Interpret Results:
The composite score represents the overall quality considering all factors
Higher scores indicate better risk-adjusted performance
The best set will have strong profitability metrics AND reasonable drawdown
'''

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np



def calculate_composite_score(row, metrics):
    """Calculate a composite score considering all important metrics"""
    weights = {
        'Profit': 0.25,
        'Profit Factor': 0.25,
        'Recovery Factor': 0.20,
        'Sharpe Ratio': 0.15,
        'Equity DD %': -0.15  # Negative weight because lower DD is better
    }

    score = 0
    for metric, weight in weights.items():
        try:
            # Normalize each metric (except DD which we handle separately)
            if metric == 'Equity DD %':
                # For DD, we want lower values to score higher
                normalized = 1 / (1 + row[metric] / 100)  # Convert % to decimal and invert
            else:
                # For other metrics, higher is better
                normalized = row[metric] / metrics[metric]['max']

            score += normalized * weight
        except:
            continue

    return score


def main():
    # Set up the file dialog
    root = tk.Tk()
    root.withdraw()

    # Ask user to select the CSV file
    file_path = filedialog.askopenfilename(
        title="Select MT5 Optimization Results CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    try:
        # Read and clean the data
        df = pd.read_csv(file_path)

        # Convert numeric columns
        numeric_cols = ['Profit', 'Profit Factor', 'Recovery Factor',
                        'Sharpe Ratio', 'Equity DD %', 'Result']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate key metrics statistics for normalization
        metrics = {
            'Profit': {'max': df['Profit'].max()},
            'Profit Factor': {'max': df['Profit Factor'].max()},
            'Recovery Factor': {'max': df['Recovery Factor'].max()},
            'Sharpe Ratio': {'max': df['Sharpe Ratio'].max()},
            'Equity DD %': {'max': df['Equity DD %'].max()}
        }

        # Calculate composite score
        df['Composite Score'] = df.apply(calculate_composite_score, axis=1, metrics=metrics)

        # Sort by composite score
        df_sorted = df.sort_values('Composite Score', ascending=False)

        # Get the best parameter set
        best_set = df_sorted.iloc[0]

        # Display results
        print("\nBEST OVERALL PARAMETER SET (Risk-Adjusted)")
        print("=" * 70)
        print(f"Composite Score: {best_set['Composite Score']:.4f}")
        print("-" * 70)
        print(f"Profit: {best_set['Profit']:.2f}")
        print(f"Profit Factor: {best_set['Profit Factor']:.2f}")
        print(f"Recovery Factor: {best_set['Recovery Factor']:.2f}")
        print(f"Sharpe Ratio: {best_set['Sharpe Ratio']:.2f}")
        print(f"Equity DD %: {best_set['Equity DD %']:.2f}%")
        print("\nPARAMETERS:")
        print("-" * 70)

        # Print all non-metric columns (assumed to be parameters)
        param_cols = [col for col in df.columns if col not in [
            'Pass', 'Result', 'Composite Score'] + numeric_cols
                      ]
        for param in param_cols:
            print(f"{param}: {best_set[param]}")

        # Save full results
        save_path = filedialog.asksaveasfilename(
            title="Save Full Analysis Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="MT5_Optimization_Full_Analysis.csv"
        )

        if save_path:
            df_sorted.to_csv(save_path, index=False)
            print(f"\nFull analysis saved to: {save_path}")

            # Save top 10 for quick review
            top10_path = save_path.replace('.csv', '_TOP10.csv')
            df_sorted.head(10).to_csv(top10_path, index=False)
            print(f"Top 10 parameter sets saved to: {top10_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()