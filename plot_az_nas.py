#!/usr/bin/env python3
"""
Script to create a scatterplot of AZ-NAS score vs accuracy from a CSV file.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_az_nas_vs_accuracy(csv_path: str, output_path: str = None):
    """
    Create two side-by-side scatterplots:
    1. Parameter count vs accuracy
    2. AZ-NAS score vs accuracy
    
    Args:
        csv_path: Path to the CSV file with columns: az_nas_score, param_count, accuracy, program
        output_path: Optional path to save the plot. If None, displays the plot.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # If accuracy is positive, set it to 0
    df['accuracy'] = df['accuracy'].apply(lambda x: 0 if x > 0 else x)
    
    # Create the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Parameter Count vs Accuracy
    scatter1 = ax1.scatter(
        df['param_count'],
        df['accuracy'],
        c=df['az_nas_score'],
        cmap='viridis',
        alpha=0.7,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('AZ-NAS Score', fontsize=12)
    ax1.set_xlabel('Parameter Count', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('Parameter Count vs Accuracy', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AZ-NAS Score vs Accuracy
    scatter2 = ax2.scatter(
        df['az_nas_score'],
        df['accuracy'],
        c=df['param_count'],
        cmap='plasma',
        alpha=0.7,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Parameter Count', fontsize=12)
    ax2.set_xlabel('AZ-NAS Score', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.set_title('AZ-NAS Score vs Accuracy', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Plot AZ-NAS score vs accuracy from CSV')
    parser.add_argument('--csv_path', type=str, default = "az_nas_validity_data.csv", help='Path to the CSV file')
    parser.add_argument('--output', '-o', type=str, default="az_nas_plot.png",
                        help='Output path for the plot (e.g., plot.png). If not provided, displays the plot.')
    args = parser.parse_args()
    
    plot_az_nas_vs_accuracy(args.csv_path, args.output)


if __name__ == '__main__':
    main()
