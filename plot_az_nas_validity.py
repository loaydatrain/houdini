import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import sys

def plot_validity(csv_path="az_nas_validity_data.csv", output_path="az_nas_validity_plot.png", 
                  invert_metrics=False, color_by_params=False):
    """
    Plot AZ-NAS validity data.
    
    Args:
        csv_path: Path to the CSV file with AZ-NAS data
        output_path: Path to save the plot
        invert_metrics: If True, multiply all AZ-NAS metrics by -1 (except param_count and accuracy)
                       Use this when lower metric values indicate better architectures.
        color_by_params: If True, color markers by param_count using plasma colormap
                        If False, color by classification (blue) vs regression (red)
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run the experiment first.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("CSV is empty.")
        return

    # Invert metrics if requested (so higher = better)
    if invert_metrics:
        print("\n*** INVERTING METRICS (multiplying by -1) ***")
        metrics_to_invert = ['az_nas_score', 'expressivity', 'progressivity', 'trainability',
                            'norm_expr', 'norm_prog', 'norm_train']
        for col in metrics_to_invert:
            if col in df.columns:
                df[col] = -df[col]
        print(f"Inverted columns: {[c for c in metrics_to_invert if c in df.columns]}")

    # Check if we have the new format with constituent metrics
    has_metrics = 'expressivity' in df.columns and 'trainability' in df.columns
    has_progressivity = 'progressivity' in df.columns

    # Separate by metric type: accuracy values in [0,1] are classification, negative values are -RMSE
    df_classification = df[df['accuracy'] >= 0].copy()
    df_regression = df[df['accuracy'] < 0].copy()
    
    print(f"\n=== Dataset Split ===")
    print(f"Classification tasks (accuracy >= 0): {len(df_classification)} samples")
    print(f"Regression tasks (-RMSE < 0):         {len(df_regression)} samples")
    
    # Setup for param_count coloring
    if color_by_params and 'param_count' in df.columns:
        min_params = df['param_count'].min()
        max_params = df['param_count'].max()
        norm = mcolors.Normalize(vmin=min_params, vmax=max_params)
        cmap = cm.plasma
        print(f"Color by param_count: {min_params:,} - {max_params:,}")
    
    # Helper to plot with classification/regression colors
    def dual_scatter(ax, x_col, y_col='accuracy', xlabel='', ylabel='Accuracy / -RMSE', title=''):
        if x_col not in df.columns:
            ax.text(0.5, 0.5, f'Column {x_col} not found', ha='center', va='center', transform=ax.transAxes)
            return
        if len(df_classification) > 0:
            ax.scatter(df_classification[x_col], df_classification[y_col], 
                      alpha=0.7, color='blue', label='Classification (acc)', marker='o', s=50)
        if len(df_regression) > 0:
            ax.scatter(df_regression[x_col], df_regression[y_col], 
                      alpha=0.7, color='red', label='Regression (-RMSE)', marker='x', s=50)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
    
    # Helper to plot with param_count color gradient
    def param_color_scatter(ax, x_col, y_col='accuracy', xlabel='', ylabel='Accuracy / -RMSE', title=''):
        if x_col not in df.columns:
            ax.text(0.5, 0.5, f'Column {x_col} not found', ha='center', va='center', transform=ax.transAxes)
            return None
        
        # Use different markers for classification vs regression, but color by params
        scatter = None
        if len(df_classification) > 0:
            scatter = ax.scatter(df_classification[x_col], df_classification[y_col], 
                                c=df_classification['param_count'], cmap=cmap, norm=norm,
                                alpha=0.8, marker='o', s=60, edgecolors='white', linewidths=0.5)
        if len(df_regression) > 0:
            scatter = ax.scatter(df_regression[x_col], df_regression[y_col], 
                                c=df_regression['param_count'], cmap=cmap, norm=norm,
                                alpha=0.8, marker='s', s=60, edgecolors='white', linewidths=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add shape legend - use 'best' to auto-find least obstructive location
        from matplotlib.lines import Line2D
        legend_elements = []
        if len(df_classification) > 0:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                         markersize=8, label='Classification'))
        if len(df_regression) > 0:
            legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                                         markersize=8, label='Regression'))
        if legend_elements:
            ax.legend(handles=legend_elements, fontsize=7, loc='best', framealpha=0.9)
        
        return scatter
    
    # Track if we need a colorbar
    scatter_for_cbar = None
    
    if has_metrics:
        if has_progressivity:
            # New format with progressivity: 2x4 grid
            fig, axes = plt.subplots(2, 4, figsize=(22 if color_by_params else 20, 10))
            
            if color_by_params:
                # Row 1: Raw metrics
                param_color_scatter(axes[0, 0], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')
                param_color_scatter(axes[0, 1], 'expressivity', xlabel='Expressivity', title='Expressivity vs Performance')
                param_color_scatter(axes[0, 2], 'progressivity', xlabel='Progressivity', title='Progressivity vs Performance')
                param_color_scatter(axes[0, 3], 'trainability', xlabel='Trainability', title='Trainability vs Performance')

                # Row 2: Normalized metrics
                param_color_scatter(axes[1, 0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
                param_color_scatter(axes[1, 1], 'norm_expr', xlabel='Normalized Expressivity', title='Norm Expr vs Performance')
                param_color_scatter(axes[1, 2], 'norm_prog', xlabel='Normalized Progressivity', title='Norm Prog vs Performance')
                scatter_for_cbar = param_color_scatter(axes[1, 3], 'norm_train', xlabel='Normalized Trainability', title='Norm Train vs Performance')
            else:
                # Standard classification/regression colors
                dual_scatter(axes[0, 0], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')
                dual_scatter(axes[0, 1], 'expressivity', xlabel='Expressivity', title='Expressivity vs Performance')
                dual_scatter(axes[0, 2], 'progressivity', xlabel='Progressivity', title='Progressivity vs Performance')
                dual_scatter(axes[0, 3], 'trainability', xlabel='Trainability', title='Trainability vs Performance')
                dual_scatter(axes[1, 0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
                dual_scatter(axes[1, 1], 'norm_expr', xlabel='Normalized Expressivity', title='Norm Expr vs Performance')
                dual_scatter(axes[1, 2], 'norm_prog', xlabel='Normalized Progressivity', title='Norm Prog vs Performance')
                dual_scatter(axes[1, 3], 'norm_train', xlabel='Normalized Trainability', title='Norm Train vs Performance')
        else:
            # Old format without progressivity: 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(17 if color_by_params else 15, 10))
            
            if color_by_params:
                param_color_scatter(axes[0, 0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
                param_color_scatter(axes[0, 1], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')
                param_color_scatter(axes[0, 2], 'expressivity', xlabel='Expressivity', title='Expressivity vs Performance')
                param_color_scatter(axes[1, 0], 'trainability', xlabel='Trainability', title='Trainability vs Performance')
                param_color_scatter(axes[1, 1], 'norm_expr', xlabel='Normalized Expressivity', title='Norm Expr vs Performance')
                scatter_for_cbar = param_color_scatter(axes[1, 2], 'norm_train', xlabel='Normalized Trainability', title='Norm Train vs Performance')
            else:
                dual_scatter(axes[0, 0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
                dual_scatter(axes[0, 1], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')
                dual_scatter(axes[0, 2], 'expressivity', xlabel='Expressivity', title='Expressivity vs Performance')
                dual_scatter(axes[1, 0], 'trainability', xlabel='Trainability', title='Trainability vs Performance')
                dual_scatter(axes[1, 1], 'norm_expr', xlabel='Normalized Expressivity', title='Norm Expr vs Performance')
                dual_scatter(axes[1, 2], 'norm_train', xlabel='Normalized Trainability', title='Norm Train vs Performance')

    else:
        # Old format - simple 2 plots
        fig, axes = plt.subplots(1, 2, figsize=(14 if color_by_params else 12, 5))

        if color_by_params:
            param_color_scatter(axes[0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
            scatter_for_cbar = param_color_scatter(axes[1], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')
        else:
            dual_scatter(axes[0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
            dual_scatter(axes[1], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')

    # Apply tight_layout first, then add colorbar
    plt.tight_layout()
    
    # Add colorbar on the right if needed
    if color_by_params and scatter_for_cbar is not None:
        # Make room on the right for colorbar
        fig.subplots_adjust(right=0.92)
        # Add colorbar axes on the right
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter_for_cbar, cax=cbar_ax)
        cbar.set_label('Parameter Count', fontsize=12)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Print correlation analysis if we have metrics
    if has_metrics:
        def print_correlations(data, label, acc_name="Accuracy"):
            print(f"\n=== Correlation Analysis ({label}) ===")
            print(f"AZ-NAS Score vs {acc_name}: {data['az_nas_score'].corr(data['accuracy']):.3f}")
            print(f"Expressivity vs {acc_name}: {data['expressivity'].corr(data['accuracy']):.3f}")
            if has_progressivity and 'progressivity' in data.columns:
                print(f"Progressivity vs {acc_name}: {data['progressivity'].corr(data['accuracy']):.3f}")
            print(f"Trainability vs {acc_name}: {data['trainability'].corr(data['accuracy']):.3f}")
            print(f"Param Count vs {acc_name}:  {data['param_count'].corr(data['accuracy']):.3f}")
            if has_progressivity:
                print(f"--- Normalized metrics ---")
                print(f"Norm Expr vs {acc_name}:    {data['norm_expr'].corr(data['accuracy']):.3f}")
                if 'norm_prog' in data.columns:
                    print(f"Norm Prog vs {acc_name}:    {data['norm_prog'].corr(data['accuracy']):.3f}")
                print(f"Norm Train vs {acc_name}:   {data['norm_train'].corr(data['accuracy']):.3f}")
        
        print_correlations(df, "ALL DATA - CAUTION: mixes metrics!")
        
        if len(df_classification) > 2:
            print_correlations(df_classification, "CLASSIFICATION ONLY", "Accuracy")
        
        if len(df_regression) > 2:
            print_correlations(df_regression, "REGRESSION ONLY", "-RMSE")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot AZ-NAS validity data")
    parser.add_argument("--csv", default="az_nas_validity_data.csv", help="Path to CSV file")
    parser.add_argument("--output", default="az_nas_validity_plot.png", help="Output plot path")
    parser.add_argument("--invert", action="store_true", 
                        help="Multiply all AZ-NAS metrics by -1 (use when lower=better)")
    parser.add_argument("--color-by-params", action="store_true",
                        help="Color markers by param_count (plasma colormap) instead of task type")
    args = parser.parse_args()
    
    plot_validity(csv_path=args.csv, output_path=args.output, 
                  invert_metrics=args.invert, color_by_params=args.color_by_params)
