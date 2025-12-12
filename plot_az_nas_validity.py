import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_validity(csv_path="az_nas_validity_data.csv", output_path="az_nas_validity_plot.png"):
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

    # Check if we have the new format with constituent metrics
    has_metrics = 'expressivity' in df.columns and 'trainability' in df.columns
    has_progressivity = 'progressivity' in df.columns

    # Separate by metric type: accuracy values in [0,1] are classification, negative values are -RMSE
    df_classification = df[df['accuracy'] >= 0].copy()
    df_regression = df[df['accuracy'] < 0].copy()
    
    print(f"\n=== Dataset Split ===")
    print(f"Classification tasks (accuracy >= 0): {len(df_classification)} samples")
    print(f"Regression tasks (-RMSE < 0):         {len(df_regression)} samples")
    
    # Helper to plot both metric types with different colors
    def dual_scatter(ax, x_col, y_col='accuracy', xlabel='', ylabel='Accuracy / -RMSE', title=''):
        if x_col not in df.columns:
            ax.text(0.5, 0.5, f'Column {x_col} not found', ha='center', va='center', transform=ax.transAxes)
            return
        if len(df_classification) > 0:
            ax.scatter(df_classification[x_col], df_classification[y_col], 
                      alpha=0.7, color='blue', label='Classification (acc)', marker='o')
        if len(df_regression) > 0:
            ax.scatter(df_regression[x_col], df_regression[y_col], 
                      alpha=0.7, color='red', label='Regression (-RMSE)', marker='x')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
    
    if has_metrics:
        if has_progressivity:
            # New format with progressivity: 2x4 grid
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # Row 1: Raw metrics
            dual_scatter(axes[0, 0], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')
            dual_scatter(axes[0, 1], 'expressivity', xlabel='Expressivity (log unique patterns)', title='Expressivity vs Performance')
            dual_scatter(axes[0, 2], 'progressivity', xlabel='Progressivity (rank expansion)', title='Progressivity vs Performance')
            dual_scatter(axes[0, 3], 'trainability', xlabel='Trainability (Jacobian stability)', title='Trainability vs Performance')

            # Row 2: Normalized metrics
            dual_scatter(axes[1, 0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
            dual_scatter(axes[1, 1], 'norm_expr', xlabel='Normalized Expressivity', title='Norm Expr vs Performance')
            dual_scatter(axes[1, 2], 'norm_prog', xlabel='Normalized Progressivity', title='Norm Prog vs Performance')
            dual_scatter(axes[1, 3], 'norm_train', xlabel='Normalized Trainability', title='Norm Train vs Performance')
        else:
            # Old format without progressivity: 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Overall metrics
            dual_scatter(axes[0, 0], 'param_count', xlabel='Parameter Count', title='Params vs Performance')
            dual_scatter(axes[0, 1], 'az_nas_score', xlabel='AZ-NAS Score', title='AZ-NAS Score vs Performance')
            dual_scatter(axes[0, 2], 'expressivity', xlabel='Expressivity (unique patterns)', title='Expressivity vs Performance')

            # Row 2: Constituent metrics
            dual_scatter(axes[1, 0], 'trainability', xlabel='Trainability (gradient norm sum)', title='Trainability vs Performance')
            dual_scatter(axes[1, 1], 'norm_expr', xlabel='Normalized Expressivity', title='Norm Expressivity vs Performance')
            dual_scatter(axes[1, 2], 'norm_train', xlabel='Normalized Trainability', title='Norm Trainability vs Performance')

    else:
        # Old format - simple 2 plots with metric type separation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot A: Parameter Count vs Accuracy
        if len(df_classification) > 0:
            axes[0].scatter(df_classification['param_count'], df_classification['accuracy'], 
                           alpha=0.7, color='blue', label='Classification', marker='o')
        if len(df_regression) > 0:
            axes[0].scatter(df_regression['param_count'], df_regression['accuracy'], 
                           alpha=0.7, color='red', label='Regression (-RMSE)', marker='x')
        axes[0].set_xlabel('Parameter Count')
        axes[0].set_ylabel('Accuracy / -RMSE')
        axes[0].set_title('Params vs Performance')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].legend()

        # Plot B: AZ-NAS Score vs Accuracy
        if len(df_classification) > 0:
            axes[1].scatter(df_classification['az_nas_score'], df_classification['accuracy'], 
                           alpha=0.7, color='blue', label='Classification', marker='o')
        if len(df_regression) > 0:
            axes[1].scatter(df_regression['az_nas_score'], df_regression['accuracy'], 
                           alpha=0.7, color='red', label='Regression (-RMSE)', marker='x')
        axes[1].set_xlabel('AZ-NAS Score')
        axes[1].set_ylabel('Accuracy / -RMSE')
        axes[1].set_title('AZ-NAS Score vs Performance')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
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
    plot_validity()
