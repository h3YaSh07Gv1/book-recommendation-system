import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the results
df = pd.read_csv('model_comparison_results.csv')

# All models ran successfully, so use all data
successful_models = df.copy()

# Clean model names for display
def clean_model_name(name):
    return name.replace('sentence-transformers/', '').replace('-v2', '').replace('-v1', '')

successful_models['model_clean'] = successful_models['model'].apply(clean_model_name)

# Create visualizations directory
Path('visualizations').mkdir(exist_ok=True)

def create_accuracy_metrics_plot():
    """Create bar plot for accuracy metrics (F1, Precision@5, MRR, NDCG)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Accuracy Comparison - Top Metrics', fontsize=16, fontweight='bold')

    # F1 Score
    bars1 = ax1.bar(successful_models['model_clean'], successful_models['weighted_f1'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], alpha=0.8)
    ax1.set_title('Weighted F1 Score', fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0.65, 0.9)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Precision@5
    bars2 = ax2.bar(successful_models['model_clean'], successful_models['precision_at_k'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], alpha=0.8)
    ax2.set_title('Precision@5', fontweight='bold')
    ax2.set_ylabel('Precision')
    ax2.set_ylim(0.63, 0.71)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # MRR
    bars3 = ax3.bar(successful_models['model_clean'], successful_models['mrr'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], alpha=0.8)
    ax3.set_title('Mean Reciprocal Rank (MRR)', fontweight='bold')
    ax3.set_ylabel('MRR')
    ax3.set_ylim(0.96, 1.0)
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # NDCG@5
    bars4 = ax4.bar(successful_models['model_clean'], successful_models['ndcg'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], alpha=0.8)
    ax4.set_title('NDCG@5', fontweight='bold')
    ax4.set_ylabel('NDCG')
    ax4.set_ylim(0.9, 0.96)
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/accuracy_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_metrics_plot():
    """Create bar plot for performance metrics (Load Time, Inference Time)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Load Time
    bars1 = ax1.bar(successful_models['model_clean'], successful_models['load_time'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], alpha=0.8)
    ax1.set_title('Model Load Time', fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_ylim(0, 65)
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

    # Inference Time
    bars2 = ax2.bar(successful_models['model_clean'], successful_models['avg_inference_time'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], alpha=0.8)
    ax2.set_title('Average Inference Time per Query', fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_ylim(0.07, 0.11)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_plot():
    """Create radar plot showing all metrics normalized"""
    # Normalize metrics to 0-1 scale
    metrics_to_plot = ['weighted_f1', 'precision_at_k', 'mrr', 'ndcg']
    normalized_data = successful_models.copy()

    for metric in metrics_to_plot:
        min_val = successful_models[metric].min()
        max_val = successful_models[metric].max()
        normalized_data[metric] = (successful_models[metric] - min_val) / (max_val - min_val)

    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Plot each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    for i, (_, row) in enumerate(normalized_data.iterrows()):
        values = row[metrics_to_plot].tolist()
        values += values[:1]  # Close the loop

        ax.plot(angles, values, 'o-', linewidth=3, label=row['model_clean'],
               color=colors[i], alpha=0.8)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Set labels
    metric_labels = ['F1 Score', 'Precision@5', 'MRR', 'NDCG@5']
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison - Normalized Accuracy Metrics\n(Higher is Better)', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/radar_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_improvement_plot():
    """Create plot showing percentage improvements over baseline"""
    baseline_model = successful_models.iloc[0]  # First model as baseline
    improvements = successful_models.copy()

    metrics_to_compare = ['weighted_f1', 'precision_at_k', 'mrr', 'ndcg']

    for metric in metrics_to_compare:
        baseline_value = baseline_model[metric]
        improvements[f'{metric}_improvement'] = ((successful_models[metric] - baseline_value) / baseline_value) * 100

    fig, ax = plt.subplots(figsize=(12, 8))

    improvement_metrics = [f'{metric}_improvement' for metric in metrics_to_compare]
    metric_labels = ['F1 Score', 'Precision@5', 'MRR', 'NDCG@5']

    x = np.arange(len(successful_models))
    width = 0.2

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    for i, (metric, label) in enumerate(zip(improvement_metrics, metric_labels)):
        bars = ax.bar(x + i*width, improvements[metric], width,
                     label=label, color=colors[i], alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                   height + (0.5 if height > 0 else -1.5),
                   f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold', fontsize=10)

    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Percentage Improvement (%)', fontweight='bold')
    ax.set_title('Model Improvements Over Baseline (all-MiniLM-L6-v2)', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(successful_models['model_clean'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)

    plt.tight_layout()
    plt.savefig('visualizations/improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table():
    """Create a summary table image"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    # Prepare data for table
    table_data = successful_models[['model_clean', 'weighted_f1', 'precision_at_k', 'mrr', 'ndcg', 'load_time', 'avg_inference_time']].copy()
    table_data.columns = ['Model', 'F1 Score', 'Precision@5', 'MRR', 'NDCG@5', 'Load Time (s)', 'Inference Time (s)']

    # Create table
    table = ax.table(cellText=table_data.round(4).values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#f5f5f5'] * len(table_data.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight best values
    for col_idx, col in enumerate(['F1 Score', 'Precision@5', 'MRR', 'NDCG@5']):
        best_idx = table_data[col].idxmax()
        table[(best_idx+1, col_idx+1)].set_facecolor('#FFE66D')

    plt.title('Model Performance Summary Table', fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/model_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating model comparison visualizations...")

    try:
        create_accuracy_metrics_plot()
        print("✓ Created accuracy metrics comparison chart")

        create_performance_metrics_plot()
        print("✓ Created performance metrics comparison chart")

        create_radar_plot()
        print("✓ Created radar plot for accuracy metrics")

        create_improvement_plot()
        print("✓ Created improvement analysis chart")

        create_summary_table()
        print("✓ Created summary table")

        print("\n✅ All visualizations created successfully!")
        print("📁 Check the 'visualizations/' folder for PNG files")

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
