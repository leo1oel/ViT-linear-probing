import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def plot_training_history(history: Dict, save_path: str):
    """Create a beautiful training history plot with loss and accuracy metrics.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    # Set the style
    sns.set_theme(style="darkgrid")
    
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot loss on primary y-axis
    color = sns.color_palette()[0]
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(history['epochs'], history['train_loss'], color=color, label='Train Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()
    color = sns.color_palette()[1]
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12)
    ax2.plot(history['epochs'], [acc * 100 for acc in history['train_acc']], 
            color=color, label='Train Accuracy', linestyle='--', marker='s')
    ax2.plot(history['epochs'], [acc * 100 for acc in history['val_acc']], 
            color=sns.color_palette()[2], label='Val Accuracy', linestyle='--', marker='^')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title
    plt.title('Training History', pad=20, fontsize=14, fontweight='bold')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()