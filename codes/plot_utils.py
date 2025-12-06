import os
import matplotlib.pyplot as plt
import time
from pathlib import Path
from datetime import datetime


def plot_losses(train_losses, val_losses=None, title="Training Loss", save_path=None, ylim:list=None):
    """
    Plot training loss and optional validation loss over epochs, with optional saving to file.
    
    Parameters
    ----------
    train_losses : list or array
        Training loss values per epoch.
    val_losses : list or array, optional
        Validation loss values per epoch.
    title : str, default="Training Loss"
        Title of the plot.
    save_path : str or Path, optional
        If given, saves the plot to this location.
    """

    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label='Train', color='tab:blue')

    if val_losses is not None:
        plt.plot(val_losses, label='Validation', color='tab:orange')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title, fontsize=12
              # , fontweight='bold'
             )
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

