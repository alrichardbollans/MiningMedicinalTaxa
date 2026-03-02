#!/usr/bin/env python3
# plot_training.py
"""
Plot training loss and learning rate for each available fold.

USAGE:
    python SciBert/plot_training.py

OUTPUT:
    outputs/plots/ner_training_loss.png
    outputs/plots/ner_learning_rate.png
    outputs/plots/re_training_loss.png
    outputs/plots/re_learning_rate.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from SciBert.config import Config


def load_fold_logs(prefix: str) -> dict:
    """Load training logs from CSV for each available fold.
    prefix: 'ner' or 're'
    """
    logs = {}
    for fold in range(1, Config.N_FOLDS + 1):
        path = Config.OUTPUTS / f"{prefix}_training_log_fold{fold}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Keep only rows with loss (excludes final row with train_runtime)
            df = df[df['loss'].notna()].copy()
            logs[fold] = df
    return logs


def plot_loss(logs: dict, prefix: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for fold, df in logs.items():
        ax.plot(df['epoch'], df['loss'], label=f'Fold {fold}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title(f'{prefix.upper()} Training Loss per Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = Config.PLOTS / f"{prefix}_training_loss.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_lr(logs: dict, prefix: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for fold, df in logs.items():
        ax.plot(df['epoch'], df['learning_rate'], label=f'Fold {fold}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{prefix.upper()} Learning Rate Schedule per Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = Config.PLOTS / f"{prefix}_learning_rate.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    for prefix in ('ner', 're'):
        logs = load_fold_logs(prefix)
        if not logs:
            print(f"No {prefix.upper()} training logs found in {Config.OUTPUTS}")
            continue
        print(f"Found {prefix.upper()} logs for folds: {sorted(logs.keys())}")
        plot_loss(logs, prefix)
        plot_lr(logs, prefix)


if __name__ == "__main__":
    main()