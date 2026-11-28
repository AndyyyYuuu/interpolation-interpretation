"""
Usage:
  python results/plot.py results/runs/<run_dir>/output.pt
"""

import csv
import math
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

COLORS = ["#ebc026", "#54cc2f", "#f04343", "#a7a9d6"]


def get_model_prop(model: str, prop: str, path: str | Path = "models.csv"):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == model:
                return row[prop]
    raise KeyError(f"{model!r} not found in {path}")


def plot_embed(data: dict) -> None:
    results = data["results"]   # (n_steps, n_classes)
    weights = data["weights"]
    labels  = data["labels"]
    model   = data["model"].split("/")[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, label in enumerate(labels):
        ax.plot(weights, results[:, i].float().detach().numpy(), label=label, color=COLORS[i % len(COLORS)])
    ax.set_xlabel("Interpolation weight")
    ax.set_ylabel("Next-token probability")
    ax.set_title(model)
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_hidden(data: dict, interval: int = 150) -> None:
    results = data["results"]   # (n_layers, n_steps, n_classes)
    weights = data["weights"]
    labels  = data["labels"]
    model   = data["model"].split("/")[-1]
    n_layers = results.shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Interpolation weight")
    ax.set_ylabel("Next-token probability")

    lines = [ax.plot([], [], label=label, color=COLORS[i % len(COLORS)])[0] for i, label in enumerate(labels)]
    ax.legend()

    def update(layer):
        for i, line in enumerate(lines):
            line.set_data(weights, results[layer, :, i].detach().numpy())
        ax.set_title(f"{model}  |  Layer {layer + 1}/{n_layers}")
        return lines

    ani = animation.FuncAnimation(fig, update, frames=range(n_layers), interval=interval, blit=False)
    plt.tight_layout()
    plt.show()


def plot_embed_dir(dir: str | Path, cols: int = 3) -> None:
    paths = sorted(Path(dir).glob("*.pt"))
    if not paths:
        print(f"No .pt files found in {dir}")
        return

    dataset = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        try:
            params_b = float(get_model_prop(data["model"], "params_b"))
        except KeyError:
            params_b = 0.0
        dataset.append((params_b, data))
    dataset.sort(key=lambda e: e[0])

    n = len(dataset)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(min(3 * cols, 14), min(2.2 * rows, 7)), squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    for ax, (_, data) in zip(axes, dataset):
        results = data.get("partial_probs", data.get("results"))
        weights = data["weights"]
        labels  = data["labels"]
        model   = data["model"].split("/")[-1]

        for i, label in enumerate(labels):
            ax.plot(weights, results[:, i].float().detach().numpy(), label=label, color=COLORS[i % len(COLORS)])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Interpolation weight", fontsize=7)
        ax.set_ylabel("Probability", fontsize=7)
        ax.set_title(model, fontsize=8)
        ax.tick_params(labelsize=6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=7)
    

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.show()

if __name__ == "__main__":
    path = sys.argv[1]
    if path[-3:] == ".pt":
        data = torch.load(path, map_location="cpu")
        results = data["results"]

        if results.dim() == 2:
            plot_embed(data)
        elif results.dim() == 3:
            animate_hidden(data)
        else:
            raise ValueError(f"Unexpected results shape: {results.shape}")
    else: 
        plot_embed_dir(path, 7)

