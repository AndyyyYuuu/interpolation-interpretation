"""
Usage:
  python results/plot.py results/runs/<run_dir>/output.pt
"""

import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

COLORS = ["#ebc026", "#54cc2f", "#f04343", "#a7a9d6"]


def plot_embed(data: dict) -> None:
    results = data["results"]   # (n_steps, n_classes)
    weights = data["weights"]
    labels  = data["labels"]
    model   = data["model"].split("/")[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, label in enumerate(labels):
        ax.plot(weights, results[:, i].numpy(), label=label, color=COLORS[i % len(COLORS)])
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
            line.set_data(weights, results[layer, :, i].numpy())
        ax.set_title(f"{model}  |  Layer {layer + 1}/{n_layers}")
        return lines

    ani = animation.FuncAnimation(fig, update, frames=range(n_layers), interval=interval, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1]
    data = torch.load(path, map_location="cpu")
    results = data["results"]

    if results.dim() == 2:
        plot_embed(data)
    elif results.dim() == 3:
        animate_hidden(data)
    else:
        raise ValueError(f"Unexpected results shape: {results.shape}")
