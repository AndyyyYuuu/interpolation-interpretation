import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from einops import rearrange


def get_model_prop(model: str, prop: str, path: str | Path = "models.csv"):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == model:
                return row[prop]
    raise KeyError(f"{model!r} not found in {path}")


def plot_max_slopes(dir: str | Path, x_prop: str="params_b") -> None:
    paths = sorted(Path(dir).glob("*.pt"))

    if not paths:
        print(f"No .pt files found in {dir}")
        return

    x, y, families = [], [], []
    for p in paths:
        data = torch.load(p, map_location="cpu", weights_only=False)
        probs = data.get("partial_probs", data.get("results"))  # (steps+1, n_classes)
        weights = torch.tensor(data.get("weights"))
        maxdiff = ((probs[:-1] - probs[1:]) / rearrange((weights[:-1] - weights[1:]), "s -> s 1")).abs().max()
        x.append(float(get_model_prop(data["model"], x_prop)))
        y.append(maxdiff)
        families.append(get_model_prop(data["model"], "family"))

    unique_families = list(dict.fromkeys(families))
    cmap = plt.colormaps["tab10"]
    family_color = {f: cmap(i / max(len(unique_families) - 1, 1)) for i, f in enumerate(unique_families)}

    fig, ax = plt.subplots()
    ax.set_xlabel(x_prop)
    ax.set_ylabel("max |slope|")

    for xi, yi, family in zip(x, y, families):
        ax.scatter(xi, yi, s=8, color=family_color[family])

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=family_color[f], markersize=6, label=f) for f in unique_families]
    ax.legend(handles=handles)
    plt.show()

if __name__ == "__main__":
    plot_max_slopes(sys.argv[1], sys.argv[2])


