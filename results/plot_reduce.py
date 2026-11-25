"""
Usage:
  python results/plot_reduce.py results/runs/<run_dir>/output.pt
"""

import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def plane_proj(data: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> torch.Tensor:
    assert data.dim() == 2
    assert v1.dim() == v2.dim() == v3.dim() == 1

    e1 = v2 - v1
    e1 = e1 / e1.norm()

    e2 = v3 - v1
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / e2.norm()

    centered = data - v1
    return torch.stack([centered @ e1, centered @ e2], dim=1)  # (steps+1, 2)


# data format for reference:
# {"partial_probs": partial_probs, "norm_resid": norm_resid, "weights": weights, "labels": args.classes, "model": args.model_name}

def graph_plane_proj(path: str, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, v1_label: str="", v2_label: str="", v3_label: str=""):
    data = torch.load(path, map_location="cpu", weights_only=True)
    norm_resid = data["norm_resid"]
    weights = data["weights"]
    is_hidden = norm_resid.dim() == 4

    anchors = plane_proj(torch.stack([v1, v2, v3]).float(), v1, v2, v3).numpy()
    anchor_labels = [v1_label, v2_label, v3_label]

    def plot_anchors(ax):
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="+", s=200, c="red", zorder=5)
        for (x, y), label in zip(anchors, anchor_labels):
            if label:
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

    if is_hidden:
        trajs = norm_resid[:, :, -1, :].float()  # (n_layers, steps+1, hidden)
        n_layers = trajs.shape[0]

        fig, ax = plt.subplots()

        def update(layer):
            ax.clear()
            traj = trajs[layer]
            coords = plane_proj(traj, v1, v2, v3).numpy()
            ax.scatter(coords[:, 0], coords[:, 1], c=weights, cmap="viridis")
            ax.plot(coords[:, 0], coords[:, 1], alpha=0.4)
            plot_anchors(ax)
            ax.set_title(f"{data['model']} — layer {layer}")

        ani = animation.FuncAnimation(fig, update, frames=n_layers, interval=200)
        plt.show()
        return ani
    else:
        traj = norm_resid[:, -1, :].float()  # (steps+1, hidden)
        coords = plane_proj(traj, v1, v2, v3).numpy()

        fig, ax = plt.subplots()
        ax.scatter(coords[:, 0], coords[:, 1], c=weights, cmap="viridis")
        ax.plot(coords[:, 0], coords[:, 1], alpha=0.4)
        plot_anchors(ax)
        ax.set_title(data["model"])
        plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, label="weight")
        plt.show()


def graph_plane_proj_3d(path: str | Path, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, v1_label: str = "", v2_label: str = "", v3_label: str = "", layer: int | None = None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    norm_resid = data["norm_resid"]
    weights = data["weights"]
    is_hidden = norm_resid.dim() == 4

    if is_hidden and layer is None:
        raise ValueError("layer must be specified for hidden experiments")

    traj = (norm_resid[layer, :, -1, :] if is_hidden else norm_resid[:, -1, :]).float()  # (steps+1, hidden)
    v1, v2, v3 = v1.float(), v2.float(), v3.float()

    e1 = v2 - v1
    e1 = e1 / e1.norm()
    e2 = v3 - v1
    e2 = e2 - (e2 @ e1) * e1
    e2 = e2 / e2.norm()

    centered = traj - v1
    coord1 = centered @ e1
    coord2 = centered @ e2
    residual = centered - coord1.unsqueeze(1) * e1 - coord2.unsqueeze(1) * e2

    _, _, V = torch.pca_lowrank(residual, q=1)
    coord3 = residual @ V[:, 0]

    coords = torch.stack([coord1, coord2, coord3], dim=1).numpy()
    anchors = plane_proj(torch.stack([v1, v2, v3]), v1, v2, v3).numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=weights, cmap="viridis", s=3)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], alpha=0.4)
    ax.scatter(anchors[:, 0], anchors[:, 1], [0, 0, 0], marker="+", s=200, c="red", zorder=5)
    for (x, y), label in zip(anchors, [v1_label, v2_label, v3_label]):
        if label:
            ax.text(x, y, 0, label)

    ax.set_title(data["model"] + (f" — layer {layer}" if is_hidden else ""))
    plt.colorbar(sc, ax=ax, label="weight")

    def rotate(frame):
        ax.view_init(azim=frame % 360)

    ani = animation.FuncAnimation(fig, rotate, frames=360, interval=50)
    plt.show()
    return ani


if __name__ == "__main__":
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM

    classes = [" yellow", " green", " red"]
    dir = Path(sys.argv[1])

    config = json.loads((dir / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map="cpu", torch_dtype=torch.float32)
    embed_weight = model.model.embed_tokens.weight.detach()

    def class_vec(cls: str) -> torch.Tensor:
        tok_id = tokenizer(cls, return_tensors="pt").input_ids[0, -1]
        return embed_weight[tok_id]

    v1, v2, v3 = [class_vec(c) for c in classes]
    graph_plane_proj_3d(dir / "output.pt", v1, v2, v3, *classes, layer=0)
