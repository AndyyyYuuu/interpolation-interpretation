import argparse
import torch
import tqdm
from datetime import datetime
from dotenv import load_dotenv

from linterp.experiment import HiddenLerp
from scripts.prep_run import prep_run

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--str-0", default="The most common color for bananas is")
    parser.add_argument("--str-1", default="The most common color for apples is")
    parser.add_argument("--classes", nargs="+", default=[" yellow", " green", " red"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--job-id", default=datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument("--output-root", default="results/runs")
    return parser.parse_args()


args = parse_args()

config = {
    "experiment": "hidden_lerp",
    "job_id": args.job_id,
    "model_name": args.model_name,
    "str_0": args.str_0,
    "str_1": args.str_1,
    "classes": args.classes,
    "steps": args.steps,
}

run_dir = prep_run(config, output_root=args.output_root)

exp = HiddenLerp(args.model_name, args.str_0, args.str_1, args.classes)

weights = [i / args.steps for i in range(args.steps + 1)]
results = torch.stack([
    torch.stack([exp.forward(w, layer) for w in weights])
    for layer in tqdm.tqdm(range(exp.n_layers), desc="layers")
])  # (n_layers, steps+1, n_classes)

torch.save(
    {"results": results, "weights": weights, "labels": args.classes, "model": args.model_name},
    run_dir / "output.pt",
)
