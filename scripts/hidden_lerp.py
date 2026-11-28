import argparse
import torch
import tqdm
from datetime import datetime
from dotenv import load_dotenv

from linterp.experiment import INTERP_FNS, HiddenLerp
from scripts.prep_run import prep_run

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--str-0", default="The most common color for bananas is")
    parser.add_argument("--str-1", default="The most common color for apples is")
    parser.add_argument("--classes", nargs="+", default=[" yellow", " green", " red"])
    parser.add_argument("--interp", default="lerp", choices=list(INTERP_FNS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--job-id", default=datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument("--output-root", default="results/runs")
    return parser.parse_args()


args = parse_args()

config = {
    "experiment": "hidden_lerp",
    "job_id": args.job_id,
    "interp": args.interp,
    "model_name": args.model_name,
    "str_0": args.str_0,
    "str_1": args.str_1,
    "classes": args.classes,
    "steps": args.steps,
}

run_dir = prep_run(config, output_root=args.output_root)

experiment = HiddenLerp(args.model_name, args.str_0, args.str_1, args.classes, interp_fn=INTERP_FNS[args.interp])

weights = [i / args.steps for i in range(args.steps + 1)]
layer_resids, layer_probs = [], []
for layer in tqdm.tqdm(range(experiment.n_layers), desc="layers"):
    norm_resid, partial_probs = zip(*[experiment.forward(w, layer) for w in weights])
    layer_resids.append(torch.stack(norm_resid))
    layer_probs.append(torch.stack(partial_probs))

norm_resid = torch.stack(layer_resids)    # (n_layers, steps+1, seq_len, hidden)
partial_probs = torch.stack(layer_probs)  # (n_layers, steps+1, n_classes)

torch.save(
    {"partial_probs": partial_probs, "norm_resid": norm_resid, "weights": weights, "labels": args.classes, "model": args.model_name},
    run_dir / "output.pt",
)
