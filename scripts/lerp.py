import argparse
import torch
import tqdm
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from linterp.experiment import INTERP_FNS, EmbedLerp, HiddenLerp
from scripts.prep_run import prep_run

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # which experiments
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--hidden", action="store_true")
    # select models
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", nargs="+")
    group.add_argument("--models-path")
    parser.add_argument("--models-variant", default=None)
    # shared stuff
    parser.add_argument("--classes", nargs="+", default=[" yellow", " green", " red"])
    parser.add_argument("--interp", default="lerp", choices=list(INTERP_FNS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--job-id", default=datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument("--output-root", default="results/runs")
    # embed stuff
    parser.add_argument("--prompt", default="The most common color for{} is")
    parser.add_argument("--tok-0", default=" apples")
    parser.add_argument("--tok-1", default=" bananas")
    # hidden stuff
    parser.add_argument("--str-0", default="The most common color for bananas is")
    parser.add_argument("--str-1", default="The most common color for apples is")
    return parser.parse_args()


args = parse_args()

if not args.embed and not args.hidden:
    raise ValueError("At least one of --embed or --hidden must be set")

if args.model_name is not None:
    models_to_test = args.model_name
else:
    df = pd.read_csv(args.models_path, usecols=["model", "variant"])
    if args.models_variant is not None:
        df = df[df['variant'] == args.models_variant]
        if df.empty:
            print(f"Warning: no models of variant {args.models_variant}")
    models_to_test = df['model'].tolist()

config = {
    "experiment": "lerp",
    "job_id": args.job_id,
    "modes": [m for m in ("embed", "hidden") if getattr(args, m)],
    "interp": args.interp,
    "model_name": models_to_test,
    "prompt": args.prompt,
    "tok_0": args.tok_0,
    "tok_1": args.tok_1,
    "str_0": args.str_0,
    "str_1": args.str_1,
    "classes": args.classes,
    "steps": args.steps,
}

run_dir = prep_run(config, output_root=args.output_root)

weights = [i / args.steps for i in range(args.steps + 1)]
interp_fn = INTERP_FNS[args.interp]

for model_name in tqdm.tqdm(models_to_test, desc="models", disable=len(models_to_test) == 1):
    model_slug = model_name.replace("/", "_")

    if args.embed:
        try:
            experiment = EmbedLerp(model_name, args.prompt, args.tok_0, args.tok_1, args.classes, interp_fn=interp_fn)
            norm_resid, partial_probs = zip(*[experiment.forward(w) for w in tqdm.tqdm(weights, desc="embed")])
            torch.save(
                {"partial_probs": torch.stack(partial_probs), "norm_resid": torch.stack(norm_resid), "weights": weights, "labels": args.classes, "model": model_name},
                run_dir / f"output_{model_slug}_embed.pt",
            )
        except Exception as e:  # sorry but I'm just gonna catch all of it
            tqdm.tqdm.write(f"[{model_name}] embed failed: {e}")

    if args.hidden:
        try:
            experiment = HiddenLerp(model_name, args.str_0, args.str_1, args.classes, interp_fn=interp_fn)
            layer_resids, layer_probs = [], []
            for layer in tqdm.tqdm(range(experiment.n_layers), desc="hidden"):
                norm_resid, partial_probs = zip(*[experiment.forward(w, layer) for w in weights])
                layer_resids.append(torch.stack(norm_resid))
                layer_probs.append(torch.stack(partial_probs))
            torch.save(
                {"partial_probs": torch.stack(layer_probs), "norm_resid": torch.stack(layer_resids), "weights": weights, "labels": args.classes, "model": model_name},
                run_dir / f"output_{model_slug}_hidden.pt",
            )
        except Exception as e:  # sorry but I'm just gonna catch all of it
            tqdm.tqdm.write(f"[{model_name}] hidden failed: {e}")
