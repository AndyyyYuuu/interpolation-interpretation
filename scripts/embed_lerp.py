import argparse
import torch
import tqdm
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from linterp.experiment import INTERP_FNS, EmbedLerp
from scripts.prep_run import prep_run

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", nargs="+", default=None)
    parser.add_argument("--models-path", default=None)
    parser.add_argument("--models-variant", default=None)
    parser.add_argument("--prompt", default="The most common color for{} is")
    parser.add_argument("--tok-0", default=" apples")
    parser.add_argument("--tok-1", default=" bananas")
    parser.add_argument("--classes", nargs="+", default=[" yellow", " green", " red"])
    parser.add_argument("--interp", default="lerp", choices=list(INTERP_FNS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--job-id", default=datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument("--output-root", default="results/runs")
    return parser.parse_args()


args = parse_args()

if (args.model_name is not None) == (args.models_path is not None):
    raise ValueError("Exactly one of --model-name or --models-path must be set")

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
    "experiment": "embed_lerp",
    "job_id": args.job_id,
    "interp": args.interp,
    "model_name": models_to_test,
    "prompt": args.prompt,
    "tok_0": args.tok_0,
    "tok_1": args.tok_1,
    "classes": args.classes,
    "steps": args.steps,
}

run_dir = prep_run(config, output_root=args.output_root)

weights = [i / args.steps for i in range(args.steps + 1)]

for model_name in tqdm.tqdm(models_to_test, desc="models", disable=len(models_to_test) == 1):
    try:
        experiment = EmbedLerp(model_name, args.prompt, args.tok_0, args.tok_1, args.classes, interp_fn=INTERP_FNS[args.interp])

        norm_resid, partial_probs = zip(*[experiment.forward(w) for w in tqdm.tqdm(weights, desc="steps")])
        partial_probs = torch.stack(partial_probs)  # (steps+1, n_classes)
        norm_resid = torch.stack(norm_resid)  # (steps+1, seq_len, hidden)

        model_slug = model_name.replace("/", "_")
        torch.save(
            {"partial_probs": partial_probs, "norm_resid": norm_resid, "weights": weights, "labels": args.classes, "model": model_name},
            run_dir / f"output_{model_slug}.pt",
        )
    except Exception as e: # sorry but I'm just gonna catch all of it
        tqdm.tqdm.write(f"[{model_name}] failed: {e}")
