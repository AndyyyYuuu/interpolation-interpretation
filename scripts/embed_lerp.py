import argparse
import torch
import tqdm
from datetime import datetime
from dotenv import load_dotenv

from linterp.experiment import EmbedLerp
from scripts.prep_run import prep_run

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", nargs="+", default=["meta-llama/Llama-3.2-1B-Instruct"])
    parser.add_argument("--prompt", default="The most common color for{} is")
    parser.add_argument("--tok-0", default=" apples")
    parser.add_argument("--tok-1", default=" bananas")
    parser.add_argument("--classes", nargs="+", default=[" yellow", " green", " red"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--job-id", default=datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument("--output-root", default="results/runs")
    return parser.parse_args()


args = parse_args()

config = {
    "experiment": "embed_lerp",
    "job_id": args.job_id,
    "model_name": args.model_name,
    "prompt": args.prompt,
    "tok_0": args.tok_0,
    "tok_1": args.tok_1,
    "classes": args.classes,
    "steps": args.steps,
}

run_dir = prep_run(config, output_root=args.output_root)

weights = [i / args.steps for i in range(args.steps + 1)]

for model_name in tqdm.tqdm(args.model_name, desc="models", disable=len(args.model_name) == 1):
    try:
        experiment = EmbedLerp(model_name, args.prompt, args.tok_0, args.tok_1, args.classes)

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
