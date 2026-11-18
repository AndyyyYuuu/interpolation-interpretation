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
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
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

exp = EmbedLerp(args.model_name, args.prompt, args.tok_0, args.tok_1, args.classes)

weights = [i / args.steps for i in range(args.steps + 1)]
results = torch.stack([exp.forward(w) for w in tqdm.tqdm(weights)])  # (steps+1, n_classes)

torch.save(
    {"results": results, "weights": weights, "labels": args.classes, "model": args.model_name},
    run_dir / "output.pt",
)
