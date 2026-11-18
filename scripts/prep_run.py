import json
from datetime import datetime
from pathlib import Path


def prep_run(config: dict, output_root: str = "results/runs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    job_id = str(config["job_id"])
    experiment = config["experiment"]
    run_dir = Path(output_root) / f"{timestamp}_{job_id}_{experiment}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {"timestamp": timestamp, **config}
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"writing outputs to {run_dir}")
    return run_dir
