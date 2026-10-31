"""Usage: python run_experiment.py --config config.json [--quick]."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RUL regression benchmark for NASA FD001")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--quick", action="store_true", help="one candidate/configuration; verification only")
    parser.add_argument("--output", help="override output_dir; must be new or empty")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if args.output:
        config["output_dir"] = args.output
    if config["cv_folds"] != 5 or config["rul_cap"] != 125:
        raise ValueError("This thesis protocol uses 5 folds and RUL cap 125")
    if "raw_scaled" not in {c["name"] for c in config["configurations"]}:
        raise ValueError("raw_scaled anchor configuration is required")
    if config["threads"] < 1 or config["timing_repeats"] < 1:
        raise ValueError("threads and timing_repeats must be positive")
    # Set before importing NumPy, BLAS or XGBoost.
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = str(config["threads"])
    from pdm.experiments import run
    try:
        run(config, path.parent, quick=args.quick)
    except Exception:
        import logging
        logging.exception("Experiment failed")
        raise


if __name__ == "__main__":
    main()
