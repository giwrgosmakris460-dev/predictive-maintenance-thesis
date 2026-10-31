"""Inference on complete observed FD001 histories using a saved pipeline."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from pdm.core import load_history, endpoint_positions
from pdm.experiments import endpoint_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="26-column file, all observed cycles per engine")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    history = load_history(args.input)
    fitted = joblib.load(args.model)
    endpoints = history.iloc[endpoint_positions(history)]
    result = endpoints[["unit", "cycle"]].copy()
    result["predicted_RUL"] = endpoint_predict(fitted, history, 125)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} engine predictions to {args.output}")


if __name__ == "__main__":
    main()
