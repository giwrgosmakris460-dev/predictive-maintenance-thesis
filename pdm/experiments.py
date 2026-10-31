"""Model selection, fixed ablations, robustness and reproducible timing."""
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_info, threadpool_limits

from .core import (SENSORS, make_pipeline, selected_names, clipped_predict,
                   load_history, training_target, load_test_target, endpoint_positions)


def json_write(path, obj):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert), encoding="utf-8")


def csv_write(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def clipped_rmse(y, p, cap=125):
    return rmse(y, np.clip(p, 0, cap))


def clipped_mae(y, p, cap=125):
    return mean_absolute_error(y, np.clip(p, 0, cap))


def metrics(y, p):
    error = np.asarray(p) - np.asarray(y)
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": rmse(y, p),
            "bias_pred_minus_true": float(error.mean()),
            "overestimation_fraction": float((error > 0).mean()),
            "underestimation_fraction": float((error < 0).mean())}


def model_specs(config):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("XGBoost is required. Run: python -m pip install -r requirements.txt") from exc
    seed, threads = config["seed"], config["threads"]
    return {
        "Ridge": Ridge(solver="svd"),
        "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=threads),
        "XGBoost": XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                random_state=seed, n_jobs=threads, device="cpu"),
        "SVR": SVR(kernel="rbf", cache_size=512, max_iter=-1),
        "KNN": KNeighborsRegressor(n_jobs=threads, algorithm="brute"),
    }


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"median_s": float(np.median(values)),
            "iqr_s": float(np.percentile(values, 75) - np.percentile(values, 25))}


def fit_with_times(template, X, y):
    pipe = clone(template)
    start = time.perf_counter()
    Z = pipe[:-1].fit_transform(X, y)
    pre = time.perf_counter() - start
    start_model = time.perf_counter()
    pipe.named_steps["model"].fit(Z, y)
    fit = time.perf_counter() - start_model
    total = time.perf_counter() - start
    return pipe, {"preprocessing_fit_s": pre, "estimator_fit_s": fit, "full_fit_s": total}


def endpoint_predict(pipe, history, cap):
    # Build features from each entire observed history, infer only at its endpoint.
    Z = pipe[:-1].transform(history)
    positions = endpoint_positions(history)
    return np.clip(pipe.named_steps["model"].predict(Z[positions]), 0, cap)


def benchmark(template, train, y, test, config, model):
    # Untimed warm-up fit; all timed repetitions use exactly the same rows.
    warm, _ = fit_with_times(template, train, y)
    endpoint_predict(warm, test, config["rul_cap"])
    fitted, rows = None, []
    for repeat in range(config["timing_repeats"]):
        fitted, duration = fit_with_times(template, train, y)
        rows.append({"model": model, "repeat": repeat, "phase": "fit", **duration})
    Z = fitted[:-1].transform(test)[endpoint_positions(test)]
    fitted.named_steps["model"].predict(Z)
    endpoint_predict(fitted, test, config["rul_cap"])
    for repeat in range(config["timing_repeats"]):
        start = time.perf_counter()
        fitted.named_steps["model"].predict(Z)
        estimator = time.perf_counter() - start
        start = time.perf_counter()
        endpoint_predict(fitted, test, config["rul_cap"])
        full = time.perf_counter() - start
        rows.append({"model": model, "repeat": repeat, "phase": "predict",
                     "estimator_predict_s": estimator, "full_history_predict_s": full})
    summary = {"model": model, "batch_endpoints": len(Z), "input_history_rows": len(test),
               "features_selected": len(selected_names(fitted))}
    for key in ["preprocessing_fit_s", "estimator_fit_s", "full_fit_s",
                "estimator_predict_s", "full_history_predict_s"]:
        values = [r[key] for r in rows if key in r]
        summary.update({f"{key}_{k}": v for k, v in stats(values).items()})
    summary["estimator_ms_per_endpoint"] = summary["estimator_predict_s_median_s"] * 1000 / len(Z)
    summary["full_history_ms_per_endpoint"] = summary["full_history_predict_s_median_s"] * 1000 / len(Z)
    # Full-history time is batch amortization, not online single-cycle latency.
    return fitted, rows, summary


def error_stages(y, p, model):
    stages = [("RUL_0_30", y <= 30), ("RUL_31_60", (y > 30) & (y <= 60)),
              ("RUL_61_125", y > 60)]
    return [{"model": model, "stage": label, "n": int(mask.sum()), **metrics(y[mask], p[mask])}
            for label, mask in stages if mask.any()]


def validation_importance(template, train, y, folds, config, model):
    """Held-out-fold permutation of selected engineered features, not test data.

    Fold means may be optimistic after hyperparameter selection on the same
    folds. These values are descriptive explanations, not independent tests.
    """
    rows = []
    score = make_scorer(clipped_rmse, greater_is_better=False, cap=config["rul_cap"])
    for fold, (a, b) in enumerate(folds):
        fitted = clone(template).fit(train.iloc[a], y[a])
        Z = fitted[:-1].transform(train.iloc[b])
        target = y[b]
        if len(Z) > config["importance_max_rows"]:
            rng = np.random.default_rng(config["seed"] + fold)
            idx = rng.choice(len(Z), config["importance_max_rows"], replace=False)
            Z, target = Z[idx], target[idx]
        result = permutation_importance(fitted.named_steps["model"], Z, target,
            scoring=score, n_repeats=config["importance_repeats"], random_state=config["seed"] + fold,
            n_jobs=1)
        for name, mean, sd in zip(selected_names(fitted), result.importances_mean, result.importances_std):
            rows.append({"model": model, "fold": fold, "feature": name,
                         "RMSE_increase_mean": mean, "RMSE_increase_std": sd, "validation_rows": len(Z)})
    return rows


def disturb(frame, kind, rate, seed, train_std, sensor, block_length):
    rng = np.random.default_rng(seed)
    altered = frame.copy()
    values = altered[SENSORS].to_numpy(copy=True)
    details = []
    if kind == "noise":
        values += rng.normal(size=values.shape) * train_std.to_numpy()[None, :] * rate
    elif kind == "missing":
        count = int(round(rate * values.size))
        indices = rng.choice(values.size, size=count, replace=False)
        values.flat[indices] = np.nan
    elif kind == "block":
        col = SENSORS.index(sensor)
        for unit, indices in altered.groupby("unit", sort=True).indices.items():
            length = min(block_length, len(indices))
            start = int(rng.integers(0, len(indices) - length + 1))
            chosen = indices[start:start + length]
            values[chosen, col] = np.nan
            details.append({"seed": seed, "unit": int(unit), "sensor": sensor,
                            "start_cycle": int(altered.iloc[chosen[0]].cycle), "length": length})
    else:
        raise ValueError(kind)
    altered[SENSORS] = values
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    return altered, digest, details


def robustness(fitted, train, test, y, config, out):
    clean = {name: metrics(y, endpoint_predict(pipe, test, config["rul_cap"]))
             for name, pipe in fitted.items()}
    raw_std = train[SENSORS].std(ddof=0).fillna(0.0)
    csv_write(pd.DataFrame({"sensor": SENSORS, "training_population_std": raw_std.to_numpy()}), out / "noise_training_std.csv")
    scenarios = [(kind, rate) for kind in ["noise", "missing"] for rate in config["perturbation_rates"]]
    scenarios.append(("block", 0.0))
    rows, block_rows = [], []
    for kind, rate in scenarios:
        logging.info("Robustness %s %.3f", kind, rate)
        for seed in config["robustness_seeds"]:
            altered, digest, blocks = disturb(test, kind, rate, seed, raw_std,
                    config["block_sensor"], config["block_length"])
            block_rows.extend(blocks)
            # One realization, reused unchanged for ALL algorithms.
            for model, pipe in fitted.items():
                p = endpoint_predict(pipe, altered, config["rul_cap"])
                m = metrics(y, p)
                rows.append({"model": model, "scenario": kind, "rate": rate, "seed": seed,
                             "input_sha256": digest, **m,
                             "delta_MAE": m["MAE"] - clean[model]["MAE"],
                             "delta_RMSE": m["RMSE"] - clean[model]["RMSE"]})
    frame = pd.DataFrame(rows)
    csv_write(frame, out / "robustness_trials.csv")
    csv_write(pd.DataFrame(block_rows), out / "missing_blocks.csv")
    summary = frame.groupby(["model", "scenario", "rate"])[["MAE", "RMSE", "delta_MAE", "delta_RMSE"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    csv_write(summary, out / "robustness_summary.csv")
    return summary


def plots(test_metrics, predictions, robust, stages, out):
    palette = plt.get_cmap("tab10")
    colors = {model: palette(i) for i, model in enumerate(test_metrics.model)}
    residual_bins = np.linspace(predictions.error_pred_minus_true.min(), predictions.error_pred_minus_true.max(), 21)
    if residual_bins[0] == residual_bins[-1]:
        residual_bins = np.linspace(residual_bins[0] - 1, residual_bins[-1] + 1, 21)
    fig, ax = plt.subplots(figsize=(8, 4))
    test_metrics.set_index("model")[["MAE", "RMSE"]].plot.bar(ax=ax)
    ax.set_ylabel("Error (cycles)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out / "test_errors.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model, g in predictions.groupby("model", sort=False):
        axes[0].scatter(g.true_RUL_capped, g.predicted_RUL, s=10, alpha=.4, label=model, color=colors[model])
        axes[1].hist(g.error_pred_minus_true, bins=residual_bins, histtype="step", label=model, color=colors[model])
    axes[0].plot([0, 125], [0, 125], "k--", lw=1)
    axes[0].set(xlabel="True capped RUL", ylabel="Predicted RUL")
    axes[1].set(xlabel="Predicted minus true RUL", ylabel="Engines"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "predictions_residuals.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind in zip(axes, ["noise", "missing"]):
        for model, g in robust[robust.scenario == kind].groupby("model", sort=False):
            g = g.sort_values("rate")
            ax.errorbar(g.rate * 100, g.delta_RMSE_mean, yerr=g.delta_RMSE_std, marker="o", label=model, color=colors[model])
        ax.set(xlabel=f"{kind} (%)", ylabel="Change in RMSE (cycles)")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "robustness.png", dpi=160); plt.close(fig)


def eda(train, y, test, out):
    for name, frame in [("train", train), ("test", test)]:
        csv_write(frame.groupby("unit").cycle.max().rename("trajectory_cycles").reset_index(), out / f"{name}_trajectory_lengths.csv")
        csv_write(frame[SENSORS].isna().sum().rename("missing_count").rename_axis("sensor").reset_index(), out / f"{name}_missing.csv")
    csv_write(train[SENSORS].describe().T.reset_index(names="sensor"), out / "training_descriptives.csv")
    corr = train[SENSORS].copy(); corr["RUL"] = y
    corr.corr().to_csv(out / "training_correlations.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(y, bins=25); axes[0].set(xlabel="Training capped RUL", ylabel="Rows")
    for unit in train.unit.unique()[:3]:
        g = train[train.unit == unit]
        axes[1].plot(g.cycle, g.s2, label=f"Engine {unit}")
    axes[1].set(xlabel="Cycle", ylabel="Sensor s2"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "training_eda.png", dpi=160); plt.close(fig)


def run(config, root, quick=False):
    root = Path(root)
    out = root / config["output_dir"]
    # Prevent accidental mixing of results from different runs.
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose another output_dir.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "experiment.log", encoding="utf-8")])
    data = root / config["data_dir"]
    paths = {name: data / filename for name, filename in config["files"].items()}
    train, test = load_history(paths["train"]), load_history(paths["test"])
    y = training_target(train, config["rul_cap"])
    if config["strict_fd001"]:
        if train.unit.nunique() != 100 or test.unit.nunique() != 100 or len(train) != 20631 or len(test) != 13096:
            raise ValueError("Data dimensions do not match NASA FD001 (100/100 engines, 20631/13096 rows)")
    models = model_specs(config)
    configs = config["configurations"]
    grids = config["grids"]
    if quick:
        configs = [c for c in configs if c["name"] == "raw_scaled"]
        grids = {name: {param: [values[0]] for param, values in grid.items()} for name, grid in grids.items()}
        logging.warning("QUICK VERIFICATION: one configuration and one candidate per algorithm; not the full study.")
    actual = dict(config, effective_configurations=configs, effective_grids=grids, quick_verification=quick)
    json_write(out / "config_used.json", actual)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in
            ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "joblib", "threadpoolctl"]},
        "threads": config["threads"], "threadpool_info": threadpool_info(),
        "input_files": {name: {"filename": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in paths.items()}}
    try:
        import psutil
        environment["memory"] = {"total_bytes": psutil.virtual_memory().total, "available_bytes": psutil.virtual_memory().available}
    except ImportError:
        if hasattr(os, "sysconf"):
            environment["memory_total_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if Path("/proc/meminfo").exists():
            info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            if "MemAvailable" in info:
                environment["memory_available_bytes"] = int(info["MemAvailable"].strip().split()[0]) * 1024
    if Path("/proc/cpuinfo").exists():
        environment["cpu_model"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                         if line.startswith("model name")), "unknown")
    json_write(out / "environment.json", environment)
    eda(train, y, test, out)
    folds = list(GroupKFold(n_splits=config["cv_folds"]).split(train, y, groups=train.unit))
    split_records = []
    for fold, (a, b) in enumerate(folds):
        a_units, b_units = sorted(train.iloc[a].unit.unique()), sorted(train.iloc[b].unit.unique())
        assert not set(a_units) & set(b_units)
        split_records.append({"fold": fold, "train_units": a_units, "validation_units": b_units})
    json_write(out / "fold_assignments.json", split_records)
    scorers = {"RMSE": make_scorer(clipped_rmse, greater_is_better=False, cap=config["rul_cap"]),
               "MAE": make_scorer(clipped_mae, greater_is_better=False, cap=config["rul_cap"])}
    comparisons, tuning_times, templates, best_params = [], {}, {}, {}
    with threadpool_limits(limits=config["threads"]):
        # No test labels are loaded anywhere during development/selection.
        for model, estimator in models.items():
            searches = {}
            for c in configs:
                logging.info("Tune %s / %s", model, c["name"])
                template = make_pipeline(clone(estimator), c)
                grid = {"model__" + k: v for k, v in grids[model].items()}
                search = GridSearchCV(template, grid, scoring=scorers, refit=False,
                        cv=folds, n_jobs=1, error_score="raise", return_train_score=False)
                start = time.perf_counter(); search.fit(train, y)
                elapsed = time.perf_counter() - start
                tuning_times[model] = tuning_times.get(model, 0) + elapsed
                cv = pd.DataFrame(search.cv_results_)
                csv_write(cv, out / f"grid_{model}_{c['name']}.csv")
                idx = int(np.argmax(search.cv_results_["mean_test_RMSE"]))
                params = search.cv_results_["params"][idx]
                searches[c["name"]] = (template, params)
                comparisons.append({"model": model, "configuration": c["name"], "mode": "retuned",
                    "CV_RMSE": -search.cv_results_["mean_test_RMSE"][idx],
                    "CV_MAE": -search.cv_results_["mean_test_MAE"][idx],
                    "CV_RMSE_fold_std": search.cv_results_["std_test_RMSE"][idx],
                    "search_s": elapsed, "params": json.dumps(params)})
            # Fix estimator hyperparameters chosen with raw scaled sensors.
            anchor = searches["raw_scaled"][1]
            for c in configs:
                logging.info("Fixed ablation %s / %s", model, c["name"])
                template = make_pipeline(clone(estimator), c).set_params(**anchor)
                errors, durations, counts = [], [], []
                for a, b in folds:
                    start = time.perf_counter(); fitted = clone(template).fit(train.iloc[a], y[a])
                    durations.append(time.perf_counter() - start)
                    p = clipped_predict(fitted, train.iloc[b], config["rul_cap"])
                    errors.append(metrics(y[b], p)); counts.append(len(selected_names(fitted)))
                comparisons.append({"model": model, "configuration": c["name"], "mode": "fixed",
                    "CV_RMSE": np.mean([e["RMSE"] for e in errors]),
                    "CV_MAE": np.mean([e["MAE"] for e in errors]),
                    "CV_RMSE_fold_std": np.std([e["RMSE"] for e in errors]),
                    "fold_fit_s_mean": np.mean(durations), "features_mean": np.mean(counts),
                    "params": json.dumps(anchor)})
            candidates = [r for r in comparisons if r["model"] == model and r["mode"] == "retuned"]
            chosen = min(candidates, key=lambda r: r["CV_RMSE"])
            template, params = searches[chosen["configuration"]]
            templates[model] = clone(template).set_params(**params)
            best_params[model] = {"configuration": chosen["configuration"], "parameters": params,
                                  "selected_by": "mean clipped validation RMSE", "CV_RMSE": chosen["CV_RMSE"]}
            csv_write(pd.DataFrame(comparisons), out / "ablation_comparison.csv")
        json_write(out / "best_models.json", best_params)
        importance_rows = []
        for model, template in templates.items():
            logging.info("Validation permutation importance %s", model)
            importance_rows += validation_importance(template, train, y, folds, config, model)
        csv_write(pd.DataFrame(importance_rows), out / "validation_permutation_importance.csv")
        fitted_models, time_rows, timing, native = {}, [], [], []
        for model, template in templates.items():
            logging.info("Final repeated fit and timing %s", model)
            fitted, rows, summary = benchmark(template, train, y, test, config, model)
            fitted_models[model] = fitted; time_rows += rows
            summary["grid_search_total_s"] = tuning_times[model]; timing.append(summary)
            joblib.dump(fitted, out / "models" / f"{model}.joblib")
            estimator = fitted.named_steps["model"]
            names = selected_names(fitted)
            if hasattr(estimator, "coef_"):
                kind = "standardized_coefficient" if fitted.named_steps["scale"] != "passthrough" else "raw_coefficient"
                values = estimator.coef_
            elif hasattr(estimator, "feature_importances_"):
                kind, values = "internal_feature_importance", estimator.feature_importances_
            else:
                continue
            native += [{"model": model, "feature": n, "kind": kind, "value": float(v)} for n, v in zip(names, values)]
        csv_write(pd.DataFrame(time_rows), out / "timing_repetitions.csv")
        csv_write(pd.DataFrame(timing), out / "timing_summary.csv")
        csv_write(pd.DataFrame(native), out / "native_feature_importance.csv")
        # Test RUL is read only now, after all five model choices are frozen.
        raw_y, test_y = load_test_target(paths["rul"], test, config["rul_cap"])
        predictions, results, stages = [], [], []
        endpoints = test.iloc[endpoint_positions(test)]
        for model, fitted in fitted_models.items():
            p = endpoint_predict(fitted, test, config["rul_cap"])
            results.append({"model": model, **metrics(test_y, p), "test_engines": len(test_y),
                            "configuration": best_params[model]["configuration"],
                            "features_selected": len(selected_names(fitted))})
            stages += error_stages(test_y, p, model)
            for unit, cycle, raw, target, pred in zip(endpoints.unit, endpoints.cycle, raw_y, test_y, p):
                predictions.append({"model": model, "unit": unit, "last_cycle": cycle,
                    "true_RUL_original": raw, "true_RUL_capped": target, "predicted_RUL": pred,
                    "error_pred_minus_true": pred - target})
        results, predictions = pd.DataFrame(results), pd.DataFrame(predictions)
        csv_write(results, out / "test_metrics.csv")
        csv_write(predictions, out / "test_predictions.csv")
        csv_write(pd.DataFrame(stages), out / "error_by_RUL_stage.csv")
        robust = robustness(fitted_models, train, test, test_y, config, out)
        plots(results, predictions, robust, stages, out)
    json_write(out / "run_status.json", {"completed": True, "quick_verification": quick,
               "dataset": "NASA C-MAPSS FD001", "validation_unit": "engine",
               "validation_scoring": "all cycles of held-out engines; equal-weight fold means",
               "test_scoring": "one endpoint per engine; capped RUL",
               "maritime_validation_performed": False})
    logging.info("Completed. Output: %s", out)
    print(results.to_string(index=False))
    return out
