"""Data checks and causal, fold-local sklearn transformers.

Neither engine identifiers nor cycle numbers enter the regressor.  Identifiers
are used only to group histories; cycles determine their chronological order.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", "setting1", "setting2", "setting3"] + SENSORS


def load_history(path):
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 26:
        raise ValueError(f"{path}: expected 26 columns, got {frame.shape[1]}")
    frame.columns = COLUMNS
    frame = frame.apply(pd.to_numeric, errors="raise")
    for col in ["unit", "cycle"]:
        values = frame[col].to_numpy()
        if not np.isfinite(values).all() or np.any(values < 1) or np.any(values != np.floor(values)):
            raise ValueError(f"Invalid positive integer {col} in {path}")
        frame[col] = frame[col].astype(int)
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError(f"Duplicate unit/cycle pair in {path}")
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    for unit, group in frame.groupby("unit", sort=False):
        if not np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError(f"Engine {unit}: cycles must start at 1 and be consecutive")
    # Invalid sensor readings become gaps, filled causally inside each fold.
    frame[SENSORS] = frame[SENSORS].replace([np.inf, -np.inf], np.nan)
    return frame


def training_target(frame, cap=125):
    last = frame.groupby("unit")["cycle"].transform("max")
    return np.minimum(last.to_numpy() - frame.cycle.to_numpy(), cap).astype(float)


def load_test_target(path, frame, cap=125):
    values = pd.read_csv(path, sep=r"\s+", header=None)
    units = np.sort(frame.unit.unique())
    if values.shape != (len(units), 1):
        raise ValueError("RUL file must contain one value per test engine")
    if not np.array_equal(units, np.arange(1, len(units) + 1)):
        raise ValueError("Test engine IDs must be 1..N for NASA RUL row alignment")
    raw = values.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid test RUL values")
    return raw, np.minimum(raw, cap)


def endpoint_positions(frame):
    return frame.groupby("unit", sort=True).tail(1).index.to_numpy()


class CausalFeatures(TransformerMixin, BaseEstimator):
    """Forward fill within engines, training medians for leading gaps.

    A window includes the current measurement and at most window-1 preceding
    measurements. First differences are undefined on the first cycle; these
    gaps are handled by the subsequent fold-local feature imputer.
    """
    def __init__(self, window=0):
        self.window = window

    def fit(self, X, y=None):
        self.medians_ = X[SENSORS].median().fillna(0.0)
        self.all_missing_sensors_ = X[SENSORS].isna().all().to_dict()
        self.feature_names_ = list(SENSORS)
        if self.window:
            self.feature_names_ += [f"{s}_{kind}{self.window}" for kind in ["mean", "std"] for s in SENSORS]
            self.feature_names_ += [f"{s}_diff" for s in SENSORS]
        return self

    def transform(self, X):
        if self.window < 0 or self.window == 1:
            raise ValueError("window must be 0 or >=2")
        if X.duplicated(["unit", "cycle"]).any():
            raise ValueError("Duplicate trajectory samples")
        # Work on a unique positional index, then restore the input row order.
        work = X.reset_index(drop=True).sort_values(["unit", "cycle"])
        clean = work[SENSORS].groupby(work.unit, sort=False).ffill().fillna(self.medians_)
        parts = [clean]
        if self.window:
            groups = clean.groupby(work.unit, sort=False)
            means = groups.transform(lambda s: s.rolling(self.window, min_periods=1).mean())
            stds = groups.transform(lambda s: s.rolling(self.window, min_periods=1).std(ddof=0))
            diffs = groups.diff()
            means.columns = [f"{s}_mean{self.window}" for s in SENSORS]
            stds.columns = [f"{s}_std{self.window}" for s in SENSORS]
            diffs.columns = [f"{s}_diff" for s in SENSORS]
            parts += [means, stds, diffs]
        return pd.concat(parts, axis=1).sort_index().to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


class CorrelationSelector(TransformerMixin, BaseEstimator):
    """Select the top-k absolute Pearson correlations, fitted on training only."""
    def __init__(self, k="all"):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        centered = X - X.mean(axis=0)
        target = np.asarray(y, dtype=float) - np.mean(y)
        denominator = np.linalg.norm(centered, axis=0) * np.linalg.norm(target)
        scores = np.divide(np.abs(centered.T @ target), denominator,
                           out=np.zeros(X.shape[1]), where=denominator > 0)
        self.scores_ = scores
        count = X.shape[1] if self.k == "all" else min(int(self.k), X.shape[1])
        if count < 1:
            raise ValueError("k must select at least one feature")
        ranked = np.argsort(-scores, kind="stable")[:count]
        self.indices_ = np.sort(ranked)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices_]


def make_pipeline(estimator, configuration):
    return Pipeline([
        ("features", CausalFeatures(configuration["window"])),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", CorrelationSelector(configuration["k"])),
        ("scale", StandardScaler() if configuration["scale"] else "passthrough"),
        ("model", estimator),
    ])


def selected_names(fitted):
    names = fitted.named_steps["features"].get_feature_names_out()
    names = names[fitted.named_steps["variance"].get_support()]
    return names[fitted.named_steps["select"].indices_]


def clipped_predict(fitted, X, cap=125):
    return np.clip(fitted.predict(X), 0, cap)
