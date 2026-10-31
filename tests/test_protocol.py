"""Checks for leakage, chronology, row alignment and perturbation pairing."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from pdm.core import COLUMNS, SENSORS, CausalFeatures, CorrelationSelector, make_pipeline, training_target
from pdm.experiments import disturb, endpoint_predict


def history():
    rows = []
    for unit in range(1, 11):
        for cycle in range(1, 17):
            row = [unit, cycle, 0, 0, 100]
            row += [unit + cycle * (sensor + 1) / 10 for sensor in range(21)]
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


class ProtocolTests(unittest.TestCase):
    def test_future_cannot_change_prefix_features(self):
        X = history(); transformer = CausalFeatures(window=5).fit(X)
        prefix = X[(X.unit == 2) & (X.cycle <= 8)]
        whole = X[X.unit == 2].copy()
        np.testing.assert_allclose(transformer.transform(prefix), transformer.transform(whole)[:8], equal_nan=True)
        whole.loc[whole.cycle > 8, SENSORS] = 1e9
        np.testing.assert_allclose(transformer.transform(prefix), transformer.transform(whole)[:8], equal_nan=True)

    def test_missing_values_use_past_not_next_engine(self):
        train = history(); transformer = CausalFeatures(window=0).fit(train)
        test = train[train.unit.isin([1, 2])].copy().reset_index(drop=True)
        test.loc[0, "s2"] = np.nan
        test.loc[1:2, "s2"] = np.nan
        test.loc[16, "s2"] = np.nan
        values = transformer.transform(test)
        self.assertEqual(values[0, 1], transformer.medians_["s2"])
        self.assertEqual(values[16, 1], transformer.medians_["s2"])
        self.assertEqual(values[2, 1], transformer.medians_["s2"])

    def test_groups_do_not_overlap(self):
        X = history(); y = training_target(X)
        for a, b in GroupKFold(5).split(X, y, groups=X.unit):
            self.assertFalse(set(X.iloc[a].unit) & set(X.iloc[b].unit))

    def test_validation_does_not_fit_medians_or_constants(self):
        X = history(); X.loc[X.unit <= 8, "s1"] = 7
        train, test = X[X.unit <= 8], X[X.unit > 8]
        pipe = make_pipeline(Ridge(), {"window": 0, "scale": True, "k": "all"})
        pipe.fit(train, training_target(train))
        before = pipe.named_steps["features"].medians_.copy()
        extreme = test.copy(); extreme[SENSORS] += 1e6
        pipe.predict(extreme)
        pd.testing.assert_series_equal(before, pipe.named_steps["features"].medians_)
        self.assertFalse(pipe.named_steps["variance"].get_support()[0])

    def test_row_order_and_engine_identity_excluded(self):
        X = history()
        transformer = CausalFeatures(5).fit(X)
        shuffled = X.sample(frac=1, random_state=5)
        # Causal transformer restores input row order after sorting histories.
        np.testing.assert_allclose(transformer.transform(shuffled), transformer.transform(X)[shuffled.index], equal_nan=True)
        changed = X.copy(); changed.unit += 1000
        np.testing.assert_allclose(transformer.transform(X), transformer.transform(changed), equal_nan=True)
        self.assertNotIn("cycle", transformer.get_feature_names_out())
        self.assertNotIn("unit", transformer.get_feature_names_out())

    def test_correlation_selector_and_rul_cap(self):
        selector = CorrelationSelector(1).fit(np.array([[1, 1], [2, -2], [3, 1]]), [1, 2, 3])
        self.assertEqual(selector.indices_.tolist(), [0])
        X = history(); y = training_target(X, cap=5)
        self.assertEqual(y.max(), 5)
        self.assertEqual(y[15], 0)

    def test_perturbations_reproducible_and_labels_unchanged(self):
        X = history(); std = X[SENSORS].std(ddof=0)
        for kind, rate in [("noise", .1), ("missing", .1), ("block", 0)]:
            a, h1, _ = disturb(X, kind, rate, 42, std, "s2", 5)
            b, h2, _ = disturb(X, kind, rate, 42, std, "s2", 5)
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(h1, h2)
            pd.testing.assert_frame_equal(a[["unit", "cycle"]], X[["unit", "cycle"]])
            np.testing.assert_array_equal(training_target(a), training_target(X))
        missing, _, _ = disturb(X, "missing", .1, 42, std, "s2", 5)
        self.assertEqual(int(missing[SENSORS].isna().sum().sum()), round(.1 * len(X) * 21))


if __name__ == "__main__":
    unittest.main()
