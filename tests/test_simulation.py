from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from avinor.feature_engineering import build_base_index
from avinor.models.delay_cancellation import CancellationEstimator, DelaySampler
from avinor.simulation import run_monte_carlo


def _sample_schedule() -> pd.DataFrame:
    base_time = pd.Timestamp("2025-07-01 08:00:00")
    rows = []
    for idx in range(6):
        rows.append(
            {
                "flight_id": f"FL{idx:03d}",
                "dep_airport_group": "X" if idx % 2 == 0 else "Y",
                "arr_airport_group": "Y" if idx % 2 == 0 else "X",
                "service_type": "REG",
                "std": base_time + pd.Timedelta(hours=idx),
                "sta": base_time + pd.Timedelta(hours=idx, minutes=55),
                "cancelled": 0,
                "atd": base_time + pd.Timedelta(hours=idx, minutes=5),
                "ata": base_time + pd.Timedelta(hours=idx, minutes=50),
            }
        )
    return pd.DataFrame(rows)


def _sample_base() -> pd.DataFrame:
    base_rows = []
    for airport in ("X", "Y"):
        for hour in range(8, 14):
            base_rows.append(
                {
                    "airport_group": airport,
                    "date": pd.Timestamp("2025-07-01"),
                    "hour": hour,
                }
            )
    return build_base_index(pd.DataFrame(base_rows))


class MonteCarloSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = _sample_schedule()
        self.base_index = _sample_base()
        self.delay_sampler = DelaySampler()
        self.delay_sampler.fit(self.schedule)
        self.cancel_model = CancellationEstimator()
        self.cancel_model.fit(self.schedule)

    def test_run_monte_carlo_adaptive(self) -> None:
        result = run_monte_carlo(
            self.schedule,
            self.base_index,
            self.delay_sampler,
            self.cancel_model,
            n_simulations=40,
            random_state=0,
            progress_name=None,
            min_simulations=10,
            max_error=0.05,
            adaptive=True,
            n_jobs=1,
        )

        self.assertIn("sim_prob_any_overlap", result.columns)
        self.assertIn("simulations_used", result.columns)
        self.assertLessEqual(result["simulations_used"].iloc[0], 40)
        self.assertTrue(result.filter(regex="^sim_.*_mean$").shape[1] > 0)

    def test_delay_sampler_uniform_draws(self) -> None:
        subset = self.schedule.head(3)
        uniforms = np.full(len(subset), 0.25, dtype=np.float32)
        rng = np.random.default_rng(123)
        sample_one = self.delay_sampler.sample(subset, "dep", rng, uniform_draws=uniforms)
        rng = np.random.default_rng(123)
        sample_two = self.delay_sampler.sample(subset, "dep", rng, uniform_draws=uniforms)
        np.testing.assert_allclose(sample_one, sample_two)

    def test_cancellation_uniform_draws(self) -> None:
        subset = self.schedule.head(4)
        uniforms = np.linspace(0.1, 0.9, len(subset), dtype=np.float32)
        rng = np.random.default_rng(7)
        canc_one = self.cancel_model.sample(subset, rng, uniform_draws=uniforms)
        rng = np.random.default_rng(7)
        canc_two = self.cancel_model.sample(subset, rng, uniform_draws=uniforms)
        np.testing.assert_array_equal(canc_one, canc_two)


if __name__ == "__main__":  # pragma: no cover - manual execution hook
    unittest.main()
