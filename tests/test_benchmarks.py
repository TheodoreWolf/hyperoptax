"""Integration benchmarks: run every Optimizer through the shared harness.

Any new Optimizer subclass added to ``benchmarks.OPTIMIZERS`` is automatically
exercised here — it must run end-to-end, produce valid regret, and make
progress on an easy problem.
"""

import numpy as np
import pytest
from benchmarks import BENCHMARKS, OPTIMIZERS, run

N_BUDGET = 30
N_SEEDS = 4


@pytest.mark.parametrize("opt_name", list(OPTIMIZERS))
def test_optimizer_runs_and_regret_valid(opt_name):
    """Harness works for the optimizer; regret is finite, >= 0, non-increasing."""
    cls, extra = OPTIMIZERS[opt_name]
    regret, run_sec = run(
        cls, BENCHMARKS["sphere_2d"], n_budget=N_BUDGET, n_seeds=N_SEEDS, **extra
    )
    assert regret.shape == (N_SEEDS, N_BUDGET)
    assert np.all(np.isfinite(regret))
    assert np.all(regret >= 0.0)
    # Regret tracks a running best, so it can only stay flat or drop.
    assert np.all(np.diff(regret, axis=1) <= 1e-6)


@pytest.mark.parametrize("opt_name", list(OPTIMIZERS))
def test_optimizer_makes_progress_on_sphere(opt_name):
    """Every optimizer should clearly beat a single random guess on 2D sphere.

    Mean single-sample regret over the domain is ~16.7; requiring final median
    regret < 5 catches an optimizer that fails to explore while staying robust
    for the high-variance RandomSearch baseline.
    """
    cls, extra = OPTIMIZERS[opt_name]
    regret, _ = run(
        cls, BENCHMARKS["sphere_2d"], n_budget=N_BUDGET, n_seeds=N_SEEDS, **extra
    )
    assert float(np.median(regret[:, -1])) < 5.0


def test_bayesian_beats_random_on_smooth():
    """The GP surrogate should clearly outperform random on a smooth problem."""
    bench = BENCHMARKS["sphere_2d"]
    r_random, _ = run(
        OPTIMIZERS["random"][0], bench, n_budget=N_BUDGET, n_seeds=N_SEEDS
    )
    r_bayes, _ = run(
        OPTIMIZERS["bayesian"][0], bench, n_budget=N_BUDGET, n_seeds=N_SEEDS
    )
    assert np.median(r_bayes[:, -1]) < np.median(r_random[:, -1])
