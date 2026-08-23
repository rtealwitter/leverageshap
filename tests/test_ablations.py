import numpy as np
import pytest
import scipy.special

from leverageshap.estimators.ablations import RegressionEstimator


@pytest.mark.parametrize("n", [12, 60, 79, 101])
def test_binomial_budget_at_5n_within_2_percent(n):
    """RegressionEstimator.find_constant_for_bernoulli must not
    integer-round the oversampling constant C.

    At budget m = 5n the exact (unrounded) C is structurally close to a
    half-integer (~2.5, independent of n -- see
    run_logs/rerun/ROUNDING_FIX_REPORT.md and the DIAG_REPORT it links), so
    ``self.C = round(C)`` used to bump C from ~2.5 to 3 and inflate the
    realized game-evaluation budget by 13-20% relative to the nominal m =
    5n. This checks the *deterministic* expected sample count implied by
    self.C (the same sum find_constant_for_bernoulli's bisection targets,
    no RNG involved), so it is not flaky and would have failed under the
    old ``self.C = round(C)`` for every n tested here.
    """
    num_samples = 5 * n
    estimator = RegressionEstimator(
        model=None,
        baseline=np.zeros((1, n)),
        explicand=np.ones((1, n)),
        num_samples=num_samples,
        paired_sampling=True,
        leverage_sampling=True,
        bernoulli_sampling=True,
    )
    estimator.find_constant_for_bernoulli()
    nominal_m = estimator.num_samples  # budget after find_constant_for_bernoulli's own evenness adjustment
    expected_samples = sum(
        min(scipy.special.binom(n, s), 2 * estimator.C * estimator.sample_weight(s))
        for s in range(1, n)
    )
    assert expected_samples == pytest.approx(nominal_m, rel=0.02)


def test_C_is_not_rounded_to_an_integer():
    """Directly guard against the regression: C should generally be a
    genuine float, not silently snapped back to an integer value."""
    n = 60
    estimator = RegressionEstimator(
        model=None,
        baseline=np.zeros((1, n)),
        explicand=np.ones((1, n)),
        num_samples=5 * n,
        paired_sampling=True,
        leverage_sampling=True,
        bernoulli_sampling=True,
    )
    estimator.find_constant_for_bernoulli()
    assert estimator.C != round(estimator.C)
