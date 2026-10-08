import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simgreeks.core import (PutSpec, greeks_regression, isd_sample, lsm,  # noqa: E402
                            simulate_growth)


def test_epanechnikov_isd_moments():
    X = isd_sample(200_000, 0.0, 1.0, "epanechnikov")
    assert abs(X.mean()) < 1e-12
    assert abs(X.var() - 0.2) < 1e-4          # Var of Epanechnikov on [-1,1] = 1/5
    assert X.min() >= -1 and X.max() <= 1


def test_greeks_regression_recovers_polynomial():
    X = np.linspace(30, 50, 1001)
    Y = 3.0 - 0.5 * (X - 40) + 0.04 * (X - 40) ** 2 + 1e-3 * (X - 40) ** 3
    p, d, g = greeks_regression(X, Y, 40.0, M0=5)
    assert np.allclose([p, d, g], [3.0, -0.5, 0.08], atol=1e-8)


def test_lsm_price_close_to_known_value():
    # Longstaff-Schwartz (2001) table 1 case: S0=K=40, sigma=0.2, T=1 -> ~2.314
    spec = PutSpec(K=40)
    rng = np.random.default_rng(0)
    G = simulate_growth(spec, 100_000, rng)
    res = lsm(spec, 40.0 * G, M_tau=9)
    assert abs(res.Y_naive.mean() - 2.314) < 0.03


def test_selector_constants_literal_and_fg():
    from simgreeks.selector import constants
    # literal (as printed): b = diagonal of Q = mu_{2 nu}; q = M + 1
    lit = constants(9, 2, "literal")
    assert lit["q"] == 10 and np.isclose(lit["b"], 0.2)
    # (A.3) integral int t^(M+1) K*_1 vanishes for nu = 1 (symmetric kernel)
    assert np.isclose(constants(9, 1, "literal")["b_rot"], 0.0)
    # Fan & Gijbels: local linear (p = 1), nu = 0 -> q = 2, a = 1/2, b = mu_2 = 1/3
    fg = constants(1, 0, "fg")
    assert fg["q"] == 2 and np.isclose(fg["a"], 0.5) and np.isclose(fg["b"], 1 / 3)


def test_selector_recovers_known_curvature_without_noise():
    from simgreeks.selector import select_alpha
    X = isd_sample(100_000, 40.0, 10.0, "epanechnikov")
    Y = 1.0 + 0.5 * (X - 40) + 1e-12 * (X - 40) ** 10
    r = select_alpha(X, Y, 40.0, 10.0, "epanechnikov", 9, 2, "literal")
    assert np.isclose(r["beta_q"], 1e-12, rtol=1e-3)
