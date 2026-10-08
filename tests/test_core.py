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


def _bs_put(spec):
    from scipy.stats import norm
    S, K, T, s, r, d = spec.S0, spec.K, spec.T, spec.sigma, spec.r, spec.d
    d1 = (np.log(S / K) + (r - d + 0.5 * s * s) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-d * T) * norm.cdf(-d1)
    delta = -np.exp(-d * T) * norm.cdf(-d1)
    gamma = np.exp(-d * T) * norm.pdf(d1) / (S * s * np.sqrt(T))
    theta = (-np.exp(-d * T) * S * norm.pdf(d1) * s / (2 * np.sqrt(T))
             + r * K * np.exp(-r * T) * norm.cdf(-d2) - d * S * np.exp(-d * T) * norm.cdf(-d1))
    return price, delta, gamma, theta


def test_alpha_star_rules():
    import pytest
    from simgreeks.methods import AlgoConfig, alpha_star
    spec = PutSpec(S0=40, sigma=0.2, T=1.0)
    assert AlgoConfig().alpha_star_rule == "selector"
    cfg = AlgoConfig(alpha_star_rule="heuristic")
    assert np.isclose(alpha_star(spec, cfg, None, None, 9, 2), 0.6 * 40 * 0.2)
    capped = alpha_star(PutSpec(S0=40, sigma=1.0, T=4.0), cfg, None, None, 9, 2)
    assert np.isclose(capped, 0.75 * 40)
    assert alpha_star(spec, AlgoConfig(alpha_star_rule="fixed", alpha_star_fixed=5.0),
                      None, None, 9, 2) == 5.0
    with pytest.raises(ValueError):
        AlgoConfig(alpha_star_rule="fixed")
    with pytest.raises(ValueError):
        AlgoConfig(alpha_star_fixed=5.0)
    with pytest.raises(ValueError):
        AlgoConfig(alpha_star_rule="rot")


def test_theta_pde_matches_black_scholes():
    from simgreeks.core import theta_pde
    for spec in (PutSpec(K=40), PutSpec(K=36, sigma=0.4, T=2.0, d=0.03)):
        p, de, g, th = _bs_put(spec)
        theta, ex = theta_pde(spec, p, de, g)
        assert not ex and np.isclose(theta, th, rtol=1e-12)
    theta, ex = theta_pde(PutSpec(K=44), 3.95, -1.0, 0.0)   # below intrinsic 4
    assert ex and theta == 0.0


def test_reference_european_matches_closed_form():
    from simgreeks.reference import put_fd
    spec = PutSpec(K=40, sigma=0.2, T=1.0, d=0.02)
    f = put_fd(spec, "european")
    for q, v in zip(("price", "delta", "gamma", "theta"), _bs_put(spec)):
        assert abs(f[q] - v) < 1e-4, q


def test_reference_bermudan_matches_paper_benchmark():
    # Paper Table 5, K = 40, sigma = 20%, T = 1 (binomial, 50,000 steps)
    from simgreeks.reference import put_fd
    f = put_fd(PutSpec(K=40))
    assert abs(f["price"] - 2.3141) < 2e-4
    assert abs(f["delta"] + 0.4040) < 1e-4
    assert abs(f["gamma"] - 0.0597) < 1e-4


def test_multi_basis_sizes():
    from simgreeks.multi import basis_terms
    assert len(basis_terms(9, 3, 3, 3)) == 65     # 10 + 7*3 + 4*6 + 1*10
    assert len(basis_terms(9, 1, 1, 3)) == 34     # 10 + 9 + 8 + 7
    assert len(basis_terms(9, 0, 1, 3)) == 10


def test_multi_taylor_regression_recovers_derivatives():
    from simgreeks.multi import MultiConfig, greeks_multi, isd_multi
    spec = PutSpec(S0=40.0, sigma=0.2)
    cfg = MultiConfig(N=20_000, groups=(("S", "sigma"),))
    z = isd_multi(spec, cfg, ("S", "sigma"), np.random.default_rng(0))
    x, s = z["S"] - 40.0, z["sigma"] - 0.2
    Y = 2.0 - 0.4 * x + 0.03 * x ** 2 + 15.0 * s + 0.7 * x * s + 4.0 * s ** 2
    g = greeks_multi(spec, cfg, ("S", "sigma"), z, Y)
    expect = dict(price=2.0, delta=-0.4, gamma=0.06, vega=15.0, vanna=0.7, volga=8.0)
    for k, v in expect.items():
        assert np.isclose(g[k], v, rtol=1e-6, atol=1e-8), k
    assert np.isnan(g["rho"])


def test_multi_control_variate_is_exact_for_european():
    from simgreeks.multi import MultiConfig, label
    from simgreeks.reference import bs_put
    spec = PutSpec(K=42, sigma=0.3, T=0.5, d=0.02)
    out = label(spec, MultiConfig(N=20_000, style="european"), 1)
    bs = bs_put(spec)
    for k in ("price", "delta", "gamma", "vega", "rho", "rho_d", "vanna"):
        assert np.isclose(out[k], bs[k], rtol=1e-6, atol=1e-8), k


def test_multi_american_t0_flag():
    from simgreeks.multi import MultiConfig, label
    out = label(PutSpec(K=44, sigma=0.1), MultiConfig(N=20_000, american_t0=True), 1)
    assert out["ex_region"] and out["price"] == 4.0 and out["delta"] == -1.0
    assert out["gamma"] == out["vega"] == out["theta"] == 0.0


def test_reference_bumped_greeks_match_closed_form():
    from simgreeks.reference import bs_put, put_fd_greeks
    spec = PutSpec(K=44, sigma=0.3, T=1.5, r=0.05, d=0.02)
    f, b = put_fd_greeks(spec, "european"), bs_put(spec)
    for k in ("vega", "rho", "rho_d", "vanna", "delta_r", "delta_d"):
        assert abs(f[k] - b[k]) < 1e-3 * max(1.0, abs(b[k])), k
