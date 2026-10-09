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
    assert AlgoConfig().selector_order == "nu+1"
    with pytest.raises(ValueError):
        AlgoConfig(selector_order="M")
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
    cfg = MultiConfig(N=20_000)
    z = isd_multi(spec, cfg, {"S": 5.0, "sigma": 0.05}, np.random.default_rng(0))
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
    for k in ("price", "delta", "gamma", "vega", "volga", "rho", "phi", "vanna"):
        assert np.isclose(out[k], bs[k], rtol=1e-6, atol=1e-8), k
    assert "vera" not in out                       # no default group has sigma and r
    groups = (("S", "sigma", "r"),)
    out = label(spec, MultiConfig(N=20_000, style="european", groups=groups), 1)
    assert np.isclose(out["vera"], bs["vera"], rtol=1e-6, atol=1e-8)


def test_multi_exercise_region_label():
    from simgreeks.multi import MultiConfig, label
    spec = PutSpec(K=44, sigma=0.1)                # continuation 3.947 < K - S0 = 4
    out = label(spec, MultiConfig(N=20_000), 1)
    assert out["ex_region"] and out["price"] == 4.0 and out["delta"] == -1.0
    assert out["gamma"] == out["vega"] == out["volga"] == out["theta"] == 0.0
    out = label(spec, MultiConfig(N=20_000, american_t0=False), 1)
    assert out["ex_region"] and 3.9 < out["price"] < 4.0


def test_reference_bumped_greeks_match_closed_form():
    from simgreeks.reference import bs_put, put_fd_greeks
    spec = PutSpec(K=44, sigma=0.3, T=1.5, r=0.05, d=0.02)
    f, b = put_fd_greeks(spec, "european"), bs_put(spec)
    for k in ("vega", "rho", "phi", "vanna", "vera", "volga", "delta_r", "delta_d"):
        assert abs(f[k] - b[k]) < 1e-3 * max(1.0, abs(b[k])), k


def test_multi_isd_design():
    from simgreeks.multi import MultiConfig, isd_multi
    spec = PutSpec(sigma=0.2, r=0.05)
    cfg = MultiConfig(N=50_000)
    a = {"S": 4.8, "sigma": 0.05, "r": 0.02}
    z = isd_multi(spec, cfg, a, np.random.default_rng(0))
    # S: Epanechnikov (variance alpha^2 / 5); parameters: uniform (variance alpha^2 / 3)
    assert abs(np.var(z["S"]) / a["S"] ** 2 - 0.2) < 2e-3
    for k in ("sigma", "r"):
        assert abs(np.var(z[k]) / a[k] ** 2 - 1 / 3) < 2e-3
        assert np.max(np.abs(z[k] - getattr(spec, k))) <= a[k]
    assert np.all(z["d"] == spec.d)
    assert len(isd_multi(spec, cfg, a, np.random.default_rng(0), N=1000)["S"]) == 1000


def test_multi_pilot_widths():
    from simgreeks.multi import MultiConfig, fixed_widths, pilot_widths
    spec = PutSpec(sigma=0.2, T=1.0)
    cfg = MultiConfig()
    assert pilot_widths(spec, cfg, ("S", "sigma")) == {"S": 10.0, "sigma": 0.6 * 0.2}
    assert np.isclose(pilot_widths(spec, cfg, ("S", "r"))["r"], 1.3 * 0.02)
    fixed = MultiConfig(widths="fixed")
    assert pilot_widths(spec, fixed, ("S", "sigma")) == {
        "S": fixed_widths(spec, fixed, ("S",))["S"], "sigma": 1.3 * 0.25 * 0.2}


def test_multi_taylor_regression_recovers_vera():
    from simgreeks.multi import MultiConfig, greeks_multi, isd_multi
    spec = PutSpec(S0=40.0, sigma=0.2, r=0.06)
    cfg = MultiConfig(N=20_000)
    z = isd_multi(spec, cfg, {"S": 5.0, "sigma": 0.05, "r": 0.02}, np.random.default_rng(1))
    x, s, r = z["S"] - 40.0, z["sigma"] - 0.2, z["r"] - 0.06
    Y = 2.0 - 0.4 * x + 15.0 * s - 9.0 * r + 3.0 * s * r + 0.7 * x * s + 4.0 * s ** 2
    g = greeks_multi(spec, cfg, ("S", "sigma", "r"), z, Y)
    expect = dict(price=2.0, delta=-0.4, vega=15.0, rho=-9.0, vera=3.0, vanna=0.7, volga=8.0)
    for k, v in expect.items():
        assert np.isclose(g[k], v, rtol=1e-6, atol=1e-8), k
    assert np.isnan(g["phi"])


def test_multi_partial_residual():
    from simgreeks.multi import MultiConfig, isd_multi, partial_residual, taylor_fit
    spec = PutSpec(S0=40.0, sigma=0.2)
    cfg = MultiConfig()
    z = isd_multi(spec, cfg, {"S": 5.0, "sigma": 0.05}, np.random.default_rng(2), N=20_000)
    x, s = z["S"] - 40.0, z["sigma"] - 0.2
    f_S, f_sigma = 2.0 - 0.4 * x + 0.03 * x ** 2, 15.0 * s + 4.0 * s ** 2
    Y = f_S + f_sigma + 0.7 * x * s
    fit = taylor_fit(spec, cfg, z, Y)
    assert np.allclose(partial_residual(fit, Y, "S"), f_S, atol=1e-8)
    assert np.allclose(partial_residual(fit, Y, "sigma"), 2.0 + f_sigma, atol=1e-8)


def test_multi_select_widths_on_a_known_curve():
    # Exact premium-like curve with a quartic term in S and in sigma: the
    # selector must return widths inside the caps and smaller where the
    # quartic is larger.
    from simgreeks.multi import MultiConfig, isd_multi, pilot_widths, select_widths
    spec = PutSpec(S0=40.0, sigma=0.2)
    cfg = MultiConfig()
    a0 = pilot_widths(spec, cfg, ("S", "sigma"))
    rng = np.random.default_rng(4)
    z = isd_multi(spec, cfg, a0, rng, N=50_000)
    x, s = z["S"] - 40.0, z["sigma"] - 0.2
    noise = 0.05 * rng.standard_normal(50_000)
    widths = []
    for c4 in (1e-5, 1e-3):
        Y = 0.1 * x + 0.01 * x ** 2 + c4 * x ** 4 + 5.0 * s + c4 * 1e8 * s ** 4 + noise
        widths.append(select_widths(spec, cfg, z, Y, a0))
    for w in widths:
        assert 0 < w["S"] <= a0["S"] and 0 < w["sigma"] <= a0["sigma"] / cfg.pilot_widen
    assert widths[1]["S"] < widths[0]["S"] and widths[1]["sigma"] < widths[0]["sigma"]


def test_multi_exercise_rules_out_of_sample():
    from simgreeks.multi import MultiConfig, brownian, isd_multi, lsm_multi
    spec = PutSpec(K=40, T=0.5)
    cfg = MultiConfig(N=20_000)
    rng = np.random.default_rng(3)
    a0, a = {"S": 10.0, "sigma": 0.12}, {"S": 4.0, "sigma": 0.05}
    zp = isd_multi(spec, cfg, a0, rng)
    _, Yp, rules = lsm_multi(spec, cfg, zp, brownian(spec, cfg.N, rng), a0,
                             value_function=False)
    assert Yp is None and set(rules) == set(range(1, spec.J))
    z = isd_multi(spec, cfg, a, rng)
    W = brownian(spec, cfg.N, rng)
    Yn_out, Yvf_out, used = lsm_multi(spec, cfg, z, W, a0, rules)
    assert used is not rules and all(used[j] is rules[j] for j in rules)
    assert np.all(np.isfinite(Yn_out)) and np.all(np.isfinite(Yvf_out))
    _, _, fits = lsm_multi(spec, cfg, z, W, a)               # in-sample: own fits
    assert not np.allclose(fits[10][0], rules[10][0])


def test_multi_label_widths():
    from simgreeks.multi import MultiConfig, label
    spec = PutSpec(K=40, T=0.5)
    out = label(spec, MultiConfig(N=20_000), 5)
    assert 0 < out["alpha_S"] <= 10.0 and 0 < out["alpha_sigma"] <= 0.6 * 0.2 / 1.3
    assert out["alpha_r"] == out["alpha_d"] == 0.02
    fixed = label(spec, MultiConfig(N=20_000, widths="fixed"), 5)
    assert np.isclose(fixed["alpha_S"], 0.6 * 40 * 0.2 * np.sqrt(0.5))
    assert np.isclose(fixed["alpha_sigma"], 0.25 * 0.2)


def test_multi_config_validation():
    import pytest
    from simgreeks.multi import MultiConfig
    for bad in (dict(exercise_rule="oos"), dict(param_kernel="gauss"), dict(widths="auto"),
                dict(select=("r",)), dict(c_sigma=0.8, pilot_widen=1.3),
                dict(c0_sigma=1.0), dict(pilot_widen=0.9)):
        with pytest.raises(ValueError):
            MultiConfig(**bad)
    MultiConfig(c_sigma=0.8, pilot_widen=1.2)


def test_selector_scales_with_the_units_of_S():
    # (A.3) as printed is not scale-equivariant; with x_unit the selector runs
    # in fixed units, so alpha* scales exactly with S0.
    from simgreeks.selector import S_REF, select_alpha
    X = isd_sample(50_000, 40.0, 10.0, "epanechnikov")
    x = X - 40.0
    Y = 2.0 - 0.4 * x + 0.03 * x ** 2 - 1e-3 * x ** 3 + 2e-5 * x ** 4 \
        + 0.05 * np.random.default_rng(6).standard_normal(len(X))
    lam = 2.5
    a = select_alpha(X, Y, 40.0, 10.0, "epanechnikov", 3, 2)
    b = select_alpha(lam * X, lam * Y, 40.0 * lam, 10.0 * lam, "epanechnikov", 3, 2,
                     x_unit=lam * 40.0 / S_REF)
    assert np.isclose(b["alpha_star"], lam * a["alpha_star"], rtol=1e-9)
    assert np.isclose(b["h_rot"], lam * a["h_rot"], rtol=1e-9)
    raw = select_alpha(lam * X, lam * Y, 40.0 * lam, 10.0 * lam, "epanechnikov", 3, 2)
    assert np.isclose(raw["h_rot"], lam ** (10 / 9) * a["h_rot"], rtol=1e-9)


def test_labels_scale_with_S0_and_K():
    # P(lam S, lam K) = lam P(S, K): each output scales by lam to its power
    from simgreeks.methods import AlgoConfig, run_once
    from simgreeks.multi import MultiConfig, label
    lam = 2.5
    power = {"price": 1, "delta": 0, "gamma": -1, "theta": 1, "vega": 1, "volga": 1,
             "vanna": 0, "rho": 1, "phi": 1, "delta_r": 0, "delta_d": 0,
             "alpha_S": 1, "alpha_sigma": 0}
    cfg = MultiConfig(N=20_000)
    a = label(PutSpec(S0=40.0, K=44.0, T=0.5), cfg, 9)
    b = label(PutSpec(S0=40.0 * lam, K=44.0 * lam, T=0.5), cfg, 9)
    for k, p in power.items():
        assert np.isclose(b[k], a[k] * lam ** p, rtol=1e-6, atol=1e-12), k
    ra = run_once(PutSpec(S0=40.0, K=44.0, T=0.5), AlgoConfig(N=20_000), 9)
    rb = run_once(PutSpec(S0=40.0 * lam, K=44.0 * lam, T=0.5),
                  AlgoConfig(N=20_000, alpha=10.0 * lam), 9)
    for x, y in zip(ra, rb):
        for k, p in (("price", 1), ("delta", 0), ("gamma", -1), ("alpha_star", 1)):
            if np.isfinite(x[k]):
                assert np.isclose(y[k], x[k] * lam ** p, rtol=1e-6), (x["method"], k)
