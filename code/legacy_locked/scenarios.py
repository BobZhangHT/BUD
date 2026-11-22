"""Locked covariate/outcome generators for the BUD comparison.

Design-facing columns are ``x`` only.  ``extra`` contains diagnostic quantities
and must never be passed to a sampling rule.  Optional proxy survival outcomes
are phase-I measurements for a separately labelled, equal-information analysis.

The eight validation settings and all parameters are fixed before their results
are examined.  Development and validation use different distributions or
parameters; independent seeds are supplied by the experiment driver.
"""

from __future__ import annotations

import numpy as np


DEVELOPMENT = (
    "dev_pc1",
    "dev_additive",
    "dev_banana",
    "dev_crossing",
)

VALIDATION = (
    "val_pc1",
    "val_multivariate",
    "val_banana",
    "val_saddle",
    "val_mixture",
    "val_nuisance",
    "val_lowvar",
    "val_null",
)

SCENARIOS = DEVELOPMENT + VALIDATION

DESCRIPTIONS = {
    "dev_pc1": "Development: leading factor, correlated blocks, modest second-factor signal",
    "dev_additive": "Development: additive three-factor response, correlated measurement blocks",
    "dev_banana": "Development: perturbed banana with a third noisy coordinate",
    "dev_crossing": "Development: curved covariates, nonlinear signal, group-specific Weibull shapes",
    "val_pc1": "Validation: leading-factor signal; intended favorable setting for PC1",
    "val_multivariate": "Validation: three-factor additive and interaction signal",
    "val_banana": "Validation: exact PDDS reference X1~N(0,1), X2=X1^2+N(0,1)",
    "val_saddle": "Validation: saddle geometry with additive, interaction and quadratic signal",
    "val_mixture": "Validation: unequal three-component mixture, favorable to adaptive clustering",
    "val_nuisance": "Validation stress: two informative coordinates among 24 observed covariates",
    "val_lowvar": "Validation stress: outcome signal lies in a low-variance contrast discarded by PCA",
    "val_null": "Validation negative control: outcome independent of phase-I covariates",
}


def _blocks(rng, factors, sizes, noise):
    """Repeated noisy observed measurements; no response information is used."""
    cols = []
    for j, size in enumerate(sizes):
        for _ in range(size):
            cols.append(factors[:, j] + noise * rng.normal(size=factors.shape[0]))
    return np.column_stack(cols)


def _covariates(name, rng, n):
    if name == "dev_pc1":
        z = rng.normal(size=(n, 3))
        x = _blocks(rng, z, (5, 2, 1), 0.42)
        risk = (z[:, 0] + 0.30 * z[:, 1]) / np.sqrt(1.09)
    elif name == "dev_additive":
        z = rng.normal(size=(n, 3))
        x = _blocks(rng, z, (4, 3, 2), 0.36)
        risk = (0.70 * z[:, 0] + 0.60 * z[:, 1] + 0.50 * z[:, 2]) / np.sqrt(1.10)
    elif name == "dev_banana":
        z = rng.normal(size=(n, 3))
        u, e = z[:, 0], z[:, 1]
        curved = 0.65 * (u * u - 1.0) + 0.65 * e
        x = np.column_stack((u, curved, u + 0.45 * z[:, 2]))
        risk = (0.65 * u + 0.65 * e + 0.30 * u * e) / np.sqrt(0.935)
    elif name == "dev_crossing":
        z = rng.normal(size=(n, 5))
        u, v = z[:, 0], z[:, 1]
        x = np.column_stack((u, v, 0.90 * u * u + 0.35 * v + 0.20 * z[:, 2],
                             u + 0.40 * z[:, 3], v + 0.70 * z[:, 4]))
        # Var(sin Z)=(1-exp(-2))/2 and covariances below are zero.
        risk = (0.65 * np.sin(u) + 0.40 * v + 0.35 * u * v) / np.sqrt(
            0.65**2 * (1.0 - np.exp(-2.0)) / 2.0 + 0.40**2 + 0.35**2)
    elif name == "val_pc1":
        z = rng.normal(size=(n, 3))
        x = _blocks(rng, z, (6, 3, 2), 0.25)
        risk = z[:, 0]
    elif name == "val_multivariate":
        z = rng.normal(size=(n, 3))
        x = _blocks(rng, z, (5, 3, 2), 0.30)
        risk = (0.40 * z[:, 0] + 0.60 * z[:, 1] + 0.60 * z[:, 2]
                + 0.60 * z[:, 1] * z[:, 2]) / np.sqrt(1.24)
    elif name == "val_banana":
        z = rng.normal(size=(n, 2))
        u, e = z[:, 0], z[:, 1]
        # Exact two-dimensional illustration from Zhou et al. (2024),
        # Efficient Model-Free Subsampling Method for Massive Data, Sec. 2.
        # Marginal standardization makes the population covariance identity;
        # this is deliberately retained as a PC1-identifiability stress case.
        x = np.column_stack((u, u * u + e))
        risk = 0.60 * u + 0.80 * e
    elif name == "val_saddle":
        z = rng.normal(size=(n, 6))
        u, v = z[:, 0], z[:, 1]
        x = np.column_stack((u, v, u * u - v * v + 0.45 * z[:, 2],
                             u + 0.40 * z[:, 3], v + 0.40 * z[:, 4],
                             u + v + 0.40 * z[:, 5]))
        risk = (0.60 * u + 0.60 * v + 0.70 * u * v
                + 0.35 * (u * u - v * v)) / np.sqrt(1.70)
    elif name == "val_mixture":
        z = rng.normal(size=(n, 6))
        component = rng.choice(3, size=n, p=(0.65, 0.25, 0.10))
        centers = np.array(((0.0, 0.0), (3.2, 0.0), (-2.8, 3.6)))
        u = centers[component, 0] + 0.65 * z[:, 0]
        v = centers[component, 1] + 0.65 * z[:, 1]
        x = np.column_stack((u, v, u + 0.35 * z[:, 2], v + 0.35 * z[:, 3],
                             0.5 * u + 0.5 * v + 0.45 * z[:, 4], z[:, 5]))
        # Population centering and variance use the fixed mixture weights.
        component_score = np.array((-0.5, 1.0, 2.5))[component]
        component_mean = -0.5 * 0.65 + 1.0 * 0.25 + 2.5 * 0.10
        component_var = (0.25 * 0.65 + 1.0 * 0.25 + 6.25 * 0.10
                         - component_mean**2)
        risk = (0.70 * (component_score - component_mean) / np.sqrt(component_var)
                + 0.60 * z[:, 0] + 0.35 * z[:, 1]) / np.sqrt(0.9725)
        z = np.column_stack((z, component))
    elif name == "val_nuisance":
        z = rng.normal(size=(n, 17))
        # 2 signal coordinates + 6 measurements of an independent nuisance
        # factor + 16 independent nuisance coordinates = 24 observed columns.
        nuisance_block = z[:, 2, None] + 0.32 * rng.normal(size=(n, 6))
        x = np.column_stack((z[:, :2], nuisance_block, rng.normal(size=(n, 16))))
        risk = (0.75 * z[:, 0] + 0.50 * np.sin(z[:, 1])
                + 0.50 * z[:, 0] * z[:, 1]) / np.sqrt(
                    0.75**2 + 0.50**2 * (1.0 - np.exp(-2.0)) / 2.0 + 0.50**2)
    elif name == "val_lowvar":
        z = rng.normal(size=(n, 3))
        signs = np.array((1.0, -1.0, 1.0, -1.0))
        x_first = z[:, 0, None] + 0.12 * z[:, 1, None] * signs + 0.05 * rng.normal(size=(n, 4))
        x_second = z[:, 0, None] + 0.30 * z[:, 2, None] + 0.05 * rng.normal(size=(n, 4))
        x = np.column_stack((x_first, x_second))
        risk = z[:, 1]
    elif name == "val_null":
        z = rng.normal(size=(n, 3))
        x = _blocks(rng, z, (4, 2, 1), 0.50)
        risk = np.zeros(n)
    else:
        raise ValueError(f"Unknown scenario {name!r}; choose one of {SCENARIOS}")
    return np.asarray(x, dtype=float), np.asarray(risk, dtype=float), z


def generate(name: str, rng: np.random.Generator, n_group: int, *, proxy: bool = True):
    """Return one complete finite cohort for an independent simulation replicate.

    Parameters
    ----------
    name : str
        One of DEVELOPMENT or VALIDATION.
    rng : numpy.random.Generator
        Driver-owned generator. No global RNG is read or changed.
    n_group : int
        Exact cohort size per group; group 0 rows precede group 1 rows.
    proxy : bool, default True
        Return optional noisy phase-I proxy survival. Drawing these measurements
        happens last and therefore never changes the primary outcomes/covariates.

    Returns
    -------
    dict
        x, group, time, event, tau, scenario, extra. If requested, proxy_time and
        proxy_event are observed noisy validation-survival measurements. Only x
        is admissible to the primary four unsupervised sampling methods.

    Notes
    -----
    Exact continuous-time RMST is integral_0^tau KM(t) dt; no time grid is part
    of this data generator. The finite-cohort target must be computed separately
    on the complete generated cohort, not from a population analytic survival.
    The Weibull noise has nonzero conditional variance in every setting.
    """
    if not isinstance(n_group, (int, np.integer)) or n_group < 2:
        raise ValueError("n_group must be an integer at least two")
    n = 2 * int(n_group)
    group = np.repeat(np.arange(2, dtype=int), n_group)
    x, risk, latent = _covariates(name, rng, n)
    tau = 3.0
    # Saturation prevents a few Gaussian-product outliers from creating nearly
    # deterministic vanishing/huge lifetimes. No empirical outcome calibration.
    risk_used = np.clip(risk, -3.0, 3.0)
    if name == "dev_crossing":
        shape = np.where(group == 0, 1.30, 2.55)
        log_scale = np.log(2.05) + 0.08 * group - 0.70 * risk_used
        censor_rate = 0.16
    else:
        shape = np.full(n, 2.0 if name in DEVELOPMENT else 2.25)
        strength = 0.70 if name in DEVELOPMENT else 0.80
        log_scale = np.log(1.95 if name in DEVELOPMENT else 2.10) + 0.16 * group - strength * risk_used
        censor_rate = 0.14 if name in DEVELOPMENT else 0.12
    # rng.weibull accepts an array of positive shape parameters.
    latent_event_time = np.exp(log_scale) * rng.weibull(shape)
    latent_censor_time = rng.exponential(scale=1.0 / censor_rate, size=n)
    time = np.minimum(np.minimum(latent_event_time, latent_censor_time), tau)
    event = (latent_event_time <= latent_censor_time) & (latent_event_time <= tau)
    result = {
        "x": x,
        "group": group,
        "time": time,
        "event": event.astype(np.int8),
        "tau": tau,
        "scenario": name,
        "extra": {
            "risk": risk_used,
            "latent": latent,
            "event_time": latent_event_time,
            "censor_time": latent_censor_time,
            "description": DESCRIPTIONS[name],
            "stage": "development" if name in DEVELOPMENT else "validation",
        },
    }
    if proxy:
        # Cheap unvalidated survival measurement; independent multiplicative
        # measurement noise, not the conditional risk or oracle influence score.
        # A separate proxy-information comparison must give every method the
        # same proxy information and label its target/extra resources explicitly.
        proxy_event_time = latent_event_time * np.exp(0.85 * rng.normal(size=n))
        proxy_censor_time = rng.exponential(scale=1.0 / 0.10, size=n)
        result["proxy_time"] = np.minimum(np.minimum(proxy_event_time, proxy_censor_time), tau)
        result["proxy_event"] = ((proxy_event_time <= proxy_censor_time)
                                 & (proxy_event_time <= tau)).astype(np.int8)
    return result


if __name__ == "__main__":
    for i, name in enumerate(SCENARIOS):
        data = generate(name, np.random.default_rng(823112 + i), 960)
        print(name, data["x"].shape,
              "events", round(float(data["event"].mean()), 3),
              "at_tau", round(float(np.mean(data["time"] >= data["tau"])), 3))
