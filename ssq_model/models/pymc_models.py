from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..data import Draw
from .base import Prediction, normalize_blue, scale_red_to_six

MACOS_INSTALL_HINT = (
    "PyMC backend requires the Apple Silicon Bayesian environment. "
    "On macOS arm64 run: python3 -m pip install '.[macos-arm64]' "
    "or use scripts/bootstrap_macos.sh."
)


class PyMCNotInstalledError(ImportError):
    """Raised when the optional PyMC backend is requested without PyMC installed."""

    def __init__(self) -> None:
        super().__init__(MACOS_INSTALL_HINT)


def require_pymc():
    try:
        import pymc as pm  # type: ignore
    except ImportError as exc:
        raise PyMCNotInstalledError() from exc
    return pm


def require_arviz():
    try:
        import arviz as az  # type: ignore
    except ImportError as exc:
        raise PyMCNotInstalledError() from exc
    return az


@dataclass
class PyMCSamplingConfig:
    draws: int = 500
    tune: int = 500
    chains: int = 2
    cores: int = 2
    random_seed: int = 20260529
    target_accept: float = 0.9
    sampler: str = "pymc"  # "pymc" or "numpyro"


@dataclass
class PyMCBlueDirichletModel:
    """PyMC-compatible Dirichlet posterior model for blue ball probabilities."""

    alpha: float = 1.0
    name: str = "pymc_blue_dirichlet"
    _draws: list[Draw] = field(default_factory=list, init=False, repr=False)
    trace: Any = field(default=None, init=False, repr=False)

    def fit(self, draws: list[Draw]) -> "PyMCBlueDirichletModel":
        if not draws:
            raise ValueError("PyMCBlueDirichletModel 至少需要 1 期训练数据")
        self._draws = list(draws)
        return self

    def posterior_blue_mean(self) -> dict[int, float]:
        counts = Counter(draw.blue_ball for draw in self._draws)
        weights = {i: counts.get(i, 0) + self.alpha for i in range(1, 17)}
        return normalize_blue(weights)

    def predict_proba(self) -> Prediction:
        pred = Prediction(
            red_probs={i: 6 / 33 for i in range(1, 34)},
            blue_probs=self.posterior_blue_mean(),
            model_name=self.name,
        )
        pred.validate()
        return pred

    def build_model(self):
        pm = require_pymc()
        counts = [sum(1 for draw in self._draws if draw.blue_ball == i) for i in range(1, 17)]
        with pm.Model() as model:
            theta = pm.Dirichlet("blue_theta", a=[self.alpha] * 16)
            pm.Multinomial("blue_counts", n=len(self._draws), p=theta, observed=counts)
        return model

    def sample(self, config: PyMCSamplingConfig | None = None):
        config = config or PyMCSamplingConfig()
        pm = require_pymc()
        with self.build_model() as model:
            if config.sampler == "numpyro":
                from pymc.sampling.jax import sample_numpyro_nuts  # type: ignore

                self.trace = sample_numpyro_nuts(
                    draws=config.draws,
                    tune=config.tune,
                    chains=config.chains,
                    random_seed=config.random_seed,
                    target_accept=config.target_accept,
                )
            else:
                self.trace = pm.sample(
                    draws=config.draws,
                    tune=config.tune,
                    chains=config.chains,
                    cores=config.cores,
                    random_seed=config.random_seed,
                    target_accept=config.target_accept,
                    progressbar=False,
                )
        return self.trace


@dataclass
class PyMCRedBetaBinomialModel:
    """PyMC-compatible independent Beta-Binomial marginal model for red balls."""

    alpha: float = 1.0
    beta: float = 1.0
    name: str = "pymc_red_beta_binomial"
    _draws: list[Draw] = field(default_factory=list, init=False, repr=False)
    trace: Any = field(default=None, init=False, repr=False)

    def fit(self, draws: list[Draw]) -> "PyMCRedBetaBinomialModel":
        if not draws:
            raise ValueError("PyMCRedBetaBinomialModel 至少需要 1 期训练数据")
        self._draws = list(draws)
        return self

    def posterior_red_mean(self) -> dict[int, float]:
        counts = Counter(ball for draw in self._draws for ball in draw.red_balls)
        n = len(self._draws)
        raw = {i: (counts.get(i, 0) + self.alpha) / (n + self.alpha + self.beta) for i in range(1, 34)}
        return scale_red_to_six(raw)

    def predict_proba(self) -> Prediction:
        pred = Prediction(
            red_probs=self.posterior_red_mean(),
            blue_probs={i: 1 / 16 for i in range(1, 17)},
            model_name=self.name,
        )
        pred.validate()
        return pred

    def build_model(self):
        pm = require_pymc()
        counts = [sum(1 for draw in self._draws if i in draw.red_balls) for i in range(1, 34)]
        n = len(self._draws)
        with pm.Model() as model:
            p = pm.Beta("red_p", alpha=self.alpha, beta=self.beta, shape=33)
            pm.Binomial("red_counts", n=n, p=p, observed=counts)
        return model

    def sample(self, config: PyMCSamplingConfig | None = None):
        config = config or PyMCSamplingConfig()
        pm = require_pymc()
        with self.build_model() as model:
            if config.sampler == "numpyro":
                from pymc.sampling.jax import sample_numpyro_nuts  # type: ignore

                self.trace = sample_numpyro_nuts(
                    draws=config.draws,
                    tune=config.tune,
                    chains=config.chains,
                    random_seed=config.random_seed,
                    target_accept=config.target_accept,
                )
            else:
                self.trace = pm.sample(
                    draws=config.draws,
                    tune=config.tune,
                    chains=config.chains,
                    cores=config.cores,
                    random_seed=config.random_seed,
                    target_accept=config.target_accept,
                    progressbar=False,
                )
        return self.trace


@dataclass
class PyMCJointBayesianModel:
    """Joint prediction wrapper over red Beta-Binomial and blue Dirichlet PyMC models."""

    alpha: float = 1.0
    beta: float = 1.0
    name: str = "pymc_joint_bayesian"
    red_model: PyMCRedBetaBinomialModel = field(init=False)
    blue_model: PyMCBlueDirichletModel = field(init=False)

    def __post_init__(self) -> None:
        self.red_model = PyMCRedBetaBinomialModel(alpha=self.alpha, beta=self.beta)
        self.blue_model = PyMCBlueDirichletModel(alpha=self.alpha)

    def fit(self, draws: list[Draw]) -> "PyMCJointBayesianModel":
        self.red_model.fit(draws)
        self.blue_model.fit(draws)
        return self

    def predict_proba(self) -> Prediction:
        pred = Prediction(
            red_probs=self.red_model.posterior_red_mean(),
            blue_probs=self.blue_model.posterior_blue_mean(),
            model_name=self.name,
        )
        pred.validate()
        return pred

    def sample(self, config: PyMCSamplingConfig | None = None) -> dict[str, Any]:
        return {
            "red": self.red_model.sample(config),
            "blue": self.blue_model.sample(config),
        }
