"""MCMC proposal distributions and weighted proposal mixture containers."""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from impulse.chain_stats import ChainStats

logger = logging.getLogger(__name__)

_SQRT2_INV = 2.4 / math.sqrt(2)
DE_MIN_FILL = 100

_COPPUCCINO_HINT = (
    "NormalizingFlowProposal requires the optional `coppuccino` package "
    "(https://github.com/AaronDJohnson/coppuccino). Install with `pip install coppuccino`."
)


def _require_coppuccino():
    try:
        from coppuccino import log_prob, normalizing_flows_fit, sample
    except ImportError as e:
        raise ImportError(_COPPUCCINO_HINT) from e
    return normalizing_flows_fit, sample, log_prob


class JumpProposals:
    """Weighted collection of proposal distributions for a single temperature chain."""

    def __init__(
        self,
        chain_stats: ChainStats,
        proposal_list: list | None = None,
        proposal_weights: list | None = None,
        proposal_probs: np.ndarray | None = None,
    ):
        self.chain_stats = chain_stats
        self.proposal_list = proposal_list if proposal_list is not None else []
        self.proposal_weights = proposal_weights if proposal_weights is not None else []
        n = len(self.proposal_list)
        if proposal_probs is None and n > 0:
            if not self.proposal_weights:
                self.proposal_weights = [1.0] * n
            if len(self.proposal_weights) != n:
                raise ValueError(
                    f"proposal_weights has {len(self.proposal_weights)} entries but "
                    f"proposal_list has {n}."
                )
            total = sum(self.proposal_weights)
            proposal_probs = (
                np.array(self.proposal_weights, dtype=float) / total
                if total > 0
                else np.ones(n) / n
            )
        self.proposal_probs = proposal_probs
        self._last_proposal_idx = -1
        self._proposal_calls = np.zeros(n, dtype=np.int64)
        self._proposal_accepts = np.zeros(n, dtype=np.int64)
        self._cumprobs_arr: np.ndarray = np.zeros(0)
        self._cumprobs_list: list[float] = []
        self._refresh_cumprobs()

    @property
    def _proposal_cumprobs(self) -> np.ndarray:
        return self._cumprobs_arr

    @_proposal_cumprobs.setter
    def _proposal_cumprobs(self, value: np.ndarray) -> None:
        self._cumprobs_arr = np.asarray(value, dtype=float)
        self._cumprobs_list = self._cumprobs_arr.tolist()

    def add_jump(self, jump: Callable, weight: float) -> None:
        if jump not in self.proposal_list:
            self.proposal_list.append(jump)
            self.proposal_weights.append(weight)
            self._proposal_calls = np.append(self._proposal_calls, 0)
            self._proposal_accepts = np.append(self._proposal_accepts, 0)
        elif weight != self.proposal_weights[self.proposal_list.index(jump)]:
            self.proposal_weights[self.proposal_list.index(jump)] = weight
        total = sum(self.proposal_weights)
        if total > 0:
            self.proposal_probs = np.array(self.proposal_weights) / total
        else:
            self.proposal_probs = np.ones(len(self.proposal_weights)) / len(self.proposal_weights)
        self._refresh_cumprobs()

    def _refresh_cumprobs(self) -> None:
        probs = self.proposal_probs
        self._proposal_cumprobs = np.zeros(0) if probs is None else np.cumsum(probs)

    def _select_proposal_index(self, rng) -> int:
        u = rng.random()
        cum = self._cumprobs_list
        n = len(cum)
        if n == 3:
            if u < cum[0]:
                return 0
            return 1 if u < cum[1] else 2
        return min(bisect.bisect_right(cum, u), len(self.proposal_list) - 1)

    def __call__(self, state) -> Tuple[np.ndarray, float]:
        chain_stats = self.chain_stats
        chain_stats.update_sample(state.positions[chain_stats.chain_index])
        idx = self._select_proposal_index(chain_stats.rng)
        self._last_proposal_idx = idx
        self._proposal_calls[idx] += 1
        return self.proposal_list[idx](chain_stats)

    def report_accept(self, accepted: bool) -> None:
        if accepted and self._last_proposal_idx >= 0:
            self._proposal_accepts[self._last_proposal_idx] += 1

    def acceptance_rates(self) -> dict:
        result = {}
        for i, proposal in enumerate(self.proposal_list):
            calls = int(self._proposal_calls[i])
            accepts = int(self._proposal_accepts[i])
            rate = accepts / calls if calls > 0 else 0.0
            result[proposal.__name__] = {"calls": calls, "accepts": accepts, "rate": rate}
        return result


@dataclass
class ProposalBundle:
    """Coordinates proposal generation across all temperature chains."""

    jump_proposals: List[JumpProposals]

    def get_new_position(self, state) -> Tuple[np.ndarray, np.ndarray]:
        jump_proposals = self.jump_proposals
        new_samples = np.empty_like(state.positions)
        qxys = np.empty(len(jump_proposals))
        for i, jp in enumerate(jump_proposals):
            new_samples[i], qxys[i] = jp(state)
        return new_samples, qxys

    def add_jump(self, jump: Callable, weight: float) -> None:
        for jp in self.jump_proposals:
            jp.add_jump(jump, weight)

    def __call__(self, state) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_new_position(state)

    def report_accepts(self, accepts: np.ndarray) -> None:
        for jp, acc in zip(self.jump_proposals, np.asarray(accepts).tolist()):
            jp.report_accept(bool(acc))

    def acceptance_report(self) -> dict:
        chain_reports = [jp.acceptance_rates() for jp in self.jump_proposals]
        all_names = list(chain_reports[0].keys()) if chain_reports else []
        result = {}
        for name in all_names:
            total_calls = sum(cr[name]["calls"] for cr in chain_reports if name in cr)
            total_accepts = sum(cr[name]["accepts"] for cr in chain_reports if name in cr)
            per_chain = [cr[name] for cr in chain_reports if name in cr]
            rate = total_accepts / total_calls if total_calls > 0 else 0.0
            result[name] = {
                "calls": total_calls,
                "accepts": total_accepts,
                "rate": rate,
                "per_chain": per_chain,
            }
        return result

    def chain_acceptance_rates(self) -> list:
        out = []
        for jp in self.jump_proposals:
            rates = jp.acceptance_rates()
            total_calls = int(sum(r["calls"] for r in rates.values()))
            total_accepts = int(sum(r["accepts"] for r in rates.values()))
            out.append(
                {
                    "calls": total_calls,
                    "accepts": total_accepts,
                    "rate": (total_accepts / total_calls) if total_calls > 0 else 0.0,
                    "per_proposal": {name: r["rate"] for name, r in rates.items()},
                }
            )
        return out


def _draw_jump_scale(rng: np.random.Generator) -> float:
    prob = rng.random()
    if prob > 0.97:
        return 10.0
    if prob > 0.9:
        return 0.2
    return 1.0


def _draw_group_index(rng: np.random.Generator, n_groups: int):
    # integers(0, 1) draws no random bits, so skipping the call for a single
    # group leaves the stream unchanged and saves ~0.7 us of call overhead.
    return 0 if n_groups == 1 else rng.integers(0, n_groups)


def _get_group_specs(chain_stats: ChainStats) -> list:
    if chain_stats._group_specs_ref is not chain_stats.groups:
        chain_stats._rebuild_group_specs()
    return chain_stats._group_specs


def am(chain_stats: ChainStats) -> Tuple[np.ndarray, float]:
    """Adaptive Metropolis proposal using group covariance factor ``L = U * sqrt(S)``."""
    rng = chain_stats.rng
    assert chain_stats.current_sample is not None
    assert chain_stats.groups is not None and chain_stats.proposal_L is not None
    specs = _get_group_specs(chain_stats)
    jumpind = _draw_group_index(rng, len(specs))
    scale = _draw_jump_scale(rng)
    indexer, neff, sqrt_2neff, is_full = specs[jumpind]
    cd = 2.4 / sqrt_2neff * scale
    proposal_L = chain_stats.proposal_L[jumpind]
    assert proposal_L is not None
    step = cd * (proposal_L @ rng.standard_normal(neff))
    if is_full:
        return chain_stats.current_sample + step, 0.0
    q = chain_stats.current_sample.copy()
    q[indexer] += step
    return q, 0.0


def scam(chain_stats: ChainStats) -> Tuple[np.ndarray, float]:
    """Single-component Adaptive Metropolis proposal along one principal axis."""
    rng = chain_stats.rng
    assert chain_stats.current_sample is not None
    assert chain_stats.groups is not None and chain_stats.proposal_L is not None
    specs = _get_group_specs(chain_stats)
    jumpind = _draw_group_index(rng, len(specs))
    indexer, ndim, _, is_full = specs[jumpind]
    scale = _draw_jump_scale(rng)
    ind = rng.integers(0, ndim)
    cd = _SQRT2_INV * scale
    proposal_L = chain_stats.proposal_L[jumpind]
    assert proposal_L is not None
    step = rng.standard_normal() * cd * proposal_L[:, ind]
    if is_full:
        return chain_stats.current_sample + step, 0.0
    q = chain_stats.current_sample.copy()
    q[indexer] += step
    return q, 0.0


def de(chain_stats: ChainStats, min_fill: int = DE_MIN_FILL) -> Tuple[np.ndarray, float]:
    """Differential-evolution proposal gated on a minimum history-buffer fill."""
    rng = chain_stats.rng
    assert chain_stats.current_sample is not None and chain_stats.groups is not None
    buf = chain_stats._buffer
    n_filled = min(chain_stats.sample_total, len(buf))
    if n_filled < max(min_fill, 2):
        return chain_stats.current_sample.copy(), 0.0

    specs = _get_group_specs(chain_stats)
    jumpind = _draw_group_index(rng, len(specs))
    indexer, _, sqrt_2neff, is_full = specs[jumpind]

    mm = rng.integers(0, n_filled)
    nn = rng.integers(0, n_filled)
    while mm == nn:
        nn = rng.integers(0, n_filled)

    scale = 1.0 if rng.random() > 0.5 else rng.random() * 2.4 / sqrt_2neff
    offset = len(buf) - n_filled
    if is_full:
        return (
            chain_stats.current_sample + scale * (buf[offset + mm] - buf[offset + nn]),
            0.0,
        )
    q = chain_stats.current_sample.copy()
    q[indexer] += scale * (buf[offset + mm, indexer] - buf[offset + nn, indexer])
    return q, 0.0


class DEProposal:
    """Differential-evolution proposal with a configurable ``min_fill`` threshold."""

    __name__ = "de"

    def __init__(self, min_fill: int = DE_MIN_FILL):
        if min_fill < 2:
            raise ValueError(
                f"min_fill must be >= 2 (two distinct buffer rows are "
                f"needed for a difference move), got {min_fill}"
            )
        self.min_fill = int(min_fill)

    def __call__(self, chain_stats: ChainStats) -> Tuple[np.ndarray, float]:
        return de(chain_stats, min_fill=self.min_fill)


def gaussian(chain_stats: ChainStats) -> Tuple[np.ndarray, float]:
    """Isotropic Gaussian random-walk proposal."""
    rng = chain_stats.rng
    assert chain_stats.current_sample is not None and chain_stats.groups is not None
    specs = _get_group_specs(chain_stats)
    jumpind = _draw_group_index(rng, len(specs))
    indexer, ndim, sqrt_2neff, is_full = specs[jumpind]
    scale = 1.0 if rng.random() > 0.5 else rng.random() * 2.4 / sqrt_2neff
    if is_full:
        return chain_stats.current_sample + rng.standard_normal(ndim) * scale, 0.0
    q = chain_stats.current_sample.copy()
    q[indexer] += rng.standard_normal(ndim) * scale
    return q, 0.0


class NormalizingFlowProposal:
    """Independence Metropolis proposal using a ``coppuccino`` normalizing flow."""

    __name__ = "nf_flow"

    def __init__(
        self,
        *,
        flow=None,
        refit_interval: int = 2000,
        min_samples: int = 500,
        max_epochs: int = 200,
        prior_bounds: Optional[np.ndarray] = None,
        rng_seed: int = 0,
        fit_kwargs: Optional[dict] = None,
        cold_chain_only: bool = True,
    ):
        self._fit_fn, self._sample_fn, self._logprob_fn = _require_coppuccino()
        self.refit_interval = int(refit_interval)
        self.min_samples = int(min_samples)
        self.max_epochs = int(max_epochs)
        self.prior_bounds = None if prior_bounds is None else np.asarray(prior_bounds, dtype=float)
        self.rng_seed = int(rng_seed)
        self.fit_kwargs = dict(fit_kwargs or {})
        self.cold_chain_only = bool(cold_chain_only)
        self.fixed = flow is not None
        self.flow = flow
        self.frozen = False
        self._call_count = 0
        self._last_fit_at = -(10**9)
        self._fit_count = 0

    def freeze_adaptation(self) -> None:
        self.frozen = True

    def set_flow(self, flow) -> None:
        self.flow = flow
        self.fixed = True

    def get_checkpoint_state(self) -> dict:
        return {
            "frozen": bool(self.frozen),
            "fixed": bool(self.fixed),
            "call_count": int(self._call_count),
            "last_fit_at": int(self._last_fit_at),
            "fit_count": int(self._fit_count),
        }

    def set_checkpoint_state(self, state: dict) -> None:
        self.frozen = bool(state["frozen"])
        self.fixed = bool(state["fixed"])
        self._call_count = int(state["call_count"])
        self._last_fit_at = int(state["last_fit_at"])
        self._fit_count = int(state["fit_count"])

    def _maybe_fit(self, chain_stats: ChainStats) -> None:
        if self.frozen or self.fixed:
            return
        if self.cold_chain_only and chain_stats.chain_index != 0:
            return
        n_avail = min(chain_stats.sample_total, chain_stats.buffer_size)
        if n_avail < self.min_samples:
            return
        if (self._call_count - self._last_fit_at) < self.refit_interval and self.flow is not None:
            return
        # A copy: the history buffer is shifted in place, so a view handed to a
        # fit that keeps a reference to its training data would change under it.
        buf = np.array(chain_stats._buffer[-n_avail:], dtype=np.float64)
        seed = self.rng_seed + self._fit_count
        try:
            self.flow = self._fit_fn(
                buf,
                max_epochs=self.max_epochs,
                prior_bounds=self.prior_bounds,
                rng_seed=seed,
                **self.fit_kwargs,
            )
            self._fit_count += 1
            self._last_fit_at = self._call_count
            logger.info("NF proposal refit #%d on %d samples", self._fit_count, n_avail)
        except Exception as e:
            logger.warning("NF proposal refit failed (%s); keeping previous flow", e)

    def __call__(self, chain_stats: ChainStats) -> Tuple[np.ndarray, float]:
        if chain_stats.chain_index == 0:
            self._call_count += 1
        self._maybe_fit(chain_stats)
        assert chain_stats.current_sample is not None
        if self.flow is None:
            return chain_stats.current_sample.copy(), 0.0

        seed = int(chain_stats.rng.integers(0, 2**31 - 1))
        proposal = np.asarray(
            self._sample_fn(self.flow, n_samples=1, rng_seed=seed), dtype=np.float64
        )[0]
        x_curr = np.asarray(chain_stats.current_sample, dtype=np.float64)
        lp = np.asarray(self._logprob_fn(self.flow, np.stack([x_curr, proposal])))
        qxy = float(lp[0] - lp[1])
        if not np.isfinite(qxy):
            return x_curr.copy(), -np.inf
        return proposal, qxy
