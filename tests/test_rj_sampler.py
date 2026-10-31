"""RJPTSampler wiring: construction checks, user functions, storage, reports."""

import json

import numpy as np
import pytest

from impulse import IndependentComponentProposal, RJPTSampler

P, K = 2, 3  # rows: 1 shared + 3 slots of 2 = D = 7


def _draw(rng):
    return rng.random(P)


def _logpdf(u):
    u = np.asarray(u)
    return 0.0 if np.all((u >= 0) & (u <= 1)) else -np.inf


UNIFORM = IndependentComponentProposal(_draw, _logpdf)


def _lnlike(x, k):
    comps = x[1:].reshape(K, P)[:k]
    return -0.5 * ((x[0] - 0.3) / 0.1) ** 2 - 0.5 * float(np.sum(((comps - 0.5) / 0.2) ** 2))


def _lnprior(x, k):
    comps = x[1:].reshape(K, P)[:k]
    if not 0.0 <= x[0] <= 1.0 or np.any((comps < 0) | (comps > 1)):
        return -np.inf
    return 0.0  # uniform over k and over numbered slots


def _sampler(outdir, **kwargs):
    options = dict(
        n_shared=1,
        n_component=P,
        max_components=K,
        lnlike=_lnlike,
        lnprior=_lnprior,
        component_proposal=UNIFORM,
        ntemps=2,
        seed=7,
        outdir=str(outdir),
        save_freq=100,
        verbose=False,
    )
    options.update(kwargs)
    return RJPTSampler(**options)


X0 = np.full(1 + K * P, 0.5)


def _names(sampler):
    return [p.__name__ for p in sampler.proposal_bundle.jump_proposals[0].proposal_list]


def test_per_row_functions_get_a_vector_and_an_int(tmp_path):
    seen = set()

    def lnlike(x, k):
        seen.add((x.shape, type(k)))
        return _lnlike(x, k)

    _sampler(tmp_path, lnlike=lnlike).sample(X0, 30, initial_k=1)
    assert seen == {((7,), int)}


def test_vectorized_functions_get_a_batch_and_an_int_array(tmp_path):
    seen = set()

    def lnlike(x, k):
        seen.add((x.shape[1], x.flags.c_contiguous, k.dtype.kind, k.shape == (len(x),)))
        return np.array([_lnlike(xi, int(ki)) for xi, ki in zip(x, k)])

    def lnprior(x, k):
        return np.array([_lnprior(xi, int(ki)) for xi, ki in zip(x, k)])

    sampler = _sampler(tmp_path, lnlike=lnlike, lnprior=lnprior, vectorized=True)
    sampler.sample(X0, 30, initial_k=1)
    assert seen == {(7, True, "i", True)}


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(ndim=7), TypeError, "ndim"),
        (dict(component_proposal=object()), ValueError, "component_proposal"),
        (dict(sample_cov=np.eye(8)), ValueError, r"sample_cov must have shape \(7, 7\)"),
        (dict(groups=[np.arange(8)]), ValueError, "out-of-range"),
        (dict(periodic={7: 1.0}), ValueError, "periodic index 7"),
        (dict(component_periodic={2: 1.0}), ValueError, "component_periodic index 2"),
        (
            dict(periodic={2: 1.0}, component_periodic={1: 1.0}),
            ValueError,
            "both periodic and component_periodic",
        ),
        (dict(birth_death_weight=-1.0), ValueError, "birth_death_weight"),
        (dict(component_redraw_weight=np.nan), ValueError, "component_redraw_weight"),
        (dict(min_components=4), ValueError, "min_components"),
        (dict(n_component=0), ValueError, "n_component"),
    ],
)
def test_construction_checks(tmp_path, kwargs, error, match):
    with pytest.raises(error, match=match):
        _sampler(tmp_path, **kwargs)


def test_registered_moves(tmp_path):
    assert _names(_sampler(tmp_path / "a")) == ["rj_am", "rj_scam", "rj_de", "birth_death"]
    with_redraw = _sampler(tmp_path / "b", component_redraw_weight=5)
    assert _names(with_redraw) == ["rj_am", "rj_scam", "rj_de", "birth_death", "component_redraw"]
    assert _names(_sampler(tmp_path / "c", min_components=3)) == ["rj_am", "rj_scam", "rj_de"]
    assert _names(_sampler(tmp_path / "d", birth_death_weight=0)) == ["rj_am", "rj_scam", "rj_de"]


def test_component_periodic_expands_to_every_slot(tmp_path):
    sampler = _sampler(tmp_path, periodic={0: 1.0}, component_periodic={1: 2 * np.pi})
    assert sampler.wrap.indices.tolist() == [0, 2, 4, 6]


def test_initial_state_checks(tmp_path):
    sampler = _sampler(tmp_path)
    with pytest.raises(ValueError, match="k must be an integer"):
        sampler.sample(X0, 5, initial_k=4)
    with pytest.raises(ValueError, match="initial_position must have shape"):
        sampler.sample(np.zeros(8), 5)
    bad = X0.copy()
    bad[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sampler.sample(bad, 5, initial_k=1)
    with pytest.raises(ValueError, match="initial_k"):
        sampler.sample(X0, 5, initial_k=[1, 1, 1])


def test_initial_k_defaults_to_min_components(tmp_path):
    sampler = _sampler(tmp_path, min_components=1, birth_death_weight=0)  # k cannot change
    sampler.sample(X0, 1)
    assert sampler.load_chain()["k"][:, 0].tolist() == [1, 1]


@pytest.mark.parametrize("chain_format", ["binary", "text"])
def test_load_chain_layout(tmp_path, chain_format):
    sampler = _sampler(tmp_path, chain_format=chain_format)
    sampler.sample(X0, 400, initial_k=1)
    chain = sampler.load_chain()
    assert chain["samples"].shape == (2, 400, 7)
    assert chain["k"].shape == (2, 400) and chain["k"].dtype.kind == "i"
    assert chain["shared"].shape == (2, 400, 1)
    assert chain["components"].shape == (2, 400, 3, 2)
    assert np.shares_memory(chain["components"], chain["samples"])
    for t in range(2):
        for row, k in zip(chain["samples"][t], chain["k"][t]):
            n_act = 1 + P * k
            assert np.all(np.isfinite(row[:n_act])) and np.all(np.isnan(row[n_act:]))
    assert len(np.unique(chain["k"][0])) > 1


def test_acceptance_report_splits_birth_and_death(tmp_path):
    sampler = _sampler(tmp_path)
    sampler.sample(X0, 400, initial_k=1)
    rates = sampler.proposal_acceptance_rates()
    assert {"rj_am", "rj_scam", "rj_de", "birth", "death"} <= set(rates)
    assert "birth_death" not in rates
    assert len(rates["birth"]["per_chain"]) == 2
    bd_calls = sum(int(jp._proposal_calls[3]) for jp in sampler.proposal_bundle.jump_proposals)
    assert rates["birth"]["calls"] + rates["death"]["calls"] == bd_calls
    with open(tmp_path / "chain_acceptance.json") as fp:
        assert "birth" in json.load(fp)["per_proposal"]


def test_zero_density_deaths_warn_once(tmp_path):
    def narrow_draw(rng):
        return rng.uniform(0.0, 0.5, P)

    def narrow_logpdf(u):
        u = np.asarray(u)
        return np.log(4.0) if np.all((u >= 0) & (u <= 0.5)) else -np.inf

    narrow = IndependentComponentProposal(narrow_draw, narrow_logpdf)
    sampler = _sampler(tmp_path, component_proposal=narrow, ntemps=1)
    with pytest.warns(RuntimeWarning, match="death proposal") as record:
        sampler.sample(X0, 3000, initial_k=1)
    assert sum("death proposal" in str(w.message) for w in record) == 1
    assert sampler.birth_death.rejected_deaths > 0


def _grows(cs):
    return cs.layout.insert(cs.current_sample, 0, [0.5, 0.5]), 0.0


def test_custom_within_model_jump_is_checked(tmp_path):
    sampler = _sampler(tmp_path, am_weight=0, scam_weight=0, de_weight=0, birth_death_weight=0)
    sampler.add_custom_jump(_grows, 1.0)
    with pytest.raises(ValueError, match="'_grows' changed k"):
        sampler.sample(X0, 5, initial_k=1)


def test_registering_a_move_twice_keeps_one_entry(tmp_path):
    sampler = _sampler(tmp_path)

    def mine(cs):
        return cs.current_sample.copy(), 0.0

    sampler.add_custom_jump(mine, 1.0)
    sampler.add_custom_jump(mine, 2.0)
    assert _names(sampler).count("mine") == 1
    assert sampler.proposal_bundle.jump_proposals[0].proposal_weights[-1] == 2.0
    with pytest.raises(ValueError, match="already registered"):
        sampler.add_model_jump(mine, 1.0)
    with pytest.raises(ValueError, match="log_jacobian"):
        sampler.add_model_jump(_grows, 1.0, log_jacobian=np.inf)


def test_moves_named_birth_or_death_are_refused(tmp_path):
    sampler = _sampler(tmp_path)
    for name in ("birth", "death"):

        def move(cs):
            return cs.current_sample.copy(), 0.0

        move.__name__ = name
        with pytest.raises(ValueError, match=f"named {name!r}"):
            sampler.add_custom_jump(move, 1.0)
        with pytest.raises(ValueError, match=f"named {name!r}"):
            sampler.add_model_jump(move, 1.0)
    assert _names(sampler) == ["rj_am", "rj_scam", "rj_de", "birth_death"]
