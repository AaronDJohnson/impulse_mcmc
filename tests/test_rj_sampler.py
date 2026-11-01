"""RJPTSampler wiring: construction checks, user functions, storage, reports."""

import json
import warnings

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
        (dict(sample_mean=np.zeros(8)), ValueError, r"sample_mean must have shape \(7,\)"),
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
    i = _names(sampler).index("birth_death")
    bd_calls = sum(int(jp._proposal_calls[i]) for jp in sampler.proposal_bundle.jump_proposals)
    assert rates["birth"]["calls"] + rates["death"]["calls"] == bd_calls
    with open(tmp_path / "chain_acceptance.json") as fp:
        assert "birth" in json.load(fp)["per_proposal"]


def _narrow_draw(rng):
    return rng.uniform(0.0, 0.5, P)


def _narrow_logpdf(u):
    u = np.asarray(u)
    return np.log(4.0) if np.all((u >= 0) & (u <= 0.5)) else -np.inf


def _death_warnings(sampler, *args, **kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sampler.sample(*args, **kwargs)
    return [str(w.message) for w in record if "death proposal" in str(w.message)]


def test_zero_density_deaths_warn_once_per_call(tmp_path):
    narrow = IndependentComponentProposal(_narrow_draw, _narrow_logpdf)
    sampler = _sampler(tmp_path, component_proposal=narrow, ntemps=1)
    with pytest.warns(RuntimeWarning, match="death proposal") as record:
        sampler.sample(X0, 3000, initial_k=1)
    assert sum("death proposal" in str(w.message) for w in record) == 1
    total = sampler.birth_death.rejected_deaths
    assert total > 0
    assert f"{total} death proposal(s) in this sample() call ({total} in total)" in str(
        record[0].message
    )
    # One iteration from k=1 can only remove the (0.5, 0.5) component, which has a
    # finite density, so this call rejects no death and must not re-report earlier ones.
    assert _death_warnings(sampler, X0, 1, initial_k=1) == []
    assert sampler.birth_death.rejected_deaths == total


def test_zero_density_death_count_restarts_after_resume(tmp_path):
    narrow = IndependentComponentProposal(_narrow_draw, _narrow_logpdf)
    first = _sampler(tmp_path, component_proposal=narrow, ntemps=1)
    assert len(_death_warnings(first, X0, 3000, initial_k=1)) == 1
    # Same class (so the checkpoint verifies), but a density that is never zero.
    fixed = IndependentComponentProposal(_narrow_draw, lambda u: np.log(4.0))
    resumed = _sampler(tmp_path, component_proposal=fixed, ntemps=1, resume=True)
    assert _death_warnings(resumed, X0, 3300, initial_k=1) == []
    assert resumed.birth_death.rejected_deaths > 0  # restored from the checkpoint


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


def test_moves_see_a_read_only_current_sample(tmp_path):
    sampler = _sampler(tmp_path, am_weight=0, scam_weight=0, de_weight=0, birth_death_weight=0)

    def writes_in_place(cs):
        cs.current_sample[0] = 0.4
        return cs.current_sample.copy(), 0.0

    sampler.add_custom_jump(writes_in_place, 1.0)
    with pytest.raises(ValueError, match="read-only"):
        sampler.sample(X0, 5, initial_k=1)


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("which", ["lnlike", "lnprior"])
def test_user_functions_get_read_only_arrays(tmp_path, vectorized, which):
    def writes(x, k):
        x[np.isnan(x)] = 0.0
        return 0.0 if vectorized is False else np.zeros(len(x))

    kwargs = {which: writes}
    if which == "lnlike":
        kwargs["lnprior"] = _lnprior
    else:
        kwargs["lnlike"] = _lnlike
    if vectorized:
        other = "lnprior" if which == "lnlike" else "lnlike"
        f = kwargs[other]
        kwargs[other] = lambda x, k: np.array([f(xi, int(ki)) for xi, ki in zip(x, k)])
    sampler = _sampler(tmp_path, vectorized=vectorized, ntemps=1, **kwargs)
    with pytest.raises(ValueError, match="read-only"):
        sampler.sample(X0, 5, initial_k=1)


@pytest.mark.parametrize(
    "name", ["am", "scam", "de", "gaussian", "DEProposal", "NormalizingFlowProposal"]
)
def test_ptsampler_proposals_are_refused(tmp_path, name):
    import impulse
    from impulse import proposals

    obj = getattr(proposals, name)
    move = obj() if isinstance(obj, type) and name == "DEProposal" else obj
    if name == "NormalizingFlowProposal":
        move = object.__new__(obj)  # the flow itself needs optional deps; isinstance is enough
    assert getattr(impulse, name, obj) is obj
    sampler = _sampler(tmp_path)
    with pytest.raises(ValueError, match="rj_am"):
        sampler.add_custom_jump(move, 1.0)


def test_jump_weights_are_checked(tmp_path):
    sampler = _sampler(tmp_path)

    def stay(cs):
        return cs.current_sample.copy(), 0.0

    for weight in (-5.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="weight of jump 'stay'"):
            sampler.add_custom_jump(stay, weight)
        with pytest.raises(ValueError, match="weight of model jump 'stay'"):
            sampler.add_model_jump(stay, weight)
    assert _names(sampler) == ["rj_am", "rj_scam", "rj_de", "birth_death"]


def test_component_proposal_state_must_be_json(tmp_path):
    class ArrayState(IndependentComponentProposal):
        def get_checkpoint_state(self):
            return {"centre": np.zeros(P)}

        def set_checkpoint_state(self, state):
            pass

    sampler = _sampler(tmp_path, component_proposal=ArrayState(_draw, _logpdf))
    with pytest.raises(TypeError, match=r"ArrayState\.get_checkpoint_state\(\) must return JSON"):
        sampler.sample(X0, 101, initial_k=1)  # the first checkpoint is at iteration 100


PERIODIC_PHASE = dict(component_periodic={1: 1.0})
ONLY_BIRTHS = dict(am_weight=0, scam_weight=0, de_weight=0)  # k starts at min_components = 0
ONLY_REDRAWS = dict(
    am_weight=0, scam_weight=0, de_weight=0, birth_death_weight=0, component_redraw_weight=1
)


def _fixed_draw(value):
    return IndependentComponentProposal(lambda rng: np.array([0.5, value]), _logpdf)


@pytest.mark.parametrize("moves, k0", [(ONLY_BIRTHS, 0), (ONLY_REDRAWS, 1)])
def test_periodic_draws_outside_the_interval_raise(tmp_path, moves, k0):
    sampler = _sampler(tmp_path, component_proposal=_fixed_draw(1.5), **PERIODIC_PHASE, **moves)
    with pytest.raises(ValueError, match=r"returned 1\.5 for periodic component parameter 1"):
        sampler.sample(X0, 1, initial_k=k0)


def test_periodic_draw_at_the_top_of_the_interval_is_accepted(tmp_path):
    top = np.mod(-1e-20, 1.0)
    assert top == 1.0  # np.mod can return the period itself, so the check is closed
    sampler = _sampler(
        tmp_path, component_proposal=_fixed_draw(top), **PERIODIC_PHASE, **ONLY_BIRTHS
    )
    sampler.sample(X0, 1, initial_k=0)
    assert sampler.birth_death.branch_counts(0)["birth"][0] == 1


def _inserting(shared, phase, qxy):
    def inserts(cs):
        row = cs.layout.insert(cs.current_sample, 0, [0.5, phase])
        row[0] = shared
        return row, qxy

    return inserts


NO_BUILT_IN_MOVES = dict(am_weight=0, scam_weight=0, de_weight=0, birth_death_weight=0)


@pytest.mark.parametrize(
    "periodic, shared, phase, index",
    [(PERIODIC_PHASE, 0.5, 1.5, 2), (dict(periodic={0: 1.0}), 1.5, 0.5, 0)],
)
def test_model_jump_rows_must_be_wrapped(tmp_path, periodic, shared, phase, index):
    sampler = _sampler(tmp_path, **periodic, **NO_BUILT_IN_MOVES)
    sampler.add_model_jump(_inserting(shared, phase, 0.0), 1.0)
    with pytest.raises(ValueError, match=f"'inserts' returned 1.5 for periodic parameter {index}"):
        sampler.sample(X0, 1, initial_k=1)


def test_model_jump_rows_at_the_interval_edge_or_rejected_pass(tmp_path):
    at_top = _sampler(tmp_path / "top", **PERIODIC_PHASE, **NO_BUILT_IN_MOVES)
    at_top.add_model_jump(_inserting(0.5, 1.0, 0.0), 1.0)
    at_top.sample(X0, 1, initial_k=1)
    # qxy = -inf: always rejected, so an unwrapped row is allowed and never stored.
    rejected = _sampler(tmp_path / "rejected", **PERIODIC_PHASE, **NO_BUILT_IN_MOVES)
    rejected.add_model_jump(_inserting(0.5, 1.5, -np.inf), 1.0)
    rejected.sample(X0, 20, initial_k=1)
    chain = rejected.load_chain()
    assert np.all(chain["k"] == 1) and not np.any(chain["accepted"])
