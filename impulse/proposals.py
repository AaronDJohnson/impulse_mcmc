import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple
from impulse.chain_stats import ChainStats
from impulse.sampler_state import SamplerState


class JumpProposals:
    """
    Called to get a proposal distribution based on weights
    """
    def __init__(self,
                 chain_stats: ChainStats,
                 proposal_list: list = [],
                 proposal_weights: list = [],
                 proposal_probs: np.ndarray|None = None):
        self.chain_stats = chain_stats
        self.proposal_list = proposal_list
        self.proposal_weights = proposal_weights
        self.proposal_probs = proposal_probs

    def add_jump(self,
                 jump: Callable,
                 weight: float
                 ) -> None:
        if jump not in self.proposal_list:
            self.proposal_list.append(jump)
            self.proposal_weights.append(weight)
            self.proposal_probs = np.array(self.proposal_weights) / sum(self.proposal_weights)  # normalize probabilities
        elif weight != self.proposal_weights[self.proposal_list.index(jump)]:
            self.proposal_weights[self.proposal_list.index(jump)] = weight
            self.proposal_probs = np.array(self.proposal_weights) / sum(self.proposal_weights)

    def __call__(self,
                 state: SamplerState
                 ) -> Tuple[np.ndarray, float]:
        old_sample = state.positions[self.chain_stats.chain_index]
        self.chain_stats.update_sample(old_sample)
        rng = self.chain_stats.rng
        proposal = rng.choice(self.proposal_list, p=self.proposal_probs)
        # don't let DE jumps happen until after buffer is full
        # TODO: change this so that DE isn't possible until buffer is full
        while proposal.__name__ == 'de' and self.chain_stats.buffer_full is False:
            proposal = rng.choice(self.proposal_list, p=self.proposal_probs)
        new_sample, qxy = proposal(self.chain_stats)
        return new_sample, qxy

@dataclass
class ProposalBundle:
    """
    """
    jump_proposals: List['JumpProposals']

    def get_new_position(self, state: SamplerState) -> Tuple[np.ndarray, np.ndarray]:
        new_samples = np.zeros_like(state.positions)
        qxys = np.zeros((len(self.jump_proposals),))
        for i, jp in enumerate(self.jump_proposals):
            new_samples[i], qxys[i] = jp(state)
        return new_samples, qxys

    def add_jump(self, jump: Callable, weight: float) -> None:
        for jp in self.jump_proposals:
            jp.add_jump(jump, weight)

    def __call__(self, state: SamplerState) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_new_position(state)

def am(chain_stats: ChainStats) -> Tuple[np.ndarray, float]:
    """
    Adaptive Jump Proposal. This function will occasionally
    use different jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability
    """
    rng = chain_stats.rng
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    # jumpind = np.random.randint(0, len(groups))

    # adjust step size
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # standard medium jump
    else:
        scale = 1.0

    # adjust scale based on temperature
    # if chain_stats.temp <= 100:
    #     if not np.isinf(chain_stats.temp):
    #         scale *= np.sqrt(1e40)
    #     else:
    #         scale *= np.sqrt(chain_stats.temp)

    # get parmeters in new diagonalized basis
    y = np.dot(chain_stats.svd_U[jumpind].T, chain_stats.current_sample[chain_stats.groups[jumpind]])

    # make correlated componentwise adaptive jump
    ind = np.arange(len(chain_stats.groups[jumpind]))
    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    y[ind] = y[ind] + rng.standard_normal(neff) * cd * np.sqrt(chain_stats.svd_S[jumpind][ind])
    q[chain_stats.groups[jumpind]] = np.dot(chain_stats.svd_U[jumpind], y)

    return q, qxy


def scam(chain_stats: ChainStats) -> tuple[np.ndarray, float]:
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability
    """
    rng = chain_stats.rng
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    ndim = len(chain_stats.groups[jumpind])

    # adjust step size
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # standard medium jump
    else:
        scale = 1.0

    # adjust scale based on temperature
    # if chain_stats.temp <= 100:
    #     if not np.isinf(chain_stats.temp):
    #         scale *= np.sqrt(1e40)
    #     else:
    #         scale *= np.sqrt(chain_stats.temp)

    # make correlated componentwise adaptive jump
    ind = np.unique(rng.integers(0, ndim, 1))

    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    q[chain_stats.groups[jumpind]] += (
        rng.standard_normal() * cd * np.sqrt(chain_stats.svd_S[jumpind][ind]) * chain_stats.svd_U[jumpind][:, ind].flatten()
    )

    return q, qxy


def de(chain_stats: ChainStats) -> tuple[np.ndarray, float]:
    """
    Differential Evolution Jump. This function will  occasionally
    use different jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability
    """
    rng = chain_stats.rng
    # get old parameters
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    ndim = len(chain_stats.groups[jumpind])

    bufsize = chain_stats.buffer_size

    # draw a random integer from 0 - iter
    mm = rng.integers(0, bufsize)
    nn = rng.integers(0, bufsize)

    # make sure mm and nn are not the same iteration
    while mm == nn:
        nn = rng.integers(0, bufsize)

    # get jump scale size
    prob = rng.random()

    # mode jump
    if prob > 0.5:
        scale = 1.0
    
    else:
        scale = rng.random() * 2.4 / np.sqrt(2 * ndim)

    # else:
    #     if np.isinf(chain_stats.temp):
    #         scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1e40)
    #     else:
    #         scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(chain_stats.temp)

    for ii in range(ndim):

        # jump size
        sigma = (chain_stats._buffer[mm, chain_stats.groups[jumpind][ii]] -
                 chain_stats._buffer[nn, chain_stats.groups[jumpind][ii]])

        # jump
        q[chain_stats.groups[jumpind][ii]] += scale * sigma

    return q, qxy

def gaussian(chain_stats: ChainStats) -> tuple[np.ndarray, float]:
    """
    Gaussian Jump. This function will occasionally
    use different jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability
    """
    rng = chain_stats.rng
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    ndim = len(chain_stats.groups[jumpind])

    # get jump scale size
    prob = rng.random()

    # mode jump
    if prob > 0.5:
        scale = 1.0

    else:
        scale = rng.random() * 2.4 / np.sqrt(2 * ndim)

    # else:
    #     if np.isinf(chain_stats.temp):
    #         scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1e40)
    #     else:
    #         scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(chain_stats.temp)

    # make jump
    q[chain_stats.groups[jumpind]] += rng.standard_normal(ndim) * scale

    return q, qxy

def source_swap_proposal(chain_stats: ChainStats) -> tuple[np.ndarray, float]:
    rng = chain_stats.rng
    q = chain_stats.current_sample.copy()
    qxy = 0
    nmodel = int(np.rint(q[-1]))
    if nmodel == 0:
        return q, qxy
    swap_source_1 = rng.integers(0, nmodel + 1)
    swap_source_2 = rng.integers(0, nmodel + 1)
    if swap_source_1 == swap_source_2:
        return q, qxy
    # print(nmodel)
    # print(swap_source_1, swap_source_2)
    # print(q[self.num_params * swap_source_1:self.num_params * (swap_source_1 + 1)])
    # print(q[self.num_params * swap_source_2:self.num_params * (swap_source_2 + 1)])
    # print()
    x = q[3 * swap_source_1:3 * (swap_source_1 + 1)].copy()
    y = q[3 * swap_source_2:3 * (swap_source_2 + 1)].copy()

    q[3 * swap_source_1:3 * (swap_source_1 + 1)] = y
    q[3 * swap_source_2:3 * (swap_source_2 + 1)] = x
    return q, qxy
