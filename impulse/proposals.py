import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
from impulse.mhsampler import MHState

from impulse.online_updates import update_covariance, svd_groups
from loguru import logger

def shift_array(arr: np.ndarray,
                num: int,
                fill_value: float = 0
                ) -> np.ndarray:
    """
    Shift an array (arr) by to the left (negative num) or the right (positive num)
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

@dataclass
class ChainData:
    """
    Data to be used to propose new samples
    """
    ndim: int
    groups: list = [np.arange(0, ndim)]
    sample_cov: np.ndarray = np.identity(ndim)
    svd_U: list = [[]] * len(groups)  # U in the standard SVD of samples
    svd_S: list = [[]] * len(groups)  # Sigma in the standard SVD of samples
    sample_mean: float = 0
    temp: float = 1
    current_sample: np.ndarray = None
    rng: np.random.Generator

    # DEBuffer pieces:
    sample_total: int = 0
    buffer_size: int = 50_000
    _buffer: np.ndarray = np.zeros((buffer_size, ndim))
    buffer_full: bool = False

    def update_buffer(self,
                      new_samples: np.ndarray
                      ) -> None:
        self._buffer = shift_array(self._buffer, -len(new_samples))
        self._buffer[-len(new_samples):] = new_samples
        if not self.buffer_full:
            if self.sample_total > self.buffer_size:
                self.buffer_full = True

    def recursive_update(self,
                         sample_num: int,
                         new_samples: np.ndarray
                         ) -> None:
        # update buffer
        self.sample_total += len(new_samples)
        self.update_buffer(new_samples)
        # get new sample mean and covariance
        self.sample_mean, self.sample_cov = update_covariance(sample_num, self.sample_cov, self.sample_mean, self._buffer)
        # new SVD on groups
        self.svd_U, self.svd_S = svd_groups(self.svd_U, self.svd_S, self.groups, self.sample_cov)

    def update_sample(self,
                      state: MHState):
        self.current_sample = state.position

    def update_temp(self,
                    state: MHState):
        self.temp = state.temp

@dataclass
class JumpProposals:
    """
    Called to get a proposal distribution based on weights
    """
    proposal_list: list
    proposal_weights: list
    proposal_probs: np.ndarray
    chain_data: ChainData

    def add_jump(self,
                 jump: Callable,
                 weight: float
                 ) -> None:
        self.proposal_list.append(jump)
        self.proposal_weights.append(weight)
        self.proposal_probs = np.array(self.proposal_weights) / sum(self.proposal_weights)  # normalize probabilities

    def __call__(self,
                 old_sample: np.ndarray,
                 ) -> np.ndarray:
        self.chain_data.update_sample(old_sample)
        rng = self.chain_data.rng
        proposal = rng.choice(self.proposal_list, p=self.proposal_probs)
        # don't let DE jumps happen until after buffer is full
        while proposal.__name__ == 'de' and self.chain_data.buffer_full == False:
            proposal = rng.choice(self.proposal_list, p=self.proposal_probs)
        new_sample = proposal(self.chain_data)
        return new_sample

def am(chain_data: ChainData) -> Tuple(np.ndarray, float):
    """
    Adaptive Jump Proposal. This function will occasionally
    use different jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability
    """
    rng = chain_data.rng
    q = chain_data.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_data.groups))
    # jumpind = np.random.randint(0, len(groups))

    # adjust step size
    prob = rng.random()
    # prob = np.random.rand()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # small-medium jump
    # elif prob > 0.6:
    #    scale = 0.5

    # standard medium jump
    else:
        scale = 1.0

    # adjust scale based on temperature
    if chain_data.temp <= 100:
        scale *= np.sqrt(chain_data.temp)

    # get parmeters in new diagonalized basis
    y = np.dot(chain_data.svd_U[jumpind].T, chain_data.current_sample[chain_data.groups[jumpind]])

    # make correlated componentwise adaptive jump
    ind = np.arange(len(chain_data.groups[jumpind]))
    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    y[ind] = y[ind] + rng.standard_normal(neff) * cd * np.sqrt(chain_data.svd_S[jumpind][ind])
    q[chain_data.groups[jumpind]] = np.dot(chain_data.svd_U[jumpind], y)

    return q, qxy


def scam(chain_data: ChainData) -> Tuple(np.ndarray, float):
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
    rng = chain_data.rng
    q = chain_data.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_data.groups))
    ndim = len(chain_data.groups[jumpind])

    # adjust step size
    # prob = np.random.rand()
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # small-medium jump
    # elif prob > 0.6:

    # standard medium jump
    else:
        scale = 1.0

    # scale = np.random.uniform(0.5, 10)

    # adjust scale based on temperature
    if chain_data.temp <= 100:
        scale *= np.sqrt(chain_data.temp)

    # get parmeters in new diagonalized basis
    # y = np.dot(self.U.T, x[self.covinds])

    # make correlated componentwise adaptive jump
    # ind = np.unique(np.random.randint(0, ndim, 1))
    ind = np.unique(rng.integers(0, ndim, 1))

    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    # y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.S[ind])
    # q[self.covinds] = np.dot(self.U, y)
    q[chain_data.groups[jumpind]] += (
        rng.standard_normal() * cd * np.sqrt(chain_data.svd_S[jumpind][ind]) * chain_data.svd_U[jumpind][:, ind].flatten()
    )

    return q, qxy


def de(chain_data:ChainData) -> Tuple(np.ndarray, float):
    """
    Differential Evolution Jump. This function will  occasionally
    use different jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability
    """
    rng = chain_data.rng
    # get old parameters
    q = chain_data.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = np.random.randint(0, len(chain_data.groups))
    ndim = len(chain_data.groups[jumpind])

    bufsize = chain_data.buffer_size

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
        scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(chain_data.temp)

    for ii in range(ndim):

        # jump size
        sigma = (chain_data._buffer[mm, chain_data.groups[jumpind][ii]] -
                 chain_data._buffer[nn, chain_data.groups[jumpind][ii]])

        # jump
        q[chain_data.groups[jumpind][ii]] += scale * sigma

    return q, qxy




