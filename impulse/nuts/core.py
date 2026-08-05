"""Core No-U-Turn Sampler (NUTS) algorithm.

Implements the multinomial NUTS variant from Betancourt (2017) with
U-turn detection and divergence tracking.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from impulse.nuts.mass_matrix import MassMatrix


@dataclass
class NUTSState:
    """State of the NUTS sampler at a single iteration.

    Attributes
    ----------
    position : np.ndarray
        Current parameter values.
    logp : float
        Log-probability at current position.
    grad : np.ndarray
        Gradient of log-probability at current position.
    step_size : float
        Leapfrog step size.
    mass_matrix : MassMatrix
        Current mass matrix.
    iteration : int
        Current iteration number.
    accepted : bool
        Whether the last proposal was accepted (always True for NUTS
        unless the entire tree diverged).
    divergent : bool
        Whether a divergence was detected in the last transition.
    tree_depth : int
        Depth of the trajectory tree in the last transition.
    energy_error : float
        Hamiltonian error of the selected proposal: H at the proposal
        leaf (potential + kinetic) minus H at the trajectory start.
        Zero if the transition kept the current position.
    mean_accept_prob : float
        Mean acceptance probability across the tree.
    """

    position: np.ndarray
    logp: float
    grad: np.ndarray
    step_size: float
    mass_matrix: MassMatrix
    iteration: int = 0
    accepted: bool = True
    divergent: bool = False
    tree_depth: int = 0
    energy_error: float = 0.0
    mean_accept_prob: float = 0.0


def leapfrog(
    position: np.ndarray,
    momentum: np.ndarray,
    grad: np.ndarray,
    step_size: float,
    mass_matrix: MassMatrix,
    logp_and_grad: Callable,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Single leapfrog integration step.

    Parameters
    ----------
    position : np.ndarray
        Current position.
    momentum : np.ndarray
        Current momentum.
    grad : np.ndarray
        Gradient of logp at current position.
    step_size : float
        Integration step size.
    mass_matrix : MassMatrix
        Mass matrix for velocity computation.
    logp_and_grad : callable
        Function (x) -> (logp, grad).

    Returns
    -------
    new_position : np.ndarray
    new_momentum : np.ndarray
    new_logp : float
    new_grad : np.ndarray
    """
    # half-step momentum
    momentum = momentum + 0.5 * step_size * grad
    # full-step position
    position = position + step_size * mass_matrix.inverse_multiply(momentum)
    # evaluate at new position
    new_logp, new_grad = logp_and_grad(position)
    # half-step momentum
    momentum = momentum + 0.5 * step_size * new_grad

    return position, momentum, new_logp, new_grad


def _build_tree(
    position: np.ndarray,
    momentum: np.ndarray,
    grad: np.ndarray,
    logp: float,
    depth: int,
    step_size: float,
    direction: int,
    mass_matrix: MassMatrix,
    logp_and_grad: Callable,
    H0: float,
    max_delta_energy: float,
    rng: np.random.Generator,
) -> dict:
    """Recursively build a binary trajectory tree (multinomial sampling).

    Parameters
    ----------
    position, momentum, grad, logp : current state at tree leaf
    depth : int
        Remaining tree depth to build.
    step_size : float
        Base step size (sign determined by direction).
    direction : int
        +1 or -1, integration direction.
    mass_matrix : MassMatrix
        Mass matrix.
    logp_and_grad : callable
        Function (x) -> (logp, grad).
    H0 : float
        Initial Hamiltonian for divergence check.
    max_delta_energy : float
        Max allowed energy error before flagging divergence.
    rng : np.random.Generator
        Random state.

    Returns
    -------
    dict with keys:
        position_left, momentum_left, grad_left, logp_left,
        position_right, momentum_right, grad_right, logp_right,
        proposal_position, proposal_logp, proposal_grad, proposal_H,
        log_sum_weight, n_leapfrog, divergent, turning, sum_accept_prob
    """
    if depth == 0:
        # Base case: single leapfrog step
        new_pos, new_mom, new_logp, new_grad = leapfrog(
            position, momentum, grad, direction * step_size, mass_matrix, logp_and_grad
        )
        H_new = -new_logp + mass_matrix.kinetic_energy(new_mom)
        delta_energy = H_new - H0
        divergent = delta_energy > max_delta_energy

        # log weight is -H (up to constant)
        log_weight = -H_new if np.isfinite(H_new) else -np.inf
        # acceptance probability
        accept_prob = min(1.0, np.exp(-delta_energy)) if np.isfinite(delta_energy) else 0.0

        return {
            "position_left": new_pos,
            "momentum_left": new_mom,
            "grad_left": new_grad,
            "logp_left": new_logp,
            "position_right": new_pos,
            "momentum_right": new_mom,
            "grad_right": new_grad,
            "logp_right": new_logp,
            "proposal_position": new_pos,
            "proposal_logp": new_logp,
            "proposal_grad": new_grad,
            "proposal_H": H_new,
            "log_sum_weight": log_weight,
            "n_leapfrog": 1,
            "divergent": divergent,
            "turning": False,
            "sum_accept_prob": accept_prob,
        }

    # Recursion: build first half-tree
    inner = _build_tree(
        position,
        momentum,
        grad,
        logp,
        depth - 1,
        step_size,
        direction,
        mass_matrix,
        logp_and_grad,
        H0,
        max_delta_energy,
        rng,
    )

    if inner["divergent"] or inner["turning"]:
        return inner

    # Build second half-tree from the appropriate endpoint
    if direction == 1:
        outer = _build_tree(
            inner["position_right"],
            inner["momentum_right"],
            inner["grad_right"],
            inner["logp_right"],
            depth - 1,
            step_size,
            direction,
            mass_matrix,
            logp_and_grad,
            H0,
            max_delta_energy,
            rng,
        )
    else:
        outer = _build_tree(
            inner["position_left"],
            inner["momentum_left"],
            inner["grad_left"],
            inner["logp_left"],
            depth - 1,
            step_size,
            direction,
            mass_matrix,
            logp_and_grad,
            H0,
            max_delta_energy,
            rng,
        )

    if outer["divergent"] or outer["turning"]:
        # Keep inner's proposal but propagate stop signal
        inner["divergent"] = inner["divergent"] or outer["divergent"]
        inner["turning"] = True
        inner["n_leapfrog"] += outer["n_leapfrog"]
        inner["sum_accept_prob"] += outer["sum_accept_prob"]
        return inner

    # Multinomial sampling: combine proposals weighted by exp(log_weight)
    log_sum_weight = np.logaddexp(inner["log_sum_weight"], outer["log_sum_weight"])
    # Accept outer proposal with probability proportional to its weight
    log_accept = outer["log_sum_weight"] - log_sum_weight
    if np.log(rng.random()) < log_accept:
        inner["proposal_position"] = outer["proposal_position"]
        inner["proposal_logp"] = outer["proposal_logp"]
        inner["proposal_grad"] = outer["proposal_grad"]
        inner["proposal_H"] = outer["proposal_H"]

    inner["log_sum_weight"] = log_sum_weight
    inner["n_leapfrog"] += outer["n_leapfrog"]
    inner["sum_accept_prob"] += outer["sum_accept_prob"]

    # Update endpoints based on direction
    if direction == 1:
        inner["position_right"] = outer["position_right"]
        inner["momentum_right"] = outer["momentum_right"]
        inner["grad_right"] = outer["grad_right"]
        inner["logp_right"] = outer["logp_right"]
    else:
        inner["position_left"] = outer["position_left"]
        inner["momentum_left"] = outer["momentum_left"]
        inner["grad_left"] = outer["grad_left"]
        inner["logp_left"] = outer["logp_left"]

    # U-turn check on full combined tree
    span = inner["position_right"] - inner["position_left"]
    v_left = mass_matrix.inverse_multiply(inner["momentum_left"])
    v_right = mass_matrix.inverse_multiply(inner["momentum_right"])
    inner["turning"] = (np.dot(span, v_left) < 0) or (np.dot(span, v_right) < 0)

    return inner


def nuts_step(
    state: NUTSState,
    logp_and_grad: Callable,
    rng: np.random.Generator,
    max_tree_depth: int = 10,
    max_delta_energy: float = 1000.0,
) -> NUTSState:
    """Perform one full NUTS transition.

    Parameters
    ----------
    state : NUTSState
        Current sampler state.
    logp_and_grad : callable
        Function (x) -> (logp, grad).
    rng : np.random.Generator
        Random number generator.
    max_tree_depth : int
        Maximum trajectory tree depth.
    max_delta_energy : float
        Energy error threshold for divergence detection.

    Returns
    -------
    NUTSState
        Updated state after one NUTS transition.
    """
    position = state.position
    logp = state.logp
    grad = state.grad
    mass_matrix = state.mass_matrix
    step_size = state.step_size

    # Sample momentum
    momentum = mass_matrix.sample_momentum(rng)

    # Initial Hamiltonian
    H0 = -logp + mass_matrix.kinetic_energy(momentum)

    # Initialize tree endpoints
    pos_left = pos_right = position
    mom_left = mom_right = momentum
    grad_left = grad_right = grad
    logp_left = logp_right = logp

    proposal_position = position
    proposal_logp = logp
    proposal_grad = grad
    proposal_H = H0

    log_sum_weight = -H0
    depth = 0
    n_leapfrog = 0
    divergent = False
    sum_accept_prob = 0.0

    while depth < max_tree_depth:
        # Choose direction uniformly
        direction = 2 * int(rng.random() < 0.5) - 1

        if direction == 1:
            tree = _build_tree(
                pos_right,
                mom_right,
                grad_right,
                logp_right,
                depth,
                step_size,
                direction,
                mass_matrix,
                logp_and_grad,
                H0,
                max_delta_energy,
                rng,
            )
            pos_right = tree["position_right"]
            mom_right = tree["momentum_right"]
            grad_right = tree["grad_right"]
            logp_right = tree["logp_right"]
        else:
            tree = _build_tree(
                pos_left,
                mom_left,
                grad_left,
                logp_left,
                depth,
                step_size,
                direction,
                mass_matrix,
                logp_and_grad,
                H0,
                max_delta_energy,
                rng,
            )
            pos_left = tree["position_left"]
            mom_left = tree["momentum_left"]
            grad_left = tree["grad_left"]
            logp_left = tree["logp_left"]

        if tree["divergent"]:
            divergent = True
            n_leapfrog += tree["n_leapfrog"]
            sum_accept_prob += tree["sum_accept_prob"]
            break

        if tree["turning"]:
            n_leapfrog += tree["n_leapfrog"]
            sum_accept_prob += tree["sum_accept_prob"]
            break

        # Multinomial: accept new subtree's proposal with appropriate weight
        log_accept = tree["log_sum_weight"] - log_sum_weight
        if np.log(rng.random()) < log_accept:
            proposal_position = tree["proposal_position"]
            proposal_logp = tree["proposal_logp"]
            proposal_grad = tree["proposal_grad"]
            proposal_H = tree["proposal_H"]

        log_sum_weight = np.logaddexp(log_sum_weight, tree["log_sum_weight"])
        n_leapfrog += tree["n_leapfrog"]
        sum_accept_prob += tree["sum_accept_prob"]

        # U-turn check on full trajectory
        span = pos_right - pos_left
        v_left = mass_matrix.inverse_multiply(mom_left)
        v_right = mass_matrix.inverse_multiply(mom_right)
        if (np.dot(span, v_left) < 0) or (np.dot(span, v_right) < 0):
            break

        depth += 1

    # True Hamiltonian error of the selected proposal relative to the
    # trajectory start (both include the kinetic term)
    energy_error = proposal_H - H0

    mean_accept_prob = sum_accept_prob / max(n_leapfrog, 1)

    return NUTSState(
        position=proposal_position,
        logp=proposal_logp,
        grad=proposal_grad,
        step_size=step_size,
        mass_matrix=mass_matrix,
        iteration=state.iteration + 1,
        accepted=not np.array_equal(proposal_position, position),
        divergent=divergent,
        tree_depth=depth,
        energy_error=energy_error,
        mean_accept_prob=mean_accept_prob,
    )
