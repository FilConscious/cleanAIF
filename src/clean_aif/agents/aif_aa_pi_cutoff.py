"""
Created at 18:00 on 21st July 2024
@author: Filippo Torresan
"""

# Standard libraries imports
import argparse
import sys

# import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
import importlib
import gymnasium
import gymnasium_env  # Needed otherwise NamespaceNotFound error
from itertools import product

# import math
import numpy as np
import os
from typing import TypedDict, cast, Tuple
from scipy import special
from pathlib import Path
import time

# Custom imports
from ..config import LOG_DIR
from ..utils_agents.utils_aa_cutoff import *

# Set the print options for NumPy
np.set_printoptions(precision=3, suppress=True)


class params(TypedDict):
    """
    This class defines the type of the dictionary of parameters passed to the agent class below. It is used
    to provide the correct type annotation to the parameters the agent receives.
    """

    # Parameters from the command line, they overwrite corresponding ones in class Args above if different
    exp_name: str
    gym_id: str  # also included in Args
    num_runs: int
    num_episodes: int
    num_steps: int  # also included in Args
    learning_rate: float
    seed: int
    inf_steps: int
    num_policies: int  # also included in Args
    plan_horizon: int  # also included in Args
    policy_prior: bool
    action_selection: str
    learn_A: bool
    learn_B: bool
    learn_D: bool
    num_videos: int
    task_type: str
    # Parameters unique to class Args above
    exp_name: str
    num_states: int
    obs_channels: int
    obs_dim: tuple
    factors: int
    factors_dims: tuple
    start_state: int
    goal_state: int
    num_actions: int
    policies: np.ndarray
    policies_indices: np.ndarray
    pref_type: str
    A_params: np.ndarray
    B_params: np.ndarray
    C_params: np.ndarray


class Agent(object):
    """AifAgent class to implement active inference algorithm in a discrete POMDP setting."""

    def __init__(self, params: params):
        """
        Inputs:
        - params (dict): the parameters used to initialize the agent (see description below)
        """

        # Getting some relevant data from params and using default values
        # if nothing was passed in params for these variables.
        self.env_name = params.get("gym_id")
        self.num_states: int = params.get("num_states")
        self.num_actions: int = params.get("num_actions")
        self.start_state: int = params.get("start_state")
        self.steps: int = params.get("num_steps")
        self.inf_iters: int = params.get("inf_steps")
        self.efe_tsteps: int = params.get("plan_horizon")
        self.pref_type: str = params["pref_type"]
        self.policies: np.ndarray = params["policies"]
        self.num_policies: int = params["num_policies"]
        self.policy_prior: bool = params["policy_prior"]
        self.policies_indices: np.ndarray = params["policies_indices"]
        self.as_mechanism: str = params["action_selection"]
        self.learning_A: bool = params["learn_A"]
        self.learning_B: bool = params["learn_B"]
        self.learning_D: bool = params["learn_D"]
        self.seed: int = params["seed"]
        self.task_type: str = params["task_type"]
        self.rng = np.random.default_rng(seed=self.seed)

        # 1. Generative Model, initializing the relevant components used in the computation
        # of free energy and expected free energy:
        #
        # - self.A: observation matrix, i.e. P(o|s) (each column is a categorical distribution);
        # - self.B: transition matrices, i.e. P(s'|s, pi), one for each action (each column
        # is a categorical distribution);
        # - self.C: agent's preferences represented by vector of probabilities C.
        # - self.D: categorical distribution over the initial state, i.e. P(S_1).

        # Observation matrix, stored in numpy array A, and randomly initialised parameters of
        # their Dirichlet prior distributions.
        # Note 1: the Dirichlet parameters must be > 0.
        # Note 2: the values in matrix A are sampled using the corresponding Dirichlet parameters
        # in the for loop.
        # Note 3: if the agent has no uncertainty in the mapping from states to observations,
        # one can initialise A's parameters so that the entries in A's diagonal are close
        # to one or just set A equal to the identity matrix (but the latter may cause some
        # computational errors).
        if self.learning_A == True:
            # With learning over A's parameters, initialise matrix A, its Dirichlet parameters,
            # and sample from them to fill in A
            self.A = np.zeros((self.num_states, self.num_states))
            # Parameters initialised uniformly
            self.A_params = np.ones((self.num_states, self.num_states))
            # For every state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

        elif self.learning_A == False:
            # Without learning over A's parameters, initialise matrix A and its parameters so
            # that the entries in A's diagonal will be close to 1
            self.A = np.zeros((self.num_states, self.num_states))
            # Retrieve true parameters form dictionary
            self.A_params = params.get("A_params")
            # For every state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

        # Transition matrices, stored in tensor B, and randomly initialised parameters of
        # their Dirichlet prior distributions.
        # Note 1: the Dirichlet parameters must be > 0.
        # Note 2: the values in the tensor B are sampled using the corresponding Dirichlet
        # parameters in the for loop.
        # Note 3: if the agent has no uncertainty in the transitions, one can initialise
        # B's parameters so that certain entries in the B matrices are close to one or can
        # just set them equal to one, e.g. to indicate that action 1 from state 1 brings you
        # to state 2 for sure.
        if self.learning_B == True:
            # With learning over B's parameters, initialise tensor B, its Dirichlet parameters,
            # and sample from them to fill in B
            self.B = np.zeros((self.num_actions, self.num_states, self.num_states))
            # Parameters initialised uniformly
            self.B_params = np.ones(
                (self.num_actions, self.num_states, self.num_states)
            )

            # For every action and state draw one sample from the Dirichlet distribution using
            # the corresponding column of parameters
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        elif self.learning_B == False:
            # Without learning over B's parameters, initialise matrix B and its parameters so
            # that entries in B reflect true transitions
            self.B = np.zeros((self.num_actions, self.num_states, self.num_states))
            self.B_params = params.get("B_params")

            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        # Agent's preferences represented by matrix C. Each column stores the agent's preferences for a
        # certain time step. Specifically, each column is a categorical distribution with the probability
        # mass concentrated on the state(s) the agent wants to be in at every time step in an episode.
        # These preference/probabilities could either be over states or observations, and they are defined
        # in the corresponding phenotype module.
        self.C = params["C_params"]

        # Initial state distribution, D (if the state is fixed, then D has the probability mass almost
        # totally concentrated on start_state).
        if self.learning_D == True:

            raise NotImplementedError

        elif self.learning_D == False:

            # Sharp prior on initial
            self.D = np.ones(self.num_states) * 0.01
            self.D[self.start_state] = 1 - (0.01 * (self.num_states - 1))

            # Uniform prior on initial state
            # self.D = np.ones(self.num_states) / self.num_states
            # print(self.D)

        # 2. Variational Distribution, initializing the relevant components used in the computation
        # of free energy and expected free energy:
        # - self.actions: list of actions;
        # - self.policies: numpy array (policy x actions) with pre-determined policies,
        # i.e. rows of num_steps actions;
        # - self.Qpi: categorical distribution over the policies, i.e. Q(pi);
        # - self.Qs_ps: categorical distributions over the states for each policy, i.e. Q(S_t|pi);
        # - self.Qt_pi: categorical distributions over the states for one specific S_t for each policy,
        # i.e. Q(S_t|pi);
        # - self.Qs: policy-independent states probabilities distributions, i.e. Q(S_t).

        # List of actions, dictionary with all the policies, and array with their corresponding probabilities.
        self.actions = list(range(self.num_actions))
        self.policies = params.get("policies")
        self.ordered_policies = np.zeros(
            (self.steps, self.num_policies, self.efe_tsteps), dtype=np.int64
        )

        # Array to store policies probability
        # NOTE: these are reassigned at every time step when new policies are computed/selected
        # (!!! IMPORTANT !!!): the probabilities over policies are updated at every time step and the
        # updated values are saved at the corresponding time step in self.Qpi; this means that the first
        # updated values computed at `self.current_tstep = 0` overwrite the initialized value below.
        # In other words: at each time step we save the updated policies probabilities that become the prior
        # for the NEXT time step.
        self.Qpi = np.zeros((self.num_policies, self.steps))
        self.Qpi[:, 0] = np.ones(self.num_policies) * 1 / self.num_policies
        self.action_probs = np.zeros((self.num_actions, self.steps))

        # Policy-independent states probability distributions, numpy array of size (num_states, timesteps).
        # NOTE: these are used in free energy minimization, they are state beliefs of the common past each
        # policy/plan shares; see perception() method below.
        self.Qs = np.zeros((self.num_states, self.steps))
        # Initialize prior beliefs for the initial step
        self.Qs[:, 0] = 1 / self.num_states
        # Policy-independent state probability distributions after the perception (free energy minimization)
        # step, i.e. after the policy-dependent beliefs have been updated, saved at each time step
        self.Qs_fe = np.zeros((self.num_states, self.steps))

        # Policy-dependent state probabilities distributions, or future state beliefs for EACH policy,
        # numpy array of size (num_policies, num_states, plan_horizon).
        # NOTE: they are computed anew at every free energy minimization step (because the policy also change
        # every time); see perception() method below.
        self.Qs_ps = np.zeros((self.num_policies, self.num_states, self.efe_tsteps))
        # We save in the following array the policy-dependent future state probabilities distributions
        # computed at EVERY time step of each episode; see perception() method below.
        self.Qs_all_ps = np.zeros(
            (self.steps, self.num_policies, self.num_states, self.efe_tsteps)
        )

        # 3. Initialising arrays where to store agent's data during the experiment.
        # Numpy arrays where at every time step the computed free energies and expected free energies
        # for each policy and the total free energy are stored. For how these are calculated see the
        # methods below. We also stored separately the various EFE components.
        self.free_energies = np.zeros((self.num_policies, self.steps))
        self.state_logprob = np.zeros((self.num_policies, self.steps))
        self.state_logprob_first = np.zeros((self.num_policies, self.steps))
        self.obs_loglik = np.zeros((self.num_policies, self.steps))
        self.transit_loglik = np.zeros((self.num_policies, self.steps))
        # EFE and its components
        self.expected_free_energies = np.zeros((self.num_policies, self.steps))
        self.efe_ambiguity = np.zeros((self.num_policies, self.steps))
        self.efe_risk = np.zeros((self.num_policies, self.steps))
        self.efe_Anovelty = np.zeros((self.num_policies, self.steps))
        self.efe_Bnovelty = np.zeros((self.num_policies, self.steps))
        self.efe_Bnovelty_t = np.zeros((self.steps, self.num_policies, self.efe_tsteps))
        self.total_free_energies = np.zeros((self.steps))

        # Matrix of one-hot columns indicating the observation at each time step and matrix
        # of one-hot columns indicating the agent observation at each time step.
        # Note 1 (IMPORTANT): if the agent does not learn A and the environment is not stochastic,
        # then current_obs is the same as agent_obs.
        # Note 2: If the agent learns A, then at every time step what the agent actually observes is sampled
        # from A given the environment state.
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))

        # Numpy array that stores the sequence of actions performed by the agent during the episode
        self.actual_action_sequence = np.zeros((self.steps - 1), dtype=np.int64)
        # Numpy array that stores the index of the policy picked at each action selection step
        # NOTE: the index is with respect to the list of policies computed on the fly at that time step
        self.actual_pi_indices = np.zeros((self.steps - 1))
        # Numpy array that stores the full policy picked at each action selection step, minimizing EFE,
        # from which only the first action was executed
        self.actual_pi = np.empty((0, self.num_actions))

        # Learning rate for the gradient descent on free energy (i.e. it multiplies grad_F_pi in the
        # perception method below)
        self.learning_rate_F = 0.56  # 0.33 #0.4 #0.75

        # Integers indicating the current action and the current time step
        # Note 1 (IMPORTANT!): self.current_tstep is a counter that starts from 0 (zero) in
        # the agent_start method, i.e. the first step in every episode is going to be step 0.
        # Note 2: the advantage of counting from 0 is that, for instance, if you do
        # self.current_obs[:, self.current_tstep] you get the observation indexed by
        # self.current_tstep which corresponds to the observation you got at the same time step;
        # if you were counting the time steps from 1, you would get the observation you got at the
        # next time step because of how arrays are indexed in Python.
        # Note 3: the disadvantage is that when you slice an array by doing
        # self.current_obs[:, 0:self.current_tstep] the slicing actually excludes the column
        # indexed by self.current_tstep, so the right slicing is self.current_obs[:, 0:self.current_tstep+1].
        self.current_action = None
        self.current_tstep = -1

        # 4. Setting the action selection mechanism
        if self.as_mechanism == "kd":
            # Action selection mechanism with Kronecker delta (KD) as described in Da Costa et. al. 2020,
            # (DOI: 10.1016/j.jmp.2020.102447).
            self.select_action = "self.action_selection_KD()"

        elif self.as_mechanism == "kl":
            # Action selection mechanism with Kullback-Leibler divergence (KL) as described
            # in Sales et. al. 2019, 'Locus Coeruleus tracking of prediction errors optimises
            # cognitive flexibility: An Active Inference model'.
            self.select_action = "self.action_selection_KL()"

        elif self.as_mechanism == "probs":
            # Action selection mechanism naively based on updated policy probabilities
            self.select_action = "self.action_selection_probs()"

        else:

            raise Exception("Invalid action selection mechanism.")

    from math import factorial

    def set_policies(
        self,
        num_policies: int,
        policy_len: int,
        num_actions: int,
        shuffle_policies: bool = True,
    ) -> np.ndarray:
        """Function to create and select the policies the agent will use to plan and pick an action at
        one step in the interaction with the environment.

        The policies are sequences of actions that correspond to all the k-tuples over the set S of actions,
        S = [0,.., (num_actions - 1)] and k = policy_len, with repetitions allowed. They amount to the elements
        of the k-fold Cartesian product [0,.., (num_actions - 1)] x .. x [0,.., (num_actions - 1)] for a total
        of (num_actions**policy_len) sequences.

        Inputs:
        - num_policies: number of policies to use (if one does not want to consider all the permutations)
        - policy_len: number of actions in a policy (length of a policy)
        - num_actions: number of available actions, represented by the integers in [0, .. , (num_actions - 1)]

        Output:
        - policy_array: array of shape (num_policies, policy_len), all the policies stored as rows
        """

        # Set of actions
        actions = np.arange(num_actions, dtype=np.int64)
        # Create all the policies
        policies_list = [p for p in product(actions, repeat=policy_len)]
        # Convert list into array
        policies_array = np.array(policies_list, dtype=np.int64)
        # Number of all the sequences
        num_all_pol = num_actions**policy_len
        # Don't shuffle policies (FALSE) when the agent has a policy prior (TRUE)
        # shuffle_policies = not self.policy_prior

        if num_policies == 1:
            sel_policies = np.array([[3, 3, 2]])

        elif num_policies == 2:

            sel_policies = np.array([[3, 3, 2], [0, 3, 1]])

        else:
            if shuffle_policies:
                if self.task_type == "episodic" and self.current_tstep == 0:
                    # Set policy order using list of indices passed to agent at init
                    # NOTE 1: this makes sure that the array of policies is ordered the same way across
                    # runs/agents at the first step of each episode.
                    print(
                        f"Policies indices at first step of each episode: {self.policies_indices}"
                    )
                    sel_policies = policies_array[
                        self.policies_indices[:num_policies], :
                    ]
                else:
                    # All the row indices of policies_array
                    indices = np.arange(num_all_pol)
                    # Shuffle the indices
                    self.rng.shuffle(indices)
                    # Randomly select num_policies from the array with all the policies
                    # NOTE 1: if num_policies equals the number of all sequences, the end result is just
                    # policies_array with its rows shuffled
                    # NOTE 2 (!!!ATTENTION!!!): if num_policies is NOT equal to the number of all sequencies,
                    # the selected policies may not include the optimal policy in this implementation
                    sel_policies = policies_array[indices[:num_policies], :]

            else:
                # Set policy order using list of indices passed to agent at init
                # NOTE 1: this makes sure that the array of policies is ordered the same way across runs/agents
                # *and* across time steps/episodes.
                # NOTE 2 (!!!ATTENTION!!!): if num_policies is NOT equal to the number of all sequencies,
                # the selected policies may not include the optimal policy in this implementation
                sel_policies = policies_array[self.policies_indices[:num_policies], :]

        return sel_policies

    def perception(self):
        """Method that performs a gradient descent on free energy for every policy to update
        the various Q(S_t|pi) by filling in the corresponding entries in self.Q_s.
        It also performs policy-independent state estimation (perceptual inference) by storing in
        self.states_beliefs the states the agent believed most likely to be in during the episode.

        Inputs:

        - None.

        Outputs:

        - state_belief (integer), the state the agent believes it is more likely to be in.

        """

        print("---------------------")
        print("--- 1. PERCEPTION ---")

        # Length of a trajectory, from past to future time steps considered by each policy
        # NOTE: current_tstep is the index for the current time step; since we start counting from 0, an index
        # of 2 implies that *three* times teps have passed. Therefore we add 1 below for computing the total
        # number of time steps in a trajectory.
        efe_tsteps = self.steps - self.current_tstep - 1
        # trajectory_len = self.current_tstep + 1 + self.efe_tsteps
        trajectory_len = self.current_tstep + 1 + efe_tsteps

        # Initialize empty arrays to store policy-dependent state probabilities over full trajectories
        # NOTE: these values are saved for the planning step involving expected free energy minimization,
        # also note that this attribute is re-assigned at every time step following the computation of a
        # new set of policies
        self.Qs_ps_traj = np.empty((self.num_policies, self.num_states, trajectory_len))

        # Loop over policies to calculate the policy-conditioned free energies and their gradients.
        for pi, pi_actions in enumerate(self.policies):

            # Compute future state beliefs for current policy
            # NOTE: this attribute is re-assigned at every step following the computation of a new set of
            # policies, therefore note that across steps the beliefs stored in row i do not refer to the
            # same policy
            if self.current_tstep == 0:
                # At the first time step we are using the agent's prior probabilities on its location
                # NOTE: also we are saving them in self.Qs
                self.Qs[:, 0] = np.copy(self.D)
                current_state_probs = np.copy(self.D)
            else:
                # At other time step we are using the prior state probabilities which were computed
                # at the previous time step but saved at the current one, see the planning() method.
                current_state_probs = self.Qs[:, self.current_tstep]

            # Compute policy-dependent future state beliefs using prior state probabilities at current time step
            # NOTE: this is overwritten at every time step and the array of values computed at the LAST step
            # of the episode is logged
            # self.Qs_ps[pi] = compute_future_beliefs(
            #     self.num_states, current_state_probs, pi_actions, self.B
            # )

            if efe_tsteps != 0:
                Qs_p_future_traj = (
                    np.ones((self.num_states, efe_tsteps)) * 1 / self.num_states
                )

                # Concatenate past, present and future state beliefs for CURRENT policy
                # NOTE: each policy will have a shared past and present
                Qs_p_traj = np.concatenate(
                    (self.Qs[:, 0 : self.current_tstep + 1], Qs_p_future_traj), axis=1
                )
            else:
                # Concatenate past, present beliefs for CURRENT policy
                # NOTE: each policy will have a shared past and present
                Qs_p_traj = np.copy(self.Qs[:, 0 : self.current_tstep + 1])

            # Qs_p_traj = np.ones((self.num_states, trajectory_len)) * 1 / self.num_states
            # Qs_p_traj[:, 0] = np.copy(self.D)
            # Check that Qs_p has correct shape
            assert (
                Qs_p_traj.shape[1] == trajectory_len
            ), f"Axis 1 of Qs_p should be of size {trajectory_len}, but it is of size {Qs_p_traj.shape[1]} instead."

            # Compute action sequence from past to future given current policy
            # NOTE: this is a combination of already taken actions and the policy's actions
            if efe_tsteps != 0:

                pi_actions = np.concatenate(
                    (
                        self.actual_action_sequence[: self.current_tstep],
                        pi_actions[-efe_tsteps:],
                    ),
                    axis=0,
                )

            else:
                pi_actions = self.actual_action_sequence[: self.current_tstep]

            # Compute variational free energy with prior distributions Q(S_t|pi)
            # NOTE: if B parameters are learned then you need to pass in self.B_params and
            # self.learning_B (the same applies for A)
            i = 0
            (
                st_log_st,
                ot_logA_st,
                s1_logD,
                st_logB_stp,
                logA_pi,
                logB_pi,
                logD_pi,
                F_pi_old,
            ) = vfe(
                i,
                trajectory_len,
                self.num_states,
                self.current_tstep,
                self.current_obs,
                pi,
                pi_actions,
                self.A,
                self.B,
                self.D,
                Qs_p_traj,
                A_params=self.A_params,
                learning_A=self.learning_A,
                B_params=self.B_params,
                learning_B=self.learning_B,
            )

            ### DEBUG ###
            if self.current_tstep == 0:
                if pi_actions[0] == pi_actions[1] == 3 and pi_actions[2] == 2:
                    print(f"Prior beliefs: {Qs_p_traj}")
                    print(f"Prior FE: {F_pi_old}")
            ### END ###

            # Inference loop
            while True:
                i += 1
                # Computing the free energy gradient for the current policy
                grad_F_pi = grad_vfe(
                    trajectory_len,
                    self.num_states,
                    self.current_tstep,
                    self.current_obs,
                    pi,
                    pi_actions,
                    Qs_p_traj,
                    logA_pi,
                    logB_pi,
                    logD_pi,
                )

                # Update Q(S_t|pi), for all t in [1, T], by setting gradients to zero
                # NOTE: the update equation below is based on the computations of Da Costa, 2020, p. 9,
                # by setting the gradient to zero one can solve for the parameters that minimize the gradient,
                # here we are recovering those solutions *from* the gradient (by subtraction) before applying
                # a softmax to make sure we get valid probabilities.
                Qs_p_traj = special.softmax(
                    (-1) * (grad_F_pi - np.log(Qs_p_traj) - 1) - 1, axis=0
                )

                # Compute value of policy-conditioned free energy with updated Q(S_t|pi)
                # NOTE: at convergence this value will be the free energy minimum
                (
                    st_log_st,
                    ot_logA_st,
                    s1_logD,
                    st_logB_stp,
                    logA_pi,
                    logB_pi,
                    logD_pi,
                    F_pi,
                ) = vfe(
                    i,
                    trajectory_len,
                    self.num_states,
                    self.current_tstep,
                    self.current_obs,
                    pi,
                    pi_actions,
                    self.A,
                    self.B,
                    self.D,
                    Qs_p_traj,
                    A_params=self.A_params,
                    learning_A=self.learning_A,
                    B_params=self.B_params,
                    learning_B=self.learning_B,
                )

                # Stop at convergence or when hit the max number of iterations
                if abs(F_pi - F_pi_old) < 0.001 or i >= self.inf_iters:
                    break

                F_pi_old = F_pi

            # Save the policy-conditioned future beliefs to be used in the planning method (EFE)
            # print(efe_tsteps)
            # print(self.Qs_ps[pi][:, -efe_tsteps:].shape)
            # print(Qs_p_traj[:, -efe_tsteps:].shape)
            if (
                self.current_tstep == 0
                or self.current_tstep == 1
                or self.current_tstep == 2
            ):
                if pi_actions[0] == pi_actions[2] == 3 and pi_actions[1] == 2:
                    print(f"Policy {pi_actions}")
                    print(f"Last FE: {F_pi}")
                    print("Updated Beliefs")
                    print(Qs_p_traj)

            if efe_tsteps != 0:
                self.Qs_ps[pi][:, -efe_tsteps:] = Qs_p_traj[:, -efe_tsteps:]

            # Save in a separate array the policy-dependent future state beliefs the agent computes
            # at EACH time step in every episode
            # TODO: this does not seem to be used, likely to be REMOVED
            self.Qs_all_ps[self.current_tstep, pi, :, :] = np.copy(self.Qs_ps[pi])
            # Store the last update of Qs_p_traj for the current policy
            # NB: this is then sliced at the current step to compute an updated self.Qs
            # print("Belifes for trajectory:")
            # print(f"{Qs_p_traj}")
            self.Qs_ps_traj[pi] = Qs_p_traj
            # Storing the last computed free energy and components
            self.free_energies[pi, self.current_tstep] = F_pi
            self.state_logprob[pi, self.current_tstep] = st_log_st
            self.state_logprob_first[pi, self.current_tstep] = s1_logD
            self.obs_loglik[pi, self.current_tstep] = ot_logA_st
            self.transit_loglik[pi, self.current_tstep] = st_logB_stp

    def planning(self, unfolding):
        """Method for planning, which involves computing the expected free energy for all the policies
        to update their probabilities, i.e. Q(pi), which are then used for action selection.

        Inputs:
        - None.

        Outputs:
        - None.

        """
        print("---------------------")
        print("--- 2. PLANNING ---")
        # Retrieving the softmax function needed to normalise the values of vector Q(pi) after
        # updating them with the expected free energies.
        sigma = special.softmax
        # Prior over policies
        if self.policy_prior:
            # With prior check if we are at the beginning of the experiment
            if self.current_tstep == 0:
                pi_prior_probs = np.log(self.Qpi[:, 0])
            else:
                pi_prior_probs = np.log(self.Qpi[:, self.current_tstep - 1])
        else:
            # No prior when updating policy probabilities
            pi_prior_probs = 0

        for pi, pi_actions in enumerate(self.policies):

            # At the last time step only update Q(pi) with the computed free energy
            # (because there is no expected free energy then). for all the other steps
            # compute the total expected free energy over the remaining time steps.
            if self.current_tstep == (self.steps - 1) or not unfolding:
                # Storing the free energy for the corresponding policy as the corresponding entry
                # in self.Qpi, that will be normalised below using a softmax to get update probability
                # over the policies (e.g. sigma(-F_pi))
                F_pi = self.free_energies[pi, self.current_tstep]
                self.Qpi[pi, self.current_tstep] = F_pi
                # Storing the zero expected free energy for reference in self.expected_free_energies
                self.expected_free_energies[pi, self.current_tstep] = 0

            else:
                # Note 1: if B parameters are learned then you need to pass in self.B_params
                # and self.learning_B (the same applies for A)
                ### DEBUGGING ###
                # print(
                #     f"The B params for action 2 (frist column): {self.B_params[2,:,0]}"
                # )
                # print(
                #     f"The B params for action 2 (frist column): {self.B_params[2,:,3]}"
                # )
                ### END ###
                efe_tsteps = self.steps - self.current_tstep - 1
                G_pi, tot_Hs, tot_slog_s_over_C, tot_AsW_As, tot_AsW_Bs, sq_AsW_Bs = (
                    efe(
                        self.steps,
                        efe_tsteps,
                        self.current_tstep,
                        pi,
                        pi_actions,
                        self.A,
                        self.C,
                        self.Qs_ps,
                        self.pref_type,
                        self.A_params,
                        self.B_params,
                        self.learning_A,
                        self.learning_B,
                    )
                )

                # Storing the expected free energy and the free energy for the corresponding policy
                # as the corresponding entry in self.Qpi, that will be normalised below using a
                # softmax to get update probability over the policies (e.g. sigma(-F_pi-G_pi))
                F_pi = self.free_energies[pi, self.current_tstep]
                self.Qpi[pi, self.current_tstep] = G_pi + F_pi
                # Storing the expected free energy for reference in self.expected_free_energies
                self.expected_free_energies[pi, self.current_tstep] = G_pi
                # Storing the expected free energy components for later visualizations
                self.efe_ambiguity[pi, self.current_tstep] = tot_Hs
                self.efe_risk[pi, self.current_tstep] = tot_slog_s_over_C
                self.efe_Anovelty[pi, self.current_tstep] = tot_AsW_As
                self.efe_Bnovelty[pi, self.current_tstep] = tot_AsW_Bs
                #### DEBUGGING ####
                # print(f"--- Summary of planning at time step {self.current_tstep} ---")
                # print(f"FE_{pi}: {F_pi}")
                # print(f"EFE_{pi}: {G_pi}")
                # print(f"Risk_{pi}: {tot_slog_s_over_C}")
                # print(f"Ambiguity {pi}: {tot_Hs}")
                # print(f"A-novelty {pi}: {tot_AsW_As}")
                # print(f"B-novelty {pi}: {tot_AsW_Bs}")
                ##### END ####

                # if self.current_tstep == 0:
                #     print(f"B-novelty sequence at t ZERO: {sq_AsW_Bs}")
                #     self.efe_Bnovelty_t[pi] += sq_AsW_Bs
                #     print(
                #         f"B-novelty sequence by policy (stored): {self.efe_Bnovelty_t}"
                #     )
                # self.efe_Bnovelty_t[self.current_tstep, pi, :] += sq_AsW_Bs

        # Normalising the negative expected free energies stored as column in self.Qpi to get
        # the posterior over policies Q(pi) to be used for action selection
        print(f"Computing posterior over policy Q(pi)...")
        self.Qpi[:, self.current_tstep] = sigma(
            -self.Qpi[:, self.current_tstep] + pi_prior_probs
        )
        if self.current_tstep == 1 or self.current_tstep == 2:
            print(self.Qpi[:, self.current_tstep])

        print("Policies Probs > 0.005")
        pi_indices = np.where(self.Qpi[:, self.current_tstep] > 0.005)[0]
        print(f"Policies indices: {pi_indices}")

        # print(f"Before adding noise - Q(pi): {self.Qpi}")
        # Replacing zeroes with 0.0001, to avoid the creation of nan values, and replacing 1 with 5 to make
        # sure a similar concentration of probabilities is preserved when reapplying the softmax
        # self.Qpi[:, self.current_tstep] = np.where(
        #     self.Qpi[:, self.current_tstep] == 1, 5, self.Qpi[:, self.current_tstep]
        # )
        # self.Qpi[:, self.current_tstep] = np.where(
        #     self.Qpi[:, self.current_tstep] == 0,
        #     0.0001,
        #     self.Qpi[:, self.current_tstep],
        # )
        # self.Qpi[:, self.current_tstep] = sigma(self.Qpi[:, self.current_tstep])
        # print(f"After adding noise - Q(pi): {self.Qpi}")

        # Computing updated policy-independent state probabilities by marginalizing w.r.t to policies
        # NOTE (!!! IMPORTANT !!!): here we are computing two kinds of beliefs:
        #
        # 1. CURRENT time step: $Q(S_{t}) =  \sum_{\pi} Q(S_{t} | \pi) Q(\pi)$, used as observation-grounded
        # beliefs for learning, these are crucial for learning to occur properly and they should be considered
        # as the correct beliefs of the agent as to where it is located at each time step in the environment.
        #
        # 2. NEXT time step: $Q(S_{t+1}) =  \sum_{\pi} Q(S_{t+1} | \pi) Q(\pi)$, used as prior probabilities
        # over states for the NEXT time step and saved in self.Qs at the index corresponding to the NEXT time
        # step, these beliefs will be used at the NEXT time step by the method compute_future_beliefs() to
        # predict the consequences of each policy;

        # (1) Computed at ALL time steps
        self.Qs_fe[:, self.current_tstep] = (
            self.Qs_ps_traj[:, :, self.current_tstep].T
            @ self.Qpi[:, self.current_tstep]
        )

        # # (2) Not computed at the truncation or terminal point
        if self.current_tstep != (self.steps - 1) and unfolding:
            self.Qs[:, self.current_tstep + 1] = (
                self.Qs_ps_traj[:, :, self.current_tstep + 1].T
                @ self.Qpi[:, self.current_tstep]
            )

        ### DEBUGGING ###
        # print(f"Time step {self.current_tstep}")
        # index_most_pp = np.argmax(self.Qpi[:, self.current_tstep])
        # print(f"Index of most probable policy: {index_most_pp}")
        # print(f"Most probable policy: {self.policies[index_most_pp]}")
        # if self.current_tstep != (self.steps - 1) and unfolding:
        #     print(
        #         f"At next step agent believes to be in state: {np.argmax(self.Qs[:, self.current_tstep + 1])}"
        #     )
        #     print("Current state beliefs:")
        #     print(self.Qs_fe[:, self.current_tstep])

        #     print("Next state beliefs:")
        #     print(self.Qs[:, self.current_tstep + 1])
        ### END ###

        # At the terminal/truncation point set new prior over initial states depending on task type
        if self.task_type == "continuing" and not unfolding:
            # Save the last policy-independent state probabilities as prior state probabilities for
            # the next episode
            self.D = np.copy(self.Qs_fe[:, self.current_tstep])
            print(f"Prior for next episode:")
            print(self.Qs_fe[:, self.current_tstep])
            print(
                f"Most probable state for next episode: {np.argmax(self.Qs_fe[:, self.current_tstep])}"
            )

    def action_selection_KD(self):
        """Method for action selection based on the Kronecker delta, as described in Da Costa et. al. 2020,
        (DOI: 10.1016/j.jmp.2020.102447). It involves using the approximate posterior Q(pi) to select the most
        likely action, this is done through a Bayesian model average.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.

        """

        # Matrix of shape (num_actions, num_policies) with each row being populated by the same integer,
        # i.e. the index for an action, e.g. np.array([[0,0,0,0,0],[1,1,1,1,1],..]) if there are
        # five policies.
        actions_matrix = np.array([self.actions] * self.num_policies).T

        # By using '==' inside the np.matmul(), we get a boolean matrix in which each row tells us
        # how many policies dictate a certain action at the current time step. Then, these counts are
        # weighed by the probabilities of the respective policies, by doing a matrix-vector multiply
        # between the boolean matrix and the vector Q(pi). In this way, we can pick the favourite action
        # among the most probable policies, by looking for the argmax.

        # Note 1: here we consider the actions the policies dictate at self.current_tstep (say, step 0,
        # so that would be the first action), but we slice self.Qpi using (self.current_tstep + 1) because
        # for action selection we use the computed approx. posterior over policies which was stored as the
        # prior for the next time step in self.Qpi.

        # Note 2: What if there is more than one argmax? To break ties we cannot use np.argmax because it
        # returns the index of the first max value encountered in an array. Instead, we compute actions_probs
        # and look for all the indices corresponding to the max value using np.argwhere; with more than one
        # index np.argwhere places them in a column vector so squeeze() is used to flatten them into an array
        # of shape (num, ), whereas with one index only np.argwhere returns an unsized object, i.e. an integer.
        # Finally, we check if argmax_actions has indeed more than one element or not by looking at its shape:
        # in the former case we randomly break ties with self.rng.choice and set action_selected equal to the
        # randomly picked index, in the latter case action_selected is set to be equal to argmax_actions
        # which is simply the wanted index (an integer).

        actions_probs = np.matmul(
            (self.policies[:, self.current_tstep] == actions_matrix),
            self.Qpi[:, self.current_tstep],
        )
        self.action_probs[:, self.current_tstep] = actions_probs
        print(f"Action probabilities: {actions_probs}")
        argmax_actions = np.argwhere(actions_probs == np.amax(actions_probs)).squeeze()

        if argmax_actions.shape == ():

            action_selected = argmax_actions

        else:

            action_selected = self.rng.choice(argmax_actions)

        print(f"Step {self.current_tstep}")
        print(f"Action selected: {int(action_selected)}")
        # if self.current_tstep == 3:
        #     print(a)

        return int(action_selected)

    def action_selection_KL(self):
        """Method for action selection based on the Kullback-Leibler divergence, as described in
        Sales et. al. 2019 (10.1371/journal.pcbi.1006267). That is, action selection involves computing
        the KL divergence between expected observations and expected observations *conditioned* on performing
        a certain action.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        """

        kl_div = np.zeros(self.num_actions)

        for a in range(self.num_actions):

            # Computing categorical distributions
            # print(f'Qs is {self.Qs[:, self.current_tstep+1]}')

            # Computing policy independent states for t+1
            # Stp1 = np.dot(
            #     self.Qs_pi[:, :, self.current_tstep + 1].T,
            #     self.Qpi[:, self.current_tstep],
            # )
            if self.current_tstep == 0:
                Stp1 = self.Qs[:, self.current_tstep]
                St = self.D
            else:
                Stp1 = self.Qs[:, self.current_tstep]
                St = self.Qs[:, self.current_tstep - 1]

            # Computing distributions over observations
            AS_tp1 = np.dot(self.A, Stp1)
            ABS_t = np.dot(self.A, np.dot(self.B[a, :, :], St))
            # Computing KL divergence for action a and storing it in kl_div
            div = cat_KL(AS_tp1, ABS_t, axis=0)
            kl_div[a] = div

        argmin_actions = np.argwhere(kl_div == np.amin(kl_div)).squeeze()

        if argmin_actions.shape == ():

            action_selected = argmin_actions

        else:

            action_selected = self.rng.choice(argmin_actions)

        return action_selected

    def action_selection_probs(self):
        """Method for action selection based on update policies probabilities. That is, action selection
        simply involves picking the action dictated by the most probable policy.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        """

        argmax_policies = np.argwhere(
            self.Qpi[:, self.current_tstep] == np.amax(self.Qpi[:, self.current_tstep])
        ).squeeze()

        if argmax_policies.shape == ():

            action_selected = self.policies[argmax_policies, 0]

        else:

            action_selected = self.rng.choice(self.policies[argmax_policies, 0])

        # if self.current_tstep == 10:
        #     print(a)
        return action_selected

    def learning(self):
        """Method for parameters learning. This occurs when the agent reaches the goal state or the truncation
        point, i.e. the max number of time steps in the environment..
        Inputs:
            - None.
        Outputs:
            - None.
        """

        print("---------------------")
        print("--- 4. LEARNING ---")
        # Getting the updated parameters for matrices A and B using dirichlet_update().
        # Note 1: if A or B parameters are *not* learned the update method simply return self.A_params or
        # self.B_params
        # NOTE: below we retrieve the second last Qpi because that corresponds to the last time step an
        # action was selected by the agent (no action is selected at the truncation or termination point)
        # and that is all that is needed by the dirichlet_update() method
        print("Updating Dirichlet parameters...")
        self.A_params, self.B_params = dirichlet_update(
            self.num_states,
            self.num_actions,
            self.current_tstep,
            self.current_obs,
            self.actual_action_sequence,
            self.Qs_fe,
            self.A_params,
            self.B_params,
            self.learning_A,
            self.learning_B,
        )

        print(f"Qs size: {self.Qs.shape}")
        print("Agent beliefs:")
        print(np.argmax(self.Qs, axis=0))
        # print(self.Qs)
        print("Actual observations:")
        print(np.argmax(self.current_obs, axis=0))
        # print(self.current_obs)

        # After getting the new parameters, you need to sample from the corresponding Dirichlet distributions
        # to get new approximate posteriors P(A) and P(B). Below we distinguish between different learning
        # scenarios.
        # Note 1: if a set of parameters is not learned, the corresponding matrix(ces) are not considered
        # below (they do not change from their initialised form).
        if self.learning_A == True and self.learning_B == True:

            print("Updated parameters for matrices A and Bs.")
            # After learning A's parameters, for every state draw one sample from the Dirichlet
            # distribution using the corresponding column of parameters.
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

            # After learning B's parameters, sample from them to update the B matrices, i.e. for every
            # action and state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters.
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        elif self.learning_A == True and self.learning_B == False:

            print("Updated parameters for matrix A (no Bs learning).")
            # After learning A's parameters, for every state draw one sample from the Dirichlet
            # distribution using the corresponding column of parameters.
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

        elif self.learning_A == False and self.learning_B == True:

            print("Updated parameters for matrices Bs (no A learning).")
            # After learning B's parameters, sample from them to update the B matrices, i.e. for every
            # action and state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters.
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

            print(self.B[0])

        elif self.learning_A == False and self.learning_B == False:

            print("No update (matrices A and Bs not subject to learning).")

    def step(self, new_obs, unfolding):
        """This method brings together all computational processes defined above, forming the
        perception-action loop of an active inference agent at every time step during an episode.

        Inputs:
        - new_obs: the state from the environment's env_step method (based on where the agent ended up
        after the last step, e.g., an integer indicating the tile index for the agent in the maze).

        Outputs:
        - self.current_action: the action chosen by the agent.
        """

        # Retrieve done information
        # truncated, terminated = done

        # During an episode the counter self.current_tstep goes up by one unit at every time step
        self.current_tstep += 1
        # Updating the matrix of observations and agent obs with the observations at the first time step
        self.current_obs[new_obs, self.current_tstep] = 1
        # Create new set of policies for this time step
        self.policies = self.set_policies(
            self.num_policies, self.efe_tsteps, self.num_actions
        )
        print(f"Computing policies at time step {self.current_tstep}")
        # Save new set of policy for current time step
        self.ordered_policies[self.current_tstep, :, :] = np.copy(self.policies)
        # Initialize total free energy variable
        total_F = 0

        ### ATTENTION: NOT NEEDED/CONSIDER REMOVING ###
        # Sampling from the categorical distribution, i.e. corresponding column of A.
        # Note 1: the agent_observation is a one-hot vector.
        # Note 2 (IMPORTANT!): The computation below presupposes an interpretation of matrix A as a
        # mapping from the environmental stimulus to the agent observation, i.e., as the perceptual
        # processing that gives rise to an observation for the agent. However, in the active inference
        # literature (in discrete state-spaces) the environment stimulus is typically regarded as that
        # observation already.
        # Note 3: for the above reason the computed values are not used/relevant as things stand.
        # To make them relevant, we should pass self.agent_obs to the various methods that require it,
        # e.g. the methods used to minimise free energy in self.perception().
        # TODO: consider commenting out these two lines
        # agent_observation = np.random.multinomial(1, self.A[:, new_obs], size=None)
        # self.agent_obs[:, self.current_tstep] = agent_observation
        ### END ###

        # During an episode perform perception, planning, and action selection based on current observation
        if self.current_tstep < self.steps - 1 and unfolding:
            # Free-energy minimization, i.e. inference or state estimation
            self.perception()
            # Expected free energy minimization, i.e. planning and update of policies probabilities
            self.planning(unfolding)
            # Compute total free energy, i.e. sum of free energies scaled by policies' probabilities Q(pi)
            # NOTE: this should be computed after the planning stage because it requires old policy
            # probabilities as well as the updated ones
            total_F = total_free_energy(
                self.current_tstep, unfolding, self.free_energies, self.Qpi
            )

            print("---------------------")
            print("--- 3. ACTING ---")
            # Select an action
            self.current_action = eval(self.select_action)
            # Storing the selected action in self.actual_action_sequence
            self.actual_action_sequence[self.current_tstep] = self.current_action

        # At the end of the episode (terminal state), do perception and update the A and/or B's parameters
        # (an instance of learning)
        elif self.current_tstep == self.steps - 1 or not unfolding:
            # Saving the P(A) and/or P(B) used during the episode before parameter learning,
            # in this way we conserve the priors for computing the KL divergence(s) for the
            # total free energy at the end of the episode (see below).
            prior_A = self.A_params
            prior_B = self.B_params
            # Free-energy minimization, i.e. inference or state estimation
            self.perception()
            # Expected free energy minimization, i.e. planning and update of policies probabilities
            # Note 1 (IMPORTANT): at the last time step self.planning() only serves to update Q(pi) based on
            # the past as there is no expected free energy to compute.
            self.planning(unfolding)
            # Learning (parameter's updates)
            self.learning()
            self.current_action = None
            # Compute total free energy, i.e. sum of free energies scaled by policies' probabilities Q(pi)
            # NOTE: this should be computed after the planning and learning stage because it requires old
            # policy probabilities as well as the updated ones and the old Dirichlet parameters as well as
            # the updated ones
            total_F = total_free_energy(
                self.current_tstep,
                unfolding,
                self.free_energies,
                self.Qpi,
                prior_A=prior_A,
                prior_B=prior_B,
                A_params=self.A_params,
                learning_A=self.learning_A,
                B_params=self.B_params,
                learning_B=self.learning_B,
            )

        # Store total free energy in self.total_free_energies (as a reference for the agent performance)
        self.total_free_energies[self.current_tstep] = total_F

        return self.current_action

    def reset(self):
        """
        Method to reset agent's attributes used to store different metrics before starting a new episode.
        """

        # Initializing current action and step variables
        self.current_action = None
        self.current_tstep = -1
        # Setting self.current_obs and self.agent_obs to a zero array
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))
        # Setting self.Qs to a zero array
        self.Qs = np.zeros((self.num_states, self.steps))
        # Reset prior probabilities over states to be uniform
        self.Qs[:, 0] = 1 / self.num_states
        # Setting self.Qpi to a zero array
        self.Qpi = np.zeros((self.num_policies, self.steps))
        # Reset prior probabilities over policies to be uniform
        self.Qpi[:, 0] = np.ones(self.num_policies) * 1 / self.num_policies
        # Rest array for ordered list of policies at each times step
        self.ordered_policies = np.zeros(
            (self.steps, self.num_policies, self.efe_tsteps), dtype=np.int64
        )
        # Reset other attributes used to store episodic data related to free energy
        self.free_energies = np.zeros((self.num_policies, self.steps))
        self.expected_free_energies = np.zeros((self.num_policies, self.steps))
        self.efe_ambiguity = np.zeros((self.num_policies, self.steps))
        self.efe_risk = np.zeros((self.num_policies, self.steps))
        self.efe_Anovelty = np.zeros((self.num_policies, self.steps))
        self.efe_Bnovelty = np.zeros((self.num_policies, self.steps))
        self.efe_Bnovelty_t = np.zeros((self.steps, self.num_policies, self.efe_tsteps))
        self.total_free_energies = np.zeros((self.steps))
        # Reset array for sequence of actions performed by the agent during an episode
        self.actual_action_sequence = np.zeros((self.steps - 1), dtype=np.int64)
        # Reset array for the index of the policy picked at each action selection step
        self.actual_pi_indices = np.zeros((self.steps - 1))
        # Reset array for storing the full policy picked at each action selection step, minimizing EFE
        self.actual_pi = np.empty((0, self.num_actions))


class LogData(object):
    """
    Class that defines and stores data collected from an experiment with the above agent class.
    """

    def __init__(self, params: params) -> None:
        ### Retrieve relevant experiment/agent parameters ###
        self.num_runs: int = params.get("num_runs")
        self.num_episodes: int = params.get("num_episodes")
        self.num_states: int = params.get("num_states")
        self.num_max_steps: int = params.get("num_steps")
        self.num_actions: int = params.get("num_actions")
        self.num_policies: int = params.get("num_policies")
        self.efe_tsteps: int = params.get("plan_horizon")
        self.num_videos: int = params.get("num_videos")
        self.learnA: bool = params.get("learn_A")
        self.learnB: bool = params.get("learn_B")

        ### Define attributes to store various metrics ###
        """Counts of how many times maze tiles have been visited"""
        self.state_visits: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_states)
        )
        """Counts of how many times the goal state is reached"""
        self.reward_counts: np.ndarray = np.zeros((self.num_runs, self.num_episodes))
        """Counts of how many steps the episode last (i.e. steps to termination or truncation)"""
        self.steps_count: np.ndarray = np.zeros((self.num_runs, self.num_episodes))
        """Policy dependent free energies at each step during every episode"""
        self.pi_free_energies: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        self.state_logprob: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        self.state_logprob_first: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        self.obs_loglik: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        self.transit_loglik: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        """Total free energies at each step during every episode"""
        self.total_free_energies: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_max_steps)
        )
        """Policy dependent expected free energies at each step during every episode"""
        self.expected_free_energies: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        """Ambiguity term of policy dependent expected free energies at each step during every episode"""
        self.efe_ambiguity: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        """Risk term of policy dependent expected free energies at each step during every episode"""
        self.efe_risk: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        """A-novelty term of policy dependent expected free energies at each step during every episode"""
        self.efe_Anovelty: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        """B-novelty term of policy dependent expected free energies at each step during every episode"""
        self.efe_Bnovelty: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        self.efe_Bnovelty_t: np.ndarray = np.zeros(
            (
                self.num_runs,
                self.num_episodes,
                self.num_max_steps,
                self.num_policies,
                self.efe_tsteps,
            )
        )
        """Observations collected by the agent at each step during an episode"""
        self.observations: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_states, self.num_max_steps)
        )
        """Policy independent probabilistic beliefs about environmental states"""
        self.states_beliefs: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_states, self.num_max_steps)
        )
        """Sequence of action performed by the agent during each episode"""
        self.actual_action_sequence: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_max_steps - 1)
        )
        """Policy dependent probabilistic beliefs about environmental states (last episode step)"""
        self.policy_state_prob: np.ndarray = np.zeros(
            (
                self.num_runs,
                self.num_episodes,
                self.num_policies,
                self.num_states,
                self.efe_tsteps,
            )
        )
        """Policy dependent probabilistic beliefs about environmental states (first episode step)"""
        self.policy_state_prob_first: np.ndarray = np.zeros(
            (
                self.num_runs,
                self.num_episodes,
                self.num_policies,
                self.num_states,
                self.efe_tsteps,
            )
        )
        """Q(S_i|pi) recorded at every time step for every belief state"""
        self.every_tstep_prob: np.ndarray = np.zeros(
            (
                self.num_runs,
                self.num_episodes,
                self.num_max_steps,
                self.num_policies,
                self.num_states,
                self.efe_tsteps,
            )
        )
        """Probabilities of the policies at each time step during every episode"""
        self.pi_probabilities: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        self.action_probs: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_actions, self.num_max_steps)
        )
        """State-observation mappings (matrix A) at the end of each episode"""
        self.so_mappings: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_states, self.num_states)
        )
        """State-trainsition probabilities (matrix B) at the end of each episode"""
        self.transitions_prob: np.ndarray = np.zeros(
            (
                self.num_runs,
                self.num_episodes,
                self.num_actions,
                self.num_states,
                self.num_states,
            )
        )
        """Policies in the order in which they are sampled at each time step"""
        self.ordered_policies: np.ndarray = np.zeros(
            (
                self.num_runs,
                self.num_episodes,
                self.num_max_steps,
                self.num_policies,
                self.efe_tsteps,
            )
        )

    def log_step(self, run, episode, next_state):
        """Method to log data at every step"""

        # Adding a unit to the state counter visits for the new state reached
        self.state_visits[run][episode][next_state] += 1

    def log_episode(self, run: int, episode: int, **kwargs):
        """Method to log data at end of every episode"""

        # At the end of the episode, storing the total reward in reward_counts and other info
        # accumulated by the agent, e.g the total free energies, expected free energies etc.
        # (this is done for every episode and for every run).
        self.steps_count[run][episode] = kwargs["steps_count"]
        self.reward_counts[run][episode] = kwargs["total_reward"]
        self.pi_free_energies[run, episode, :, :] = kwargs["free_energies"]
        self.state_logprob[run, episode, :, :] = kwargs["state_logprob"]
        self.state_logprob_first[run, episode, :, :] = kwargs["state_logprob_first"]
        self.obs_loglik[run, episode, :, :] = kwargs["obs_loglik"]
        self.transit_loglik[run, episode, :, :] = kwargs["transit_loglik"]
        self.total_free_energies[run, episode, :] = kwargs["total_free_energies"]
        self.expected_free_energies[run, episode, :, :] = kwargs[
            "expected_free_energies"
        ]
        self.efe_ambiguity[run, episode, :, :] = kwargs["efe_ambiguity"]
        self.efe_risk[run, episode, :, :] = kwargs["efe_risk"]
        self.efe_Anovelty[run, episode, :, :] = kwargs["efe_Anovelty"]
        self.efe_Bnovelty[run, episode, :, :] = kwargs["efe_Bnovelty"]
        self.efe_Bnovelty_t[run, episode, :, :, :] = kwargs["efe_Bnovelty_t"]
        self.observations[run, episode, :, :] = kwargs["current_obs"]
        self.states_beliefs[run, episode, :, :] = kwargs["Qs"]
        self.actual_action_sequence[run, episode, :] = kwargs["actual_action_sequence"]
        self.policy_state_prob[run, episode, :, :, :] = kwargs[
            "Qs_ps"
        ]  # at the last time step
        self.policy_state_prob_first[run, episode, :, :, :] = kwargs["Qs_all_ps"][
            0
        ]  # at the first time step
        self.every_tstep_prob[run, episode, :, :, :, :] = kwargs[
            "Qs_all_ps"
        ]  # at every time step
        self.pi_probabilities[run, episode, :, :] = kwargs["Qpi"]
        self.action_probs[run, episode, :, :] = kwargs["action_probs"]
        self.so_mappings[run, episode, :, :] = kwargs["A"]
        self.transitions_prob[run, episode, :, :, :] = kwargs["B"]
        self.ordered_policies[run, episode, :, :, :] = kwargs["ordered_policies"]

    def save_data(self, log_dir, file_name="data"):
        """Method to save to file the collected data"""

        # Dictionary to store the data
        data = {}
        # Populate dictionary with corresponding key
        data["exp_name"] = "aif_plans"
        data["num_runs"] = self.num_runs
        data["num_episodes"] = self.num_episodes
        data["num_states"] = self.num_states
        data["num_steps"] = self.num_max_steps
        data["num_policies"] = self.num_policies
        data["learn_A"] = self.learnA
        data["learn_B"] = self.learnB
        data["state_visits"] = self.state_visits
        data["reward_counts"] = self.reward_counts
        data["steps_count"] = self.steps_count
        data["pi_free_energies"] = self.pi_free_energies
        data["state_logprob"] = self.state_logprob
        data["state_logprob_first"] = self.state_logprob_first
        data["obs_loglik"] = self.obs_loglik
        data["transit_loglik"] = self.transit_loglik
        data["total_free_energies"] = self.total_free_energies
        data["expected_free_energies"] = self.expected_free_energies
        data["efe_ambiguity"] = self.efe_ambiguity
        data["efe_risk"] = self.efe_risk
        data["efe_Anovelty"] = self.efe_Anovelty
        data["efe_Bnovelty"] = self.efe_Bnovelty
        data["efe_Bnovelty_t"] = self.efe_Bnovelty_t
        data["observations"] = self.observations
        data["states_beliefs"] = self.states_beliefs
        data["actual_action_sequence"] = self.actual_action_sequence
        data["policy_state_prob"] = self.policy_state_prob
        data["policy_state_prob_first"] = self.policy_state_prob_first
        data["every_tstep_prob"] = self.every_tstep_prob
        data["pi_probabilities"] = self.pi_probabilities
        data["action_probs"] = self.action_probs
        data["so_mappings"] = self.so_mappings
        data["transition_prob"] = self.transitions_prob
        data["ordered_policies"] = self.ordered_policies
        # Save data to local file
        file_data_path = log_dir.joinpath(file_name)
        np.save(file_data_path, data)


def main():

    ##################################
    ### 1. PARSING COMMAND LINE
    ##################################

    # Create command line parser object
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    ### General arguments ###
    parser.add_argument(
        "--exp_name",
        "-expn",
        type=str,
        default="aif-plans",
        help="the name of this experiment based on the active inference implementation",
    )
    parser.add_argument(
        "--gym_id",
        "-gid",
        type=str,
        default="GridWorld-v1",
        help="the name of the registered gym environment (choices: GridWorld-v1)",
    )
    parser.add_argument(
        "--env_layout",
        "-el",
        type=str,
        default="Tmaze3",
        help="layout of the gridworld (choices: Tmaze3, Tmaze4, Ymaze4)",
    )
    parser.add_argument(
        "--num_runs",
        "-nr",
        type=int,
        default=30,
        help="the number of times the experiment is run",
    )
    parser.add_argument(
        "--num_episodes",
        "-ne",
        type=int,
        default=100,
        help="number of episodes per run/experiment",
    )
    parser.add_argument(
        "--num_steps",
        "-ns",
        type=int,
        required=True,
        help="number of steps per episode or total number of steps (depending on type of agent)",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=2.5e-4,
        help="the learning rate for the free energy gradients",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")

    ### Agent-specific arguments ###
    # Inference
    parser.add_argument(
        "--inf_steps",
        "-infstp",
        type=int,
        default=1,
        help="number of free energy minimization steps",
    )
    # Policy
    parser.add_argument(
        "--num_policies",
        "-np",
        type=int,
        nargs="?",
        help="number of policies (i.e. sequences of actions) in planning",
    )
    parser.add_argument(
        "--plan_horizon",
        "-ph",
        type=int,
        nargs="?",
        help="planning horizon (i.e. length of a policy)",
    )
    # Action selection mechanism
    parser.add_argument(
        "--action_selection",
        "-as",
        type=str,
        default="kd",
        help="choices: probs, kl, kd",
    )
    # Enable learning of different aspects of the environment
    parser.add_argument("--learn_A", "-lA", action="store_true")
    parser.add_argument("--learn_B", "-lB", action="store_true")
    parser.add_argument("--learn_D", "-lD", action="store_true")
    # Number of videos to record
    parser.add_argument("--num_videos", "-nvs", type=int, default=0)
    # Flag to switch from episodic to continuing task (in the former at each episode the agent's location is
    # the same, set as agent's attribute whereas in the latter the location corresponds to the last reached
    # state at the previous episode)
    parser.add_argument(
        "--task_type",
        "-tsk",
        type=str,
        default="episodic",
        help="choices: episodic, continuing",
    )
    # NOTE: this is just a label used to identify the experiment, make sure it corresponds
    # to the attribute/property set in the agent Args class, see top of file
    parser.add_argument(
        "--pref_type",
        "-pft",
        type=str,
        default="states",
        help="choices: states, statesmanh, obs",
    )
    # Whether to use a policy prior when udapting the policies' probabilities
    parser.add_argument("--policy_prior", "-ppr", action="store_true")
    # Whether to shuffle policies at each times step in an episode
    parser.add_argument("--shuffle_policies", "-shpol", action="store_true")

    # Creating object holding the attributes from the command line
    args = parser.parse_args()
    # Convert args to dictionary
    cl_params = vars(args)

    ########################################################
    ### 2. CREATE DIRECTORY FOR LOGGING DATA FROM CURRENT EXP
    ########################################################

    # Datetime object containing current date and time
    now = datetime.now()
    # Converting data-time in an appropriate string: '_dd.mm.YYYY_H.M.S'
    dt_string = now.strftime("%Y.%m.%d_%H.%M.%S_")
    # Create string of experiment-specific info
    exp_info = (
        f'{cl_params["gym_id"]}_{cl_params["env_layout"]}_{cl_params["exp_name"]}_{cl_params["task_type"]}'
        f'_nr{cl_params["num_runs"]}_ne{cl_params["num_episodes"]}_steps{cl_params["num_steps"]}'
        f'_infsteps{cl_params["inf_steps"]}_preftype_{cl_params["pref_type"]}'
        f'_npol{cl_params["num_policies"]}_phor{cl_params["plan_horizon"]}_ppr{cl_params["policy_prior"]}'
        f'_pshuffle{cl_params["shuffle_policies"]}_AS{cl_params["action_selection"]}'
        f'_lA{str(cl_params["learn_A"])[0]}_lB{str(cl_params["learn_B"])[0]}_lD{str(cl_params["learn_D"])[0]}'
    )
    # Create folder (with dt_string as unique identifier) where to store data from current experiment.
    data_path = LOG_DIR.joinpath(dt_string + exp_info)
    data_path.mkdir(parents=True, exist_ok=True)

    # Save the command used to run the script
    cmd_str = " ".join(sys.argv)
    cmd_file = os.path.join(data_path, "command.txt")

    with open(cmd_file, "w") as f:
        f.write(cmd_str + "\n")

    print(f"Command saved to {cmd_file}")

    ###############################
    ### 3. INIT AGENT PARAMETERS
    ##############################

    # Importing config module dynamically based on env layout
    module_name = f'..config_agents.aif_aa_{cl_params["env_layout"]}_cfg'
    agent_config = importlib.import_module(module_name, package=__package__)
    Args = agent_config.Args
    # Create dataclass with default parameters configuration for the agent
    agent_params = Args()
    # Convert dataclass to dictionary
    agent_params = asdict(agent_params)

    # Update agent_params with corresponding key values in cl_params, and/or add key from cl_params
    # Custom update function not overwriting default parameter's value if the one from the CL is None
    def update_params(default_params, new_params):
        for key, value in new_params.items():
            if value is not None:
                default_params[key] = value

    update_params(agent_params, cl_params)
    # print(agent_params)

    # Create ordered array of policies indices
    # !!!IMPORTANT!!!: if the agent is given a prior over policies, this is used to make sure that the
    # policies are ordered in the same way across runs/agent *and* across time steps/episodes (otherwise
    # adding a prior to a shuffled array would make no sense); if the agent is not given a prior, this is
    # used to make sure that the order is the same at the first step of each episode, provided the task is
    # episodic (not continuing).
    # NOTE 1: having a consistent policies' order across agents/runs and steps/episodes is only crucial when
    # we average policy related results and plot them

    # Set rng for the experiment
    rng = np.random.default_rng(seed=333777)
    # Create array of policies indices, all possible policies are considered
    policies_indices = np.arange(
        agent_params["num_actions"] ** agent_params["plan_horizon"]
    )
    # Shuffle the indices
    rng.shuffle(policies_indices)
    # Add the shuffled array of indices to params
    agent_params["policies_indices"] = policies_indices

    ##########################
    ### 4. INIT ENV
    ##########################

    # Retrieve name of the environment
    env_module_name = cl_params["gym_id"]
    # Retrieve task type
    task_type = cl_params["task_type"]
    # Number of runs (or agents interacting with the env)
    NUM_RUNS = agent_params["num_runs"]
    # Number of episodes
    NUM_EPISODES = agent_params["num_episodes"]
    # Number of steps in one episode
    NUM_STEPS = agent_params["num_steps"]
    # Fix walls location in the environment depending on env_layout
    env_layout = agent_params["env_layout"]
    if env_layout == "tmaze3":
        SIZE = 3
        WALLS_LOC = [
            set_wall_xy(3),
            set_wall_xy(5),
            set_wall_xy(6),
            set_wall_xy(7),
            set_wall_xy(8),
        ]
    elif env_layout == "tmaze4":
        SIZE = 3
        WALLS_LOC = [
            set_wall_xy(3),
            set_wall_xy(5),
            set_wall_xy(6),
            set_wall_xy(8),
        ]
    elif env_layout == "ymaze4":
        SIZE = 3
        WALLS_LOC = [set_wall_xy(1), set_wall_xy(6), set_wall_xy(8)]

    elif env_layout == "gridw9":
        SIZE = 3
        WALLS_LOC = []

    elif env_layout == "gridw16":
        SIZE = 4
        WALLS_LOC = []

    else:
        raise ValueError(
            "Value of 'env_layout' is not among the available ones. Choose from: Tmaze3, Tmaze4, Ymaze4."
        )
    # Fix target location in the environment (the same in every episode)
    # TARGET_LOC = convert_state(agent_params["goal_state"], env_layout)
    TARGET_LOC = []
    for goal in agent_params["goal_state"]:
        g = convert_state(goal, env_layout)
        TARGET_LOC.append(g)

    # Create the environment
    env = gymnasium.make(
        "gymnasium_env/GridWorld-v1",
        max_episode_steps=NUM_STEPS - 1,
        render_mode=None,
        size=SIZE,
    )

    ##############################
    ### 5. INIT LOGGING
    ##############################

    # Create instance of LogData for logging and saving experiments metrics
    logs_writer = LogData(cast(params, agent_params))
    # Define random number generator for picking random actions
    # rng = np.random.default_rng()

    ###############################
    ### 5. TRAINING
    ###############################

    start_time = time.time()

    ### TRAINING LOOP ###
    # Loop over number of runs and episodes
    for run in range(NUM_RUNS):
        # Print iteration (run) number
        print("************************************")
        print(f"Starting Run {run}...")
        # Set a random seed for current run, used by RNG attribute in the agent
        agent_params["seed"] = run
        # Fix agent's location at the FIRST episode or at EACH episode (i.e. agent starts the first episode or
        # each episode from the same location), depending on the flag 'task_type', see below
        # NOTE: we need to convert the following various states from an index to a (x, y) representation which is
        # what the Gymnasium environment requires.
        AGENT_LOC = convert_state(
            agent_params["start_state"], env_layout
        )  # output: np.array([0, 0])
        # Create agent (`cast()` is used to tell the type checker that `agent_params` is of type `params`)
        agent = Agent(cast(params, agent_params))
        # Loop over episodes
        for e in range(NUM_EPISODES):
            # Printing episode number
            print("**********************************************************")
            print(f"****************** EPISODE {e} **************************")
            print("**********************************************************")

            # Initialize steps and reward counters
            steps_count = 0
            total_reward = 0
            # is_terminal = False

            # Reset the environment
            # NOTE: the Gymnasium environment's observation is a dictionary with the locations of the agent and
            # the goal/target, i.e. {'agent': array([0, 0]), 'target': array([2, 2])}, but we need only the
            # location of the agent!
            obs, info = env.reset(
                options={
                    "deterministic_agent_loc": AGENT_LOC,
                    "deterministic_target_loc": TARGET_LOC,
                    "deterministic_wall_loc": WALLS_LOC,
                },
            )

            terminated, truncated = False, False

            # Retrieve observation of the agent's location
            # print(f"Observation: {obs}; type {type(obs)}")
            obs = obs["agent"]
            # Convert obs into index representation
            start_state = process_obs(obs, env_layout)
            # Adding a unit to the state_visits counter for the start_state
            logs_writer.log_step(run, e, start_state)
            # Current state (updated at every step and passed to the agent)
            current_state = start_state

            # Establish different agent-environment interaction loops based on task_type
            if task_type == "episodic":
                unfolding = not terminated and not truncated
            elif task_type == "continuing":
                unfolding = not truncated
            else:
                raise ValueError(
                    f"Invalid value: {task_type}. Allowed values are: episodic, continuing"
                )

            # Agent and environment interact until the environment is truncated or terminated.
            # NOTE: the maximum number of steps is NUM_STEPS, i.e. the time step at which the environment
            # is truncated regardless of whether the agent has reached the goal state or not..
            print(f"Unfolding is {unfolding}")
            while unfolding:

                # Agent returns an action based on current observation/state
                action = agent.step(current_state, unfolding)
                # Environment outputs based on agent action
                next_obs, reward, terminated, truncated, info = env.step(action)
                # Retrieve observation of the agent's location
                next_obs = next_obs["agent"]
                # Convert observation into index representation
                next_state = process_obs(next_obs, env_layout)
                # Update total_reward
                total_reward += reward

                print("-------- STEP SUMMARY --------")
                print(f"Time step: {steps_count}")
                print(f"Observation: {current_state}")
                print(f"Agent action: {action}")
                print(f"Next state: {next_state}")
                print(f"Terminal/Goal state: {terminated}")

                # Adding a unit to the state_visits counter for the start_state
                logs_writer.log_step(run, e, next_state)
                # Update current state with next_state
                current_state = next_state
                # Update step count
                steps_count += 1
                # Update unfolding flag
                if task_type == "episodic":
                    unfolding = not terminated and not truncated
                elif task_type == "continuing":
                    unfolding = not truncated
                else:
                    raise ValueError(
                        f"Invalid value: {task_type}. Allowed values are: continuing, episodic"
                    )
                print(f"Unfolding AFTER STEP is {unfolding}")

            # After the environment is truncated/terminated we use agent.step outside the loop one more time
            # to update specific agent's parameters, those of A or B matrices, based on the trajectory the
            # agent has realized; in the active inference literature this is an instance of learning.
            print(f"Unfolding before LEARNING is {unfolding}")
            action = agent.step(current_state, unfolding)

            print("-------- EPISODE SUMMARY --------")
            print(f"Step Count: {steps_count}")
            print(f"Goal reached: {terminated}")

            # Retrieve all agent's attributes, including episodic metrics we want to save
            all_metrics = agent.__dict__
            # Add key-value pair `total_reward` which is not among the agent's attributes
            all_metrics["total_reward"] = total_reward
            # Add key-value pair step count (i.e. time steps to termination or truncation)
            all_metrics["steps_count"] = steps_count + 1
            # Call the logs_writer function to save the episodic info we want
            # NOTE: unpack dictionary with `**` to feed the function with  key-value arguments
            logs_writer.log_episode(run, e, **all_metrics)

            # In a continuing task update AGENT_LOC with the last computed policy-independent state probabilities
            # NOTE: this is done so that the environment receives the correct option at the next episode
            if task_type == "continuing":
                AGENT_LOC = convert_state(int(np.argmax(agent.D)), env_layout)

            # Reset the agent before starting a new episode
            agent.reset()

            # Record num_videos uniformly distanced throughout the experiment
            # if num_videos != 0 and num_videos <= num_episodes:

            #     rec_step = num_episodes // num_videos
            #     if ((e + 1) % rec_step) == 0:

            #         env.make_video(str(e), VIDEO_DIR)

    # Save all collected data in a dictionary
    logs_writer.save_data(data_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time for {NUM_RUNS} runs: {elapsed_time:.2f} seconds")


# if __name__ == "__main__":
# main()
