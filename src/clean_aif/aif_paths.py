"""

Created at 18:00 on 21st July 2024
@author: Filippo Torresan

"""

# Standard libraries imports

# Standard libraries imports
import argparse
import os
import sys

# import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
import gymnasium
import gymnasium_env  # Needed otherwise NamespaceNotFound error
from itertools import product

import numpy as np
import os
from typing import TypedDict, cast, Tuple
from scipy import special
from pathlib import Path

# Custom imports
from .config import LOG_DIR
from .utils_paths import *


@dataclass
class Args:
    """
    Dataclass that defines and stores default parameters for the agent class.
    """

    ### General ###
    """the name of this experiment"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    ### Environment ###
    """ Environment ID """
    gym_id: str = "GridWorld-v1"
    """ Max number of steps in an episode denoted by indices in [0, .., num_steps -1] """
    num_steps: int = 3
    """ Number of environmental states (represented by indices 0,1,2,..,8) """
    num_states: int = 9
    ### Agent ###
    """ the number of observation channels or modalities """
    obs_channels: int = 1
    """ dimensions of observations for each channel """
    obs_dims: Tuple[int] = (1,)
    """ the number of factors in the environment """
    factors: int = 1
    """ dimensions of each factor """
    factors_dims: Tuple[int] = (1,)
    """ index of starting state (agent knows start location) """
    start_state: int = 4
    """ index of goal state/location """
    goal_state: int = 0
    """ number of policies the agent considers for planning """
    num_policies: int = 16
    """ planning horizon, also the length of a policy """
    """ NOTE 1: also MAX number of future steps for which expected free energy is computed"""
    """ NOTE 2: the length of a policy should be num_steps - 1 because there is no action at the last time step"""
    plan_horizon: int = 2
    """ number of actions (represented by indices 0,1,2,3)"""
    num_actions: int = 4
    """ hard-coded agent's policies """
    policies: np.ndarray = field(
        default_factory=lambda: Args.init_policies(
            Args.num_policies, Args.plan_horizon, Args.num_actions
        )
    )
    """ preference prior type """
    pref_type: str = "states"
    """ time step(s) on which the preference prior is placed """
    pref_loc: str = "all_goal"  # "last", all_goal", "all_diff"
    ### Agent's knowledge of the environment ###
    """NOTE: using field() to generate a default value for the attribute when an instance is created,
    by using `field(init=False)` we can pass a function with arguments (not allowed if we had used
    ``field(default_factory = custom_func)``)"""
    """ C array: specifies agent's preferred state(s) in the environment """
    C_params: np.ndarray = field(
        default_factory=lambda: Args.init_C_array(
            Args.num_states,
            Args.num_steps,
            Args.goal_state,
            Args.pref_type,
            Args.pref_loc,
        )
    )
    """ B params: specifies Dirichlet parameters to compute transition probabilities """
    B_params: np.ndarray = field(
        default_factory=lambda: Args.init_B_params(Args.num_states, Args.num_actions)
    )
    """ A params: specifies Dirichlet parameters to compute observation probabilities """
    A_params: np.ndarray = field(
        default_factory=lambda: Args.init_A_params(Args.num_states)
    )

    # @staticmethod
    # def init_policies() -> np.ndarray:
    #     """
    #     Method to specify the agent's policies for the duration of an episode; the agent is given some
    #     "motor plans" (sequences of actions) to try out and perform during an episode. Note: policies
    #     are usually hard coded in the discrete active inference literature (but also see recent
    #     implementations: pymdp).
    #     """

    #     # Policies to move in Gridworld-v1
    #     # NOTE: 0: "right", 1: "up", 2: "left", 3: "down"
    #     policies = np.array([[1, 1, 0, 2], [0, 0, 1, 1]])

    #    return policies

    @staticmethod
    def init_policies(
        num_policies: int, policy_len: int, num_actions: int
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

        # Init RNG for shuffling list of policies below
        rng = np.random.default_rng()
        # Set of actions
        actions = np.arange(num_actions, dtype=np.int64)
        # Create all the policies
        policies_list = [p for p in product(actions, repeat=policy_len)]
        # Convert list into array
        policies_array = np.array(policies_list, dtype=np.int64)
        # Number of all the sequences
        num_all_pol = num_actions**policy_len
        # All the row indices of policies_array
        indices = np.arange(num_all_pol)
        # Shuffle the indices
        rng.shuffle(indices)
        # Randomly select num_policies from the array with all the policies
        # NOTE 1: if num_policies equals the number of all sequences, the end result is just
        # policies_array with its rows shuffled
        # NOTE 2 (!!!ATTENTION!!!): if num_policies is NOT equal to the number of all sequencies,
        # the selected policies may not include the optimal policy in this implementation
        sel_policies = policies_array[indices[:num_policies], :]
        # print("Policies")
        # print(sel_policies)

        return sel_policies

    # def __post_init__(self):
    #     """
    #     Class method that runs at instance creation AFTER the dataclass is initialized, useful for additional
    #     initialization logic that depends on the instance attributes.
    #     """

    #     # Create and set preference array for the agent
    #     self.pref_array = self.create_pref_array(self.num_states, self.num_steps)

    @staticmethod
    def init_C_array(
        num_states: int,
        steps: int,
        goal_state: int,
        pref_type: str = "states",
        pref_loc: str = "last",
    ) -> np.ndarray:
        """
        Initialize preference array/matrix, denoted by C in the active inference literature, where each column
        represents the preferred/desired location for the agent at one step in the episode. In other words,
        each column corresponds to a categorical distribution.

        NOTE 1: the preferences could be either dense, telling the agent to prefer a single state at each
        time step (probability mass concentrated at ), or sparse, telling the agent defined either for every single step of the correct trajectory
        leading to the goal state or just for the goal state. Below we follow the latter approach
        (the former is commented out).
        NOTE 2: preferences can be either over states (default) or observations.

        Input:
        - num_states: number of states in the environment
        - steps: number of steps in an episode
        - goal_state: index of the state the agent wants to reach
        - pref_type: preference type ("state" or "obs")

        Ouput:

        - pref_array: np.ndarray (matrix) of shape (num_states, num_steps)
        """

        # Initialize preference matrix that will store the probabilities of being located on a certain
        # maze tile at each time step during an episode
        pref_array = np.ones((num_states, steps)) * (1 / num_states)

        if pref_type == "states":

            if pref_loc == "last":
                print("Setting agent's preferences...")
                # (1) At every time step all states have uniform probabilities except at the last time step
                # when the goal state is given the highest probability
                pref_array[:, -1] = 0.1 / (num_states - 1)
                pref_array[goal_state, -1] = 0.9
                print(pref_array)

            elif pref_loc == "all_goal":
                print("Setting agent's preferences...")
                # (2) Set higher preference for the goal state at each time step
                pref_array[:, :] = 0.1 / (num_states - 1)
                pref_array[goal_state, :] = 0.9
                print(pref_array)

            elif pref_loc == "all_diff":
                print("Setting agent's preferences...")
                # (3) Define agent's preferences for each time step (i.e. a different goal for each step time)
                pref_array = np.ones((num_states, steps)) * (0.1 / (num_states - 1))
                # IMPORTANT: the probabilities below need to be set MANUALLY depending on the environment
                # in which the agent acts and based on the trajectory we want it to follow.

                # Example: trajectory in a T-maze leading to the goal (on the left arm) in 3 steps
                pref_array[0, 2] = 0.9
                pref_array[1, 1] = 0.9
                pref_array[4, 0] = 0.9
                print(pref_array)

            # Checking all the probabilities sum to one
            assert np.all(np.sum(pref_array, axis=0)) == 1, print(
                "The preferences do not sum to one!"
            )

        elif pref_type == "obs":
            # NOTE 1: we are assuming a 1-to-1 correspondence between states and observations, i.e. obs `1`
            # indicates to the agent that it is in state `1`. Thus, the agent actually deals with an MDP as
            # opposed to a POMDP, and selecting either type of preferences does not make a difference.
            # NOTE 2; implement and use this kind of preferences with an actual POMDP (i.e., an observation
            # `1` of the environment may or may not indicate state `1`).

            # At every time step all states have uniform probabilities...
            pref_array[:-1, -1] = 0.1 / (num_states - 1)
            # ...except at the last time step when the goal state is given the highest probability
            pref_array[-1, -1] = 0.9
            # Checking all the probabilities sum to one
            assert np.all(np.sum(pref_array, axis=0)) == 1, print(
                "The preferences do not sum to one!"
            )

        return pref_array

    @staticmethod
    def init_B_params(num_states: int, num_actions: int) -> np.ndarray:
        """
        Initialize the Dirichlet parameters that specify the transition probabilities of the environment,
        stored and denoted by B matrices in the active inference literature, when the agent does not have to
        learn about them. The parameters are used in the agent's class to sample correct transition probabilities
        when the 'learn_B' flag passed via the command line is False.

        Input:
        - num_state (integer): no. of states in environment
        - num_actions (integer): no. of actions available to the agent

        Output:
        - B_params (np.ndarray): hard coded parmaters for the B matrices

        """

        B_params = np.zeros((num_actions, num_states, num_states))

        # Creating a matrix of the same shape as the environment matrix filled with the tiles' labels
        env_matrix_labels = np.reshape(np.arange(9), (3, 3))

        # Assigning 1s to correct transitions for every action.
        # IMPORTANT: The code below works for a maze of size (3, 3) only and specifically for env_layout = 't-maze-3'
        # TODO: Implement an automatic way to load these B-matrices depending on layout
        # Basically, we are looping over the 3 rows of the maze (indexed from 0 to 2)
        # and assigning 1s to the correct transitions.
        for r in range(3):

            labels_ud = env_matrix_labels[r]
            labels_rl = env_matrix_labels[:, r]

            if r == 0:
                # NOTE: -1 in the y direction, from an external observer this would correspond to "up", in the
                # Gymnasium grid coordinate system the negative and positive y axes are swapped
                # Down action: 3
                B_params[3, labels_ud, labels_ud] = 1
                # Up action: 1
                B_params[1, labels_ud[0], labels_ud[0]] = (
                    1  # Hitting wall and stay at the same state
                )
                B_params[1, labels_ud[1] + 3, labels_ud[1]] = 1
                B_params[1, labels_ud[2], labels_ud[2]] = (
                    1  # Hitting wall and stay at the same state
                )
                # Right action: 0
                B_params[0, labels_rl + 1, labels_rl] = 1
                # Left action: 2
                B_params[2, labels_rl, labels_rl] = 1

            elif r == 1:
                # Down action: 3
                B_params[3, labels_ud - 3, labels_ud] = 1
                # Up action: 1
                B_params[1, labels_ud[0] + 3, labels_ud[0]] = 1
                B_params[1, labels_ud[1], labels_ud[1]] = (
                    1  # Hitting wall and stay at the same state
                )
                B_params[1, labels_ud[2] + 3, labels_ud[2]] = 1
                # Right action: 0
                B_params[0, labels_rl[0] + 1, labels_rl[0]] = 1
                B_params[0, labels_rl[1], labels_rl[1]] = (
                    1  # Hitting wall and stay at the same state
                )
                B_params[0, labels_rl[2] + 1, labels_rl[2]] = 1
                # Left action: 2
                B_params[2, labels_rl[0] - 1, labels_rl[0]] = 1
                B_params[2, labels_rl[1], labels_rl[1]] = (
                    1  # Hitting wall and stay at the same state
                )
                B_params[2, labels_rl[2] - 1, labels_rl[2]] = 1

            elif r == 2:
                # Down action: 3
                B_params[3, labels_ud - 3, labels_ud] = 1
                # Up action: 1
                B_params[1, labels_ud, labels_ud] = 1
                # Right action: 0
                B_params[0, labels_rl, labels_rl] = 1
                # Left action: 2
                B_params[2, labels_rl - 1, labels_rl] = 1

        # Increasing the magnitude of the Dirichlet parameters so that when the B matrices are sampled
        # the correct transitions for every action will have a value close to 1.
        B_params = B_params * 199 + 1
        print(B_params[0])

        return B_params

    @staticmethod
    def init_A_params(num_states: int) -> np.ndarray:
        """
        Initialize the Dirichlet parameters that specify the state-observation mapping probabilities of the
        environment, stored and denoted by A matrices in the active inference literature, when the agent does
        not have to learn about them. The parameters are used in the agent's class to sample correct
        observation probabilities when the 'learn_A' flag passed via the command line is False.

        Input:
        - num_state (integer): no. of states in environment

        Output:
        - A_params (np.ndarray): hard coded parmaters for the A matrix

        """

        # Create matrix of 1s with 200s on the diagonal
        # NOTE: with these parameters there will be 1-to-1 correspondence between states and observations
        A_params = np.identity(num_states) * 199 + 1

        return A_params


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
    num_policies: int
    plan_horizon: int  # also included in Args
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
    pref_type: str
    policies: np.ndarray
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
        self.pref_loc: str = params["pref_loc"]
        self.policies: np.ndarray = params["policies"]
        self.num_policies: int = params["num_policies"]
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

            self.D = np.ones(self.num_states) * 0.0001
            self.D[self.start_state] = 1 - (0.0001 * (self.num_states - 1))

        # 2. Variational Distribution, initializing the relevant components used in the computation
        # of free energy and expected free energy:
        # - self.actions: list of actions;
        # - self.policies: numpy array (policy x actions) with pre-determined policies,
        # i.e. rows of num_steps actions;
        # - self.Qpi: categorical distribution over the policies, i.e. Q(pi);
        # - self.Qs_pi: categorical distributions over the states for each policy, i.e. Q(S_t|pi);
        # - self.Qt_pi: categorical distributions over the states for one specific S_t for each policy,
        # i.e. Q(S_t|pi);
        # - self.Qs: policy-independent states probabilities distributions, i.e. Q(S_t).

        # List of actions, dictionary with all the policies, and array with their corresponding probabilities.
        # Note 1: the policies are chosen with equal probability at the beginning.
        # Note 2: self.policies.shape[1] = (self.steps-1), i.e. the length of a policy is such that
        # it brings the agent to visit self.steps states (including the initial state) but at the last
        # time step (or state visited) there is no action.
        # Note 3 (!!! IMPORTANT !!!): the probabilities over policies are updated at every time step and the
        # updated values are saved at the corresponding time step in self.Qpi; this means that the first
        # updated values computed at `self.current_tstep = 0` overwrite the initialized value below.
        # In other words: at each time step we save the updated policies probabilities that become the prior
        # for the NEXT time step.
        self.actions = list(range(self.num_actions))
        self.policies = params.get("policies")
        self.Qpi = np.zeros((self.num_policies, self.steps))
        self.Qpi[:, 0] = np.ones(self.num_policies) * 1 / self.policies.shape[0]

        # State probabilities given a policy for every time step, the multidimensional array
        # contains self.steps distributions for each policy (i.e. policies*states*timesteps parameters).
        # In other words, every policy has a categorical distribution over the states for each time step.
        # Note 1: a simple way to initialise these probability values is to make every categorical
        # distribution a uniform distribution.
        self.Qs_pi = (
            np.ones((self.policies.shape[0], self.num_states, self.steps))
            * 1
            / self.num_states
        )
        # This multi-dimensional array is exactly like the previous one but is used for storing/logging
        # the state probabilities given a policy for the first step of the episode (while the previous array
        # is overwritten at every step and ends up logging the last step state beliefs for the episode)
        self.Qsf_pi = (
            np.ones((self.policies.shape[0], self.num_states, self.steps))
            * 1
            / self.num_states
        )
        # Policy conditioned state-beliefs throughout an episode, i.e. these matrices show how
        # all the Q(S_i|pi) change step after step by doing perceptual inference.
        self.Qt_pi = (
            np.ones((self.steps, self.policies.shape[0], self.num_states, self.steps))
            * 1
            / self.num_states
        )
        # Policy-independent states probabilities distributions, numpy array of size (num_states, timesteps).
        # See perception() method below.
        self.Qs = np.zeros((self.num_states, self.steps))

        # 3. Initialising arrays where to store agent's data during the experiment.
        # Numpy arrays where at every time step the computed free energies and expected free energies
        # for each policy and the total free energy are stored. For how these are calculated see the
        # methods below. We also stored separately the various EFE components.
        self.free_energies = np.zeros((self.policies.shape[0], self.steps))
        self.expected_free_energies = np.zeros((self.policies.shape[0], self.steps))
        self.efe_ambiguity = np.zeros((self.policies.shape[0], self.steps))
        self.efe_risk = np.zeros((self.policies.shape[0], self.steps))
        self.efe_Anovelty = np.zeros((self.policies.shape[0], self.steps))
        self.efe_Bnovelty = np.zeros((self.policies.shape[0], self.steps))
        self.efe_Bnovelty_t = np.zeros((self.policies.shape[0], self.steps))
        self.total_free_energies = np.zeros((self.steps))

        # Where the agent believes it is at each time step
        self.states_beliefs = np.zeros((self.steps))

        # Matrix of one-hot columns indicating the observation at each time step and matrix
        # of one-hot columns indicating the agent observation at each time step.
        # Note 1 (IMPORTANT): if the agent does not learn A and the environment is not stochastic,
        # then current_obs is the same as agent_obs.
        # Note 2: If the agent learns A, then at every time step what the agent actually observes is sampled
        # from A given the environment state.
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))

        # Numpy array storing the actual sequence of actions performed by the agent during the episode
        # (updated at every time step)
        self.actual_action_sequence = np.zeros((self.steps - 1))

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
        # Retrieving the softmax function needed to normalise the updated values of the Q(S_t|pi)
        # after performing gradient descent.
        sigma = special.softmax

        # Looping over the policies to calculate the respective free energies and their gradients
        # to perform gradient descent on the Q(S_t|pi).
        for pi, pi_actions in enumerate(self.policies):

            ### DEBUGGING ###
            # print(f"FE Minimization for Policy {pi}")
            # print(f'Policy actions {pi_actions}')
            # print(f"Time Step {self.current_tstep}")
            # print(f"Pi actions {pi_actions}")
            # print(f"First action {pi_actions[0]}")
            ### END ###

            ########### Update the Q(S_t|pi) by setting gradient to zero ##############

            for _ in range(self.inf_iters):

                # IMPORTANT: here we are replacing zero probabilities with the value 0.0001
                # to avoid zeroes in logs.
                self.Qs_pi = np.where(self.Qs_pi == 0, 0.0001, self.Qs_pi)

                # Computing the variational free energy for the current policy
                # Note 1: if B parameters are learned then you need to pass in self.B_params and
                # self.learning_B (the same applies for A)
                logA_pi, logB_pi, logD_pi, F_pi = vfe(
                    self.num_states,
                    self.steps,
                    self.current_tstep,
                    self.current_obs,
                    pi,
                    pi_actions,
                    self.A,
                    self.B,
                    self.D,
                    self.Qs_pi,
                    A_params=self.A_params,
                    learning_A=self.learning_A,
                    B_params=self.B_params,
                    learning_B=self.learning_B,
                )

                # Computing the free energy gradient for the current policy
                grad_F_pi = grad_vfe(
                    self.num_states,
                    self.steps,
                    self.current_tstep,
                    self.current_obs,
                    pi,
                    self.Qs_pi,
                    logA_pi,
                    logB_pi,
                    logD_pi,
                )

                # Simultaneous beliefs updates
                # Note: the update equation below is based on the computations of Da Costa, 2020, p. 9,
                # by setting the gradient to zero one can solve for the parameters that minimize that gradient,
                # here we are recovering those solutions *from* the gradient (by subtraction) before applying
                # a softmax to make sure we get valid probabilities.
                # IMPORTANT: when using a mean-field approx. in variational inference (like it is commonly
                # done in vanilla active inference) the various factors, e.g., the Q(S_t|pi), are updated
                # one at time by keeping all the others fixed. Here, we are instead using a simultaneous
                # update of all the factors, possibly repeating this operation a few times. However,
                # results seem OK even if the for loop iterates just for one step.

                # print("Gradient for update:")
                # print(f"{grad_F_pi}")
                self.Qs_pi[pi, :, :] = sigma(
                    (-1) * (grad_F_pi - np.log(self.Qs_pi[pi, :, :]) - 1) - 1,
                    axis=0,
                )

                # print("Qs_pi after update:")
                # print(f"{self.Qs_pi}")

                # Storing the state beliefs at the first step of the episode
                if self.current_tstep == 0:
                    self.Qsf_pi[pi, :, :] = self.Qs_pi[pi, :, :]
                # self.Qs_pi[pi, :, :] = sigma(-self.Qpi[pi, -1] * grad_F_pi, axis=0)
            ######### END ###########

            # Printing the free energy value for current policy at current time step
            # print(f"Time Step: {self.current_tstep}")
            # print(f" FE_pi_{pi}: {F_pi}")
            # Storing the last computed free energy in self.free_energies
            self.free_energies[pi, self.current_tstep] = F_pi
            # Computing the policy-independent state probability at self.current_tstep and storing
            # it in self.Qs. This is commented out because it might make more sense to do it after
            # updating the probabilities over policies (in the planning method, see below), however
            # it could be also done here using the updated Q(S_t|pi) and the old Q(pi).
            # self.Qs[:, self.current_tstep] += (
            #     self.Qs_pi[pi, :, self.current_tstep] * self.Qpi[pi, self.current_tstep]
            # )

            # Storing S_i probabilities for a certain index i, e.g., the index that corresponds to the
            # final step to see whether the agent ends up believing that it will reach the goal state
            # at the end of the episode by following the corresponding policy.
            for t in range(self.steps):
                self.Qt_pi[t, pi, :, self.current_tstep] = self.Qs_pi[pi, :, t]

        ### DEBUGGING ###
        # assert np.sum(self.Qs[:, self.current_tstep], axis=0) == 1, "The values of the policy-independent state probability distribution at time " + str(self.current_tstep) + " don't sum to one!"
        ### END ###

        # Identifying and storing the state the agents believes to be in at the current time step.
        # There might be no point in doing this here if the state-independent probabilities are not
        # computed above, i.e., one is getting the most probable state the agent believes to be in
        # based on Q(S_t|pi) and the old Q(pi) updated *at the previous time step*.
        # TODO: check that commenting this out does not cause any error
        # state_belief = np.argmax(self.Qs[:, self.current_tstep], axis=0)
        # self.states_beliefs[self.current_tstep] = state_belief

    def planning(self):
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

        for pi, pi_actions in enumerate(self.policies):

            # At the last time step only update Q(pi) with the computed free energy
            # (because there is no expected free energy then). for all the other steps
            # compute the total expected free energy over the remaining time steps.
            if self.current_tstep == (self.steps - 1):
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
                G_pi, tot_Hs, tot_slog_s_over_C, tot_AsW_As, tot_AsW_Bs, sq_AsW_Bs = (
                    efe(
                        self.num_states,
                        self.steps,
                        self.current_tstep,
                        self.efe_tsteps,
                        pi,
                        pi_actions,
                        self.A,
                        self.C,
                        self.Qs_pi,
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
                print(f"--- Policy {pi} ---")
                print(f"--- Summary of planning at time step {self.current_tstep} ---")
                print(f"FE_{pi}: {F_pi}")
                print(f"EFE_{pi}: {G_pi}")
                print(f"Risk_{pi}: {tot_slog_s_over_C}")
                print(f"Ambiguity {pi}: {tot_Hs}")
                print(f"A-novelty {pi}: {tot_AsW_As}")
                print(f"B-novelty {pi}: {tot_AsW_Bs}")
                #### END ####
                # if self.current_tstep == 0:
                #     print(f"B-novelty sequence at t ZERO: {sq_AsW_Bs}")
                #     self.efe_Bnovelty_t[pi] += sq_AsW_Bs
                #     print(
                #         f"B-novelty sequence by policy (stored): {self.efe_Bnovelty_t}"
                #     )

        # Normalising the negative expected free energies stored as column in self.Qpi to get
        # the posterior over policies Q(pi) to be used for action selection
        print(f"Computing posterior over policy Q(pi)...")
        self.Qpi[:, self.current_tstep] = sigma(-self.Qpi[:, self.current_tstep])
        # print(f"Before adding noise - Q(pi): {self.Qpi}")
        # Replacing zeroes with 0.0001, to avoid the creation of nan values and multiplying by 5 to make sure
        # the concentration of probabilities is preserved when reapplying the softmax
        self.Qpi[:, self.current_tstep] = np.where(
            self.Qpi[:, self.current_tstep] == 1, 5, self.Qpi[:, self.current_tstep]
        )
        self.Qpi[:, self.current_tstep] = np.where(
            self.Qpi[:, self.current_tstep] == 0,
            0.0001,
            self.Qpi[:, self.current_tstep],
        )
        self.Qpi[:, self.current_tstep] = sigma(self.Qpi[:, self.current_tstep])
        # Computing the policy-independent state probability at self.current_tstep and storing it in self.Qs
        for pi, _ in enumerate(self.policies):
            #### DEBUGGING ####
            # print(f"For policy {pi}")
            # print("Before update:")
            # print(f"{self.Qs}")
            # print("Using:")
            # print(f"{self.Qs_pi[pi, :, :]}")
            # print(f"{self.Qpi[pi, self.current_tstep]}")
            #### END ####
            # NOTE: here we are updating the values of Qs using += because the array is initialized with zeroes
            # without any prior values whereas Qpi above was initialized with uniform probabilities at index 0
            self.Qs[:, self.current_tstep] += (
                self.Qs_pi[pi, :, self.current_tstep] * self.Qpi[pi, self.current_tstep]
            )

        ### DEBUGGING ###
        # self.Qs_test = np.zeros_like(self.Qs)
        # self.Qs_test[:, self.current_tstep] += (
        #     self.Qs_pi[:, :, self.current_tstep].T @ self.Qpi[:, self.current_tstep]
        # )
        # print(self.Qs_test)
        # print(self.policies)
        # print(self.Qs_pi[:, :, 1])
        # print(self.Qs_pi[:, :, 2])
        print(f"Updated Qs at times step {self.current_tstep}")
        # print(self.Qs[:, self.current_tstep])
        print(f"Most probable state: {np.argmax(self.Qs[:, self.current_tstep])}")
        ### END ###

        # Check if the agent has reached the last time step of the episode
        if self.task_type == "continuing" and self.current_tstep + 1 == self.steps:
            # Save the last policy-independent state probabilities as prior state probabilities for
            # the next episode
            self.D = self.Qs[:, self.current_tstep]
            print(f"Prior for next episode:")
            print(self.Qs[:, self.current_tstep])
            print(
                f"Most probable state for next episode: {np.argmax(self.Qs[:, self.current_tstep])}"
            )

    def action_selection_KD(self):
        """Method for action selection based on the Kronecker delta, as described in Da Costa et. al. 2020,
        (DOI: 10.1016/j.jmp.2020.102447). It involves using the approximate posterior Q(pi) to select the most
        likely action, this is done through a Bayesian model average.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.

        Note 1: the matrix self.Qpi is indexed with self.current_tstep + 1 (the next time step) because
        we store the approximate posterior in the next available column, with the first one being occupied
        by the initial Q(pi). This makes sense because the new Q(pi) becomes the prior for the next time step.
        But because it is also the approximate posterior at the current time step it is used for selecting
        the action.
        """

        # Matrix of shape (num_actions, num_policies) with each row being populated by the same integer,
        # i.e. the index for an action, e.g. np.array([[0,0,0,0,0],[1,1,1,1,1],..]) if there are
        # five policies.
        actions_matrix = np.array([self.actions] * self.policies.shape[0]).T

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
        # TODO: what if instead of taking the greedy action action we sample?
        argmax_actions = np.argwhere(actions_probs == np.amax(actions_probs)).squeeze()

        if argmax_actions.shape == ():

            action_selected = argmax_actions

        else:

            action_selected = self.rng.choice(argmax_actions)

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
            Stp1 = np.dot(
                self.Qs_pi[:, :, self.current_tstep + 1].T,
                self.Qpi[:, self.current_tstep],
            )
            # Computing distributions over observations
            AS_tp1 = np.dot(self.A, Stp1)
            ABS_t = np.dot(
                self.A, np.dot(self.B[a, :, :], self.Qs[:, self.current_tstep])
            )
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

        NOTE: below we are retrieving the pi probability values at the index 'current_tstep' because in the
        planning method the *updated* policy probabilities are saved in Qpi at the current time step, i.e.
        using self.current_tstep as index.

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        """

        argmax_policies = np.argwhere(
            self.Qpi[:, self.current_tstep] == np.amax(self.Qpi[:, self.current_tstep])
        ).squeeze()

        if argmax_policies.shape == ():

            action_selected = self.policies[argmax_policies, self.current_tstep]

        else:

            action_selected = self.rng.choice(
                self.policies[argmax_policies, self.current_tstep]
            )

        return action_selected

    def learning(self):
        """Method for parameters learning. This occurs at the end of the episode.
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

        #### DEBUGGING ####
        # print("Params for dirichlet update")
        # print(f"Observation list: {self.current_obs}")
        # print(f"Action sequence: {self.actual_action_sequence}")
        # print("Policy independent state probabilities:")
        # print(f"{self.Qs}")
        #### END ####
        print("Updating Dirichlet parameters...")
        # NOTE: below we retrieve the second last Qpi because that corresponds to the last time step an
        # action was selected by the agent (no action is selected at the truncation or termination point)
        # and that is all that is needed by the dirichlet_update() method
        self.A_params, self.B_params = dirichlet_update(
            self.num_states,
            self.num_actions,
            self.steps,
            self.current_obs,
            self.actual_action_sequence,
            self.policies,
            self.Qpi[:, -2],
            self.Qs_pi,
            self.Qs,
            self.A_params,
            self.B_params,
            self.learning_A,
            self.learning_B,
        )

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

            print("Updated parameters for matrix A only.")
            # After learning A's parameters, for every state draw one sample from the Dirichlet
            # distribution using the corresponding column of parameters.
            for s in range(self.num_states):
                self.A[:, s] = self.rng.dirichlet(self.A_params[:, s], size=1)

            print("New matrix A:")
            print(f" {self.A}")

        elif self.learning_A == False and self.learning_B == True:

            print("Updated parameters for matrices Bs only.")
            # After learning B's parameters, sample from them to update the B matrices, i.e. for every
            # action and state draw one sample from the Dirichlet distribution using the corresponding
            # column of parameters.
            for a in range(self.num_actions):
                for s in range(self.num_states):
                    self.B[a, :, s] = self.rng.dirichlet(self.B_params[a, :, s], size=1)

        elif self.learning_A == False and self.learning_B == False:

            print("No update (matrices A and Bs not subject to learning).")

    def update_C(self):
        """
        Function to update agent's preferences.
        """

        # Compute action probabilities, i.e. P(a)
        actions_matrix = np.array([self.actions] * self.num_policies).T
        actions_logits = np.zeros((self.num_actions))

        for t in range(self.efe_tsteps):
            actions_logits += np.matmul(
                (self.policies[:, t] == actions_matrix),
                self.Qpi[:, -1],
            )

        actions_probs = special.softmax(actions_logits)

        # Compute action-conditioned surprisal for each state value
        last_qs = self.Qs[:, -1]
        new_prefs = np.zeros_like(last_qs)

        for a in range(self.num_actions):
            new_prefs = actions_probs[a] * np.sum(
                -((self.B_params[a] * last_qs.T) @ np.log(self.B_params[a].T))
                @ np.eye(self.num_states),
                axis=1,
            )

            new_prefs += new_prefs

        new_prefs = special.softmax(new_prefs)[:, np.newaxis]
        self.C = new_prefs * np.ones((1, self.steps))
        print(self.C.shape)
        print("New preferences set to:")
        print(self.C)
        print(f"Agent new goal is: state {np.argmax(self.C[:, -1])}")

    def step(self, new_obs):
        """This method brings together all computational processes defined above, forming the
        perception-action loop of an active inference agent at every time step during an episode.

        NOTE: the computation of total free energy is done AFTER perception and planning because it involves
        both the old policy probabilities, saved in Qpi at `current_tstep - 1`, and the updated policy
        probabilities saved in Qpi at `current_tstep`.

        Inputs:
        - new_obs: the state from the environment's env_step method (based on where the agent ended up
        after the last step, e.g., an integer indicating the tile index for the agent in the maze).

        Outputs:
        - self.current_action: the action chosen by the agent.
        """

        # During an episode the counter self.current_tstep goes up by one unit at every time step
        self.current_tstep += 1
        # Updating the matrix of observations and agent obs with the observations at the first time step
        self.current_obs[new_obs, self.current_tstep] = 1

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
        agent_observation = np.random.multinomial(1, self.A[:, new_obs], size=None)
        self.agent_obs[:, self.current_tstep] = agent_observation

        # During an episode perform perception, planning, and action selection based on current observation
        if self.current_tstep < (self.steps - 1):

            self.perception()
            self.planning()
            # Computing the total free energy and store it in self.total_free_energies
            # (as a reference for the agent performance)
            total_F = total_free_energy(
                self.current_tstep, self.steps, self.free_energies, self.Qpi
            )

            print("---------------------")
            print("--- 3. ACTING ---")

            self.current_action = eval(self.select_action)
            # Storing the selected action in self.actual_action_sequence
            self.actual_action_sequence[self.current_tstep] = self.current_action

        # At the end of the episode (terminal state), do perception and update the A and/or B's parameters
        # (an instance of learning)
        elif self.current_tstep == (self.steps - 1):
            # Saving the P(A) and/or P(B) used during the episode before parameter learning,
            # in this way we conserve the priors for computing the KL divergence(s) for the
            # total free energy at the end of the episode (see below).
            prior_A = self.A_params
            prior_B = self.B_params
            # Perception (state-estimation)
            self.perception()
            # Planning (expected free energy computation)
            # Note 1 (IMPORTANT): at the last time step self.planning() only serves to update Q(pi) based on
            # the past as there is no expected free energy to compute.
            self.planning()
            # Learning (parameter's updates)
            self.learning()
            self.current_action = None
            # Computing the total free energy and store it in self.total_free_energies (as a reference
            # for the agent performance)
            total_F = total_free_energy(
                self.current_tstep,
                self.steps,
                self.free_energies,
                self.Qpi,
                prior_A,
                prior_B,
                A_params=self.A_params,
                learning_A=self.learning_A,
                B_params=self.B_params,
                learning_B=self.learning_B,
            )

        # Store total free energy in self.total_free_energies (as a reference for the agent performance)
        self.total_free_energies[self.current_tstep] = total_F

        return self.current_action

    def reset(self):
        """This method is used to reset certain variables before starting a new episode.

        Specifically, the observation matrix, self.current_obs, and self.Qs should be reset at the beginning
        of each episode to store the new sequence of observations etc.; also, the matrix with the probabilities
        over policies stored in the previous episode, self.Qpi, should be rinitialized so that at time step 0
        (zero) the prior over policies is the last computed value from the previous episode.
        """

        # Initializing current action and step variables
        self.current_action = None
        self.current_tstep = -1
        # Setting self.current_obs and self.agent_obs to a zero array before starting a new episode
        self.current_obs = np.zeros((self.num_states, self.steps))
        self.agent_obs = np.zeros((self.num_states, self.steps))
        # Setting self.Qs to a zero array before starting a new episode
        self.Qs = np.zeros((self.num_states, self.steps))
        # Resetting sequence of B-novelty values at t = 0
        self.efe_Bnovelty_t = np.zeros((self.policies.shape[0], self.steps))
        # Initializing self.Qpi so that the prior over policies is equal to the last probability distribution
        # computed.
        # Note 1: this is done at all episodes except for the very first. To single out the first episode
        # case we check whether it is populated by zeros (because it is initialized as such when the agent
        # object is instantiated).
        if np.sum(self.Qpi[:, 1], axis=0) == 0:
            # Do nothing because self.Qpi is already initialized correctly
            pass
        else:
            # New prior probability distribution over policies
            ppd_policies = self.Qpi[:, -1]
            self.Qpi = np.zeros((self.num_policies, self.steps))
            self.Qpi[:, 0] = ppd_policies


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
        self.policies: np.ndarray = params.get("policies")
        self.num_policies: int = params.get("num_policies")
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
        """Policy dependent free energies at each step during every episode"""
        self.pi_free_energies: np.ndarray = np.zeros(
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
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
        )
        """Observations collected by the agent at each step during an episode"""
        self.observations: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_states, self.num_max_steps)
        )
        """Policy independent probabilistic beliefs about environmental states"""
        self.states_beliefs: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_max_steps)
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
                self.num_max_steps,
            )
        )
        """Policy dependent probabilistic beliefs about environmental states (first episode step)"""
        self.policy_state_prob_first: np.ndarray = np.zeros(
            (
                self.num_runs,
                self.num_episodes,
                self.num_policies,
                self.num_states,
                self.num_max_steps,
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
                self.num_max_steps,
            )
        )
        """Probabilities of the policies at each time step during every episode"""
        self.pi_probabilities: np.ndarray = np.zeros(
            (self.num_runs, self.num_episodes, self.num_policies, self.num_max_steps)
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

    def log_step(self, run, episode, next_state):
        """Method to log data at every step"""

        # Adding a unit to the state counter visits for the new state reached
        self.state_visits[run][episode][next_state] += 1

    def log_episode(self, run: int, episode: int, **kwargs):
        """Method to log data at end of every episode"""

        # At the end of the episode, storing the total reward in reward_counts and other info
        # accumulated by the agent, e.g the total free energies, expected free energies etc.
        # (this is done for every episode and for every run).
        self.reward_counts[run][episode] = kwargs["total_reward"]
        self.pi_free_energies[run, episode, :, :] = kwargs["free_energies"]
        self.total_free_energies[run, episode, :] = kwargs["total_free_energies"]
        self.expected_free_energies[run, episode, :, :] = kwargs[
            "expected_free_energies"
        ]
        self.efe_ambiguity[run, episode, :, :] = kwargs["efe_ambiguity"]
        self.efe_risk[run, episode, :, :] = kwargs["efe_risk"]
        self.efe_Anovelty[run, episode, :, :] = kwargs["efe_Anovelty"]
        self.efe_Bnovelty[run, episode, :, :] = kwargs["efe_Bnovelty"]
        self.efe_Bnovelty_t[run, episode, :, :] = kwargs["efe_Bnovelty_t"]
        self.observations[run, episode, :, :] = kwargs["current_obs"]
        self.states_beliefs[run, episode, :] = kwargs["states_beliefs"]
        self.actual_action_sequence[run, episode, :] = kwargs["actual_action_sequence"]
        self.policy_state_prob[run, episode, :, :, :] = kwargs["Qs_pi"]
        # np.set_printoptions(precision=3, suppress=True)
        # print(f'Qs_pi: {kwargs["Qs_pi"]}')
        self.policy_state_prob_first[run, episode, :, :, :] = kwargs["Qsf_pi"]
        self.every_tstep_prob[run, episode, :, :, :, :] = kwargs["Qt_pi"]
        self.pi_probabilities[run, episode, :, :] = kwargs["Qpi"]
        self.so_mappings[run, episode, :, :] = kwargs["A"]
        self.transitions_prob[run, episode, :, :, :] = kwargs["B"]

    def save_data(self, log_dir, file_name="data"):
        """Method to save to file the collected data"""

        # Dictionary to store the data
        data = {}
        # Populate dictionary with corresponding key
        data["num_runs"] = self.num_runs
        data["num_episodes"] = self.num_episodes
        data["num_states"] = self.num_states
        data["num_steps"] = self.num_max_steps
        data["policies"] = self.policies
        data["num_policies"] = self.num_policies
        data["learn_A"] = self.learnA
        data["learn_B"] = self.learnB
        data["state_visits"] = self.state_visits
        data["reward_counts"] = self.reward_counts
        data["pi_free_energies"] = self.pi_free_energies
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
        data["so_mappings"] = self.so_mappings
        data["transition_prob"] = self.transitions_prob
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
        default="aif-paths",
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
        default="t-maze-2",
        help="layout of the gridworld (choices: t-maze-2, t-maze-3)",
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
        help="number of policies (i.e. sequences of actions) for planning",
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
    # Preference type prior
    # NOTE: this is just a label used to identify the experiment, make sure it corresponds
    # to the attribute/property set in the agent Args class, see top of file
    parser.add_argument(
        "--pref_type",
        "-pft",
        type=str,
        default="states",
        help="choices: states, obs",
    )
    # Time step(s) on which preference prior is placed
    # NOTE: this is just a label used to identify the experiment, make sure it corresponds
    # to the attribute/property set in the agent Args class, see top of file
    parser.add_argument(
        "--pref_loc",
        "-pfl",
        type=str,
        default="last",
        help="choices: last, all_goal, all_diff",
    )

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
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S_")
    # Create string of experiment-specific info
    exp_info = (
        f'{cl_params["gym_id"]}_{cl_params["env_layout"]}_{cl_params["exp_name"]}_{cl_params["task_type"]}'
        f'_nr{cl_params["num_runs"]}_ne{cl_params["num_episodes"]}_steps{cl_params["num_steps"]}'
        f'_infsteps{cl_params["inf_steps"]}_preftype_{cl_params["pref_type"]}_prefloc_{cl_params["pref_loc"]}'
        f'_AS{cl_params["action_selection"]}'
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

    # Create dataclass with default parameters configuration for the agent
    agent_params = Args()
    # Convert dataclass to dictionary
    agent_params = asdict(agent_params)

    # Update agent_params with corresponding key values in cl_params, and/or add key from cl_params
    # Custom update function not overwriting default parameter's value if the one from the CL is None
    def update_params(default_params, new_params):
        for key, value in new_params.items():
            # Make sure these keys have values that correspond to how the agent was initialized
            # i.e. the command line arguments do not change these agent's properties
            if key == "pref_type" or key == "pref_loc":
                assert (
                    value == default_params[key]
                ), f"Agent was initialized with {key}: {default_params[key]} but you passed {value} as argument."
            elif value is not None:
                default_params[key] = value

    update_params(agent_params, cl_params)
    # print(agent_params)

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
    if env_layout == "t-maze-2":
        WALLS_LOC = [
            convert_state(3),
            convert_state(5),
            convert_state(6),
            convert_state(7),
            convert_state(8),
        ]
    elif env_layout == "t-maze-3":
        WALLS_LOC = [convert_state(1), convert_state(6), convert_state(8)]
    else:
        raise ValueError(
            "Value of 'env_layout' is not among the available ones. Choose from: t-maze-2, t-maze-3."
        )
    # Fix target location in the environment (the same in every episode)
    TARGET_LOC = convert_state(agent_params["goal_state"])

    # Create the environment
    env = gymnasium.make(
        "gymnasium_env/GridWorld-v1", max_episode_steps=NUM_STEPS, render_mode=None
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

    np.set_printoptions(precision=3, suppress=True)

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
            agent_params["start_state"]
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

            # Retrieve observation of the agent's location
            # print(f"Observation: {obs}; type {type(obs)}")
            obs = obs["agent"]
            # Convert obs into index representation
            start_state = process_obs(obs)
            # Adding a unit to the state_visits counter for the start_state
            logs_writer.log_step(run, e, start_state)
            # Current state (updated at every step and passed to the agent)
            current_state = start_state

            # Agent and environment interact for NUM_STEPS steps.
            # NOTE: we start counting episode steps from 0 (steps_count = 0) and we have NUM_STEPS steps
            # (say, 5) so `steps_count < NUM_STEPS ensures the loop lasts for NUM_STEPS.
            while steps_count < NUM_STEPS:
                # Agent returns an action based on current observation/state
                action = agent.step(current_state)
                # Except when at the last episode's step, the agent's action affects the environment;
                # at the last time step the environment does not change but the agent engages in learning
                # (parameters update)
                if steps_count < NUM_STEPS - 1:
                    # Environment outputs based on agent action
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    # Retrieve observation of the agent's location
                    next_obs = next_obs["agent"]
                    # Convert observation into index representation
                    next_state = process_obs(next_obs)
                    # Update total_reward
                    total_reward += reward

                    print("-------- STEP SUMMARY --------")
                    print(f"Time step: {steps_count}")
                    print(f"Current state: {current_state}")
                    print(f"Agent action: {action}")
                    print(f"Next state: {next_state}")
                    print(f"Terminal/Goal state: {terminated}")

                    # Adding a unit to the state_visits counter for the start_state
                    logs_writer.log_step(run, e, next_state)
                    # Update current state with next_state
                    current_state = next_state

                # Update step count
                steps_count += 1

            # Retrieve all agent's attributes, including episodic metrics we want to save
            all_metrics = agent.__dict__
            # Adding the key-value pair `total_reward` which is not among the agent's attributes
            all_metrics["total_reward"] = total_reward
            # Call the logs_writer function to save the episodic info we want
            # NOTE: unpack dictionary with `**` to feed the function with  key-value arguments
            logs_writer.log_episode(run, e, **all_metrics)

            # In a continuing task update AGENT_LOC with the last computed policy-independent state probabilities
            # NOTE: this is done so that the environment receives the correct option at the next episode
            if task_type == "continuing":
                AGENT_LOC = convert_state(int(np.argmax(agent.D)))

            # Reset the agent before starting a new episode
            agent.reset()

            # Record num_videos uniformly distanced throughout the experiment
            # if num_videos != 0 and num_videos <= num_episodes:

            #     rec_step = num_episodes // num_videos
            #     if ((e + 1) % rec_step) == 0:

            #         env.make_video(str(e), VIDEO_DIR)

    # Save all collected data in a dictionary
    logs_writer.save_data(data_path)


# if __name__ == "__main__":
# main()
