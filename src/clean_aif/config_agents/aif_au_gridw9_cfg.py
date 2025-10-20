"""
Created on 22nd April 2025
@author: Filippo Torresan
"""

import os
from dataclasses import dataclass, field
from itertools import product
import numpy as np
from typing import Tuple


@dataclass
class Args:
    """
    Dataclass that defines and stores default parameters for the agent class.
    """

    ### General ###
    """the name of this experiment: either 'aif_au_gridw9' or 'aif_aa_gridw9'"""
    exp_name: str = "aif_au_gridw9"
    ### Environment ###
    """ Environment ID """
    gym_id: str = "GridWorld-v1"
    """ Environment layout """
    env_layout: str = "gridw9"  # choice: Tmaze3, Tmaze4, Ymaze4
    """ Max number of steps in an episode denoted by indices in [0, .., num_steps -1] """
    num_steps: int = 5
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
    start_state: int = 0
    """ index of goal state/location """
    goal_state: tuple = (8,)
    """ number of policies the agent considers for planning """
    num_policies: int = 256
    """ planning horizon, also the length of a policy """
    """ NOTE 1: also MAX number of future steps for which expected free energy is computed"""
    """ NOTE 2: the length of a policy should be num_steps - 1 because there is no action at the last time step"""
    plan_horizon: int = 4
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
        default_factory=lambda: Args.init_B_params(
            Args.num_states, Args.num_actions, Args.env_layout
        )
    )
    """ A params: specifies Dirichlet parameters to compute observation probabilities """
    A_params: np.ndarray = field(
        default_factory=lambda: Args.init_A_params(Args.num_states)
    )

    @staticmethod
    def init_policies(
        num_policies: int, policy_len: int, num_actions: int, exp_name: str
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

        if exp_name == "aif_au_gridw9":

            if num_policies == 1:

                sel_policies = np.array([[3, 3, 2]])

            elif num_policies == 2:

                sel_policies = np.array([[3, 3, 2], [0, 3, 1]])

            else:
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

        elif exp_name == "aif_aa_gridw9":
            # Array to store policies crated at the planning stage
            sel_policies = np.empty((0, policy_len))

        return sel_policies

    @staticmethod
    def goal_distribution_manh(num_states: int, goal_state: int, sharpness=1.0):
        """
        Returns a categorical distribution over `num_states` states, with highest probability at `goal_state`
        and smoothly decreasing based on the Manhattan distance of the goal with respect to agent's initial
        state. The `sharpness` parameter controls how smoothly the probability decreases.
        """

        l = np.sqrt(num_states).astype(int)
        index_repr = np.arange(num_states).reshape(l, l)
        y_goal, x_goal = np.where(index_repr == goal_state)
        manh_distances = []

        for s in np.arange(num_states):
            y, x = np.where(index_repr == s)
            manh_dist = np.abs(y - y_goal) + np.abs(x - x_goal)
            manh_distances.append(manh_dist)

        logits = -sharpness * np.array(manh_distances)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    @staticmethod
    def init_C_array(
        num_states: int,
        steps: int,
        goal_state: tuple,
        exp_name: str,
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
                # Divide remaining prob mass equally among goals
                prob_mass_goal = 0.9 / len(goal_state)
                for g in goal_state:
                    pref_array[g, -1] = prob_mass_goal
                print(pref_array)

            elif pref_loc == "all_goal":
                print("Setting agent's preferences...")
                # (2) Set higher preference for the goal state at each time step
                pref_array[:, :] = 0.1 / (num_states - 1)
                # Divide remaining prob mass equally among goals
                prob_mass_goal = 0.9 / len(goal_state)
                for g in goal_state:
                    pref_array[g, :] = prob_mass_goal
                # prob_mass_goal = [0.3, 0.6]
                # for i, g in enumerate(goal_state):
                #     pref_array[g, :] = prob_mass_goal[i]
                print(pref_array)

            elif pref_loc == "all_diff":
                print("Setting agent's preferences...")
                # (3) Define agent's preferences for each time step (i.e. a different goal for each step time)
                pref_array = np.ones((num_states, steps)) * (0.1 / (num_states - 1))

                # IMPORTANT: the probabilities below need to be set MANUALLY depending on the environment
                # in which the agent acts and based on the trajectory we want it to follow.
                # Fix a trajectory of intermediate goals leading to the goal in 5 steps
                inter_goals = [0, 1, 4, 7, 8]
                for t in range(Args.num_steps):
                    g = inter_goals[t]
                    pref_array[g, t] = 0.9

        elif pref_type == "states_manh":

            if pref_loc == "all_goal":
                print("Setting agent's preferences...")
                # NOTE: this assumes that the agent wants to reach a single goal so we take the first element of
                # the tuple goal_state
                prefs = Args.goal_distribution_manh(num_states, goal_state[0])
                # print(prefs.shape)
                pref_array[:, :] = prefs
                print(pref_array)

            elif pref_loc == "all_diff":
                print("Setting agent's preferences...")
                # Fix a trajectory of intermediate goals leading to the goal in 5 steps
                inter_goals = [0, 1, 4, 7, 8]
                for t in range(Args.num_steps):
                    g = inter_goals[t]
                    prefs = Args.goal_distribution_manh(num_states, g)
                    pref_array[:, t] = prefs.squeeze()

                print(pref_array)

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

        # Checking all the probabilities sum to one
        assert np.all(np.sum(pref_array, axis=0)) == 1, print(
            "The preferences do not sum to one!"
        )

        if exp_name == "aif_aa_gridw9":
            # Goal shaping is not implemented for action-aware agents so we select a single preference vector
            pref_array = pref_array[:, -1]

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

        # Init B_params matrix
        B_params = np.zeros((num_actions, num_states, num_states))
        # Creating a matrix of the same shape as the environment matrix filled with the tiles' labels
        n = int(np.sqrt(num_states))
        env_matrix_labels = np.reshape(np.arange(num_states), (n, n))

        # Loop over the 3 rows of the maze (indexed from 0 to 2) to assign 1s to the correct transitions.
        # IMPORTANT: The code below works for a maze of size (3, 3)
        for r in range(3):

            labels_ud = env_matrix_labels[r]
            labels_rl = env_matrix_labels[:, r]

            if r == 0:
                # NOTE: -1 in the y direction, from an external observer this would correspond to "up", in the
                # Gymnasium grid coordinate system the negative and positive y axes are swapped
                # Down action: 3
                B_params[3, labels_ud, labels_ud] = 1
                # Up action: 1
                B_params[1, labels_ud + 3, labels_ud] = 1
                # Right action: 0
                B_params[0, labels_rl + 1, labels_rl] = 1
                # Left action: 2
                B_params[2, labels_rl, labels_rl] = 1

            elif r == 1:
                # Down action: 3
                B_params[3, labels_ud - 3, labels_ud] = 1
                # Up action: 1
                B_params[1, labels_ud + 3, labels_ud] = 1
                # Right action: 0
                B_params[0, labels_rl + 1, labels_rl] = 1
                # Left action: 2
                B_params[2, labels_rl - 1, labels_rl] = 1
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
