"""
Created at 18:00 on 21st July 2024
@author: Filippo Torresan
"""

from dataclasses import dataclass, field

# import math
import numpy as np
import os
from typing import Tuple


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
    """ Environment layout """
    env_layout: str = "gridw9"  # choice: Tmaze3, Tmaze4, Ymaze4
    """ Max number of steps in an episode """
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
    """ number of policies the agent consider at each planning step """
    num_policies: int = 256
    """ planning horizon, also the length of a policy """
    """ NOTE: also MAX number of future steps for which expected free energy is computed """
    plan_horizon: int = 4
    """ number of actions (represented by indices 0,1,2,3)"""
    num_actions: int = 4
    """ init empty agent's policies arrays """
    policies: np.ndarray = field(
        default_factory=lambda: Args.init_policies(Args.plan_horizon)
    )
    """ preferences type """
    pref_type: str = "states"
    ### Agent's knowledge of the environment ###
    """NOTE: using field() to generate a default value for the attribute when an instance is created,
    by using `field(init=False)` we can pass a function with arguments (not allowed if we had used
    ``field(default_factory = custom_func)``) TODO: Clarify this note!!!!"""
    """ C array: specifies agent's preferred state(s) in the environment """
    C_params: np.ndarray = field(
        default_factory=lambda: Args.init_C_array(
            Args.num_states, Args.goal_state, Args.pref_type
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

    # !!! ATTENTION !!!: probably not needed, check if it is OK removing
    @staticmethod
    def init_policies(policy_len: int) -> np.ndarray:
        """
        Create initial agent's policies array.

        """

        # Array to store policies crated at the planning stage
        policies = np.empty((0, policy_len))

        return policies

    # def __post_init__(self):
    #     """
    #     Class method that runs at instance creation AFTER the dataclass is initialized, useful for additional
    #     initialization logic that depends on the instance attributes.
    #     """

    #     # Create and set preference array for the agent
    #     self.pref_array = self.create_pref_array(self.num_states, self.num_steps)
    #
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
    def init_C_array(num_states: int, goal_state: tuple, pref_type: str) -> np.ndarray:
        """
        Initialize preference array, denoted by C in the active inference literature. The vector
        stores the parameters of a categorical distribution with the probability mass concentrated on
        the preferred/desired location of the agent in the maze environment.

        NOTE 1: preferences can be either over states (default) or observations.

        Input:
        - num_states: number of states in the environment
        - pref_type: preference type ("state" or "obs")

        Ouput:

        - pref_array: np.ndarray (matrix) of shape (num_states, 1)
        """

        # Initialize preference vector
        pref_array = np.ones((num_states, 1)) * (1 / num_states)

        if pref_type == "states":
            print("Setting agent's preferences...")
            # Assign probability to non-goal states...
            pref_array[:, 0] = 0.1 / (num_states - 1)
            # Divide remaining prob mass equally among goals
            prob_mass_goal = 0.9 / len(goal_state)
            for g in goal_state:
                pref_array[g, 0] = prob_mass_goal
            # prob_mass_goal = [0.3, 0.6]
            # for i, g in enumerate(goal_state):
            #     pref_array[g, :] = prob_mass_goal[i]
            # print(pref_array)
            # print(pref_array.shape)

            # Checking all the probabilities sum to one
            assert np.all(np.sum(pref_array, axis=0)) == 1, print(
                "The preferences do not sum to one!"
            )

        elif pref_type == "states_manh":
            # NOTE: this assumes that the agent wants to reach a single goal so we take the first element of
            # the tuple goal_state
            prefs = Args.goal_distribution_manh(num_states, goal_state[0])
            # print(prefs)
            # print(prefs.shape)
            pref_array = prefs

        elif pref_type == "obs":
            # NOTE 1: we are assuming a 1-to-1 correspondence between states and observations, i.e. obs `1`
            # indicates to the agent that it is in state `1`. Thus, the agent actually deals with an MDP as
            # opposed to a POMDP, and selecting either type of preferences does not make a difference.
            # NOTE 2; implement and use this kind of preferences with an actual POMDP (i.e., an observation
            # `1` of the environment may or may not indicate state `1`).

            # Assign probability to non-goal observation...
            pref_array[:-1, 1] = 0.1 / (num_states - 1)
            # Assign probability to goal observation
            pref_array[-1, 1] = 0.9
            # Checking all the probabilities sum to one
            assert np.all(np.sum(pref_array, axis=0)) == 1, print(
                "The preferences do not sum to one!"
            )

        return pref_array

    @staticmethod
    def init_B_params(num_states: int, num_actions: int, env_layout: str) -> np.ndarray:
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

        # Assigning 1s to correct transitions for every action.
        if env_layout == "tmaze3":
            # IMPORTANT: The code below works for a maze of size (3, 3); with flag env_layout = 'Tmaze3'
            # only transitions to accessible states are considered/modelled

            # NOTE: -1 in the y direction, from an external observer this would correspond to "up", in the
            # Gymnasium grid coordinate system the negative and positive y axes are swapped
            # Down action: 3
            B_params[3, :, :] = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.float64,
            )
            # Left action: 2
            B_params[2, :, :] = np.array(
                [
                    [1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            # Up action: 1
            B_params[1, :, :] = np.array(
                [
                    [1, 0, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                dtype=np.float64,
            )
            # Right action: 0
            B_params[0, :, :] = np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )

        # Assigning 1s to correct transitions for every action.
        elif env_layout == "tmaze4":
            # IMPORTANT: The code below works for a maze of size (3, 3); with flag env_layout = 'Tmaze4'
            # only transitions to accessible states are considered/modelled

            # NOTE: -1 in the y direction, from an external observer this would correspond to "up", in the
            # Gymnasium grid coordinate system the negative and positive y axes are swapped
            # Down action: 3
            B_params[3, :, :] = np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float64,
            )
            # Left action: 2
            B_params[2, :, :] = np.array(
                [
                    [1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            # Up action: 1
            B_params[1, :, :] = np.array(
                [
                    [1, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            # Right action: 0
            B_params[0, :, :] = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                dtype=np.float64,
            )

        elif env_layout == "gridw9":
            # Creating a matrix of the same shape as the environment matrix filled with the tiles' labels
            n = int(np.sqrt(num_states))
            env_matrix_labels = np.reshape(np.arange(num_states), (n, n))

            # IMPORTANT: The code below works for a maze of size (3, 3); with flag env_layout = 'TmazeXall'
            # all transitions are considered/modelled, also towards not accessible states (walls)
            # TODO: Implement an automatic way to load these B-matrices depending on layout
            # We are looping over the 3 rows of the maze (indexed from 0 to 2)
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
