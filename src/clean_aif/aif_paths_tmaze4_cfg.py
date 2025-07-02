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
    """the name of this experiment"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    ### Environment ###
    """ Environment ID """
    gym_id: str = "GridWorld-v1"
    """ Environment layout """
    env_layout: str = "tmaze4"  # choice: tmaze3, tmaze4, ymaze4
    """ Max number of steps in an episode denoted by indices in [0, .., num_steps -1] """
    num_steps: int = 4
    """ Number of environmental states (represented by indices 0,1,2,..,8) """
    num_states: int = 5
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
    goal_state: tuple = (0,)
    """ number of policies the agent considers for planning """
    num_policies: int = 64
    """ planning horizon, also the length of a policy """
    """ NOTE 1: also MAX number of future steps for which expected free energy is computed"""
    """ NOTE 2: the length of a policy should be num_steps - 1 because there is no action at the last time step"""
    plan_horizon: int = 3
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
        goal_state: list,
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
                    pref_array[g, -1] = prob_mass_goal
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
            # TODO: need to be aligned with options allowed in the case of state preferences
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
        if env_layout == "Tmaze3":
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
        elif env_layout == "Tmaze4":
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
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1],
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

        # Assigning 1s to correct transitions for every action.
        elif env_layout == "Ymaze4":
            # IMPORTANT: The code below works for a maze of size (3, 3); with flag env_layout = 'Tmaze4'
            # only transitions to accessible states are considered/modelled

            # NOTE: -1 in the y direction, from an external observer this would correspond to "up", in the
            # Gymnasium grid coordinate system the negative and positive y axes are swapped
            # Down action: 3
            B_params[3, :, :] = np.array(
                [
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float64,
            )
            # Left action: 2
            B_params[2, :, :] = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            # Up action: 1
            B_params[1, :, :] = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                ],
                dtype=np.float64,
            )
            # Right action: 0
            B_params[0, :, :] = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float64,
            )

        elif env_layout == "TmazeXall":
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
