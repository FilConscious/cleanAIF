"""

Created at 18:00 on 21st July 2024
@author: Filippo Torresan

"""

# Standard libraries imports

# Standard libraries imports
import os
import argparse

# import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
import gymnasium
import gymnasium_env  # Needed otherwise NamespaceNotFound error
import numpy as np
import os
from typing import TypedDict, cast
from scipy import special
from pathlib import Path

# Custom imports
from .config import PROJECT_ROOT, LOG_DIR, RESULTS_DIR
from .utils import *


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
    """ Max number of steps in an episode """
    num_steps: int = 5
    """ Number of environmental states (represented by indices 0,1,2,..,8) """
    num_states: int = 9
    ### Agent ###
    """ the number of observation channels or modalities """
    obs_channels: int = 1
    """ dimensions of observations for each channel """
    obs_dims: list = [1]
    """ the number of factors in the environment """
    factors: int = 1
    """ dimensions of each factor """
    factors_dims: list = [1]
    """ index of starting state (agent knows start location) """
    start_state: int = 0
    """ index of goal state/location """
    goal_state: int = 8
    """ planning horizon, also the length of a policy """
    """ NOTE: also MAX number of future steps for which expected free energy is computed """
    plan_horizon: int = 4
    """ number of actions (represented by indices 0,1,2,3)"""
    num_actions: int = 4
    """ hard-coded agent's policies """
    policies: np.ndarray = field(default_factory=lambda: Args.init_policies())
    ### Agent's knowledge of the environment ###
    """NOTE: using field() to generate a default value for the attribute when an instance is created,
    by using `field(init=False)` we can pass a function with arguments (not allowed if we had used
    ``field(default_factory = custom_func)``)"""
    """ C array: specifies agent's preferred state(s) in the environment """
    C_array: np.ndarray = field(
        default_factory=lambda: Args.init_C_array(Args.num_states, Args.num_steps)
    )
    """ B params: specifies Dirichlet parameters to compute transition probabilities """
    B_params: np.ndarray = field(
        default_factory=lambda: Args.init_B_params(Args.num_states, Args.num_actions)
    )
    """ A params: specifies Dirichlet parameters to compute observation probabilities """
    A_params: np.ndarray = field(
        default_factory=lambda: Args.init_A_params(Args.num_states)
    )

    @staticmethod
    def init_policies() -> np.ndarray:
        """
        Method to specify the agent's policies for the duration of an episode; the agent is given some
        "motor plans" (sequences of actions) to try out and perform during an episode. Note: policies
        are usually hard coded in the discrete active inference literature (but also see recent
        implementations: pymdp).
        """

        # Policies to move in Gridworld-v1 TODO: make sure these corresponds to valid action in the enviroment
        policies = np.array([[2, 2, 1, 0], [1, 1, 2, 2]])

        return policies

    # def __post_init__(self):
    #     """
    #     Class method that runs at instance creation AFTER the dataclass is initialized, useful for additional
    #     initialization logic that depends on the instance attributes.
    #     """

    #     # Create and set preference array for the agent
    #     self.pref_array = self.create_pref_array(self.num_states, self.num_steps)

    @staticmethod
    def init_C_array(
        num_states: int, steps: int, pref_type: str = "state"
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
        - pref_type: preference type ("state" or "obs")

        Ouput:

        - pref_array: np.ndarray (matrix) of shape (num_states, num_steps)
        """

        # Initialize preference matrix that will store the probabilities of being located on a certain
        # maze tile at each time step during an episode
        pref_array = np.ones((num_states, steps)) * (1 / num_states)

        if pref_type == "states":

            # Defining the agent's preferences over states, these are crucial to the computation of
            # expected free energy
            # pref_array = np.ones((num_states, steps)) * (0.1 / (num_states - 1))
            # pref_array[8, 4] = 0.9
            # pref_array[5, 3] = 0.9
            # pref_array[2, 2] = 0.9
            # pref_array[1, 1] = 0.9
            # pref_array[0, 0] = 0.9
            # assert np.all(np.sum(pref_array, axis=0)) == 1, print('The preferences do not sum to one!')

            # At every time step all states have uniform probabilities...
            pref_array[:-1, -1] = 0.1 / (num_states - 1)
            # ...except at the last time step when the goal state is given the highest probability
            pref_array[-1, -1] = 0.9
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
        # IMPORTANT: The code below works for a maze of size (3, 3) only.
        # Basically, we are looping over the 3 rows of the maze (indexed from 0 to 2)
        # and assigning 1s to the correct transitions.
        for r in range(3):

            labels_ud = env_matrix_labels[r]
            labels_rl = env_matrix_labels[:, r]

            if r == 0:
                # Up action
                B_params[0, labels_ud, labels_ud] = 1
                # Down action
                B_params[2, labels_ud + 3, labels_ud] = 1
                # Right action
                B_params[1, labels_rl + 1, labels_rl] = 1
                # Left action
                B_params[3, labels_rl, labels_rl] = 1

            elif r == 1:
                # Up action
                B_params[0, labels_ud - 3, labels_ud] = 1
                # Down action
                B_params[2, labels_ud + 3, labels_ud] = 1
                # Right action
                B_params[1, labels_rl + 1, labels_rl] = 1
                # Left action
                B_params[3, labels_rl - 1, labels_rl] = 1

            elif r == 2:
                # Up action
                B_params[0, labels_ud - 3, labels_ud] = 1
                # Down action
                B_params[2, labels_ud, labels_ud] = 1
                # Right action
                B_params[1, labels_rl, labels_rl] = 1
                # Left action
                B_params[3, labels_rl - 1, labels_rl] = 1

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
    seed: float
    inf_steps: int
    pref_type: str
    num_policies: int
    plan_horizon: int  # also included in Args
    action_selection: str
    learn_A: bool
    learn_B: bool
    learn_D: bool
    num_videos: int
    # Parameters unique to class Args above
    exp_name: str
    num_states: int
    obs_channels: int
    obs_dim: list
    factors: int
    factors_dims: list
    start_state: int
    goal_state: int
    num_actions: int
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
        self.policies: np.ndarray = params["policies"]
        self.num_policies: int = params["num_policies"]
        self.as_mechanism: str = params["action_selection"]
        self.learning_A: bool = params["learn_A"]
        self.learning_B: bool = params["learn_B"]
        self.learning_D: bool = params["learn_D"]
        self.rng = np.random.default_rng(seed=params.get("random_seed", 42))

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
        # Note 3: the probabilities over policies change at every step except the last one;
        # in the self.Qpi's column corresponding to the last step we store the Q(pi) computed at
        # the previous time step.
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
            # print(f'Policy {pi}')
            # print(f'Policy actions {pi_actions}')
            # print(f'Time Step {self.current_tstep}')
            ### END ###

            # IMPORTANT: the parameters of the categorical Q(S_t|pi) can be updated by
            # gradient descent on (variational) free energy or using analytical updates
            # resulting from setting the gradient to zero. Both methods are spelled out
            # below but just one is commented out.
            # TODO: the selection of the method should occur via the command line.

            ######### 1. Update the Q(S_t|pi) with gradient descent #########
            # next_F = 1000000
            # last_F = 0
            # epsilon = 1
            # delta_w = 0
            # gamma = 0.3
            # counter = 0

            # # Gradient descent on Free Energy: while loop until the difference between the next
            # # and last free energy values becomes less than or equal to epsilon.
            # while next_F - last_F > epsilon:

            #     counter += 1

            #     # Computing the free energy for the current policy and gradient descent iteration
            #     # Note 1: if B parameters are learned then you need to pass in self.B_params and
            #     # self.learning_B (the same applies for A)
            #     logA_pi, logB_pi, logD_pi, F_pi = vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, pi_actions,
            #                                             self.A, self.B, self.D, self.Qs_pi, A_params=self.A_params, learning_A=self.learning_A,
            #                                             B_params=self.B_params, learning_B=self.learning_B)

            #     # Computing the free energy gradient for the current policy and gradient descent iteration
            #     grad_F_pi = grad_vfe(self.num_states, self.steps, self.current_tstep, self.current_obs, pi, self.Qs_pi, logA_pi, logB_pi, logD_pi)

            #     # Note 1: the updates are stored in the corresponding agent's attributes
            #     # so they are immediately available at the next iteration.

            #     # Simultaneous gradient update using momentum
            #     # Note 1 (IMPORTANT!): with momentum the gradient descent works better and leads to better results, however it still does not
            #     # prevent overshooting in certain episodes with the free energy diverging (it oscillates between two values). So, below we stop
            #     # the gradient update after a certain number of iterations.
            #     self.Qs_pi[pi, :, :] = (self.Qs_pi[pi, :, :] - self.learning_rate_F * grad_F_pi + gamma * delta_w)
            #     self.Qs_pi[pi, :, :] = sigma( self.Qs_pi[pi, :, :] - np.amax(self.Qs_pi[pi, :, :], axis=0) , axis=0)

            #     delta_w = gamma * delta_w - self.learning_rate_F * grad_F_pi

            #     # Updating temporary variables to compute the difference between previous and next free energies to decide when to stop
            #     # the gradient update (i.e. when the absolute value of the different is below epsilon).
            #     if counter == 1:

            #         next_F = F_pi

            #     elif counter > 1:

            #         last_F = next_F
            #         next_F = F_pi

            #     # IMPORTANT: stopping the gradient updates after 20 iterations to avoid free energy divergence.
            #     if counter > 20:
            #         break

            ########## END ###########

            ########### 2. Update the Q(S_t|pi) by setting gradient to zero ##############

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
                print(f"BEFORE update, Qs_{pi}: {self.Qs_pi[pi,:,3]}")
                # print(f"Gradient for update: {grad_F_pi}")
                self.Qs_pi[pi, :, :] = sigma(
                    (-1) * (grad_F_pi - np.log(self.Qs_pi[pi, :, :]) - 1) - 1, axis=0
                )
                print(f"AFTER update, Qs_{pi}: {self.Qs_pi[pi,:,3]}")

                # Storing the state beliefs at the first step of the episode
                if self.current_tstep == 0:
                    self.Qsf_pi[pi, :, :] = self.Qs_pi[pi, :, :]
                # self.Qs_pi[pi, :, :] = sigma(-self.Qpi[pi, -1] * grad_F_pi, axis=0)
            ######### END ###########

            # Printing the free energy value for current policy at current time step
            print(f"Time Step: {self.current_tstep}")
            print(f" FE_pi_{pi}: {F_pi}")
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

            print(self.Qt_pi.shape)

            # if self.current_tstep == 0 and pi==1:
            #    print(f'This is Q(S_1|pi_1): {self.Qs_pi[pi, :, 1]}')

        ### DEBUGGING ###
        # assert np.sum(self.Qs[:, self.current_tstep], axis=0) == 1, "The values of the policy-independent state probability distribution at time " + str(self.current_tstep) + " don't sum to one!"
        ### END ###

        # Identifying and storing the state the agents believes to be in at the current time step.
        # There might be no point in doing this here if the state-independent probabilities are not
        # computed above, i.e., one is getting the most probable state the agent believes to be in
        # based on Q(S_t|pi) and the old Q(pi) updated *at the previous time step*.
        # TODO: check that commenting this out does not cause any error
        state_belief = np.argmax(self.Qs[:, self.current_tstep], axis=0)
        self.states_beliefs[self.current_tstep] = state_belief

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
                print(f"--- Summary of planning at time step {self.current_tstep} ---")
                print(f"FE_{pi}: {F_pi}")
                print(f"EFE_{pi}: {G_pi}")
                print(f"Risk_{pi}: {tot_slog_s_over_C}")
                print(f"Ambiguity {pi}: {tot_Hs}")
                print(f"A-novelty {pi}: {tot_AsW_As}")
                print(f"B-novelty {pi}: {tot_AsW_Bs}")

                if self.current_tstep == 0:
                    print(f"B-novelty sequence at t ZERO: {sq_AsW_Bs}")
                    self.efe_Bnovelty_t[pi] += sq_AsW_Bs
                    print(
                        f"B-novelty sequence by policy (stored): {self.efe_Bnovelty_t}"
                    )
                    # if sq_AsW_Bs[2] > 2200:
                    #     raise Exception("B-novelty too high")

        # Normalising the negative expected free energies stored as column in self.Qpi to get
        # the posterior over policies Q(pi) to be used for action selection
        print(f"Computing posterior over policy Q(pi)...")
        self.Qpi[:, self.current_tstep] = sigma(-self.Qpi[:, self.current_tstep])
        print(f"Before adding noise - Q(pi): {self.Qpi}")
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
        print(f"After adding noise - Q(pi): {self.Qpi}")
        # Computing the policy-independent state probability at self.current_tstep and storing it in self.Qs
        self.Qs[:, self.current_tstep] += (
            self.Qs_pi[pi, :, self.current_tstep] * self.Qpi[pi, self.current_tstep]
        )

        ### DEBUGGING ###
        # if self.current_tstep == 5:
        #      print(f'Prob for policy {0}: {self.Qpi[0, self.current_tstep+1]}')
        #      print(f'Prob for policy {1} {self.Qpi[1, self.current_tstep+1]}')
        ### END ###

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
        argmax_actions = np.argwhere(actions_probs == np.amax(actions_probs)).squeeze()

        if argmax_actions.shape == ():

            action_selected = argmax_actions

        else:

            action_selected = self.rng.choice(argmax_actions)

        return action_selected

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

        Inputs:
            - None.
        Outputs:
            - action_selected (integer): the action the agent is going to perform.
        """

        argmax_policies = np.argwhere(
            self.Qpi[:, self.current_tstep + 1]
            == np.amax(self.Qpi[:, self.current_tstep + 1])
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
        print("Updating Dirichlet parameters...")
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

        elif self.learning_A == False and self.learning_B == False:

            print("No update (matrices A and Bs not subject to learning).")

    def step(self, new_obs):
        """This method brings together all computational processes defined above, forming the
        perception-action loop of an active inference agent at every time step during an episode.

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
            print("---------------------")
            print("--- 3. ACTING ---")
            self.current_action = eval(self.select_action)
            # Computing the total free energy and store it in self.total_free_energies
            # (as a reference for the agent performance)
            total_F = total_free_energy(
                self.current_tstep, self.steps, self.free_energies, self.Qpi
            )

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
        "--exp-name",
        type=str,
        default="aif-pi-paths",
        help="the name of this experiment based on the active inference implementation",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="GridWorld-v1",
        help="the name of the registered gym environment (choices: GridWorld-v1)",
    )
    parser.add_argument(
        "--num-runs",
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
        nargs="?",
        help="number of steps per episode or total number of steps (depending on type of agent)",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=2.5e-4,
        help="the learning rate for the free energy gradients",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")

    ### Agent-specific arguments ###
    # Inference
    parser.add_argument(
        "--inference_steps",
        "-inf_steps",
        type=int,
        default=1,
        help="number of free energy minimization steps",
    )
    # Agent's preferences type
    parser.add_argument(
        "--pref_type",
        "-pt",
        type=str,
        default="states",
        help="choices: states, observations",
    )
    # Policy
    parser.add_argument(
        "--num_policies",
        "-np",
        type=int,
        default=2,
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
    # Create folder (with dt_string as unique identifier) where to store data from current experiment.
    data_path = LOG_DIR.joinpath(
        dt_string
        + f'{cl_params["env_name"]}r{cl_params["num_runs"]}e{cl_params["num_episodes"]}prF{cl_params["pref_type"]}AS{cl_params["action_selection"]}lA{str(cl_params["learn_A"])[0]}lB{str(cl_params["learn_B"])[0]}lD{str(cl_params["learn_D"])[0]}'
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # if not (os.path.exists(data_path)):
    #     os.makedirs(data_path)

    ###############################
    ### 3. INIT AGENT PARAMETERS
    ##############################

    # Create dataclass with default parameters configuration for the agent
    agent_params = Args()
    # Convert dataclass to dictionary
    agent_params = asdict(agent_params)
    # Update agent_params with corresponding key values in cl_params, and/or add key from cl_params
    agent_params.update(cl_params)

    ##########################
    ### 4. INIT ENV
    ##########################

    # Retrieve name of the environment
    env_module_name = cl_params["gym-id"]
    # Number of runs (or agents interacting with the env)
    NUM_RUNS = agent_params["num_runs"]
    # Number of episodes
    NUM_EPISODES = agent_params["num_episodes"]
    # Number of steps in one episode
    NUM_STEPS = agent_params["num_steps"]
    # Fix agent's location at the beginning of every episode (i.e. agent starts always from the same location)
    # NOTE: we need to convert the following various states from an index to a (x, y) representation which is
    # what the Gymnasium environment requires.
    AGENT_LOC = convert_state(agent_params["start_state"])  # output: np.array([0, 0])
    # Fix walls location in the environment (the same in every episode)
    WALLS_LOC = [convert_state(4)]  # output: np.array([1, 1])
    # Fix target location in the environment (the same in every episode)
    TARGET_LOC = convert_state(agent_params["goal_state"])

    # Create the environment
    env = gymnasium.make(
        "gymnasium_env/GridWorld-v1", max_episode_steps=NUM_STEPS, render_mode="human"
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

    ### TRAINING LOOP ###
    # Loop over number of runs and episodes
    for run in range(NUM_RUNS):
        # Print iteration (run) number
        print("************************************")
        print(f"Starting Run {run}...")
        # Set a random seed for current run, used by RNG attribute in the agent
        agent_params["seed"] += run
        # Create agent (`cast()` is used to tell the type checker that `agent_params` is of type `params`)
        agent = Agent(cast(params, agent_params))
        # Loop over episodes
        for e in range(NUM_EPISODES):
            # Printing episode number
            print("--------------------")
            print(f"Episode {e}")
            print("--------------------")

            # Initialize steps and reward counters
            steps_count = 0
            total_reward = 0
            # is_terminal = False

            # Reset the environment
            obs, info = env.reset(
                options={
                    "deterministic_agent_loc": AGENT_LOC,
                    "deterministic_target_loc": TARGET_LOC,
                    "deterministic_wall_loc": WALLS_LOC,
                },
            )

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
                    # Convert observation into index representation
                    next_state = process_obs(next_obs)
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

            # Retrieve all agent's attributes, including episodic metrics we want to save
            all_metrics = agent.__dict__
            # Call the logs_writer function to save the episodic info we want
            # NOTE: unpack dictionary with `**` to feed the function with  key-value arguments
            logs_writer.log_episode(run, e, **all_metrics)
            # Reset the agent before starting a new episode
            agent.reset()

            # Record num_videos uniformly distanced throughout the experiment
            # if num_videos != 0 and num_videos <= num_episodes:

            #     rec_step = num_episodes // num_videos
            #     if ((e + 1) % rec_step) == 0:

            #         env.make_video(str(e), VIDEO_DIR)

    # Save all collected data in a dictionary
    logs_writer.save_data(LOG_DIR)


# if __name__ == "__main__":
# main()
