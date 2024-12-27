"""

Created at 18:00 on 21st July 2024
@author: Filippo Torresan

"""

# Standard libraries imports
import importlib
import copy
import gymnasium as gym
import numpy as np
import os
import time
import tyro
import random
from scipy import special
from dataclasses import dataclass, field

# Custom packages/modules imports
# from ..agents.utils_actinf import *


@dataclass
class Args:
    ### General ###
    """t he name of this experiment"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    ### Environment ###
    """ Environment ID """
    env_id: str = "GridWorld-v1"
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


class AifAgent(object):
    """AifAgent class to implement active inference algorithm in a discrete POMDP setting."""

    def __init__(self, params: dict):
        """
        Inputs:
        - params (dict): the parameters used to initialize the agent (see description below)
        """

        # Make sure the dictionary has the required key-value pairs
        try:
            # list of the dimensions of each observation channel
            self.obs_dims: list = params["obs_dims"]
            # list of the dimensions of each factors in the environment
            self.factors_dims: list = params["factors_dims"]
            # number of free energy minimization steps
            self.inference_steps: int = params["inference_steps"]
            # list of dictionaries, representing factor-specific action channels
            # note: each key-value pair represent the meaning and the integer associated with an action
            self.actions_dict: list = params["actions"]
            # number of policies (i.e. sequences of actions) for planning
            self.num_policies: int = params["num_policies"]
            # planning horizon (i.e. length of a policy)
            self.plan_horizon: int = params["plan_horizon"]
            # whether agent preferred distribution is over obs or states
            self.pref_type: str = params["pref_type"]
            # how an action is picked from the distribution over policies
            self.action_selection: str = params["action_selection"]
            # list of parameters for obs, transition, and initial state matrices
            # Note: if the list are empty these parameters are subject to learning
            self.a_matricesp: list = params["A"]
            self.b_matricesp: list = params["B"]
            self.c_vectorsp: list = params["C"]
            self.d_vectorsp: list = params["D"]
            # initialise the random number generator for numpy
            self.rng = np.random.default_rng(seed=params.get("random_seed", 42))
        except:
            print("Please pass all the required parameters to initialise the agent.")

        ### General info ###
        # number of observation channels or modalities
        self.num_obs_channels: int = len(self.obs_dims)
        # number of factors in the environment
        self.num_factors: int = len(self.factors_dims)
        # number of observation matrices, one for each combination of observation channel and factor
        self.num_a_matrices: int = self.num_obs_channels * self.num_factors
        # unpack actions (aka controls or control states) for each factor into list of list
        self.action_channels = [list(d.values()) for d in self.actions_dict]
        # number of transition matrices, one for each combination of action and factor
        # self.num_b_matrices = self.num_factors * self.num_actions

        ### Generative model ###
        # Initialise factor-observatins mappings (self.A), transition matrices (self.B), preference vectors
        # or preferred stationary distributions (self.C), etc. as ndarray with dtype=object, i.e. each element
        # of the array is a Python object, specifically another numpy ndarray

        # Initialise container of A matrices and Dirichlet parameters (state-obser)
        self.a_matrices = np.zeros(
            (self.num_obs_channels, self.num_factors), dtype=object
        )
        self.a_params = np.zeros(
            (self.num_obs_channels, self.num_factors), dtype=object
        )
        # Case when the A matrices are learned via an update of the Dirichlet parameters
        if len(self.a_matricesp) == 0:
            # Looping over obs channel-factor pairs
            for c in range(self.num_obs_channels):
                for f in range(self.num_factors):
                    # Init matrix of Dirichlet parameters for matrix A of obs channel c and factor f
                    self.a_params[c, f] = np.ones(
                        (self.obs_dims[c], self.factors_dims[f])
                    )
                    # Init matrix A of obs channel c and factor f
                    self.a_matrices[c, f] = np.ones(
                        (self.obs_dims[c], self.factors_dims[f])
                    )
                    # Draw one sample from the associated Dirichlet distribution so that each column of
                    # matrix A encodes a proper categorical distribution P(o^c|s^f)
                    for s in range(self.factors_dims[f]):
                        self.a_matrices[c, f][:, s] = self.rng.dirichlet(
                            self.a_params[c, f][:, s], size=1
                        )
        # Case when the A matrices are NOT learned (need to be provided)
        else:
            self.set_a_matrix(self.a_matricesp)

        # Initialise container of B matrices and Dirichlet parameters
        self.b_matrices = np.zeros((1, self.num_factors), dtype=object)
        self.b_params = np.zeros((1, self.num_factors), dtype=object)
        # Case when the B matrices are learned via an update of the Dirichlet parameters
        if len(self.b_matricesp) == 0:
            # Looping over action-factor pairs
            for f, actions in enumerate(self.action_channels):
                for a in actions:
                    # Init matrix of Dirichlet parameters for matrix B of action a and factor f
                    self.b_params[f] = np.ones(
                        (a, self.factors_dims[f], self.factors_dims[f])
                    )
                    # Init matrix B of action a and factor f
                    self.b_matrices[f] = np.ones(
                        (a, self.factors_dims[f], self.factors_dims[f])
                    )
                    # Draw one sample from the associated Dirichlet distribution so that each column of
                    # matrix B encodes a proper categorical distribution P(s^f|s^f, a)
                    for s in range(self.factors_dims[f]):
                        self.b_matrices[f][a, :, s] = self.rng.dirichlet(
                            self.B_params[f][a, :, s], size=1
                        )
        # Case when the B matrices are NOT learned (need to be provided)
        else:
            self.set_b_matrix(self.b_matricesp)

        # Initialise container of C vectors and Dirichlet parameters
        self.c_vectors = np.zeros((1, self.num_factors), dtype=object)
        self.c_params = np.zeros((1, self.num_factors), dtype=object)
        # Case when the C vectors are learned via an update of the Dirichlet parameters
        if len(self.c_vectorsp) == 0:
            raise NotImplementedError
        # Case when the C vectors are provided
        else:
            self.set_c_vectors(self.c_vectorsp)

        # Initialise container of D vectors and Dirichlet parameters
        self.d_vectors = np.zeros((1, self.num_factors), dtype=object)
        self.d_params = np.zeros((1, self.num_factors), dtype=object)
        # Case when the D vectors are learned via an update of the Dirichlet parameters
        if len(self.d_vectorsp) == 0:
            raise NotImplementedError
        # Case when the D vectors are provided
        else:
            self.set_d_vectors(self.d_vectorsp)

        ### Variational Distribution ###
        # Agent's categorical distributions of factors at initial time step (prior beliefs),
        # Note: we will stack in here the posteriors at subsequent time steps up to the present
        self.q_beliefs = copy.deepcopy(self.d_vectors)

        ### Logging arrays ###
        # Free energies (FE) for each factor at each time step (array's shape: time x factors)
        self.free_energies = np.zeros((1, self.num_factors))
        # Total expected free energies (EFEs) at each time step (each is a sum of EFEs)
        self.expected_free_energies = np.zeros((1, self.num_factors))
        # Same as above but for the EFEs terms
        self.efe_ambiguity = np.zeros((1, self.num_factors))
        self.efe_risk = np.zeros((1, self.num_factors))
        self.efe_a_novelty = np.zeros((1, self.num_factors))
        self.efe_b_novelty = np.zeros((1, self.num_factors))
        # History of observations received from environment for each factor up to the present
        # Note: each object is a one-hot vector representing the state of a factor
        self.obs_history = np.zeros((1, self.num_factors), dtype=object)
        # History of actions for each factor up to the present
        self.actions_history = np.zeros((self.steps - 1))

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
        if self.action_selection == "kd":
            # Action selection mechanism with Kronecker delta (KD) as described in Da Costa et. al. 2020,
            # (DOI: 10.1016/j.jmp.2020.102447).
            self.select_action = "self.action_selection_KD()"

        elif self.action_selection == "kl":
            # Action selection mechanism with Kullback-Leibler divergence (KL) as described
            # in Sales et. al. 2019, 'Locus Coeruleus tracking of prediction errors optimises
            # cognitive flexibility: An Active Inference model'.
            self.select_action = "self.action_selection_KL()"

        elif self.action_selection == "probs":
            # Action selection mechanism naively based on updated policy probabilities
            self.select_action = "self.action_selection_probs()"

        else:

            raise Exception("Invalid action selection mechanism.")

    def set_a_matrix(self, obs_params: list):
        """
        Set the agent's observation matrices with arrays provided externally.
        Input:

        - obs_params: list of lists

        Note: each list in obs_params should store the parameters of the factor-conditioned
        observation matrices relative to one observation channel.
        """

        # Looping over obs channel-factor pairs
        for c in range(self.num_obs_channels):
            for f in range(self.num_factors):
                # Init matrix of Dirichlet parameters for matrix A of obs channel c and factor f
                self.a_params[c, f] = obs_params[c][f]
                # Init matrix A of obs channel c and factor f
                self.a_matrices[c, f] = np.zeros(
                    (self.obs_dims[c], self.factors_dims[f])
                )
                # Draw one sample from the associated Dirichlet distribution so that each column of
                # matrix A encodes a proper categorical distribution P(o^c|s^f)
                for s in range(self.factors_dims[f]):
                    self.a_matrices[c, f][:, s] = self.rng.dirichlet(
                        self.a_params[c, f][:, s], size=1
                    )

    def set_b_matrix(self, trs_params: list):
        """
        Set the agent's transitions (or control) matrices with arrays provided externally.
        Input:

        - trs_params: list of lists

        Note: each list in trs_params should store the parameters of the factor-conditioned
        transition matrices relative to one action.
        """

        # Looping over action-factor pairs
        for a in range(self.num_actions):
            for f in range(self.num_factors):
                # Init matrix of Dirichlet parameters for matrix B of action a and factor f
                self.b_params[a, f] = trs_params[a][f]
                # Init matrix B of action a and factor f
                self.b_matrices[a, f] = np.zeros(
                    (self.factors_dims[f], self.factors_dims[f])
                )
                # Draw one sample from the associated Dirichlet distribution so that each column of
                # matrix B encodes a proper categorical distribution P(s^f|s^f, a)
                for s in range(self.factors_dims[f]):
                    self.b_matrices[a, f][:, s] = self.rng.dirichlet(
                        self.b_params[a, f][:, s], size=1
                    )

    def set_c_vectors(self, pref_params: list):
        """
        Set the agent's preferences vectors with arrays provided externally.
        Input:

        - pref_params: list of arrays

        Note: each array in pref_params should store the parameters of the factor-conditioned
        preferences matrices.
        """

        # Looping over factors
        for f in range(self.num_factors):
            # Init vector of Dirichlet parameters for vector C of factor f
            self.c_params[1, f] = pref_params[f]
            # Init vector C of preferences for factor f
            self.c_vectors[1, f] = np.zeros((1, self.factors_dims[f]))
            # Draw one sample from the associated Dirichlet distribution so that vector C encodes
            # a proper categorical distribution P(s^f)
            self.c_vectors[1, f][:] = self.rng.dirichlet(self.c_params[1, f][:], size=1)

    def set_d_vectors(self, start_params: list):
        """
        Set the agent's beliefs over initial factor states with arrays provided externally.
        Input:

        - start_params: list of arrays

        Note: each array in start_params should store the parameters of the factor-conditioned initial beliefs.
        """

        # Looping over factors
        for f in range(self.num_factors):
            # Init vector of Dirichlet parameters for vector D of factor f
            self.d_params[1, f] = start_params[f]
            # Init vector D of initial factor beliefs for factor f
            self.d_vectors[1, f] = np.zeros((1, self.factors_dims[f]))
            # Draw one sample from the associated Dirichlet distribution so that vector D encodes
            # a proper categorical distribution P(s^f)
            self.d_vectors[1, f][:] = self.rng.dirichlet(self.d_params[1, f][:], size=1)

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


def GridWorldv0_pt(env_name, pref_type, action_selection, learn_A, learn_B, learn_D):
    """Function that defines the phenotype of the agent in the GridWorldv0 environment.
    The phenotype consists simply in a bunch of parameters that constrain the ways the
    agent behaves in the environment.

    Inputs:
    - env_name (string): name of the environment passed to the command line
    - pref_type (string): either the string "states" or "observations", determining whether the agent strives
    towards preferred states or observations
    - action_selection (string): the way the agent picks an action
    - learn_A (Boolean): whether the state-observation mappings are given or to be learned
    - learn_B (Boolean): whether the state-transitions probabilities are given or to be learned
    - learn_D (Boolean): whether the initial-state probabilities are given or to be learned

    Outputs:

    1. Agent parameters:

      - num_states (integer): no. of states the agent can be in, corresponding to the no. of gridworld
      tiles (so it is determined by the configuration of the gridworld);
      - start_state (integer): no. of maze tile on which the agent is in at the beginning of every episode;
      - steps (integer): no. of steps every episode lasts;
      - inf_iters (integer): no. of iterations to minimize free energy (see perception method of the agent class);
      - efe_steps (integer): no. of steps the agent 'looks' into the future to compute expected free energy;
      - num_action (integer): no. of action the agent can perform in the maze
        (move up=0, right=1, down=2, left=3);
      - num_policies (integer): no. of policies the agent starts with;
      - policies (numpy array): sequences of actions (same number as steps) stored as rows of a matrix;
      - action_selection (string): the way the agent picks an action;
      - pref_type (string): either the string "states" or "observations", determining whether the agent strives
      towards preferred states or observations;
      - preferences (array): matrix of agent's probabilistic beliefs about the most likely state(s);
      - index_Qt_Si (integer): index to save the agent's probabilistic beliefs of where it is located
      at the final time step
      - learn_A (Boolean): whether the state-observation mappings are given or to be learned
      - learn_B (Boolean): whether the state-transitions probabilities are given or to be learned
      - learn_D (Boolean): whether the initial-state probabilities are given or to be learned
      - random_seed (integer): random seed for the operations involving numpy RNG;
      - B (numpy array): matrices determining the transition probabilities when there is no learning over them;
      - B_params (numpy array): corresponding parameters for the B matrix (no subject to learning though).

    2. Environment parameters:
      - height (integer): height of the grid;
      - width (integer): width of the grid;
      - rand_conf (Boolean): whether configuration of (wall) tiles will be random;
      - goal_state (integer): the goal state for the agent;
      - rand_pos (Boolean): whether the position of the agent will be random;
      - start_agent_pos (integer): tile number on which the agent starts;
      - walls (list): list of coordinates for wall tiles.

    """

    # Some general parameters that define key features of the agent, environment, and their interaction
    # Height of the grid
    height = 3
    # Width of the grid
    width = 3
    # Goal state for the agent
    goal_state = 8
    # Whether the position of the agent will be random
    rand_pos = False
    # Whether configuration of tiles will be random
    rand_conf = False
    # List of tiles coordinates (used if rand_conf = False)
    walls = [(1, 1)]
    # Total number of tiles (therefore, states) in the world
    num_states = height * width
    # Fixed start state (used if rand_pos = False)
    start_state = 0
    # Number of time steps in an episode
    steps = 5
    # No. of free energy minimization iterations
    inf_iters = 5
    # Maximum number of time steps into the future for computing expected free energy
    efe_steps = 4
    # Number of actions
    num_action = 4
    # Number of policies
    num_policies = 2

    # Specifying the agent's preferences, either over states or observations.
    # Note: the preferences could be defined either for every single step of the correct trajectory
    # leading to the goal state or just for the goal state. Below we follow the latter approach
    # (the former is commented out).

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

        # Creating a preference matrix with the probabilities of being located on a certain maze tile
        # at each time step during an episode
        pref_array = np.ones((num_states, steps)) * (1 / num_states)
        # At every time step all states have uniform probabilities...
        pref_array[:-1, -1] = 0.1 / (num_states - 1)
        # ...except at the last time step when the goal state is given the highest probability
        pref_array[-1, -1] = 0.9
        # Checking all the probabilities sum to one
        assert np.all(np.sum(pref_array, axis=0)) == 1, print(
            "The preferences do not sum to one!"
        )

    elif pref_type == "observations":

        # TODO: this has to be implemented with an appropriate task
        pass

    else:

        raise Exception(
            "Invalid agent's preferences type; that should be either 'states' or 'observations'."
        )

    # Specifying the agent's policies for the duration of an episode; the agent is given some
    # "motor plans" (sequences of actions) to try out and perform during an episode.
    # Note: policies are usually hard coded in the discrete active inference literature.
    policies = np.array(
        [[2, 2, 1, 0], [1, 1, 2, 2]]
    )  # np.array([[2,2,1,0], [2,1,2,3], [1,1,2,3], [1,1,2,2]])

    # Defining the dictionary storing the agent's parameters.
    # All the parameters can be modified, however note:
    # 1) 'num_states' and 'num_actions' are based on the current environment
    # 2) changing some parameters might require updating others as well, e.g. if the steps of an
    # episode are increased, then the length of the policies and the preference should be amended as well.
    agent_params = {
        "env_name": env_name,
        "num_states": num_states,
        "start_state": start_state,
        "steps": steps,
        "inf_iters": inf_iters,
        "efe_tsteps": efe_steps,
        "num_actions": num_action,
        "num_policies": num_policies,
        "policies": policies,
        "action_selection": action_selection,
        "pref_type": pref_type,
        "preferences": pref_array,
        "learn_A": learn_A,
        "learn_B": learn_B,
        "learn_D": learn_D,
    }

    env_params = {
        "height": height,
        "width": width,
        "rand_conf": rand_conf,
        "goal_state": goal_state,
        "rand_pos": rand_pos,
        "start_agent_pos": start_state,
        "walls": walls,
    }

    return env_params, agent_params


def B_init_GridWorldv0(num_states, num_actions):
    """Function to initialize the Dirichlet parameters that are going to specify the B matrices of
    the agent's generative model when there is no learning over transition probabilities.
    In vanilla active inference this has to be *hard coded* for every agent/environment types.

    Input:
    - num_state (integer): no. of states in the simple grid
    - num_actions (integer): no. of actions available to the agent

    Output:
    - B_params (numpy array): hard coded parmaters for the B matrix

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
