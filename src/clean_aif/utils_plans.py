"""
File with utility functions used by the ActInfAgent class.

Created on Wed Aug  5 16:16:00 2020
@author: Filippo Torresan
"""

import numpy as np
import math
from scipy import special

# Set the print options for NumPy
np.set_printoptions(precision=3, suppress=True)


# def faster_permutations(n: int) -> np.ndarray:
#     """Function to create all the permutations of n integers. i.e. those in the sequence [0,.., n-1].
#     The strategy is to first create all the permutations of n-1 integers (those in the sequence [0,..,n-1])
#     then copy those sequences and add the nth element in all possible positions, by appropriately assigning
#     the elements on the right and left of the newly introduced element based on the copied sequences.

#     Source of this fast and efficient implementation:

#     https://stackoverflow.com/questions/64291076/generating-all-permutations-efficiently/64341432#64341432

#     """
#     # empty() is fast because it does not initialize the values of the array
#     # order='F' uses Fortran ordering, which makes accessing elements in the same column fast
#     perms = np.empty((math.factorial(n), n), dtype=np.uint8, order="F")
#     perms[0, 0] = 0

#     rows_to_copy = 1
#     for i in range(1, n):
#         perms[:rows_to_copy, i] = i
#         for j in range(1, i + 1):
#             start_row = rows_to_copy * j
#             end_row = rows_to_copy * (j + 1)
#             splitter = i - j
#             perms[start_row:end_row, splitter] = i
#             perms[start_row:end_row, :splitter] = perms[
#                 :rows_to_copy, :splitter
#             ]  # left side
#             perms[start_row:end_row, splitter + 1 : i + 1] = perms[
#                 :rows_to_copy, splitter:i
#             ]  # right side

#         rows_to_copy *= i + 1

#     return perms
#


def compute_future_beliefs(
    num_states: int,
    current_state_probs: np.ndarray,
    pi_actions: list,
    B_matrices: np.ndarray,
):
    """
    Function to compute the future state beliefs for each policy. These are computed as a matrix-vector
    multiplication between the transition probability, B matrix, associated with the action a policy
    dictates at a certain time step ($Q(S_t | S_(t-1),  a_(t-1))$) and the vector of state beliefs at the
    same time step ($Q(S_(t-1) | a_(t-2))$). The result is the vector of state probabilities at the next time
    step ($Q(S_t | a_(t-1), a_(t-2))$).

    E.g. Q(S_t | a_(t-1), a_(t-2)) = Q(S_t | S_(t-1),  a_(t-1)) Q(S_(t-1) | a_(t-2))

    NOTE: the vector $Q(S_t | a_(t-1), a_(t-2))$ is often written as $ Q(S_t | pi)$ where the actions
    $a_(t-1), a_(t-2)$ are those dictated by policy pi.

    Input:
    - num_states: number of states
    - current_state_probs: vector of current state probabilities (for the current time step)
    - pi_actions: list of action dictated by the policy
    - B_matrices: arrays of transition of probabilities

    Output:
    - future_beliefs: array of shape (num_states, plan_horizon)
    """
    # print(f"Computing future beliefs for policy: {pi_actions}")

    # State probabilities at the current time step
    previous_state_probs = current_state_probs

    # print(f"Prior or previous_state_probs:")
    # print(f"{previous_state_probs}")

    # Length of a policy, i.e. the future time steps for which the state beliefs need to be computed
    num_future_steps = len(pi_actions)
    # Initialize array with future state beliefs
    future_beliefs = np.empty((num_states, num_future_steps))

    # Loop over the future time steps
    for t in range(num_future_steps):
        # Select action the policy dictates at time step t in the future
        action = pi_actions[t]
        # Compute the future state beliefs

        # print(f"B_matrices.shape: {B_matrices[action].shape}")
        # print(f"previous_state_probs.shape: {previous_state_probs.shape}")

        state_beliefs = B_matrices[action] @ previous_state_probs
        # Store the state beliefs in the array
        future_beliefs[:, t] = state_beliefs
        # Make the state beliefs the new "current" state beliefs for the next iteration
        previous_state_probs = state_beliefs

    # print(f"Policy future beliefs shape: {future_beliefs.shape}")
    # print(f"{future_beliefs}")

    return future_beliefs


def vfe(
    trajectory_len,
    num_states,
    current_tstep,
    current_obs,
    pi,
    pi_actions,
    A,
    B,
    D,
    Qs_p,
    A_params,
    B_params,
    learning_A=False,
    learning_B=False,
):
    """Function to compute the variational free energy (vfe) w.r.t. one policy
    over a trajectory which is composed of a shared past (shared among all the policies)
    and a policy-dependent future, based on the actions each policy dictates.

    Inputs:

    - trajectory_len (integer): number of time steps in a trajectory, sum of past and future time steps;
    - num_state (integer): no. of states in the environment;
    - current_tstep (integer): current time step;
    - current_obs (numpy array): matrix of shape (num_state, num_steps) with
      one-hot columns indicating the observation at each time step;
    - pi (integer): index identifying the current policy, i.e. one of the rows
      in the agent's attribute self.policies;
    - pi_actions (numpy array): array of shape (num_steps, ) storing the sequence
      of actions dictated by the policy;
    - A,B,D (numpy arrays): matrices and/or tensors storing the relevant probabilities
      of the agent's generative model (see the agent class);
    - Qs_p (numpy array): variational probability distribution over past and future states for each
      policy, depends on the policy sequence of actions;
    - A_params, B_params (numpy arrays): matrices and/or tensors storing the parameters
      over which learning occurs;
    - learning_A, learning_B (boolean): boolean variables indicating whether learning over
      correspondingparameters occurs or not.

    Outputs:

    - logB_pi, logA_pi etc. (numpy arrays): (element-wise) logarithms of the transition
      and observation matrices, if these are learned, then their columns are Dirichlet
      distributed random variables and the corresponding expectations (still matrices)
      are returned (note: these arrays are used for computing free energy gradients).
    - F_pi (float): the free energy for the current policy.
    """

    # Computing the free energy for the current policy term by term based on
    # the treatment of Da Costa et. al. (2020), p. 8, Equation (7)
    # (DOI: 10.1016/j.jmp.2020.102447):
    #
    # F_pi =
    #  # First term: sum of dot products between the categoricals from t=1 to T.
    #  S_1T(Q(s_t|pi) @ log(Q(s_t|pi)))
    #  # Second term: sum of dot products up to the current time step.
    #  - S_1t(o_t @ A @ Q(s_t|pi))
    #  # Third term: dot product between between categoricals over the initial state.
    #  - Q(s_1|pi) @ log(D)
    #  # Fourth term: sum of dot products from t=2 to T.
    #  - S_2T(Q(s_t|pi) @ log(B_t-1) @ Q(s_t-1|pi)).

    # print("Computing FE...")

    # # Length of a policy, i.e. the number of time steps to plan/predict the future
    # plan_steps = len(pi_actions)
    # # Length of a trajectory, from past to future time steps considered by each policy
    # # NOTE: current_tstep is the index for the current time step; since we start counting from 0, an index
    # # of 2 implies that *three* times teps have passed. Therefore we add 1 below for computing the total
    # # number of time steps in a trajectory.
    # trajectory_len = current_tstep + 1 + plan_steps
    # # Concatenate past, present and future state beliefs for current policy
    # # NOTE: each policy will have a shared past and present
    # Qs_p = np.concatenate((Qs[:, 0 : current_tstep + 1], Qs_pi[pi, :, :]), axis=1)
    # # Check that Qs_p has correct shape
    # assert (
    #     Qs_p.shape[1] == trajectory_len
    # ), f"Axis 1 of Qs_p should be of size {trajectory_len}, but it is of size {Qs_shape[1]} instead."

    # Digamma function used to compute the expectations of matrices A and B if their respective
    # parameters are learned by the agent. Those expectations turn up in the computation of the
    # free energy (see below).
    psi = special.digamma

    ######################################## First term ########################################
    # The first term, indicated by st_log_st, is computed in a vectorized way, i.e., Q_s * Q_s
    # is a element(row)-wise matrix multiplication, the inner np.sum gives you the dot products
    # between the rows of the two matrices, the outer np.sum gives you their sum. The matrices
    # need to be transposed because we want the dot products between the categorical distribution,
    # which are stored as columns in the original matrices.
    st_log_st = np.sum(np.sum(Qs_p.T * np.log(Qs_p).T, axis=1))

    ####################################### Second term #######################################
    # The second term, indicated by ot_logA_st, is computed in a vectorized way in the same way
    # as the first term.

    # Initializing the variable
    ot_logA_st = 0
    # Initializing the logA_pi array
    logA_pi = np.zeros((num_states, num_states))
    # If observation matrix A is learned, logA_pi stores its expectations, otherwise it stores
    # an unambiguous state-observation mapping.
    if learning_A == True:
        ExplogA = psi(A_params) - psi(np.sum(A_params, axis=0))
        logA_pi[:, :] = ExplogA
    else:
        # logA_pi[:,:] = np.log(A)
        ExplogA = psi(A_params) - psi(np.sum(A_params, axis=0))
        logA_pi[:, :] = ExplogA
    # Computing the second term
    ot_logA_st = np.sum(
        np.sum(
            current_obs[:, 0 : current_tstep + 1].T
            * np.matmul(logA_pi, Qs_p[:, 0 : current_tstep + 1]).T,
            axis=1,
        )
    )

    ####################################### Third term #######################################
    # The third term, indicated by s1_logD, is the dot product between Q(s_1|pi) and log(D),
    # i.e. the categoricals over the intial state under the variational and generative models
    # respectively.

    # Computing the log of matrix D
    logD_pi = np.log(D)
    # Computing the third term
    s1_logD = np.dot(Qs_p[:, 0], logD_pi)

    ####################################### Fourth term #######################################
    # The fourth term, indicated by st_ExplogB_stp, is more involved as requires the expectation
    # of Dirichlet distributed values and is a sum from t=2 to T (in Python, from t=1 to T-1).

    # Initializing the the term variable
    st_logB_stp = 0

    # If transition matrices B are learned, logB_pi stores their expectations, otherwise
    # it stores the sequence of action-dependent transition matrices defining the policy pi.
    # NOTE 1: the first dimension of logB_pi is equal to lenght of a trajectory MINUS 1, because at
    # the last time step there is no action therefore no transition probabilities.
    logB_pi = np.zeros((trajectory_len - 1, num_states, num_states))

    # For every step in the trajectory retrieve the action that the policy pi dictates at
    # time step t-1. NOTE: t ranges over the indices representing each time step
    # print(f"Traj len: {trajectory_len}")
    # print(f"Action seq len: {len(pi_actions)}")

    for t in range(trajectory_len):

        if learning_B == True:
            # There is no action at the last time step so we retrieve B matrices
            # as long as t is not the index representing the last step, i.e. trajectory-1
            # (which is the last value assigned to t in the loop with `range(trajectory_len)`).
            if t < (trajectory_len - 1):
                # Computing the expectation of the Dirichlet distributions using
                # their parameters and the digamma function
                action = pi_actions[t]
                ExplogB = psi(B_params[action]) - psi(np.sum(B_params[action], axis=0))
                logB_pi[t, :, :] = ExplogB

        else:
            # Same as above for when learning_B = False
            if t < (trajectory_len - 1):
                # Simply retrieving the corresponding transition matrix B because its
                # parameters are not learned.
                action = pi_actions[t]
                # print(f"The action is: {action}, {type(action)}")
                # print("Corresponding B matrix is")
                # print(f"{B[action, :, :]}")
                logB_pi[t, :, :] = np.log(B[action, :, :])

            # if t < (steps-1):
            #     action = pi_actions[t]
            #     ExplogB = psi(B_params[action])  - psi(np.sum(B_params[action], axis=0))
            #     logB_pi[t, :, :] = ExplogB

        # Compute the value of st_logB_stp for current time step t
        if t != 0:
            st_logB_stp += np.dot(
                Qs_p[:, t], np.matmul(logB_pi[t - 1, :, :], Qs_p[:, t - 1])
            )

    ##### DEBUGGING #####
    assert math.isnan(st_log_st) != True, "Non computable value!"
    assert math.isnan(ot_logA_st) != True, "Non computable value!"
    assert math.isnan(s1_logD) != True, "Non computable value!"
    assert math.isnan(st_logB_stp) != True, "Non computable value!"
    ##### END #####

    # Summing the four terms to get the free energy for the current policy pi
    F_pi = st_log_st - ot_logA_st - s1_logD - st_logB_stp

    # assert type(F_pi)==float, 'Free energy is not of type float; it is of type: ' + str(type(F_pi))

    return logA_pi, logB_pi, logD_pi, F_pi


def grad_vfe(
    trajectory_len,
    num_states,
    current_tstep,
    current_obs,
    pi,
    pi_actions,
    Qs_p,
    logA_pi,
    logB_pi,
    logD_pi,
):
    """Function to compute the gradient vectors of the free energy for one policy
    w.r.t each categorical Q(s_t|pi).

    Inputs:

    - trajectory_len (integer): number of time steps in a trajectory, sum of past and future time steps;
    - num_state (integer): no. of states in the environment;
    - current_tstep (integer): current time step;
    - current_obs (numpy array): matrix of shape (num_state, num_steps) with one-hot columns
      indicating the observation at each time step;
    - pi (integer): index identifying the current policy, i.e. one of the rows in the
      agent's attribute self.policies;
    - pi_actions (numpy array): array of shape (num_steps, ) storing the sequence
      of actions dictated by the policy;
    - Qs_p (numpy array): variational probability distribution over past and future states for each
      policy, depends on the policy sequence of actions;
    - logA_pi, logB_pi, logD_pi (numpy arrays): (element-wise) logarithm of the transition and
      observation matrices and the initial state probability vector; if the matrices are learned,
      then their columns are Dirichlet distributed random variables and logA_pi, logB_pi correspond
      to their expectations (still matrices).

    Outputs:

    - grad_F_pi (numpy array, size: num_state*num_steps), it stores the free energy gradient vectors
    as columns, one for each Q(s_t|pi).
    """

    # print("Data for computing gradients")
    # print(f"- Num states: {num_states}")
    # print(f"- Steps: {steps}")
    # print(f"- Current tstep: {current_tstep}")
    # print("- Current obs: ")
    # print(f"{current_obs}")
    # print(f"- Policies: {pi}")
    # print("- Qs_pi: ")
    # print(f"{Qs_pi}")
    # print("- logA_pi: ")
    # print(f"{logA_pi}")
    # print("- logB_pi: ")
    # print(f"{logB_pi}")
    # print("- logD_pi: ")
    # print(f"{logD_pi}")

    # print("Computing FE gradients...")

    # Initialising the gradient vectors for each Q(s_t|pi)
    grad_F_pi = np.zeros((num_states, trajectory_len))

    # For every step in the trajectory compute the corresponding gradient terms
    # NOTE: t ranges over the indices representing each time step
    for t in range(trajectory_len):

        # Computing the gradient w.r.t. Q(s_0|pi), i.e. the categorical over the initial state
        if t == 0:
            ##### DEBUGGING #####
            # print(f'Current timestep: {current_tstep}')
            # print(f'Q(S_0=10|pi=1): {Qs_pi[1, 10, t]}')
            # print(f'Q(S_1=15|pi=1): {Qs_pi[1, 15, t+1]}')
            # print(f'obs times logA: {np.matmul(current_obs[:,t], logA_pi)}')
            # print(f'Qsi times logB: {np.matmul(Qs_pi[pi, :, t+1], logB_pi[t,:,:])}')
            # print(f'logA_pi: {logA_pi[10]}')
            # print(f'logB_pi: {logB_pi[t,15,10]}')
            # print(f'current obs: {current_obs[10,t]}')
            # print(f'logD_pi: {logD_pi[10]}')
            ##### END #####
            grad_F_pi[:, t] = (
                np.ones(num_states)
                + np.log(Qs_p[:, t])
                - (
                    np.matmul(current_obs[:, t], logA_pi)
                    + np.matmul(Qs_p[:, t + 1], logB_pi[t, :, :])
                    + logD_pi
                )
            )

            ##### DEBUGGING #####
            # print(f'The gradient is: {grad_F_pi[10,t]}')
            ##### END #####

        # Computing the gradients w.r.t. Q(s_t|pi) where 0<t<=current_tstep
        elif t > 0 and t <= current_tstep:
            # NOTE: the agent is always considering efe_tstep after the current_tstep even if the latter
            # is the terminal or truncation point so there is no issue in the indexing Qs_p[:, t+1]
            grad_F_pi[:, t] = (
                np.ones(num_states)
                + np.log(Qs_p[:, t])
                - (
                    np.matmul(current_obs[:, t], logA_pi)
                    + np.matmul(Qs_p[:, t + 1], logB_pi[t, :, :])
                    + np.matmul(logB_pi[t - 1, :, :], Qs_p[:, t - 1])
                )
            )

        # Computing the gradients w.r.t. Q(s_t|pi) where t>current_tstep
        # (i.e., w.r.t. categorical beliefs about future time steps)
        elif t > current_tstep:
            ##### DEBUGGING #####
            # if current_tstep == 0 and t == 1 and pi==1:
            #    print(Qs_pi[pi, :, t-1])
            #    print(Qs_pi[pi, :, t])
            # if current_tstep == 0 and t == 1 and pi==1:
            #    print(f'Transition matrix at t=0: {logB_pi[t,:,:]}')
            ##### END #####

            # NOTE: the agent is planning efe_tsteps into the future, beyond the current_tstep, so here we
            # have an end point after which there is no time step. Therefore we need to make sure that the
            # indexing Qs_p[:, t+1] is possible, i.e. when t is different from the last time step in the
            # trajectory
            if t != (trajectory_len - 1):
                grad_F_pi[:, t] = (
                    np.ones(num_states)
                    + np.log(Qs_p[:, t])
                    - (
                        np.matmul(Qs_p[:, t + 1], logB_pi[t, :, :])
                        + np.matmul(logB_pi[t - 1, :, :], Qs_p[:, t - 1])
                    )
                )

            # Case when t is equal to the last time step in the trajectory
            elif t == (trajectory_len - 1):

                grad_F_pi[:, t] = (
                    np.ones(num_states)
                    + np.log(Qs_p[:, t])
                    - np.matmul(logB_pi[t - 1, :, :], Qs_p[:, t - 1])
                )
            ##### DEBUGGING #####
            # if current_tstep == 0 and t == 1 and pi==1:
            # print(f'The gradient is: {grad_F_pi[:,t]}')
            ##### END #####
    return grad_F_pi


def total_free_energy(
    current_tstep, truncated, terminated, free_energies, Qpi, **kwargs
):
    """Function that computes the total free energy,  i.e. F = KL[Q(A)|P(A)] + KL[Q(B)|P(B)] +
    + KL[Q(pi)|P(pi)] + E[F_pi], at the current time step based on Equation (5) in Da Costa et al.
    2020, pp. 7-8 (DOI: 10.1016/j.jmp.2020.102447). The value is then stored in the agent's
    self.total_free_energies (this total free energy is useful to see if the agent gets
    better over the steps and the episodes).

    Inputs:

    - current_tstep (integer): current time step;
    - trajectory_steps (integer): number of steps per trajectory;
    - free_energies (numpy array): matrix storing all the policy-dependent free energies computed during
      an episode;
    - Qpi (numpy array): matrix storing all the Q(pi) computed during an episode;
    - kwargs:
        - prior_A (numpy arrays/tensors): matrices storing prior parameters for Dirichlet distributed
          random variables
        - prior_B (numpy arrays/tensors): matrices storing prior parameters for Dirichlet distributed
          random variables
        - A_params (numpy arrays/tensors): matrices storing the parameters for Dirichlet distributed
          random variables
        - B_params (numpy arrays/tensors): matrices storing the parameters for Dirichlet distributed
          random variables
        - learning_A (boolean): whether there is learning of matrix A
        - learning_B (boolean): whether there is learning of matrix B.

    Outputs:

    - total_F (float): total free energy at current time step during an episode.

    Note 1: the first two KL divergences should be included only if the parameters of those probability
    distributions are learned and only at the last time step because parameter learning occurs only then.
    Note 2: the third KL divergence is zero at the first time step if there is no prior over policies
    (later on the previous step Q(pi) can be considered the prior and the Q(pi) actually used to perform
    the action the approximate posterior so that the KL can be computed).
    Note 3: the expectation of the policy-dependent free energies is computed under Q(pi).
    Note 4: the function dirichlet_KL() is needed to compute the KL divergences between Dirichlet probability
    distributions.
    """

    # Initialising the total free energy variable
    total_F = 0
    # Case where self.current_tstep is neither the first nor the last state
    # (no need for KL[Q(A)|P(A)] and/or KL[Q(B)|P(B)])
    if current_tstep != 0 and not truncated and not terminated:
        # Computing KL[Q(pi)|P(pi)]
        KL_Qpi_Ppi = cat_KL(Qpi[:, current_tstep], Qpi[:, current_tstep - 1])
        # Computing the E[F_pi] term
        E_Fpi = np.dot(Qpi[:, current_tstep], free_energies[:, current_tstep])
        # Computing the total
        total_F = KL_Qpi_Ppi + E_Fpi

    # Case where self.current_tstep is the first state (no need for any KL divergence)
    elif current_tstep == 0:
        # Computing the E[F_pi] term
        E_Fpi = np.dot(Qpi[:, current_tstep], free_energies[:, current_tstep])
        # Computing the total
        total_F = E_Fpi

    # Case where the terminal or end state of the episode has been reached (all KL divergences are required
    # unless there is no parameter learning)
    elif terminated or truncated:
        # Retrieve prior B_params
        prior_B = kwargs["prior_B"]
        # Retrieve B_params
        B_params = kwargs["B_params"]
        # Retrieve prior A_params
        prior_A = kwargs["prior_A"]
        # Retrieve A_params
        A_params = kwargs["A_params"]
        # Considering different learning scenarios
        if kwargs["learning_A"] == True and kwargs["learning_B"] == True:
            # Retrieving the number of actions (first dimension of B_params)
            num_actions = B_params.shape[0]
            KL_QB_PB = 0
            # Computing KL[Q(B)|P(B)]
            # Note 1: since B_params is a tensor, we need to loop over its first dimension (num_actions)
            # to compute all the Dirichlet KL divergences for the B matrices.
            for a in range(num_actions):
                KL_QB_PB_a = np.sum(dirichlet_KL(B_params[a, :, :], prior_B[a, :, :]))
                KL_QB_PB += KL_QB_PB_a

            # Computing KL[Q(A)|P(A)]
            KL_QA_PA = np.sum(dirichlet_KL(A_params, prior_A))
            # Computing KL[Q(pi)|P(pi)]
            KL_Qpi_Ppi = cat_KL(Qpi[:, current_tstep], Qpi[:, current_tstep - 1])
            # Computing the E[F_pi] term
            E_Fpi = np.dot(Qpi[:, current_tstep], free_energies[:, current_tstep])
            # Computing the total
            total_F = KL_QA_PA + KL_QB_PB + KL_Qpi_Ppi + E_Fpi

        elif kwargs["learning_A"] == True and kwargs["learning_B"] == False:

            # Initially, the line below was added to prevent some NaN values in the KL computation.
            # Then, it was discovered that the cause was another one, but this line turned out to
            # prevent some gradient overshooting. So, either keep this line or reduce the learning rate
            # when learning A.
            A_params = np.where((A_params - prior_A) == 0, (A_params + 0.01), A_params)
            # Computing KL[Q(A)|P(A)]
            KL_QA_PA = np.sum(dirichlet_KL(A_params, prior_A))
            # Computing KL[Q(pi)|P(pi)]
            KL_Qpi_Ppi = cat_KL(Qpi[:, current_tstep], Qpi[:, current_tstep - 1])
            # Computing the E[F_pi] term
            E_Fpi = np.dot(Qpi[:, current_tstep], free_energies[:, current_tstep])
            # Computing the total
            total_F = KL_QA_PA + KL_Qpi_Ppi + E_Fpi

        elif kwargs["learning_A"] == False and kwargs["learning_B"] == True:
            # Computing KL[Q(B)|P(B)]
            # Retrieving the number of actions (first dimension of B_params)
            num_actions = B_params.shape[0]
            KL_QB_PB = 0
            # Since B_params is a tensor, we need to loop over its first dimension (num_actions)
            # to compute all the Dirichlet for the B matrices.
            for a in range(num_actions):

                KL_QB_PB_a = np.sum(dirichlet_KL(B_params[a, :, :], prior_B[a, :, :]))
                KL_QB_PB += KL_QB_PB_a
            # Computing KL[Q(pi)|P(pi)]
            KL_Qpi_Ppi = cat_KL(Qpi[:, current_tstep], Qpi[:, current_tstep - 1])
            # Computing the E[F_pi] term
            E_Fpi = np.dot(Qpi[:, current_tstep], free_energies[:, current_tstep])
            # Computing the total
            total_F = KL_QB_PB + KL_Qpi_Ppi + E_Fpi

        elif kwargs["learning_A"] == False and kwargs["learning_B"] == False:
            # Computing KL[Q(pi)|P(pi)]
            KL_Qpi_Ppi = cat_KL(Qpi[:, current_tstep], Qpi[:, current_tstep - 1])
            # Computing the E[F_pi] term
            E_Fpi = np.dot(Qpi[:, current_tstep], free_energies[:, current_tstep])
            # Computing the total
            total_F = KL_Qpi_Ppi + E_Fpi

    return total_F


def efe(
    future_steps,
    pi,
    pi_actions,
    A,
    C,
    Qs_pi,
    pref_type,
    A_params,
    B_params,
    learning_A,
    learning_B,
):
    """Function to compute the expected free energy (efe) w.r.t. one policy for self.efe_tsteps
    number of steps into the future. The computation is based on the expression for the expected
    free energy stated in Equation (D.1) of Da Costa et al. (2020), p. 19
    (DOI: 10.1016/j.jmp.2020.102447).

    Inputs:

    - num_state (integer): no. of states in the environment;
    - current_tstep (integer): current time step;
    - episode_steps (integer): total time steps in the episode;
    - future_steps (integer): the number of time steps into the future for which to calculate
      the expected free energy;
    - pi (integer): index identifying the current policy, i.e. one of the rows in the
      agent's attribute self.policies;
    - pi_actions (numpy array): array of shape (num_steps, ) storing the sequence of actions
      dictated by the policy;
    - A, C (numpy arrays): matrices and/or tensors storing the relevant probabilities of the
      agent's generative model (see the agent class);
    - Qs_pi (numpy array): tensor storing the variational probability distributions over the states
      for the future time steps of each policy (see the agent class);
    - pref_type (string): string indicating the type of agent's preferences, it affects the computation
      of the risk term of the expected free energy
    - A_params, B_params (numpy arrays): matrices and/or tensors storing the parameters over
      which learning occurs;
    - learning_A, learning_B (boolean): boolean variables indicating whether learning over certain parameters
      occurs or not.

    Outputs:

    - G_pi (float): the sum of pi's expected free energies, one for each of future time step considered.
    """

    # print("Computing EFE...")
    # Initialising the expected free energy variable and its components
    G_pi = 0
    # Ambiguity
    tot_Hs = 0
    # Risk
    tot_slog_s_over_C = 0
    # A-Novelty
    tot_AsW_As = 0
    # B-Novelty
    tot_AsW_Bs = 0
    # B-Novelty list by state
    sq_AsW_Bs = np.zeros(future_steps)

    # Computing matrix H for the ambiguity term of the expected free energy.
    # H = -diag[E[A]log(E[A])]
    Exp_A = A_params / np.sum(A_params, axis=0)
    H = -np.diag(np.matmul(Exp_A.T, np.log(Exp_A)))

    # Loop over the time steps in which EFE is computed
    # NOTE: if t is the present time step and EFE is computed for 4 steps, then it is computed at
    # t, t+1, t+2, and t+3.
    for tau in range(0, future_steps):
        # Compute AMBIGUITY term in EFE
        Hs = np.dot(H, Qs_pi[pi, :, tau])
        # Add the ambiguity computed at tau to the total
        tot_Hs += Hs

        # IMPORTANT: here we are replacing zero probabilities in the policy-conditioned state probabilitis
        # with the minimum value in C to  avoid zeroes in logs when we compute the RISK term in EFE.
        # NOTE: this keeps the meaning of the KL divergence (the risk term) intact because if the Qs_pi
        # predicts the state has zero probability and the preference is low, then the KL divergence turns
        # out to be zero (with the above replacement); conversely, if the preference was high then the Qs_pi
        # got things very wrong and the KL divergence will be high.
        Qs_pi_risk = np.where(Qs_pi[pi, :, tau] == 0, np.amin(C), Qs_pi[pi, :, tau])
        # Compute RISK term in EFE
        if pref_type == "states":
            # Computing risk based on preferred states
            # NOTE (!!! IMPORTANT !!!): we have replaced `np.log(C[:, tau])` with `np.log(C[:, 0])`
            # because in this active inference implementation we don't currently have the possibility
            # to specify preferences over a trajectory, which was possible in the original one, we can
            # only give the agent a single vector of preferences, i.e. a stationary distribution
            slog_s_over_C = np.dot(Qs_pi_risk, np.log(Qs_pi_risk) - np.log(C[:, 0]))
        else:
            # Computing risk based on preferred observations
            slog_s_over_C = np.dot(
                np.dot(A, Qs_pi_risk),
                np.log(np.dot(A, Qs_pi_risk)) - np.log(C[:, tau]),
            )
        # Add risk computed at tau to the total
        tot_slog_s_over_C += slog_s_over_C

        # If A matrix is learned compute A-NOVELTY term in EFE
        if learning_A:
            # W matrix needed to compute the A-novelty term of the expected free energy:
            W_A = 0.5 * (1 / A_params - 1 / np.sum(A_params, axis=0))
            # Computing the A-novelty term
            AsW_As = np.dot(
                np.matmul(A, Qs_pi[pi, :, tau]), np.matmul(W_A, Qs_pi[pi, :, tau])
            )
            # Add the A-novelty computed at tau to the total
            tot_AsW_As += AsW_As

        # If B matrices are learned compute B-NOVELTY terms in EFE
        if learning_B:
            # At tau = episode_steps - 1 there is no action so the B-novelty terms are zero
            if tau == (future_steps - 1):
                AsW_Bs = 0
            else:
                # Action that policy pi dictates at time step tau and W_B matrix needed
                # to compute the B-novelty term.
                # NOTE: W_B depends on the parameters of the transition matrix B related to the action
                # the policy dictates at tau, that's why we need to retrieve that action.
                action = pi_actions[tau]
                W_B = 0.5 * (
                    1 / B_params[action, :, :]
                    - 1 / np.sum(B_params[action, :, :], axis=0)
                )
                # Computing the B-novelty term
                AsW_Bs = np.dot(
                    np.matmul(A, Qs_pi[pi, :, tau + 1]),
                    np.matmul(W_B, Qs_pi[pi, :, tau + 1]),
                )

            # Save B-novelty component computed at tau (to have the full sequence of B-novelties
            # for each policy)
            # print(f"EFE at future time step {tau}: {AsW_Bs}")
            sq_AsW_Bs[tau] = AsW_Bs

            # Add the B-novelty term computed at tau to the total
            tot_AsW_Bs += AsW_Bs

        # Adding all the terms appropriately to obtain (an approximation of) expected free energy G_pi_tau
        # for the given policy at tau, and adding that value to G to get the expected free energy over the
        # future ("imagined") trajectory considered by the policy.
        G_pi_tau = Hs + slog_s_over_C
        if learning_A:
            G_pi_tau -= AsW_As
        if learning_B:
            G_pi_tau -= AsW_Bs
        G_pi += G_pi_tau

    return G_pi, tot_Hs, tot_slog_s_over_C, tot_AsW_As, tot_AsW_Bs, sq_AsW_Bs


def dirichlet_update(
    num_states,
    num_actions,
    last_step,
    current_obs,
    episode_actions,
    Qs,
    A_params,
    B_params,
    learning_A,
    learning_B,
):
    """Function for parameters learning. Learning involves minimising the free energy at the end of an episode
    w.r.t. the parameters of Dirichlet distributed r.v. stored in the A and B matrices, representing
    state-observation mappings and state transitions respectively. Doing a gradient descent on free energy
    w.r.t. those parameters gives us the Dirichlet update, i.e. how much A's/B's parameters should change
    given the observations received at every time step during the episode.

    Note 1: the Dirichlet update is actually found by taking the gradient of free energy w.r.t.
    the expectation of the Dirichlet distributed random variables and amounts to the policy-independent
    state distribution at the relevant time step. This is then summed to the corresponding row of the
    parameter matrix.

    Note 2: doing the update gives you an approximate posterior over the parameters, i.e. Q(A) or Q(B).
    These will become the new priors, P(A) and P(B), in the next episode.

    Note 3: for more info on this type of update see Da Costa et al. (2020), pp. 12-13
    (DOI: 10.1016/j.jmp.2020.102447).

    Inputs:

    - num_state (integer): no. of states in the environment;
    - num_actions (integer): no. of actions the agent can perform;
    - last_step (integer): index of the termination or truncation time step;
      meaning that the length of the current trajectory is `last_step + 1`;
    - current_obs (numpy array): matrix of shape (num_state, num_steps) with one-hot columns indicating the
      observation at each time step;
    - episode_actions (numpy array): sequence of actions taken by the agent during the episode;
    - Qs (numpy array): array storing the policy independent state probabilities up to last_step;
    - A_params, B_params (numpy arrays): matrices and/or tensors storing the parameters over which learning
      occurs;
    - learning_A, learning_B (boolean): boolean variables indicating whether learning over certain
      parameters occurs or not.

    Outputs:

    -  Q_A_params, Q_B_params (numpy arrays): matrix/tensor storing the updated parameters.
    """

    # Considering different learning scenarios
    if learning_A == True and learning_B == True:
        # Both A and B are learned.
        # Initialising the approximate posterior beliefs over A and B's' parameters and the Dirichlet updates
        Q_A_params = 0
        Q_B_params = 0
        Dirichlet_update_A = np.zeros((num_states, num_states))
        Dirichlet_update_B = np.zeros((num_actions, num_states, num_states))

        # Computing the Dirichlet updates for A
        for t in range(last_step + 1):
            Dirichlet_update_A += np.outer(current_obs[:, t], Qs[:, t])

        # Getting the approximate posterior
        Q_A_params = A_params + Dirichlet_update_A

        # Computing the Dirichlet updates for B
        for action in range(num_actions):
            for t in range(1, last_step + 1):
                Dirichlet_update_B[action, :, :] += (
                    action == episode_actions[t - 1]
                ) * np.outer(Qs[:, t], Qs[:, t - 1])

        # Getting the approximate posterior
        Q_B_params = B_params + Dirichlet_update_B

        # Returning the approximate posterior for the learned parameters
        return Q_A_params, Q_B_params

    elif learning_A == True and learning_B == False:
        # A is learned but not B.
        # Initialising the approximate posterior beliefs over A and B's parameters and the Dirichlet update
        Q_A_params = 0
        Q_B_params = 0
        Dirichlet_update_A = np.zeros((num_states, num_states))

        # Computing the Dirichlet update
        for t in range(last_step + 1):
            Dirichlet_update_A += np.outer(current_obs[:, t], Qs[:, t])

        # Getting the approximate posterior
        Q_A_params = A_params + Dirichlet_update_A
        Q_B_params = B_params  # Nothing is learned here

        # Returning the approximate posterior for the learned parameters (Q_B_params is equal to
        # B_params because B is not learned)
        return Q_A_params, Q_B_params

    elif learning_A == False and learning_B == True:
        # B is learned but not A.
        # Initialising the approximate posterior beliefs over A and B's' parameters and the Dirichlet updates
        Q_A_params = 0
        Q_B_params = 0
        Dirichlet_update_B = np.zeros((num_actions, num_states, num_states))

        # Computing the Dirichlet updates for B
        for action in range(num_actions):
            # NOTE 1 (important): we want to consider all the time steps in which an action was taken,
            # i.e. from index 0 to index last_step - 1 because at the last time step, indexed by last_step,
            # no action is selected. In Python the for-loop below goes from index 1 to index last_step
            # which allows us to pick the action at the first time step with episode_actions[t - 1]
            # when t = 1, and exclude the non-existent action at the last time step because the last value
            # of t is t = last_step which gives us episode_actions[last_step - 1].
            # NOTE 2 (important): after reaching the truncation/terminal state the agent has accumulated a
            # sequence of (last_step + 1) actions and the same number of policy-independent beliefs, these
            # are stored in Qs
            for t in range(1, last_step + 1):
                Dirichlet_update_B[action, :, :] += (
                    action == episode_actions[t - 1]
                ) * np.outer(Qs[:, t], Qs[:, t - 1])

        # Getting the approximate posterior
        Q_A_params = A_params  # Nothing is learned here
        # assert np.array_equal(Dirichlet_update_B[2,:,:], Dirichlet_update_B[3,:,:]) == False, 'Updates suspiciously identical!'
        # print(f"Old B params {B_params[2,:]}")
        # print(f"Dirichlet update: {Dirichlet_update_B}")
        Q_B_params = B_params + Dirichlet_update_B
        # print(f"New B params {B_params[2,:]}")
        # Returning the approximate posterior for the learned parameters (Q_A_params is equal to
        # A_params because A is not learned)
        return Q_A_params, Q_B_params

    else:
        # Neither A nor B is learned (nothing is changed).
        Q_A_params = A_params
        Q_B_params = B_params
        return Q_A_params, Q_B_params


def dirichlet_KL(Q, P, axis=0):
    """Function to compute the KL divergence between *Dirichlet* probability distributions.

    Inputs:

    - P, Q (numpy arrays): either vectors storing Dirichlet parameters or matrices (n,m) with each
      column (row) storing Dirichlet parameters, in the former case the KL divergence is computed
      between two distributions, in the latter it is computed for m (n) distributions' pairs
      formed by one column (row) of Q and the corresponding column (row) of P;
    - axis (integer): 0 means the Dirichlet parameters for one distribution are stored in a column,
      1 means they are stored in a row.

    Outputs:

    - kl_div (float or numpy array): KL divergence for one distribution (float) or m (n) pairs of
      distributions (numpy array).

    Note 1: the function requires the gamma function (scipy.special.gamma).
    Note 2: for the expression of the KL divergence between two Dirichlet distributions and its derivation
    see Soch et al. (2022) (doi.org/10.5281/zenodo.5820411)

    """

    assert len(Q.shape) == len(
        P.shape
    ), "Inputs should have the same number of dimensions!"

    # Retrieving the gamma and digamma functions from scipy
    gamma = special.gamma
    psi = special.digamma

    # Distinguishing between different input cases:
    # 1) one-dimensional numpy arrays, or
    # 2) two-dimensional numpy arrays (vector or matrix)
    if len(Q.shape) == 1:
        # Initialising KL variable
        kl_div = 0
        # Sum of the Dirichlet parameters of Q and P
        q_0 = np.sum(Q)
        p_0 = np.sum(P)
        # Computing KL divergence using gamma and digamma functions
        kl_div = (
            np.log(gamma(q_0))
            - np.sum(np.log(gamma(Q)))
            - np.log(gamma(p_0))
            + np.sum(np.log(gamma(P)))
            + np.dot(Q - P, psi(Q) - psi(q_0))
        )

    elif len(Q.shape) > 1:
        # Considering different ways the parameters could be stored:
        # 1) as a column, i.e. over rows (n,1) or
        # 2) as a row, i.e. over columns (1,n)
        if axis == 0:
            # Initialising KL variable
            kl_div = np.zeros(Q.shape[1])
            # Parameters stored as a column, looping over the number of columns,
            # i.e. of distributions considered
            for c in range(Q.shape[1]):
                # Preventing computational overflow when using gamma() if the Dirichlet parameters
                # become too high (rescaling by 0.01)
                if np.sum(Q[:, c]) > 150:
                    Q[:, c] = Q[:, c] * 0.01
                    P[:, c] = P[:, c] * 0.01
                # Sum of the Dirichlet parameters of Q[:,c] and P[:,c]
                q_0 = np.sum(Q[:, c])
                p_0 = np.sum(P[:, c])

                # Computing KL divergence for the distribution stored in the c column using
                # gamma and digamma functions
                kl_div[c] = (
                    np.log(gamma(q_0))
                    - np.sum(np.log(gamma(Q[:, c])))
                    - np.log(gamma(p_0))
                    + np.sum(np.log(gamma(P[:, c])))
                    + np.dot(Q[:, c] - P[:, c], psi(Q[:, c]) - psi(q_0))
                )

                if kl_div[c] == 0 or np.abs(kl_div[c]) < 0.0001:
                    kl_div[c] = 0.0001
                # Assertions to check there are no NaN, inf, or other unreasonable values
                assert np.all(np.isnan(kl_div[c])) == False, print(
                    f"Column {c}, i.e. {kl_div[c]}, contains NaN values"
                )
                assert np.all(np.isinf(kl_div[c])) == False, print(
                    f"Column {c}, i.e. {kl_div[c]}, contains inf values"
                )
                if kl_div[c] < 0:
                    print(kl_div[c])
                    print(Q[:, c])
                    print(P[:, c])
                assert np.all(kl_div[c] > 0) == True, print(
                    f"Some KL divergences are negative!"
                )

        elif axis == 1:

            # Initialising KL variable
            kl_div = np.zeros(Q.shape[0])
            # Parameters stored as a row, looping over the number of rows,
            # i.e. of distributions considered
            for r in range(Q.shape[0]):
                # Sum of the Dirichlet parameters of Q[:,c] and P[:,c]
                q_0 = np.sum(Q[r, :])
                p_0 = np.sum(P[r, :])
                # Computing KL divergence for the distribution stored in the c column
                # using gamma and digamma functions
                kl_div[r] = (
                    np.log(gamma(q_0))
                    - np.sum(np.log(gamma(Q[r, :])))
                    - np.log(gamma(p_0))
                    + np.sum(np.log(gamma(P[r, :])))
                    + np.dot(Q[r, :] - P[r, :], psi(Q[r, :]) - psi(q_0))
                )

        else:

            raise ValueError("axis should be either 0 or 1.")

    else:

        raise ValueError("Inputs dimensions should be equal or greater than 1.")

    return kl_div


def cat_KL(Q, P, axis=0):
    """Function to compute the KL divergence (relative entropy) between *categorical*
    probability distributions.

    Inputs:

    - P, Q (numpy arrays): either vectors or matrices (n,m) with each column (row) storing probabilities,
      in the former case the KL divergence is computed between two distributions, in the latter it is
      computed for m (n) distributions' pairs formed by one column (row) of Q and the corresponding column
      (row) of P;
    - axis (integer): 0 means the probabilities for one distribution are stored in a column, 1 means they
      are stored in a row.

    Outputs:

    - kl_div (float or numpy array): KL divergence for one distribution (float) or m (n) pairs of
      distributions (numpy array).

    """

    assert len(Q.shape) == len(
        P.shape
    ), "Inputs should have the same number of dimensions!"

    # Distinguishing between different input cases:
    # 1) one-dimensional numpy arrays,
    # 2) two-dimensional numpy arrays (vector or matrix)
    if len(Q.shape) == 1:
        # Initialising KL variable
        kl_div = 0
        # Computing KL divergence
        kl_div = np.sum(Q * np.log(Q / P))

    elif len(Q.shape) > 1:
        # Considering different ways the probabilities could be stored:
        # 1) as a column, i.e. over rows (n,1) or
        # 2) as a row, i.e. over columns (1,n)
        if axis == 0:
            # Initialising KL variable
            kl_div = np.zeros(Q.shape[1])
            # Parameters stored as a column, looping over the number of columns,
            # i.e. number of distributions considered
            for c in range(Q.shape[1]):
                # Computing KL divergence for the distributions stored in the c column
                kl_div[c] = np.sum(Q[:, c] * np.log(Q[:, c] / P[:, c]))

                if kl_div[c] == 0 or np.abs(kl_div[c]) < 0.0001:
                    kl_div[c] = 0.0001
                # Assertions to check there are no NaN, inf, or other unreasonable values
                assert np.all(np.isnan(kl_div[c])) == False, print(
                    f"Column {c}, i.e. {kl_div[c]}, contains NaN values"
                )
                assert np.all(np.isinf(kl_div[c])) == False, print(
                    f"Column {c}, i.e. {kl_div[c]}, contains inf values"
                )
                if kl_div[c] < 0:
                    print(kl_div[c])
                    print(Q[:, c])
                    print(P[:, c])
                assert np.all(kl_div[c] > 0) == True, print(
                    f"Some KL divergences are negative!"
                )

        elif axis == 1:
            # Initialising KL variable
            kl_div = np.zeros(Q.shape[0])
            # Parameters stored as a row, looping over the number of rows,
            # i.e. number of distributions considered
            for r in range(Q.shape[0]):
                # Computing KL divergence for the distribution stored in the r row
                kl_div[r] = np.sum(Q[r, :] * np.log(Q[r, :] / P[r, :]))

                if kl_div[c] == 0 or np.abs(kl_div[c]) < 0.0001:
                    kl_div[c] = 0.0001
                # Assertions to check there are no NaN, inf, or other unreasonable values
                assert np.all(np.isnan(kl_div[c])) == False, print(
                    f"Column {c}, i.e. {kl_div[c]}, contains NaN values"
                )
                assert np.all(np.isinf(kl_div[c])) == False, print(
                    f"Column {c}, i.e. {kl_div[c]}, contains inf values"
                )
                if kl_div[c] < 0:
                    print(kl_div[c])
                    print(Q[:, c])
                    print(P[:, c])
                assert np.all(kl_div[c] > 0) == True, print(
                    f"Some KL divergences are negative!"
                )

        else:

            raise ValueError("axis should be either 0 or 1.")

    else:

        raise ValueError("Inputs dimensions should be equal or greater than 1.")

    return kl_div


def process_obs(obs: np.ndarray) -> int:
    """
    Function to convert a (x, y) representation of a state in the gridworld to an index representation
    that numbers each state from 0 to 8 starting from the top left corner and moving from left to right,
    top to bottom

    Input:
    - obs: np.ndarray, (x, y)
    Ouput:
    - index: int
    """

    index_repr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    state_index = index_repr[obs[1], obs[0]].item()

    return state_index


def convert_state(state: int) -> np.ndarray:
    """"""

    index_repr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    y, x = np.where(index_repr == state)

    return np.array([x[0], y[0]])
