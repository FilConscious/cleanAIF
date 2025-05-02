"""
File with utility functions used by the ActInfAgent class.

Created on Wed Aug  5 16:16:00 2020
@author: Filippo Torresan
"""

import numpy as np
import math
from scipy import special


def vfe(
    num_states,
    steps,
    current_tstep,
    current_obs,
    pi,
    pi_actions,
    A,
    B,
    D,
    Qs_pi,
    A_params,
    B_params,
    learning_A=False,
    learning_B=False,
):
    """Function to compute the variational free energy (vfe) w.r.t. one policy
    over its entire trajectory (the duration of the episode).

    Inputs:

    - num_state (integer): no. of states in the environment;
    - steps (integer): no. of time steps in an episode;
    - current_tstep (integer): current time step;
    - current_obs (numpy array): matrix of shape (num_state, num_steps) with
      one-hot columns indicating the observation at each time step;
    - pi (integer): index identifying the current policy, i.e. one of the rows
      in the agent's attribute self.policies;
    - pi_actions (numpy array): array of shape (num_steps, ) storing the sequence
      of actions dictated by the policy;
    - A,B,D (numpy arrays): matrices and/or tensors storing the relevant probabilities
      of the agent's generative model (see the agent class);
    - Qs_pi (numpy array): tensor storing the variational probability distribution over
      the states for each policy (see the agent class);
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

    print("Computing FE...")

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

    # IMPORTANT: here we are replacing zero probabilities with the value 0.0001 to avoid zeroes in logs.
    # Qs_pi = np.where(Qs_pi == 0, 0.0001, Qs_pi)
    # print(Qs_pi[pi, :, :])
    st_log_st = np.sum(np.sum(Qs_pi[pi, :, :].T * np.log(Qs_pi[pi, :, :]).T, axis=1))

    # st_log_st = np.sum(np.sum(Qs_pi_first.T * np.log(Qs_pi_first).T, axis=1))
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
            * np.matmul(logA_pi, Qs_pi[pi, :, 0 : current_tstep + 1]).T,
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
    s1_logD = np.dot(Qs_pi[pi, :, 0], logD_pi)

    ####################################### Fourth term #######################################
    # The fourth term, indicated by st_ExplogB_stp, is more involved as requires the expectation
    # of Dirichlet distributed values and is a sum from t=2 to T (in Python, from t=1 to T-1).

    # Initializing the the term variable
    st_logB_stp = 0

    # If transition matrices B are learned, logB_pi stores their expectations, otherwise
    # it stores the sequence of action-dependent transition matrices defining the policy pi.
    # Note 1: the first dimension of logB_pi is (steps-1) because at the last time step there
    # is no action therefore no transition probabilities.
    logB_pi = np.zeros((steps - 1, num_states, num_states))

    # For every step in the episode retrieve the action that the policy pi dictates at
    # time step t-1,
    for t in range(steps):

        if learning_B == True:
            # There is no action at the last time step (i.e. steps-1 because in Python
            # we start counting from 0) so we retrieve B matrices as long as t is not
            # the last step.
            if t < (steps - 1):
                # Computing the expectation of the Dirichlet distributions using
                # their parameters and the digamma function
                action = pi_actions[t]
                ExplogB = psi(B_params[action]) - psi(np.sum(B_params[action], axis=0))
                logB_pi[t, :, :] = ExplogB

        else:
            # There is no action at the last time step (i.e. steps-1 because in Python
            # we start counting from 0) so we retrieve B matrices as long as t is not
            # the last step.
            if t < (steps - 1):
                # Simply retrieving the corresponding transition matrix B because its
                # parameters are not learned.

                action = pi_actions[t]
                # print(f"First action is: {action}")
                # print("Corresponding B matrix is")
                # print(f"{B[action, :, :]}")
                logB_pi[t, :, :] = np.log(B[action, :, :])

            # if t < (steps-1):
            #     action = pi_actions[t]
            #     ExplogB = psi(B_params[action])  - psi(np.sum(B_params[action], axis=0))
            #     logB_pi[t, :, :] = ExplogB

        # Adding the computed value to st_logB_stp (because we are computing expectations
        # of categorical distributions).
        if t != 0:
            st_logB_stp += np.dot(
                Qs_pi[pi, :, t], np.matmul(logB_pi[t - 1, :, :], Qs_pi[pi, :, t - 1])
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
    num_states, steps, current_tstep, current_obs, pi, Qs_pi, logA_pi, logB_pi, logD_pi
):
    """Function to compute the gradient vectors of the free energy for one policy
    w.r.t each categorical Q(s_t|pi).

    Inputs:

    - num_state (integer): no. of states in the environment;
    - steps (integer): no. of time steps in an episode;
    - current_tstep (integer): current time step;
    - current_obs (numpy array): matrix of shape (num_state, num_steps) with one-hot columns
      indicating the observation at each time step;
    - pi (integer): index identifying the current policy, i.e. one of the rows in the
      agent's attribute self.policies;
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

    print("Computing FE gradients...")
    # Initialising the gradient vectors for each Q(s_t|pi)
    grad_F_pi = np.zeros((num_states, steps))

    for t in range(steps):

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
                + np.log(Qs_pi[pi, :, t])
                - (
                    np.matmul(current_obs[:, t], logA_pi)
                    + np.matmul(Qs_pi[pi, :, t + 1], logB_pi[t, :, :])
                    + logD_pi
                )
            )

            ##### DEBUGGING #####
            # print(f'The gradient is: {grad_F_pi[10,t]}')
            ##### END #####

        # Computing the gradients w.r.t. Q(s_t|pi) where 0<t<=current_tstep
        elif t > 0 and t <= current_tstep:
            # Case when t is not the terminal state (if it was, it makes no sense to index Qs_pi
            # with t+1 below)
            if t != (steps - 1):
                grad_F_pi[:, t] = (
                    np.ones(num_states)
                    + np.log(Qs_pi[pi, :, t])
                    - (
                        np.matmul(current_obs[:, t], logA_pi)
                        + np.matmul(Qs_pi[pi, :, t + 1], logB_pi[t, :, :])
                        + np.matmul(logB_pi[t - 1, :, :], Qs_pi[pi, :, t - 1])
                    )
                )
            # Case when t is the terminal state
            elif t == (steps - 1):
                grad_F_pi[:, t] = (
                    np.ones(num_states)
                    + np.log(Qs_pi[pi, :, t])
                    - (
                        np.matmul(current_obs[:, t], logA_pi)
                        + +np.matmul(logB_pi[t - 1, :, :], Qs_pi[pi, :, t - 1])
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

            # Case when t is not the terminal state (if it was, it makes no sense to index Qs_pi
            # with t+1 below)
            if t != (steps - 1):
                grad_F_pi[:, t] = (
                    np.ones(num_states)
                    + np.log(Qs_pi[pi, :, t])
                    - (
                        np.matmul(Qs_pi[pi, :, t + 1], logB_pi[t, :, :])
                        + np.matmul(logB_pi[t - 1, :, :], Qs_pi[pi, :, t - 1])
                    )
                )

            # Case when t is the terminal state
            elif t == (steps - 1):

                grad_F_pi[:, t] = (
                    np.ones(num_states)
                    + np.log(Qs_pi[pi, :, t])
                    - np.matmul(logB_pi[t - 1, :, :], Qs_pi[pi, :, t - 1])
                )
            ##### DEBUGGING #####
            # if current_tstep == 0 and t == 1 and pi==1:
            # print(f'The gradient is: {grad_F_pi[:,t]}')
            ##### END #####
    return grad_F_pi


def total_free_energy(
    current_tstep,
    episode_steps,
    free_energies,
    Qpi=None,
    prior_A=None,
    prior_B=None,
    A_params=None,
    B_params=None,
    learning_A=False,
    learning_B=False,
):
    """Function that computes the total free energy,  i.e. F = KL[Q(A)|P(A)] + KL[Q(B)|P(B)] +
    + KL[Q(pi)|P(pi)] + E[F_pi], at the current time step based on Equation (5) in Da Costa et al.
    2020, pp. 7-8 (DOI: 10.1016/j.jmp.2020.102447). The value is then stored in the agent's
    self.total_free_energies (this total free energy is useful to see if the agent gets
    better over the steps and the episodes).

    Inputs:

    - current_tstep (integer): current time step;
    - steps (integer): number of steps per episode;
    - free_energies (numpy array): matrix storing all the policy-dependent free energies computed during
      an episode;
    - Qpi (numpy array): matrix storing all the Q(pi) computed during an episode;
    - prior_A, prior_B (numpy arrays/tensors): matrices storing the parameters for Dirichlet distributed
      random variables.

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
    if current_tstep != 0 and current_tstep != (episode_steps - 1):
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

    # Case where self.current_tstep is the last state (all KL divergences are required
    # unless there is no parameter learning)
    elif current_tstep == (episode_steps - 1):
        # Considering different learning scenarios
        if learning_A == True and learning_B == True:
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

        elif learning_A == True and learning_B == False:

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

        elif learning_A == False and learning_B == True:
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

        elif learning_A == False and learning_B == False:
            # Computing KL[Q(pi)|P(pi)]
            KL_Qpi_Ppi = cat_KL(Qpi[:, current_tstep], Qpi[:, current_tstep - 1])
            # Computing the E[F_pi] term
            E_Fpi = np.dot(Qpi[:, current_tstep], free_energies[:, current_tstep])
            # Computing the total
            total_F = KL_Qpi_Ppi + E_Fpi

    return total_F


def efe(
    num_states,
    episode_steps,
    current_tstep,
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
    - Qs_pi (numpy array): tensor storing the variational probability distribution over the states
      for each policy (see the agent class);
    - pref_type (string): string indicating the type of agent's preferences, it affects the computation
      of the risk term of the expected free energy
    - A_params, B_params (numpy arrays): matrices and/or tensors storing the parameters over
      which learning occurs;
    - learning_A, learning_B (boolean): boolean variables indicating whether learning over certain parameters
      occurs or not.

    Outputs:

    - G_pi (float): the sum of pi's expected free energies, one for each of future time step considered.
    """

    print("Computing EFE...")
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
    sq_AsW_Bs = np.zeros(episode_steps)

    # Computing matrix H for the ambiguity term of the expected free energy.
    # H = -diag[E[A]log(E[A])]
    Exp_A = A_params / np.sum(A_params, axis=0)
    H = -np.diag(np.matmul(Exp_A.T, np.log(Exp_A)))

    # Set last step for which to compute EFE
    last_step = min(current_tstep + future_steps + 1, episode_steps)
    # Loop over the time steps following the current one for which EFE is computed
    # NOTE: if current_tstep = 0 and episode_steps = 5, then EFE is computed at 1,2,3,4; here 4 is the
    # index of the last (fifth) time step in the episode
    # TODO (INVESTIGATE): not entirely clear whether the start tau should be current_tstep or current_tstep + 1
    for tau in range(current_tstep + 1, last_step):
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
            slog_s_over_C = np.dot(Qs_pi_risk, np.log(Qs_pi_risk) - np.log(C[:, tau]))
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
            if tau == (episode_steps - 1):
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

            # When the EFE terms are computed at curren_tstep = 0, save all B-novelty terms for each of the
            # considered tau to see how "novel" a certain future trajectory appears
            if current_tstep == 0:
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
    steps,
    current_obs,
    episode_actions,
    policies,
    Qpi,
    Qs_pi,
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
    - steps (integer): no. of time steps in an episode;
    - current_obs (numpy array): matrix of shape (num_state, num_steps) with one-hot columns indicating the
      observation at each time step;
    - episode_actions (numpy array): sequence of actions taken by the agent during the episode;
    - policies (numpy array): matrix storing in its rows the sequences of actions corresponding to the
      different policies;
    - Qpi (numpy array): vector of probabilities over the policies;
    - Qs_pi (numpy array): tensor storing the variational probability distribution over the states for each
      policy (see the agent class);
    - Qs (numpy array): matrix of shape (num_states, steps) storing the policy-independent state probabilities;
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
        for t in range(steps):
            Dirichlet_update_A += np.outer(current_obs[:, t], Qs[:, t])

        # Getting the approximate posterior
        Q_A_params = A_params + Dirichlet_update_A

        # Computing the Dirichlet updates for B
        for action in range(num_actions):
            for t in range(1, steps):
                for policy in range(policies.shape[0]):
                    Dirichlet_update_B[action, :, :] += (
                        (action == policies[policy, t - 1])
                        * Qpi[policy]
                        * np.outer(Qs_pi[policy, :, t], Qs_pi[policy, :, t - 1])
                    )

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
        for t in range(steps):
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
            # Note (important): we want to consider all the time steps in which an action was taken,
            # i.e. from index 0 to index steps - 2 because at the last time step, indexed by steps - 1,
            # no action is selected. In Python the for-loop below goes from index 1 to index steps - 1
            # which allows us to pick the action at the first time step with episode_actions[t - 1]
            # when t = 1, and exclude the non-existent action at the last time step because the last value
            # of t is t = steps - 1 which gives us episode_actions[steps - 2].
            for t in range(1, steps):
                for policy in range(policies.shape[0]):
                    Dirichlet_update_B[action, :, :] += (
                        (action == episode_actions[t - 1])
                        * Qpi[policy]
                        * np.outer(Qs_pi[policy, :, t], Qs_pi[policy, :, t - 1])
                    )

        # Getting the approximate posterior
        Q_A_params = A_params  # Nothing is learned here
        # assert np.array_equal(Dirichlet_update_B[2,:,:], Dirichlet_update_B[3,:,:]) == False, 'Updates suspiciously identical!'
        print(f"Old B params {B_params[2,:]}")
        # print(f"Dirichlet update: {Dirichlet_update_B}")
        Q_B_params = B_params + Dirichlet_update_B
        print(f"New B params {B_params[2,:]}")
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


def process_obs(obs: np.ndarray, env_layout: str) -> int:
    """
    Function to convert a (x, y) representation of a state in the gridworld to an index representation
    that numbers each state from 0 to 8 starting from the top left corner and moving from left to right,
    top to bottom

    Input:
    - obs: np.ndarray, (x, y)
    Ouput:
    - index: int
    """

    if env_layout == "Tmaze3":

        index_repr = np.array([[0, 1, 2], [-1, 3, -1], [-1, -1, -1]])
        state_index = index_repr[obs[1], obs[0]].item()

        assert state_index != -1, print(f"Value of state index is invalid.")

    elif env_layout == "Tmaze4":

        index_repr = np.array([[0, 1, 2], [-1, 3, -1], [-1, 4, -1]])
        state_index = index_repr[obs[1], obs[0]].item()

        assert state_index != -1, print(f"Value of state index is invalid.")

    elif env_layout == "Ymaze4":

        index_repr = np.array([[0, -1, 1], [2, 3, 4], [-1, 5, -1]])
        state_index = index_repr[obs[1], obs[0]].item()

        assert state_index != -1, print(f"Value of state index is invalid.")

    else:
        raise ValueError(
            "Value of 'env_layout' is not among the available ones. Choose from: Tmaze3, Tmaze4, Ymaze4."
        )

    return state_index


def convert_state(state: int, env_layout: str) -> np.ndarray:
    """"""

    assert state != -1, print(f"Value of state is invalid.")

    if env_layout == "Tmaze3":

        index_repr = np.array([[0, 1, 2], [-1, 3, -1], [-1, -1, -1]])
        y, x = np.where(index_repr == state)

    elif env_layout == "Tmaze4":

        index_repr = np.array([[0, 1, 2], [-1, 3, -1], [-1, 4, -1]])
        y, x = np.where(index_repr == state)

    elif env_layout == "Ymaze4":

        index_repr = np.array([[0, -1, 1], [2, 3, 4], [-1, 5, -1]])
        y, x = np.where(index_repr == state)

    else:
        raise ValueError(
            "Value of 'env_layout' is not among the available ones. Choose from: Tmaze3, Tmaze4, Ymaze4."
        )

    return np.array([x[0], y[0]])


def set_wall_xy(state: int) -> np.ndarray:
    """"""

    assert state != -1, print(f"Value of state is invalid.")

    index_repr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    y, x = np.where(index_repr == state)

    return np.array([x[0], y[0]])
