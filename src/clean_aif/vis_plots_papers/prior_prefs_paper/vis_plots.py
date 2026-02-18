"""
Main file for plotting saved results.

Created on Sun Jul 11 09:31:00 2021
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import shutil
import argparse
from pathlib import Path
from glob import glob

# Custom packages/modules imports
from .vis_utils import *
from .config import LOG_DIR, RESULTS_DIR


def main():

    ##################################
    ### 1. PARSING COMMAND LINE
    ##################################

    parser = argparse.ArgumentParser()

    # Names of the environment and the layout/task
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
        required=True,
        help="layout of the gridworld (choices: Tmaze3, Tmaze4, Ymaze4)",
    )
    # Number of experiments to visualize
    parser.add_argument("--num_exps", "-nexp", type=int, default=1)
    # Name to give to the result directory
    parser.add_argument("--result_dir", "-rdir", type=str)
    # Argument for the timestep used to plot the free energies in plot_pi_fes()
    # (default: last time step of every episode)
    parser.add_argument("--step_fe_pi", "-fpi", nargs="*", type=int, default=[0])
    # Step for the ticks in the x axis for plotting a variable as a function of the number of episodes
    parser.add_argument("--x_ticks_estep", "-xtes", type=int, default=1)
    # Step for the ticks in the x axis for plotting a random variable as a function of the total number of
    # timesteps in an experiment
    parser.add_argument("--x_ticks_tstep", "-xtts", type=int, default=50)
    # Arguments for selecting a random variable S_i and its desired value g to plot the Q(S_i=g|pi)
    # at a certain episode step for every episode (see the function plot_Qs_pi_prob())
    parser.add_argument("--index_Si", "-i", type=int, default=0)
    parser.add_argument("--value_Si", "-v", type=int, required=True)
    # Arguments for selecting a random variable S_i and its desired value g to plot the Q(S_i=g|pi)
    # at *every* timestep during the experiment (see the function plot_Qt_pi_prob())
    parser.add_argument("--index_tSi", "-ti", type=int, default=0)
    parser.add_argument("--value_tSi", "-tv", type=int, required=True)

    # Argument for selecting and plotting a column of matrix A, storing the observation (or emission)
    # probabilities when in a certain state
    # NOTE: this is only required/useful when the agents learns about state-observation mappings
    parser.add_argument("--state_A", "-sa", type=int, default=0)
    # Arguments for plotting the transitions probabilities for a specific state and action
    # (i.e. a column of a B matrix) over the entire experiment
    parser.add_argument("--state_B", "-sb", type=int, default=0)
    parser.add_argument("--action_B", "-ab", nargs="*", type=int, default=[0])

    # Argument to select a subset of the runs to plot depending on the probability of a policy being > 0.5
    parser.add_argument("--select_policy", "-selp", type=int, default=-1)
    # Arguments for the lengths of the environment to plot the state visits
    parser.add_argument("--v_len", "-vl", type=int, required=True)
    parser.add_argument("--h_len", "-hl", type=int, required=True)
    # Argument for length of a policy, i.e., policy horizon (only to plot action sequences)
    parser.add_argument("--policy_horizon", "-ph", type=int)
    # Argument to select one or more runs/agents (e.g. used to plot one action sequence)
    parser.add_argument(
        "--select_run",
        "-selrun",
        nargs="*",
        type=int,
        default=[],
        help="A list of run indices",
    )
    # Argument to select one specific episode ( used to plot policy-conditioned state beliefs)
    parser.add_argument("--select_episode", "-selep", nargs="*", type=int, default=[-1])
    # Number of policies to visualize (the same for each agent/experiment)
    parser.add_argument("--num_policies_vis", "-npv", type=int, required=True)
    # Argument to select a group of policies to plot
    # NOTE: for more than one experiment the groups of indices should be specified in the order in which
    # the experiment are selected from the command line.
    parser.add_argument(
        "--policies_to_vis",
        "-polvis",
        nargs="*",
        type=int,
        default=[],
        help="A list of policies indices",
    )

    # Creating object holding the attributes from the command line
    args = parser.parse_args()
    # Convert args to dictionary
    params = vars(args)
    # print(f"Log directory: {str(LOG_DIR)}")

    #################################################
    ### 2. RETRIEVE DATA AND CREATE RESULT DIRECTORY
    #################################################

    # List of directories with results from different experiment
    dir = [t for t in enumerate(glob(str(LOG_DIR) + "/*", recursive=False))]
    # List with indices of experiments to visualize
    exps_to_vis = []

    # Asking to select one or more directory(ies) for visualization
    if len(dir) == 0:
        raise Exception(
            "Sorry, there are no results to visualize. Run an experiment first!"
        )
    else:
        print(
            f"This is a list of result directories together with their numerical identifiers: \n"
        )
        for t in dir:
            print(t)

        for _ in range(params["num_exps"]):
            n = int(
                input(
                    "Please select some results to visualize by typing the corresponding numerical identifier: "
                )
            )
            if n >= 0 or n < len(dir):
                exps_to_vis.append(n)
            else:
                raise ValueError("Invalid numerical identifier. Please try again!")

    assert len(exps_to_vis) != 0, print(
        "List of experiments to visualize cannot be emtpy."
    )

    # List with data files to visualize
    data_to_vis = []
    # Loop over data directories and retrieve corresponding paths to .npy files
    for e in exps_to_vis:
        # Retrieve the string with the directory to the results
        log_dir = dir[e][1]
        # Retrievw the path to the npy file in which data was stored
        dir_data = [t for t in glob(log_dir + "/*.npy", recursive=False)]
        file_dp = os.path.join(
            log_dir, dir_data[0]
        )  # NOTE: this is assuming one npy file in the result directory!
        data_to_vis.append(file_dp)
        print(f"Data log directory retrieved: {file_dp}.")

    # Converting log_dir to Path
    # log_dir = Path(log_dir)

    # Create result directory where to save the plots
    result_path = RESULTS_DIR / (
        params["gym_id"] + params["env_layout"] + params["result_dir"]
    )
    # Create the directory if it doesn't exist
    result_path.mkdir(parents=True, exist_ok=True)
    # Converting path to string
    result_dir = str(result_path)
    print(f"Result directory created: {result_dir}.")

    #################################################
    ### 3. CALLS TO PLOTTING FUNCTIONS
    ### (see vis_utils.py for more info).
    #################################################

    num_policies_vis = params["num_policies_vis"]
    # Loop over the list of datasets to visualize (one for each experiment)
    # NOTE: this is done to plot/create separate figures
    for i, data_path in enumerate(data_to_vis):

        # Select correct policies from the list params["policies_to_vis"]
        policies_to_vis = params["policies_to_vis"][
            i * num_policies_vis : (i * num_policies_vis) + num_policies_vis
        ]
        # Plot step count across episodes
        # plot_steps_count(
        #     data_path,
        #     params["x_ticks_estep"],
        #     result_dir,
        #     params["env_layout"],
        # )
        plot_action_probs(
            data_path,
            params["x_ticks_estep"],
            [0, 0.6], # 0.6
            0,
            result_dir,
            params["env_layout"],
        )
        plot_action_probs(
            data_path,
            params["x_ticks_estep"],
            [0, 0.6],
            1,
            result_dir,
            params["env_layout"],
        )
        plot_action_probs(
            data_path,
            params["x_ticks_estep"],
            [0, 0.6],
            2,
            result_dir,
            params["env_layout"],
        )
        plot_action_probs(
            data_path,
            params["x_ticks_estep"],
            [0, 0.8],
            3,
            result_dir,
            params["env_layout"],
        )
        plot_matrix_B(
            data_path,
            params["x_ticks_estep"],
            params["state_B"],
            params["action_B"],
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        # Plot matrices B (transitions probabilities)
        plot_matrix_B_kl(
            data_path,
            params["x_ticks_estep"],
            [0, 22],  # Tmaze4: [0, 18]; gridw9: [0, 22]
            params["state_B"],
            params["action_B"],
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        # Plot percentage of successful agents across episodes
        plot_avg_good_agents(
            data_path,
            params["x_ticks_estep"],
            result_dir,
            params["env_layout"],
        )
        # Plot marginal free energy, i.e. E_pi[F_pi]
        plot_marginal_fe(
            data_path,
            params["step_fe_pi"][0],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 3],  # Tmaze3: [0, 5.5]; Tmaze4: [0, 3]; gridw9: [0, 3]
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        plot_marginal_fe(
            data_path,
            params["step_fe_pi"][1],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 5],  # Tmaze3: [0, 5.5]; Tmaze4: [0, 4]; gridw9: [0, 5]
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        plot_marginal_fe(
            data_path,
            params["step_fe_pi"][2],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 8],  # Tmaze3: [0, 5.5]; Tmaze4: [0, 6]; gridw9: [0, 8]
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        plot_marginal_fe(
            data_path,
            params["step_fe_pi"][3],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 12],  # Tmaze3: [0, 5.5]; Tmaze4: [0, 10]; gridw9: [0, 12]
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        plot_marginal_fe(
            data_path,
            params["step_fe_pi"][4],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 16],  # gridw9: [0, 16]
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )

        # Plot policy-conditioned free energies, F_pi, on the same axis
        plot_pi_fes(
            data_path,
            params["step_fe_pi"][0],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 4],  # Tmaze4: [0, 4]; gridw9: [0, 3]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # Plot policy-conditioned free energies, F_pi, on the same axis
        plot_pi_fes(
            data_path,
            params["step_fe_pi"][1],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 10],  # Tmaze3: [0, 5]; Tmaze4: [0, 8]; gridw9: [0, 10]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # Plot policy-conditioned free energies, F_pi, on the same axis
        plot_pi_fes(
            data_path,
            params["step_fe_pi"][2],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 12],  # Tmaze4: [0, 8] gridw9: [0, 12]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # Plot policy-conditioned free energies, F_pi, on the same axis
        plot_pi_fes(
            data_path,
            params["step_fe_pi"][3],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 12],  # Tmaze3: [0, 5]; Tmaze4: [0, 10]; gridw9: [0, 12]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # Plot policy-conditioned free energies, F_pi, on the same axis
        plot_pi_fes(
            data_path,
            params["step_fe_pi"][4],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 16],  # Tmaze3: [0, 5]; gridw9: [0, 14]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # Plot expected state log-prob of the free energy
        plot_pi_state_logprob(
            data_path,
            params["step_fe_pi"][0],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [-10, 0],  # Tmaze3: [0, 5]; Tmaze4: [0, 10]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # Plot expected state log-prob for the first state of the free energy
        # plot_pi_state_logprob_first(
        #     data_path,
        #     params["step_fe_pi"][0],
        #     params["x_ticks_estep"],
        #     params["x_ticks_tstep"],
        #     [0, 10],  # Tmaze3: [0, 5]; Tmaze4: [0, 10]
        #     params["select_policy"],
        #     result_dir,
        #     params["env_layout"],
        #     policies_to_vis=policies_to_vis,
        # )
        # # Plot expected obs likelihood of the free energy
        # plot_pi_obs_loglik(
        #     data_path,
        #     params["step_fe_pi"][0],
        #     params["x_ticks_estep"],
        #     params["x_ticks_tstep"],
        #     [0, 10],  # Tmaze3: [0, 5]; Tmaze4: [0, 10]
        #     params["select_policy"],
        #     result_dir,
        #     params["env_layout"],
        #     policies_to_vis=policies_to_vis,
        # )
        # # Plot expected obs likelihood of the free energy
        plot_pi_transit_loglik(
            data_path,
            params["step_fe_pi"][0],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 10],  # Tmaze3: [0, 5]; Tmaze4: [0, 10]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        plot_pi_fes_efe(
            data_path,
            params["step_fe_pi"][1],
            params["x_ticks_estep"],
            params["x_ticks_tstep"],
            [0, 12],  # Tmaze3: [0, 5]; Tmaze4: [0, 12]; gridw9: [0, 10]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # plot_pi_fes_efe(
        #     data_path,
        #     params["step_fe_pi"][2],
        #     params["x_ticks_estep"],
        #     params["x_ticks_tstep"],
        #     [0, 10],  # Tmaze3: [0, 5]; Tmaze4: [0, 10]; gridw9: [0, 10]
        #     params["select_policy"],
        #     result_dir,
        #     params["env_layout"],
        #     policies_to_vis=policies_to_vis,
        # )
        # plot_pi_fes_efe(
        #     data_path,
        #     params["step_fe_pi"][3],
        #     params["x_ticks_estep"],
        #     params["x_ticks_tstep"],
        #     [0, 10],  # Tmaze3: [0, 5]; Tmaze4: [0, 8]; gridw9: [0, 10]
        #     params["select_policy"],
        #     result_dir,
        #     params["env_layout"],
        #     policies_to_vis=policies_to_vis,
        # )
        # Plot expected free energies, EFE, for each policy on the same axis
        plot_efe(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            params["x_ticks_estep"],
            [0, 18],  # Tmaze4: [0, 12]; gridw9 [0, 18]
            select_step=0,
            policies_to_vis=policies_to_vis,
        )
        # Plot expected free energies, EFE, for each policy on the same axis
        plot_efe(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            params["x_ticks_estep"],
            [0, 14],  # Tmaze4: [0, 8]; gridw9 [0, 14]
            select_step=1,
            policies_to_vis=policies_to_vis,
        )
        # Plot expected free energies, EFE, for each policy on the same axis
        plot_efe(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            params["x_ticks_estep"],
            [0, 10],  # Tmaze4: [0, 8]; gridw9 [0, 10]
            select_step=2,
            policies_to_vis=policies_to_vis,
        )
        # # # Plot expected free energies, EFE, for each policy on the same axis
        plot_efe(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            params["x_ticks_estep"],
            [0, 4],  # gridw9 [0, 4]
            select_step=3,
            policies_to_vis=policies_to_vis,
        )
        # Plot the expected free energy component RISK for all policies
        plot_efe_risk(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            params["x_ticks_estep"],
            [0, 18],  # Tmaze4: [0, 12]; gridw9 [0, 18]
            num_tsteps=0,
            policies_to_vis=policies_to_vis,
        )
        # Plot the expected free energy component B-NOVELTY for all policies
        plot_efe_bnov(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            params["x_ticks_estep"],
            [0, 2],  # Tmaze4: [0, 1]; gridw9 [0, 2]
            num_tsteps=0,
            policies_to_vis=policies_to_vis,
        )
        # Plot the expected free energy component AMBIGUITY for all policies
        plot_efe_ambiguity(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            num_tsteps=0,
            policies_to_vis=policies_to_vis,
        )
        # Plot the expected free energy component A-NOVELTY for all policies
        plot_efe_anov(
            data_path,
            params["select_policy"],
            result_dir,
            params["env_layout"],
            num_tsteps=0,
            policies_to_vis=policies_to_vis,
        )
        plot_pi_prob_first(
            data_path,
            params["x_ticks_estep"],
            [0, 0.6],  # Tmaze3: [0, 0.45]; Tmaze4: [0, 0.08]; gridw9: [0, 0.6]
            params["select_policy"],
            result_dir,
            params["env_layout"],
            policies_to_vis=policies_to_vis,
        )
        # Plot matrices A (state-observation mappings)
        plot_matrix_A(
            data_path,
            params["x_ticks_estep"],
            params["state_A"],
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        # Plot matrices B (transitions probabilities)
        plot_matrix_B(
            data_path,
            params["x_ticks_estep"],
            params["state_B"],
            params["action_B"],
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        # Plot state visits
        plot_state_visits(
            data_path,
            params["v_len"],
            params["h_len"],
            params["select_policy"],
            result_dir,
            params["env_layout"],
        )
        # Plot action sequence for a selected run/agent
        # plot_action_seq(
        #     data_path,
        #     params["x_ticks_estep"],
        #     params["policy_horizon"],
        #     params["select_run"],
        #     result_dir,
        #     params["env_layout"],
        # )

        # Plot categorical distributions Q(S|pi) from an episode's *first* step (averaged over the runs)
        # plot_Qs_pi_first(
        #     data_path,
        #     params["select_policy"],
        #     params["select_episode"],
        #     result_dir,
        #     params["env_layout"],
        # )
        # Plot categorical distributions Q(S|pi) from an episode's *last* step (averaged over the runs)
        # plot_Qs_pi_last(
        #     data_path,
        #     params["select_policy"],
        #     params["select_episode"],
        #     result_dir,
        #     params["env_layout"],
        # )
        # Plot categorical distributions Q(S|pi) from ALL steps in a single episode (averaged over the runs)
        # plot_Qs_pi_all(
        #     data_path,
        #     params["select_policy"],
        #     params["select_episode"],
        #     result_dir,
        #     params["env_layout"],
        # )

    # Access data from all selected experiments to plot results in subplots

    # Plot policy-conditioned free energies, F_pi, on the same axis
    # Plot Fe_pi at different time steps if the list is not empty
    # for step_fe_pi in params["step_fe_pi"]:

    #     if step_fe_pi == 0:
    #         y_range = [0, 3]  # Tmaze4 [0, 5]
    #     elif step_fe_pi == 1:
    #         y_range = [0, 8]  # Tmaze4 [0, 10]
    #     elif step_fe_pi == 2:
    #         y_range = [0, 8]  # Tmaze4 [0, 15]
    #     else:
    #         y_range = [0, 10]  # Tmaze4 [0, 15]

    #     plot_pi_fes_subplots(
    #         data_to_vis,
    #         step_fe_pi,
    #         params["x_ticks_estep"],
    #         params["x_ticks_tstep"],
    #         y_range,  # Tmaze3: [0, 5]; Tmaze4: [0, 10]
    #         params["select_policy"],
    #         result_dir,
    #         params["env_layout"],
    #         num_policies_vis,
    #         policies_to_vis=params["policies_to_vis"],
    #     )

    # plot_efe_subplots(
    #     data_to_vis,
    #     params["x_ticks_estep"],
    #     [2, 10],  # Tmaze3: [2, 6]; Tmaze4: [2, 10]
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     select_step=0,
    #     policies_to_vis=params["policies_to_vis"],
    # )
    # plot_efe_subplots(
    #     data_to_vis,
    #     params["x_ticks_estep"],
    #     [2, 10],  # Tmaze3: [2, 6]; Tmaze4: [3, 11]
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     select_step=1,
    #     policies_to_vis=params["policies_to_vis"],
    # )
    # plot_efe_subplots(
    #     data_to_vis,
    #     params["x_ticks_estep"],
    #     [0, 10],  # Tmaze3: [2, 6]; Tmaze4: [3, 11]
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     select_step=2,
    #     policies_to_vis=params["policies_to_vis"],
    # )
    # plot_efe_risk_subplots(
    #     data_to_vis,
    #     [3, 10],  # Tmaze3: [2, 6]; Tmaze4: [3, 11]
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     num_tsteps=0,
    #     policies_to_vis=params["policies_to_vis"],
    # )
    # plot_efe_bnov_subplots(
    #     data_to_vis,
    #     [0, 0.9],  # Tmaze3: [0.1, 0.5]; Tmaze4: [0, 0.9]
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     num_tsteps=0,
    #     policies_to_vis=params["policies_to_vis"],
    # )
    # plot_efe_ambiguity_subplots(
    #     data_to_vis,
    #     [0, 5],
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     num_tsteps=0,
    #     policies_to_vis=params["policies_to_vis"],
    # )
    # plot_efe_anov_subplots(
    #     data_to_vis,
    #     [0, 5],
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     num_tsteps=0,
    #     policies_to_vis=params["policies_to_vis"],
    # )
    # plot_pi_prob_first_subplots(
    #     data_to_vis,
    #     params["x_ticks_estep"],
    #     [0, 0.1],  # Tmaze3: [0, 0.45]; Tmaze4: [0, 0.09]
    #     params["select_policy"],
    #     result_dir,
    #     params["env_layout"],
    #     num_policies_vis,
    #     policies_to_vis=params["policies_to_vis"],
    # )
