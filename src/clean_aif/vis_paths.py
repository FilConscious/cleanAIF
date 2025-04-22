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
from .vis_utils_paths import *
from .config import LOG_DIR, RESULTS_DIR


def main():

    ##################################
    ### 1. PARSING COMMAND LINE
    ##################################
    parser = argparse.ArgumentParser()
    # Name of the environment layout/task
    parser.add_argument(
        "--env_layout",
        "-el",
        type=str,
        required=True,
        help="layout of the gridworld (choices: Tmaze3, Tmaze4, Ymaze4)",
    )
    # Argument for the timestep used to plot the free energy in  plot_pi_fe()
    # (default is the last time step of every episode)
    parser.add_argument("--step_fe_pi", "-fpi", type=int, default=-1)
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
    parser.add_argument("--action_B", "-ab", type=int, default=0)
    # Argument to select a subset of the runs to plot depending on the probability of a policy being > 0.5
    parser.add_argument("--select_policy", "-selp", type=int, default=-1)
    # Arguments for the lengths of the environment to plot the state visits
    parser.add_argument("--v_len", "-vl", type=int, required=True)
    parser.add_argument("--h_len", "-hl", type=int, required=True)
    # Argument for length of a policy, i.e., policy horizon (only to plot action sequences)
    parser.add_argument("--policy_horizon", "-ph", type=int)
    # Argument to select one run/agent (e.g. used to plot one action sequence)
    parser.add_argument("--select_run", "-selrun", type=int)

    # Creating object holding the attributes from the command line
    args = parser.parse_args()
    # Convert args to dictionary
    params = vars(args)
    # print(f"Log directory: {str(LOG_DIR)}")
    # List of directories with results from different experiment
    dir = [t for t in enumerate(glob(str(LOG_DIR) + "/*", recursive=False))]
    # Asking to select a directory for visualization (if there is one available)
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
        n = int(
            input(
                "Please select some results to visualize by typing the corresponding numerical identifier: "
            )
        )
        if n >= 0 or n < len(dir):
            pass
        else:
            raise ValueError("Invalid numerical identifier. Please try again!")

    # Retrieving the directory of results to be visualized
    log_dir = dir[n][1]
    # Retrieving the path for the npy file in which data was stored (saved in log_dir).
    dir_data = [t for t in glob(log_dir + "/*.npy", recursive=False)]
    file_dp = os.path.join(
        log_dir, dir_data[0]
    )  # NOTE: this is assuming one npy file in the result directory!
    print(f"Data log directory retrieved: {file_dp}.")

    # Converting log_dir to Path
    log_dir = Path(log_dir)
    # Create result directory where to save the plot
    result_path = RESULTS_DIR / log_dir.name
    # Create the directory if it doesn't exist
    result_path.mkdir(parents=True, exist_ok=True)
    # Converting path to string
    result_dir = str(result_path)
    print(f"Result directory created: {result_dir}.")

    # Plotting saved data (see utils_vis.py for more info).

    # Plotting reward counts
    plot_reward_counts(
        file_dp,
        params["x_ticks_estep"],
        result_dir,
        params["env_layout"],
    )

    plot_action_seq(
        file_dp,
        params["x_ticks_estep"],
        params["policy_horizon"],
        params["select_run"],
        result_dir,
        params["env_layout"],
    )

    # 1.a Plotting the free energy conditioned on a policy, i.e. F_pi
    # plot_pi_fe(
    #     file_dp,
    #     params["step_fe_pi"],
    #     params["x_ticks_estep"],
    #     params["x_ticks_tstep"],
    #     params["select_policy"],
    #     result_dir,
    # )
    # 1.b Plotting the policy-conditioned free energies (F_pi) in the same plot
    plot_pi_fe_compare(
        file_dp,
        params["step_fe_pi"],
        params["x_ticks_estep"],
        params["x_ticks_tstep"],
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    # 2.a Plotting the total free energy, i.e. E_pi[F_pi]
    plot_total_fe(
        file_dp,
        params["step_fe_pi"],
        params["x_ticks_estep"],
        params["x_ticks_tstep"],
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    # 2.b Plotting the expected free energy for each policy
    plot_efe(
        file_dp,
        params["select_policy"],
        result_dir,
        params["env_layout"],
        select_step=0,
    )
    # 2.c Plotting the expected free energy components for each policy
    plot_efe_comps(
        file_dp,
        params["select_policy"],
        result_dir,
        params["env_layout"],
        num_tsteps=0,
    )
    # plot_efe_Bcomps(file_dp, params["select_policy"], result_dir)
    # 3.a Plotting the policies probabilities, i.e. Q(pi)
    plot_pi_prob(
        file_dp,
        params["x_ticks_tstep"],
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    plot_pi_prob_first(
        file_dp,
        params["x_ticks_estep"],
        params["x_ticks_tstep"],
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    # 3.b Plotting beliefs over states at a certain time step for every policy, i.e. Q(s|pi)
    # plot_Qs_pi_prob(
    #     file_dp,
    #     params["x_ticks_estep"],
    #     params["index_Si"],
    #     params["value_Si"],
    #     params["select_policy"],
    #     result_dir,
    # )
    # 3.c Plotting beliefs over states at certain time step for every policy, i.e. Q(s|pi), *as a
    # function of the experiment steps*
    # plot_Qt_pi_prob(
    #     file_dp,
    #     params["x_ticks_tstep"],
    #     params["index_tSi"],
    #     params["value_tSi"],
    #     params["select_policy"],
    #     result_dir,
    # )
    # 4. Plotting related to matrices A, i.e., state-observation mappings
    plot_so_mapping(
        file_dp,
        params["x_ticks_estep"],
        params["state_A"],
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    # 5. Plotting related to matrix B, i.e., transitions probabilities
    plot_transitions(
        file_dp,
        params["x_ticks_estep"],
        params["state_B"],
        params["action_B"],
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    # 6. Plotting other heatmaps
    # Plotting categorical distributions Q(S|pi) from the last episode and *last* step (averaged over the runs)
    plot_Qs_pi_final(
        file_dp,
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    # Plotting categorical distributions Q(S|pi) from the last episode and *first* step (averaged over the runs)
    plot_Qs_pi_first(
        file_dp,
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
    # Plotting state visits (averaged over the runs)
    plot_state_visits(
        file_dp,
        params["v_len"],
        params["h_len"],
        params["select_policy"],
        result_dir,
        params["env_layout"],
    )
