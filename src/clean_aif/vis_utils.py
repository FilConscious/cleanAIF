"""
Definition of function(s) for plotting and saving data

Created on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
"""

# Standard libraries imports
from logging import raiseExceptions
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns

from .utils_agents.utils_au import cat_KL

sns.set_style("whitegrid")  # setting style
sns.set_context("paper")  # setting context
sns.set_palette("colorblind")  # setting palette

plt.rc("axes", titlesize=12, labelsize=10)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
plt.rc("legend", fontsize=12)
plt.rc("figure", titlesize=14)

POLICY_INDEX_OFFSET = 0
NUM_POLICIES_VIS = 16

# Actions in the maze for observer
actions_map = {
    0: "$\\rightarrow$",
    1: "$\\downarrow$",
    2: "$\\leftarrow$",
    3: "$\\uparrow$",
}

# Used in latest experiments for new paper testing prior preferences
POLICIES_TO_VIS_GRIDW9 = np.array(
    [
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 2, 1],
        [1, 1, 0, 3],
        [1, 0, 0, 3],
        [1, 0, 3, 2],
        [0, 1, 1, 3],
        [0, 2, 0, 0],
        [0, 2, 1, 1],
        [2, 1, 1, 0],
        [3, 2, 0, 0],
        [3, 3, 1, 1],
    ]
)

# Used for submission to arxiv (action-unaware paper)
# POLICIES_TO_VIS_GRIDW9 = np.array(
#     [
#         [0, 0, 1, 1],
#         [1, 1, 0, 0],
#         [0, 1, 0, 1],
#         [0, 1, 1, 0],
#         [1, 0, 1, 0],
#         [1, 0, 0, 1],
#         [0, 1, 2, 1],
#         [1, 1, 0, 3],
#         [1, 0, 0, 3],
#         [1, 0, 3, 2],
#         [0, 1, 1, 3],
#         [0, 2, 0, 0],
#         [0, 2, 1, 1],
#         [2, 3, 1, 0],
#         [0, 1, 3, 3],
#         [1, 2, 3, 3],
#     ]
# )

# POLICIES_TO_VIS_TMAZE4 = np.array(
# POLICIES_TO_VIS_GRIDW9 = np.array(
#     [
#         [2, 3, 3],
#         [3, 3, 1],
#         [3, 2, 2],
#         [1, 2, 0],
#         [0, 3, 3],
#         [0, 0, 3],
#         [0, 2, 2],
#         [3, 3, 2],
#         [3, 3, 3],
#         [2, 3, 0],
#         [3, 0, 2],
#         [1, 1, 1],
#         [2, 2, 3],
#         [3, 3, 0],
#         [0, 2, 3],
#         [1, 3, 3],
#     ]
# )


######################################################################################################
##### Plot percentage of agents that solve the task
######################################################################################################
def plot_avg_good_agents(file_data_path, x_ticks_estep, save_dir, env_layout):
    """
    Function to plot the percentage of agents that reach the goal state in each episode.

    Inputs:
    - file_data_path (str): file path where all metrics have been stored
    - x_ticks_estep (int): step for the ticks in the x-axis when plotting as a function of episode number
    - save_dir (str): directory where to save the figure
    - env_layout (str): layout of the environment/task

    Output:
    - line plot
    """

    # Retrieving the data dictionary and extracting the content of various keys
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    reward_counts = data["reward_counts"]
    # inf_steps = data["inf_steps"]

    avg_rewards = np.mean(reward_counts, axis=0)
    std_rewards = np.std(reward_counts, axis=0)

    plt.figure(figsize=(5, 4), tight_layout=True)
    plt.plot(
        np.arange(1, num_episodes + 1),
        avg_rewards,
        ".-",
        label="",
    )
    # plt.xticks(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    plt.xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )

    # Get current axis and set y-axis to percentages
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xlabel("Episode")
    # set y-axis from 0% to 100% (adding 0.5 to have more space at the top)
    plt.ylim(0, 1.05)
    plt.ylabel("Percentage of agents", rotation=90)
    # plt.legend(loc="upper right") # not needed for single line
    # Title
    title = "Agents solving the task\n"
    # title += "(action-unaware)" if "paths" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    plt.title(title, pad=15)

    # Add a customized grid
    # plt.grid(
    #     True,
    #     which="both",  # 'major', 'minor', or 'both'
    #     axis="both",  # 'x', 'y', or 'both'
    #     linestyle="--",
    #     linewidth=0.5,
    #     color="gray",
    #     alpha=0.7,
    # )

    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_agents_goal.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.2,
    )
    # plt.show()
    plt.close()


######################################################################################################
##### Functions to plot policy-conditioned free energies and marginalized free energies
######################################################################################################
def plot_pi_fes(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    policies_to_vis=[],
):
    """Function to visualize all policy-conditioned free energies across time steps and episodes
    (all on the same axis).

    Inputs:
    - file_data_path (str): path to the .npy file where the data was stored
    - step_fe_pi (int): timestep from which to plot the free energy
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - x_ticks_tstep (int): step for the ticks in the x axis when plotting as a function of total timesteps
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the runs (depending on policy probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - scatter plot of policy-conditioned free energies at EACH time step in the experiment
    - scatter plot of policy-conditioned free energies across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    # num_runs = data["num_runs"]
    # num_policies = data["num_policies"]
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_fe = data["pi_free_energies"][selected_runs]
    else:
        pi_fe = data["pi_free_energies"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, f"Invalid step number: {step_fe_pi}"

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Figure with policy-conditioned free energies at EACH timestep in the experiment
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # Looping over the policies for Figure 1
    for p in range(NUM_POLICIES_VIS):

        # Computing the mean (average) and std of one policy's free energies over the runs
        # TODO: handle rare case in which you train only for one episode, in that case squeeze()
        # will raise the exception
        avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
        std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
        # Making sure avg_pi_fe has the right dimensions
        assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"
        # Plotting the free energy for every time step
        x1 = np.arange(num_episodes * num_steps)
        y1 = avg_pi_fe.flatten()

        # ax.plot(x1, y1, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")
        ax.plot(x1, y1, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")

    # Completing drawing axes for Figure 1
    ax.set_xticks(
        np.arange(x_ticks_estep, (num_episodes * num_steps) + 1, step=x_ticks_tstep)
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Free Energy", rotation=90)
    ax.legend(
        ncol=4,
        loc="upper center",
        title_fontsize=16,
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
    )
    ax.set_title("Free energy at each step across episodes\n")

    # Save figures and show
    fig.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_policies_fe_every_step.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()

    ### Figure with policy-conditioned free energies across episodes
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    if len(policies_to_vis) == 0:
        # Looping over the policies for Figure 2
        for p in range(NUM_POLICIES_VIS):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            std_pi_fe = np.std(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            # Making sure avg_pi_fe has the right dimensions
            # print(avg_pi_fe.shape)
            # print((num_episodes, num_steps))
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p + POLICY_INDEX_OFFSET])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:
        # Looping over the policies for Figure 2
        for i, p in enumerate(policies_to_vis):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i + 1}}}$: {policy_action_seq_leg}",
            )

        # Confidence intervals (if needed, uncomment following lines)
        # ax.fill_between(
        #     x2,
        #     y2 - (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     y2 + (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

    # Completing drawing axes for Figure 2
    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Free energy", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])

    # Create a separate figure for the legend
    # fig_legend = plt.figure(figsize=(8, 2))
    fig_legend = plt.figure(figsize=(10, 2))
    # Use the same handles and labels
    handles, labels = ax.get_legend_handles_labels()
    fig_legend.legend(
        handles,
        labels,
        title="Policies",
        ncol=4,
        title_fontsize=12,
        handlelength=2,  # shrink the line handle
        columnspacing=1.5,  # space between columns
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        fancybox=True,
    )
    # Save the legend figure separately
    fig_legend.savefig(
        save_dir + "/" + f"{env_layout}_aif_policies_legend.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
    )

    plt.close(fig_legend)

    if step_fe_pi != -1:
        step_num = f"{step_fe_pi + 1}"
    else:
        step_num = f"{num_steps}"

    title = f"Policy-conditioned free energy at step {step_num}\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)

    fig.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_policies_fe_step{step_num}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_pi_state_logprob(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    policies_to_vis=[],
):
    """Function to visualize all policy-conditioned free energies across time steps and episodes
    (all on the same axis).

    Inputs:
    - file_data_path (str): path to the .npy file where the data was stored
    - step_fe_pi (int): timestep from which to plot the free energy
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - x_ticks_tstep (int): step for the ticks in the x axis when plotting as a function of total timesteps
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the runs (depending on policy probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - scatter plot of policy-conditioned free energies at EACH time step in the experiment
    - scatter plot of policy-conditioned free energies across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    # num_runs = data["num_runs"]
    # num_policies = data["num_policies"]
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_fe = data["state_logprob"][selected_runs]
    else:
        pi_fe = data["state_logprob"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, "Invalid step number."

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Figure with policy-conditioned free energies across episodes
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    if len(policies_to_vis) == 0:
        # Looping over the policies for Figure 2
        for p in range(NUM_POLICIES_VIS):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            std_pi_fe = np.std(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            # Making sure avg_pi_fe has the right dimensions
            # print(avg_pi_fe.shape)
            # print((num_episodes, num_steps))
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p + POLICY_INDEX_OFFSET])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:
        # Looping over the policies for Figure 2
        for i, p in enumerate(policies_to_vis):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i + 1}}}$: {policy_action_seq_leg}",
            )

        # Confidence intervals (if needed, uncomment following lines)
        # ax.fill_between(
        #     x2,
        #     y2 - (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     y2 + (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

    # Completing drawing axes for Figure 2
    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Log-probabilities", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])

    if step_fe_pi != -1:
        step_num = f"{step_fe_pi + 1}"
    else:
        step_num = f"{num_steps}"

    title = f"Expected state log-probability at step {step_num}\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"

    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)

    fig.savefig(
        save_dir
        + "/"
        + f"{env_layout}_{exp_name}_policies_state_logprob_step{step_num}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_pi_state_logprob_first(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    policies_to_vis=[],
):
    """Function to visualize all policy-conditioned free energies across time steps and episodes
    (all on the same axis).

    Inputs:
    - file_data_path (str): path to the .npy file where the data was stored
    - step_fe_pi (int): timestep from which to plot the free energy
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - x_ticks_tstep (int): step for the ticks in the x axis when plotting as a function of total timesteps
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the runs (depending on policy probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - scatter plot of policy-conditioned free energies at EACH time step in the experiment
    - scatter plot of policy-conditioned free energies across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    # num_runs = data["num_runs"]
    # num_policies = data["num_policies"]
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_fe = data["state_logprob_first"][selected_runs]
    else:
        pi_fe = data["state_logprob_first"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, "Invalid step number."

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Figure with policy-conditioned free energies across episodes
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    if len(policies_to_vis) == 0:
        # Looping over the policies for Figure 2
        for p in range(NUM_POLICIES_VIS):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            std_pi_fe = np.std(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            # Making sure avg_pi_fe has the right dimensions
            # print(avg_pi_fe.shape)
            # print((num_episodes, num_steps))
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p + POLICY_INDEX_OFFSET])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:
        # Looping over the policies for Figure 2
        for i, p in enumerate(policies_to_vis):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i + 1}}}$: {policy_action_seq_leg}",
            )

        # Confidence intervals (if needed, uncomment following lines)
        # ax.fill_between(
        #     x2,
        #     y2 - (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     y2 + (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

    # Completing drawing axes for Figure 2
    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Log-probabilities", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])

    if step_fe_pi != -1:
        step_num = f"{step_fe_pi + 1}"
    else:
        step_num = f"{num_steps}"

    title = f"Expected state log-probability at step {step_num} for the initial state\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"

    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)

    fig.savefig(
        save_dir
        + "/"
        + f"{env_layout}_{exp_name}_policies_state_logprob_first_step{step_num}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_pi_obs_loglik(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    policies_to_vis=[],
):
    """Function to visualize all policy-conditioned free energies across time steps and episodes
    (all on the same axis).

    Inputs:
    - file_data_path (str): path to the .npy file where the data was stored
    - step_fe_pi (int): timestep from which to plot the free energy
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - x_ticks_tstep (int): step for the ticks in the x axis when plotting as a function of total timesteps
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the runs (depending on policy probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - scatter plot of policy-conditioned free energies at EACH time step in the experiment
    - scatter plot of policy-conditioned free energies across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    # num_runs = data["num_runs"]
    # num_policies = data["num_policies"]
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_fe = data["obs_loglik"][selected_runs]
    else:
        pi_fe = data["obs_loglik"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, "Invalid step number."

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Figure with policy-conditioned free energies across episodes
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    if len(policies_to_vis) == 0:
        # Looping over the policies for Figure 2
        for p in range(NUM_POLICIES_VIS):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            std_pi_fe = np.std(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            # Making sure avg_pi_fe has the right dimensions
            # print(avg_pi_fe.shape)
            # print((num_episodes, num_steps))
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p + POLICY_INDEX_OFFSET])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:
        # Looping over the policies for Figure 2
        for i, p in enumerate(policies_to_vis):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i + 1}}}$: {policy_action_seq_leg}",
            )

        # Confidence intervals (if needed, uncomment following lines)
        # ax.fill_between(
        #     x2,
        #     y2 - (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     y2 + (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

    # Completing drawing axes for Figure 2
    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Log-probabilities", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])

    if step_fe_pi != -1:
        step_num = f"{step_fe_pi + 1}"
    else:
        step_num = f"{num_steps}"

    title = f"Expected observation log-likelihood at step {step_num}\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"

    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)

    fig.savefig(
        save_dir
        + "/"
        + f"{env_layout}_{exp_name}_policies_obs_loglik_step{step_num}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_pi_transit_loglik(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    policies_to_vis=[],
):
    """Function to visualize all policy-conditioned free energies across time steps and episodes
    (all on the same axis).

    Inputs:
    - file_data_path (str): path to the .npy file where the data was stored
    - step_fe_pi (int): timestep from which to plot the free energy
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - x_ticks_tstep (int): step for the ticks in the x axis when plotting as a function of total timesteps
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the runs (depending on policy probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - scatter plot of policy-conditioned free energies at EACH time step in the experiment
    - scatter plot of policy-conditioned free energies across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    # num_runs = data["num_runs"]
    # num_policies = data["num_policies"]
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_fe = data["transit_loglik"][selected_runs]
    else:
        pi_fe = data["transit_loglik"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, "Invalid step number."

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Figure with policy-conditioned free energies across episodes
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    if len(policies_to_vis) == 0:
        # Looping over the policies for Figure 2
        for p in range(NUM_POLICIES_VIS):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            std_pi_fe = np.std(
                pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
            )  # .squeeze()
            # Making sure avg_pi_fe has the right dimensions
            # print(avg_pi_fe.shape)
            # print((num_episodes, num_steps))
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = -avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p + POLICY_INDEX_OFFSET])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:
        # Looping over the policies for Figure 2
        for i, p in enumerate(policies_to_vis):

            # Computing the mean (average) and std of one policy's free energies over the runs
            # TODO: handle rare case in which you train only for one episode, in that case squeeze()
            # will raise the exception
            avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
            assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

            # Plotting the free energy at the last time step of every episode for all episodes
            # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
            x2 = np.arange(1, num_episodes + 1)
            y2 = -avg_pi_fe[:, step_fe_pi]

            # int_vals = ",".join(str(int(x)) for x in policies[p])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x2,
                y2,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i + 1}}}$: {policy_action_seq_leg}",
            )

        # Confidence intervals (if needed, uncomment following lines)
        # ax.fill_between(
        #     x2,
        #     y2 - (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     y2 + (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

    # Completing drawing axes for Figure 2
    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Log-probabilities", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])

    if step_fe_pi != -1:
        step_num = f"{step_fe_pi + 1}"
    else:
        step_num = f"{num_steps}"

    title = f"Expected transition log-likelihood at step {step_num}\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)

    fig.savefig(
        save_dir
        + "/"
        + f"{env_layout}_{exp_name}_policies_transit_loglik_step{step_num}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_pi_fes_subplots(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    num_policies_vis,
    policies_to_vis=[],
):
    """Function to visualize all policy-conditioned free energies across episodes for two or more
    experiments on different axes of the same figure.

    Inputs:
    - file_data_path (list): list of paths to the .npy files where experiments' data was stored
    - step_fe_pi (int): timestep from which to plot the free energy
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - x_ticks_tstep (int): step for the ticks in the x axis when plotting as a function of total timesteps
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the runs (depending on policy probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - num_policies_vis (int): number of policies to visualize for each agent
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - scatter plot of policy-conditioned free energies across episodes
    """

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)
    # Determine the number of subplots based on number of data files
    num_datasets = len(file_data_path)
    fig, ax = plt.subplots(1, num_datasets, figsize=(18, 6))

    for num_data, d in enumerate(file_data_path):
        # Select correct policies from the list params["policies_to_vis"]
        curr_policies_to_vis = policies_to_vis[
            num_data * num_policies_vis : (num_data * num_policies_vis)
            + num_policies_vis
        ]

        # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
        data = np.load(d, allow_pickle=True).item()
        exp_name = data["exp_name"]
        num_episodes = data["num_episodes"]
        num_steps = data["num_steps"]

        # Take care of the fact that policies are created and saved differently in the two types of agents
        if "paths" or "au" in exp_name:
            policies = data["policies"]
        elif "plans" or "aa" in exp_name:
            policies = data["ordered_policies"][0, 0, 0, :, :]
        else:
            raise ValueError("exp_name is not an accepted name for the experiment.")

        # Visualize default 16 policies, unless specified differently from CL
        if len(policies_to_vis) == 0:
            # Broadcasting comparison: compare each target with all rows
            matches = (
                policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
            )  # shape: (16, num_rows, 4)
            # Now reduce over last dimension to check full-row match
            row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
            # Each row in row_matches should contain exactly one True
            row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
            policies_to_vis = list(row_indices)

        # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
        # was passed through the command line
        if select_run != -1:

            pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
            selected_runs = (pi_runs > 0.5).nonzero()[0]
            pi_fe = data["pi_free_energies"][selected_runs]
        else:
            pi_fe = data["pi_free_energies"]

        # Checking that the step_fe_pi is within an episode
        assert (
            step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
        ) or step_fe_pi == -1, "Invalid step number."

        # Use global variable if no policy was specified from the command line
        if len(policies_to_vis) == 0:
            # Looping over the policies for Figure 2
            for p in range(NUM_POLICIES_VIS):

                # Computing the mean (average) and std of one policy's free energies over the runs
                # TODO: handle rare case in which you train only for one episode, in that case squeeze()
                # will raise the exception
                avg_pi_fe = np.mean(
                    pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
                )  # .squeeze()
                std_pi_fe = np.std(
                    pi_fe[:, :, p + POLICY_INDEX_OFFSET, :], axis=0
                )  # .squeeze()

                assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

                # Plotting the free energy at a certain time step for each episode
                x2 = np.arange(1, num_episodes + 1)
                y2 = avg_pi_fe[:, step_fe_pi]

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                ax[num_data].plot(
                    x2,
                    y2,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )

        else:
            # Looping over the policies for Figure 2
            for i, p in enumerate(curr_policies_to_vis):
                # Computing the mean (average) and std of one policy's free energies over the runs
                # TODO: handle rare case in which you train only for one episode, in that case squeeze()
                # will raise the exception
                avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
                std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
                assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

                # Plotting the free energy at the last time step of every episode for all episodes
                # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
                x2 = np.arange(1, num_episodes + 1)
                y2 = avg_pi_fe[:, step_fe_pi]

                # int_vals = ",".join(str(int(x)) for x in policies[p])

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                ax[num_data].plot(
                    x2,
                    y2,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
                )

        # Completing drawing axes for Figure 2
        ax[num_data].set_xticks(
            np.arange(x_ticks_estep, num_episodes + 1, step=x_ticks_estep)
        )
        ax[num_data].set_xlabel("Episode")
        ax[num_data].set_ylabel("Free energy", rotation=90)
        # ax[i].set_ylim(0, 5)  # Uncomment for Tmaze3 experiments, comment out for others
        # ax[i].set_ylim(0, 10)  # Uncomment for Tmaze4 experiments, comment out for others
        ax[num_data].set_ylim(y_limits[0], y_limits[1])
        step_num = f"{step_fe_pi + 1}"

        if "hardgs" in exp_name:
            stitle = "(hard with goal shaping)"
            ax[num_data].set_title(
                f"Policy-conditioned free energy at step {step_num} \n" + stitle
            )
        elif "softgs" in exp_name:
            stitle = "(soft with goal shaping)"
            ax[num_data].set_title(
                f"Policy-conditioned free energy at step {step_num} \n" + stitle
            )
        elif "hard" in exp_name:
            stitle = "(hard without goal shaping)"
            ax[num_data].set_title(
                f"Policy-conditioned free energy at step {step_num} \n" + stitle
            )
        elif "soft" in exp_name:
            stitle = "(soft without goal shaping)"
            ax[num_data].set_title(
                f"Policy-conditioned free energy at step {step_num} \n" + stitle
            )
        else:
            stitle = "(preferences)"
            ax[num_data].set_title(
                f"Policy-conditioned free energy at step {step_num} \n" + stitle
            )

        # if "paths" or "au" in exp_name:
        #     ax[num_data].set_title(
        #         f"Policy-conditioned free energy at step {step_num} (action-unaware)\n"
        #     )
        # elif "plans" or "aa" in exp_name:
        #     ax[num_data].set_title(
        #         f"Policy-conditioned free energy at step {step_num} (action-aware)\n"
        #     )

    # Padding
    plt.subplots_adjust(wspace=0.3)
    # Figure title
    # fig.suptitle(f"Policy-conditioned free energy at step {step_num}\n", y=1.1)
    # Create shared legend (both axes plot the SAME policies and in the same order)
    # so we can retrieve and use the handles/labels from one of them
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Policies",
        ncol=4,
        title_fontsize=16,
        handlelength=2,  # shrink the line handle
        columnspacing=1.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
    )

    fig.savefig(
        save_dir + "/" + f"{env_layout}_policies_fe_step{step_num}.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_marginal_fe(
    file_data_path,
    step_fe,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
):
    """Plotting the total free energy averaged over the runs.

    Inputs:

    - file_data_path (string): file path where the total free energy data was stored (i.e. where log_data
      was saved);
    - step_fe (integer): step for which to visualize free energy across episodes
    - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps;
    - y_limits (list): list with lower and upper value for the y axis
    - save_dir (string): directory where to save the images.

    Outputs:

    - scatter plot of F, showing its evolution as a function of the episodes' steps;
    - plot showing how F at the last time step changes as the agent learns about the maze (episode after episode).
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        total_fe = data["total_free_energies"][selected_runs]
    else:
        total_fe = data["total_free_energies"]

    # Computing the mean (average) and std of the total free energies over the runs
    avg_total_fe = np.mean(total_fe, axis=0)  # .squeeze()
    std_total_fe = np.std(total_fe, axis=0)  # .squeeze()
    # Making sure avg_total_fe has the right dimensions
    assert avg_total_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"
    # Taking only the last step total free energy for each episode
    # last_total_fe = avg_total_fe[0,:,-1]
    # last_total_fe = total_fe[1,:,-1]

    # Plotting the total free energy for every time step as a scatter plot
    x1 = np.arange(num_episodes * num_steps)
    y1 = avg_total_fe.flatten()

    plt.figure()
    plt.plot(x1, y1, ".-", label="Total FE")
    plt.xticks(
        np.arange(x_ticks_estep, (num_episodes * num_steps) + 1, step=x_ticks_tstep)
    )
    plt.xlabel("Step")
    plt.ylabel("Total free energy", rotation=90)
    # plt.legend(loc="upper right") # No need for legend
    plt.title("Every-step total free energy across episodes\n")
    # plt.show()
    plt.close()

    # Plotting the total free energy at the last time step of every episode
    # Note 1: another time step can be chosen by changing the index number, i, in avg_total_fe[:, i]
    x2 = np.arange(1, num_episodes + 1)
    y2 = avg_total_fe[:, step_fe]
    if step_fe != -1:
        step_num = f"{step_fe + 1}"
    else:
        step_num = f"{num_steps}"

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.plot(x2, y2, ".-", label="free energy")
    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Free energy", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])
    # Title
    title = f"Free energy at step {step_num}\n"
    title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    ax.set_title(title, pad=15)
    ax.fill_between(
        x2,
        y2 - (1.96 * std_total_fe[:, -1] / np.sqrt(num_runs)),
        y2 + (1.96 * std_total_fe[:, -1] / np.sqrt(num_runs)),
        alpha=0.3,
    )
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_marginal_fe_step{step_num}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
    )
    # plt.show()
    plt.close()


######################################################################################################
##### Functions to plot expected free energies and their component across episodes
######################################################################################################


def plot_pi_fes_efe(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    policies_to_vis=[],
):
    """Function to visualize unnormalized policy probabilities (policy-conditioned FE + EFE) at the first
    step across episodes

    Inputs:
    - file_data_path (str): path to the .npy file where the data was stored
    - step_fe_pi (int): timestep from which to plot the free energy
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - x_ticks_tstep (int): step for the ticks in the x axis when plotting as a function of total timesteps
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the runs (depending on policy probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - scatter plot of policy-conditioned free energies at EACH time step in the experiment
    - scatter plot of policy-conditioned free energies across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    # num_runs = data["num_runs"]
    # num_policies = data["num_policies"]
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_fe = data["pi_free_energies"][selected_runs]
        efe = data["expected_free_energies"][selected_runs]
    else:
        pi_fe = data["pi_free_energies"]
        efe = data["expected_free_energies"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, f"Invalid step number: {step_fe_pi}"

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Figure with policy-conditioned free energies at EACH timestep in the experiment
    fig, ax = plt.subplots(figsize=(5, 4))
    # Looping over the policies for Figure 1
    for p in range(NUM_POLICIES_VIS):

        # Computing the mean (average) and std of one policy's free energies over the runs
        # TODO: handle rare case in which you train only for one episode, in that case squeeze()
        # will raise the exception
        avg_pi_fe = np.mean(pi_fe[:, :, p, step_fe_pi], axis=0)  # .squeeze()
        std_pi_fe = np.std(pi_fe[:, :, p, step_fe_pi], axis=0)  # .squeeze()
        # Averaging the expected free energies over the runs
        avg_efe = np.mean(efe[:, :, p, step_fe_pi], axis=0)  # .squeeze()

        # Plotting the free energy for every time step
        x1 = np.arange(num_episodes)
        y1 = (avg_pi_fe + avg_efe).flatten()

        # ax.plot(x1, y1, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")
        ax.plot(x1, y1, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")

    # Completing drawing axes for Figure 1
    ax.set_xticks(np.arange(x_ticks_estep, num_episodes + 1, step=x_ticks_tstep))
    ax.set_xlabel("Step")
    ax.set_ylabel("FE + EFE (for each policy)", rotation=90)
    # ax.legend(
    #     ncol=4,
    #     loc="upper center",
    #     title_fontsize=16,
    #     bbox_to_anchor=(0.5, -0.2),
    #     fancybox=True,
    # )
    ax.set_title("Unnormalized policy probabilities at each step across episodes\n")

    # Save figures and show
    fig.savefig(
        save_dir
        + "/"
        + f"{env_layout}_{exp_name}_unorm_policies_probs_step{step_fe_pi}.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_efe(
    file_data_path,
    select_policy,
    save_dir,
    env_layout,
    x_ticks_estep,
    y_limits,
    select_step=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy (EFE) for each policy at ALL timesteps or a subset
    during the experiment.

    Inputs:
    - file_data_path (str): file path where data was stored
    - select_policy (int): index to include/exclude some policies
    - select_step (int): index to select same step at each episode to plot corresponding EFE value;
    - save_dir (string): directory where to save the figures

    Outputs:
    - plot showing evolution of EFE throughout the experiment
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        efe = data["expected_free_energies"][selected_runs]
    else:
        efe = data["expected_free_energies"]

    # Averaging the expected free energies over the runs
    avg_efe = np.mean(efe, axis=0)  # .squeeze()
    # Making sure efe has the right dimensions
    assert avg_efe.shape == (num_episodes, num_policies, num_steps), "Wrong dimenions!"

    plt.figure(figsize=(5, 4), tight_layout=True)
    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    if select_step == None:
        x_label = "Step"
        title = "Expected free energy at every step\n"
        # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
        if "hardgs" in exp_name:
            title += "(hard with goal shaping)"
        elif "softgs" in exp_name:
            title += "(soft with goal shaping)"
        elif "hard" in exp_name:
            title += "(hard without goal shaping)"
        elif "soft" in exp_name:
            title += "(soft without goal shaping)"
        else:
            title += "(preferences)"

        # Plotting all the time steps across episodes
        for p in range(num_policies):
            x = np.arange(num_episodes * num_steps)
            # x = np.arange(1*(num_steps-1))
            y = avg_efe[:, p, :].flatten()
            # y = np.reshape(-avg_efe[2, p, 0:-1], (1*(num_steps-1)))
            plt.plot(x, y, ".-", color=cmap(p), label=f"Policy $\\pi_{{{p}}}$")

    else:
        x_label = "Episode"
        title = f"Expected free energy at step {select_step + 1}\n"
        # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
        if "hardgs" in exp_name:
            title += "(hard with goal shaping)"
        elif "softgs" in exp_name:
            title += "(soft with goal shaping)"
        elif "hard" in exp_name:
            title += "(hard without goal shaping)"
        elif "soft" in exp_name:
            title += "(soft without goal shaping)"
        else:
            title += "(preferences)"

        if len(policies_to_vis) == 0:
            # Plotting EFE at single time step for each episode
            for p in range(NUM_POLICIES_VIS):
                x = np.arange(1, num_episodes + 1)
                y = avg_efe[:, p, select_step].flatten()
                # int_vals = ",".join(
                #     str(int(x)) for x in policies[p + POLICY_INDEX_OFFSET]
                # )

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                plt.plot(
                    x,
                    y,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )
        else:
            # Plotting EFE at single time step for each episode
            for i, p in enumerate(policies_to_vis):
                x = np.arange(1, num_episodes + 1)
                y = avg_efe[:, p, select_step].flatten()
                # int_vals = ",".join(str(int(x)) for x in policies[p])

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                plt.plot(
                    x,
                    y,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
                )

    # Set label for x axis
    plt.xlabel(f"{x_label}")
    plt.xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    # Set label and range for y axis
    plt.ylabel("Expected free energy", rotation=90)
    plt.ylim(y_limits[0], y_limits[1])
    # Set title for the figure
    plt.title(f"{title}", pad=15)
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_efe_step{select_step}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
    )
    # plt.show()
    plt.close()


def plot_efe_subplots(
    file_data_path,
    x_ticks_estep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    num_policies_vis,
    select_step=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy (EFE) for each policy at a certain timestep of each episode
    for two or more experiments on different axes of the same figure.

    Inputs:
    - file_data_path (str): file path where data was stored
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to select data from only a subset of the policies (depending on tjeir probs)
    - save_dir (str): string with the directory path where to save the figure
    - env_layout (str): layout of the training evironment (e.g., Tmaze3)
    - num_policies_vis (int): number of policies to visualize for each agent
    - select_step (int): index to select same step at each episode to plot corresponding EFE value;
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - plot showing evolution of EFE throughout the experiment
    """
    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)
    # Determine the number of subplots based on number of data files
    num_datasets = len(file_data_path)
    fig, ax = plt.subplots(1, num_datasets, figsize=(18, 6))

    for num_data, d in enumerate(file_data_path):
        # Select correct policies from the list params["policies_to_vis"]
        curr_policies_to_vis = policies_to_vis[
            num_data * num_policies_vis : (num_data * num_policies_vis)
            + num_policies_vis
        ]

        # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
        data = np.load(d, allow_pickle=True).item()
        exp_name = data["exp_name"]
        num_episodes = data["num_episodes"]
        num_policies = data["num_policies"]
        num_steps = data["num_steps"]

        # Take care of the fact that policies are created and saved differently in the two types of agents
        if "paths" or "au" in exp_name:
            policies = data["policies"]
        elif "plans" or "aa" in exp_name:
            policies = data["ordered_policies"][0, 0, 0, :, :]
        else:
            raise ValueError("exp_name is not an accepted name for the experiment.")

        # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
        # was passed through the command line
        if select_run != -1:
            pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
            selected_runs = (pi_runs > 0.5).nonzero()[0]
            efe = data["expected_free_energies"][selected_runs]
        else:
            efe = data["expected_free_energies"]

        # Averaging the expected free energies over the runs
        avg_efe = np.mean(efe, axis=0)  # .squeeze()
        # Making sure efe has the right dimensions
        assert avg_efe.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"

        if len(policies_to_vis) == 0:
            # Plotting EFE at single time step for each episode
            for p in range(NUM_POLICIES_VIS):
                x = np.arange(1, num_episodes + 1)
                y = avg_efe[:, p, select_step].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                ax[num_data].plot(
                    x,
                    y,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )
        else:
            # Plotting EFE at single time step for each episode
            for i, p in enumerate(curr_policies_to_vis):
                x = np.arange(1, num_episodes + 1)
                y = avg_efe[:, p, select_step].flatten()
                # int_vals = ",".join(str(int(x)) for x in policies[p])

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                ax[num_data].plot(
                    x,
                    y,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
                )

        ax[num_data].set_xticks(
            np.arange(x_ticks_estep, num_episodes + 1, step=x_ticks_estep)
        )
        ax[num_data].set_xlabel("Episode")
        ax[num_data].set_ylabel("Expected free energy", rotation=90)
        # plt.ylim(2, 6)  # Uncomment for Tmaze3 experiments, comment out for others
        # plt.ylim(3, 11)  # Uncomment for Tmaze4 experiments, comment out for others
        ax[num_data].set_ylim(y_limits[0], y_limits[1])

        if "paths" or "au" in exp_name:
            ax[num_data].set_title(
                f"Expected free energy at step {select_step + 1} (action-unaware)\n"
            )
        elif "plans" or "aa" in exp_name:
            ax[num_data].set_title(
                f"Expected free energy at step {select_step + 1} (action-aware)\n"
            )

    # Figure title
    # plot_title = f"Expected free energy at step {select_step + 1}"
    # fig.suptitle(f"{plot_title}\n", y=1.1)
    # Create shared legend (both axes plot the SAME policies and in the same order)
    # so we can retrieve and use the handles/labels from one of them
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Policies",
        ncol=4,
        title_fontsize=16,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
    )
    # Save figure and show
    fig.savefig(
        save_dir + "/" + f"{env_layout}_efe_step{select_step}.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_efe_risk(
    file_data_path,
    select_run,
    save_dir,
    env_layout,
    x_ticks_estep,
    y_lims,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component RISK for all policies of an agent (averaged over
    all the runs/agents) for one experiment.

    Inputs:
    - file_data_path (str): file path to the data file
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plot showing the evolution of RISK across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:
        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        num_runs = len(selected_runs)
        efe = data["expected_free_energies"][selected_runs]
        efe_risk = data["efe_risk"][selected_runs]
    else:
        efe = data["expected_free_energies"]
        efe_risk = data["efe_risk"]

    # Averaging the expected free energies and the RISK component over the runs
    avg_efe = np.mean(efe, axis=0)  # .squeeze()
    avg_efe_risk = np.mean(efe_risk, axis=0)  # .squeeze()
    std_efe_risk = np.std(efe_risk, axis=0)  # .squeeze()

    assert avg_efe.shape == (num_episodes, num_policies, num_steps), "Wrong dimenions!"
    assert avg_efe_risk.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Plotting risk and B-novelty in separate figures

    # Figure for RISK
    fig_1, axes_1 = plt.subplots(
        1,
        1,
        figsize=(5, 4),
        tight_layout=True,
        # gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )  # 1 row, 1 column

    x_label = ""
    title_label = ""

    if len(policies_to_vis) == 0:

        for p in range(NUM_POLICIES_VIS):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps + 1}"
                x = np.arange(1, num_episodes + 1)
                # Risk
                y_efer = avg_efe_risk[:, p, num_tsteps].flatten()
                stdy_efer = std_efe_risk[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # Risk
                y_efer = avg_efe_risk[:, p, :].flatten()
                stdy_efer = std_efe_risk[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            # Plot on figures
            axes_1.plot(
                x,
                y_efer,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )
    else:
        for i, p in enumerate(policies_to_vis):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps + 1}"
                x = np.arange(1, num_episodes + 1)
                # Risk
                y_efer = avg_efe_risk[:, p, num_tsteps].flatten()
                stdy_efer = std_efe_risk[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # Risk
                y_efer = avg_efe_risk[:, p, :].flatten()
                stdy_efer = std_efe_risk[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            # Plot on figures
            axes_1.plot(
                x,
                y_efer,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
            )
    axes_1.set_xlabel(x_label)
    axes_1.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    axes_1.set_ylabel("Risk", rotation=90)
    axes_1.set_ylim(y_lims[0], y_lims[1])

    # title_label += " (action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title_label += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title_label += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title_label += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title_label += "(soft without goal shaping)"
    else:
        title_label += "(preferences)"

    axes_1.set_title(f"Risk at {title_label}", pad=15)
    # Save figures
    fig_1.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_efe_risk.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
    )
    # plt.show()
    plt.close(fig_1)


def plot_efe_bnov(
    file_data_path,
    select_run,
    save_dir,
    env_layout,
    x_ticks_estep,
    y_lims,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component B-NOVELTY for all policies of an agent (averaged
    over all the runs/agents) for one experiment.

    Inputs:
    - file_data_path (str): file path to the data file
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plot showing the evolution of B-NOVELTY across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        num_runs = len(selected_runs)
        efe = data["expected_free_energies"][selected_runs]
        efe_Bnovelty = data["efe_Bnovelty"][selected_runs]
    else:
        efe = data["expected_free_energies"]
        efe_Bnovelty = data["efe_Bnovelty"]

    # Averaging the expected free energies component B-NOVELTY over the runs
    avg_efe = np.mean(efe, axis=0)  # .squeeze()
    avg_efe_Bnovelty = np.mean(efe_Bnovelty, axis=0)  # .squeeze()
    std_efe_Bnovelty = np.std(efe_Bnovelty, axis=0)  # .squeeze()

    assert avg_efe.shape == (num_episodes, num_policies, num_steps), "Wrong dimenions!"
    assert avg_efe_Bnovelty.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    # Figure for B-novelty
    fig_2, axes_2 = plt.subplots(
        1,
        1,
        figsize=(5, 4),
        tight_layout=True,
        # gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )  # 1 row, 1 column

    x_label = ""
    title_label = ""

    if len(policies_to_vis) == 0:
        for p in range(NUM_POLICIES_VIS):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps + 1}"
                x = np.arange(1, num_episodes + 1)
                # B-novelty
                y_efeB = avg_efe_Bnovelty[:, p, num_tsteps].flatten()
                stdy_efeB = std_efe_Bnovelty[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # B-novelty
                y_efeB = avg_efe_Bnovelty[:, p, :].flatten()
                stdy_efeB = std_efe_Bnovelty[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            axes_2.plot(
                x,
                y_efeB,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:
        for i, p in enumerate(policies_to_vis):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps + 1}"
                x = np.arange(1, num_episodes + 1)
                # B-novelty
                y_efeB = avg_efe_Bnovelty[:, p, num_tsteps].flatten()
                stdy_efeB = std_efe_Bnovelty[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # B-novelty
                y_efeB = avg_efe_Bnovelty[:, p, :].flatten()
                stdy_efeB = std_efe_Bnovelty[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            axes_2.plot(
                x,
                y_efeB,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
            )

    axes_2.set_xlabel(x_label)
    axes_2.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )

    axes_2.set_ylabel("B-novelty", rotation=90)
    axes_2.set_ylim(y_lims[0], y_lims[1])

    # title_label += " (action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title_label += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title_label += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title_label += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title_label += "(soft without goal shaping)"
    else:
        title_label += "(preferences)"

    axes_2.set_title(f"B-novelty at {title_label}", pad=15)
    # Save figure
    fig_2.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_efe_bnov.pdf",
        format="pdf",
        bbox_inches=None,
        dpi=200,
    )
    # plt.show()
    plt.close(fig_2)


def plot_efe_risk_subplots(
    file_data_path,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    num_policies_vis,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component RISK for all policies of an agent (averaged over
    all the runs/agents) for two or more experiments on different axes of the same figure.

    Inputs:
    - file_data_path (str): file path to the data file
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_policies_vis (int): number of policies to visualize for each agent
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plots showing the evolution of RISK across episodes for two or more experiments
    """

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)
    # Determine the number of subplots based on number of data files
    num_datasets = len(file_data_path)
    # Figure for RISK
    fig, axes = plt.subplots(
        1,
        num_datasets,
        figsize=(18, 6),
        # gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )

    for num_data, d in enumerate(file_data_path):
        # Select correct policies from the list params["policies_to_vis"]
        curr_policies_to_vis = policies_to_vis[
            num_data * num_policies_vis : (num_data * num_policies_vis)
            + num_policies_vis
        ]

        # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
        data = np.load(d, allow_pickle=True).item()
        exp_name = data["exp_name"]
        num_runs = data["num_runs"]
        num_episodes = data["num_episodes"]
        num_policies = data["num_policies"]
        num_steps = data["num_steps"]

        # Take care of the fact that policies are created and saved differently in the two types of agents
        if "paths" or "au" in exp_name:
            policies = data["policies"]
        elif "plans" or "aa" in exp_name:
            policies = data["ordered_policies"][0, 0, 0, :, :]
        else:
            raise ValueError("exp_name is not an accepted name for the experiment.")

        # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
        # was passed through the command line
        if select_run != -1:
            pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
            selected_runs = (pi_runs > 0.5).nonzero()[0]
            num_runs = len(selected_runs)
            efe = data["expected_free_energies"][selected_runs]
            efe_risk = data["efe_risk"][selected_runs]
        else:
            efe = data["expected_free_energies"]
            efe_risk = data["efe_risk"]

        # Averaging the expected free energies and the RISK component over the runs
        avg_efe = np.mean(efe, axis=0)  # .squeeze()
        avg_efe_risk = np.mean(efe_risk, axis=0)  # .squeeze()
        std_efe_risk = np.std(efe_risk, axis=0)  # .squeeze()

        assert avg_efe.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"
        assert avg_efe_risk.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"

        if len(policies_to_vis) == 0:

            for p in range(NUM_POLICIES_VIS):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps + 1}"
                    x = np.arange(1, num_episodes + 1)
                    # Risk
                    y_efer = avg_efe_risk[:, p, num_tsteps].flatten()
                    stdy_efer = std_efe_risk[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # Risk
                    y_efer = avg_efe_risk[:, p, :].flatten()
                    stdy_efer = std_efe_risk[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efer,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )
        else:
            for i, p in enumerate(curr_policies_to_vis):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps + 1}"
                    x = np.arange(1, num_episodes + 1)
                    # Risk
                    y_efer = avg_efe_risk[:, p, num_tsteps].flatten()
                    stdy_efer = std_efe_risk[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # Risk
                    y_efer = avg_efe_risk[:, p, :].flatten()
                    stdy_efer = std_efe_risk[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efer,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
                )
        axes[num_data].set_xlabel(x_label)
        axes[num_data].set_ylabel("Risk", rotation=90)
        # axes_1.set_ylim(2, 6)  # Uncomment for Tmaze3 experiments, comment out for others
        # axes_1.set_ylim(4, 11)  # Uncomment for Tmaze4 experiments, comment out for others
        axes[num_data].set_ylim(y_limits[0], y_limits[1])

        if "paths" or "au" in exp_name:
            axes[num_data].set_title(f"Risk at {title_label} (action-unaware)\n")
        elif "plans" or "aa" in exp_name:
            axes[num_data].set_title(f"Risk at {title_label} (action-aware)\n")

    # Figure title
    # plot_title = f"Risk at {title_label}\n"
    # fig.suptitle(f"{plot_title}\n", y=1.1)
    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Put the legend at the bottom center
    fig.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )
    # Save figures
    fig.savefig(
        save_dir + "/" + f"{env_layout}_efe_risk.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig)


def plot_efe_bnov_subplots(
    file_data_path,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    num_policies_vis,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component B-NOVELTY for all policies of an agent (averaged over
    all the runs/agents) for two or more experiments on different axes of the same figure.

    Inputs:
    - file_data_path (str): file path to the data file
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_policies_vis (int): number of policies to visualize for each agent
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plots showing the evolution of B-NOVELTY across episodes for two or more experiments
    """

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)
    # Determine the number of subplots based on number of data files
    num_datasets = len(file_data_path)
    # Figure for RISK
    fig, axes = plt.subplots(
        1,
        num_datasets,
        figsize=(18, 6),
        # gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )

    for num_data, d in enumerate(file_data_path):
        # Select correct policies from the list params["policies_to_vis"]
        curr_policies_to_vis = policies_to_vis[
            num_data * num_policies_vis : (num_data * num_policies_vis)
            + num_policies_vis
        ]

        # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
        data = np.load(d, allow_pickle=True).item()
        exp_name = data["exp_name"]
        num_runs = data["num_runs"]
        num_episodes = data["num_episodes"]
        num_policies = data["num_policies"]
        num_steps = data["num_steps"]

        # Take care of the fact that policies are created and saved differently in the two types of agents
        if "paths" or "au" in exp_name:
            policies = data["policies"]
        elif "plans" or "aa" in exp_name:
            policies = data["ordered_policies"][0, 0, 0, :, :]
        else:
            raise ValueError("exp_name is not an accepted name for the experiment.")

        # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
        # was passed through the command line
        if select_run != -1:
            pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
            selected_runs = (pi_runs > 0.5).nonzero()[0]
            num_runs = len(selected_runs)
            efe = data["expected_free_energies"][selected_runs]
            efe_Bnovelty = data["efe_Bnovelty"][selected_runs]
        else:
            efe = data["expected_free_energies"]
            efe_Bnovelty = data["efe_Bnovelty"]

        # Averaging the expected free energies and the RISK component over the runs
        avg_efe = np.mean(efe, axis=0)  # .squeeze()
        avg_efe_Bnovelty = np.mean(efe_Bnovelty, axis=0)  # .squeeze()
        std_efe_Bnovelty = np.std(efe_Bnovelty, axis=0)  # .squeeze()

        assert avg_efe.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"
        assert avg_efe_Bnovelty.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"

        if len(policies_to_vis) == 0:

            for p in range(NUM_POLICIES_VIS):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps + 1}"
                    x = np.arange(1, num_episodes + 1)
                    # Risk
                    y_efeB = avg_efe_Bnovelty[:, p, num_tsteps].flatten()
                    stdy_efeB = std_efe_Bnovelty[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # Risk
                    y_efeB = avg_efe_Bnovelty[:, p, :].flatten()
                    stdy_efeB = std_efe_Bnovelty[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efeB,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )
        else:
            for i, p in enumerate(curr_policies_to_vis):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps + 1}"
                    x = np.arange(1, num_episodes + 1)
                    # Risk
                    y_efeB = avg_efe_Bnovelty[:, p, num_tsteps].flatten()
                    stdy_efeB = std_efe_Bnovelty[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # Risk
                    y_efeB = avg_efe_Bnovelty[:, p, :].flatten()
                    stdy_efeB = std_efe_Bnovelty[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efeB,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
                )
        axes[num_data].set_xlabel(x_label)
        axes[num_data].set_ylabel("B-novelty", rotation=90)
        # axes_1.set_ylim(2, 6)  # Uncomment for Tmaze3 experiments, comment out for others
        # axes_1.set_ylim(4, 11)  # Uncomment for Tmaze4 experiments, comment out for others
        axes[num_data].set_ylim(y_limits[0], y_limits[1])

        if "paths" or "au" in exp_name:
            axes[num_data].set_title(f"B-novelty at {title_label} (action-unaware)\n")
        elif "plans" or "aa" in exp_name:
            axes[num_data].set_title(f"B-novelty at {title_label} (action-aware)\n")

    # Figure title
    # plot_title = f"B-novelty at {title_label}\n"
    # fig.suptitle(f"{plot_title}\n", y=1.1)
    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Put the legend at the bottom center
    fig.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )
    # Save figures
    fig.savefig(
        save_dir + "/" + f"{env_layout}_efe_bnov.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig)


def plot_efe_ambiguity(
    file_data_path,
    select_run,
    save_dir,
    env_layout,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component AMBIGUITY for all policies of an agent (averaged over
    all the runs/agents) for one experiment.

    Inputs:
    - file_data_path (str): file path to the data file
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plot showing the evolution of AMBIGUITY across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:
        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        num_runs = len(selected_runs)
        efe = data["expected_free_energies"][selected_runs]
        efe_ambiguity = data["efe_ambiguity"][selected_runs]
    else:
        efe = data["expected_free_energies"]
        efe_ambiguity = data["efe_ambiguity"]

    # Averaging the expected free energies and the RISK component over the runs
    avg_efe = np.mean(efe, axis=0)  # .squeeze()
    avg_efe_ambiguity = np.mean(efe_ambiguity, axis=0)  # .squeeze()
    std_efe_ambiguity = np.std(efe_ambiguity, axis=0)  # .squeeze()

    assert avg_efe.shape == (num_episodes, num_policies, num_steps), "Wrong dimenions!"
    assert avg_efe_ambiguity.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    ### Plotting risk and B-novelty in separate figures

    # Figure for RISK
    fig_1, axes_1 = plt.subplots(
        1,
        1,
        figsize=(6, 5),
        gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )  # 1 row, 1 column

    if len(policies_to_vis) == 0:

        for p in range(NUM_POLICIES_VIS):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps + 1}"
                x = np.arange(1, num_episodes + 1)
                # Risk
                y_efea = avg_efe_ambiguity[:, p, num_tsteps].flatten()
                stdy_efea = std_efe_ambiguity[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # Risk
                y_efea = avg_efe_ambiguity[:, p, :].flatten()
                stdy_efea = std_efe_ambiguity[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            # Plot on figures
            axes_1.plot(
                x,
                y_efea,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )
    else:
        for i, p in enumerate(policies_to_vis):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps + 1}"
                x = np.arange(1, num_episodes + 1)
                # Risk
                y_efea = avg_efe_ambiguity[:, p, num_tsteps].flatten()
                stdy_efea = std_efe_ambiguity[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # Risk
                y_efea = avg_efe_ambiguity[:, p, :].flatten()
                stdy_efea = std_efe_ambiguity[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            # Plot on figures
            axes_1.plot(
                x,
                y_efea,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
            )
    axes_1.set_xlabel(x_label)
    axes_1.set_ylabel("Ambiguity", rotation=90)
    # axes_1.set_ylim(2, 6)  # Uncomment for Tmaze3 experiments, comment out for others
    # axes_1.set_ylim(4, 11)  # Uncomment for Tmaze4 experiments, comment out for others

    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes_1.get_legend_handles_labels()
    # Put the legend at the bottom center
    fig_1.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )

    axes_1.set_title(f"Ambiguity at {title_label}\n")
    # Save figures
    fig_1.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_efe_ambiguity.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig_1)


def plot_efe_anov(
    file_data_path,
    select_run,
    save_dir,
    env_layout,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component A-NOVELTY for all policies of an agent (averaged
    over all the runs/agents) for one experiment.

    Inputs:
    - file_data_path (str): file path to the data file
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plot showing the evolution of A-NOVELTY across episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        num_runs = len(selected_runs)
        efe = data["expected_free_energies"][selected_runs]
        efe_Anovelty = data["efe_Anovelty"][selected_runs]
    else:
        efe = data["expected_free_energies"]
        efe_Anovelty = data["efe_Anovelty"]

    # Averaging the expected free energies component B-NOVELTY over the runs
    avg_efe = np.mean(efe, axis=0)  # .squeeze()
    avg_efe_Anovelty = np.mean(efe_Anovelty, axis=0)  # .squeeze()
    std_efe_Anovelty = np.std(efe_Anovelty, axis=0)  # .squeeze()

    assert avg_efe.shape == (num_episodes, num_policies, num_steps), "Wrong dimenions!"
    assert avg_efe_Anovelty.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    # Figure for B-novelty
    fig_2, axes_2 = plt.subplots(
        1,
        1,
        figsize=(6, 5),
        gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )  # 1 row, 1 column

    if len(policies_to_vis) == 0:
        for p in range(NUM_POLICIES_VIS):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps}"
                x = np.arange(1, num_episodes + 1)
                # A-novelty
                y_efeA = avg_efe_Anovelty[:, p, num_tsteps].flatten()
                stdy_efeA = std_efe_Anovelty[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # A-novelty
                y_efeA = avg_efe_Anovelty[:, p, :].flatten()
                stdy_efeA = std_efe_Anovelty[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            axes_2.plot(
                x,
                y_efeA,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:
        for i, p in enumerate(policies_to_vis):
            # Plotting all time steps unless a specific time step is provided
            if num_tsteps != None:
                x_label = "Episode"
                title_label = f"step {num_tsteps}"
                x = np.arange(1, num_episodes + 1)
                # B-novelty
                y_efeA = avg_efe_Anovelty[:, p, num_tsteps].flatten()
                stdy_efeA = std_efe_Anovelty[:, p, num_tsteps].flatten()
            else:
                x_label = "Step"
                title_label = "all steps"
                x = np.arange(num_episodes * num_steps)
                # B-novelty
                y_efeA = avg_efe_Anovelty[:, p, :].flatten()
                stdy_efeA = std_efe_Anovelty[:, p, :].flatten()

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            axes_2.plot(
                x,
                y_efeA,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
            )

    axes_2.set_xlabel(x_label)
    axes_2.set_ylabel("A-novelty", rotation=90)
    # axes_2.set_ylim(0.1, 0.5)  # Uncomment for Tmaze3 experiments, comment out for others
    # axes_2.set_ylim(0, 0.9)  # Uncomment for Tmaze4 experiments, comment out for others

    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes_2.get_legend_handles_labels()
    # Put the legend at the bottom center
    fig_2.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )

    axes_2.set_title(f"A-novelty at {title_label}\n")
    # Save figure
    fig_2.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_efe_anov.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig_2)


def plot_efe_ambiguity_subplots(
    file_data_path,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    num_policies_vis,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component AMBIGUITY for all policies of an agent (averaged over
    all the runs/agents) for two or more experiments on different axes of the same figure.

    Inputs:
    - file_data_path (str): file path to the data file
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_policies_vis (int): number of policies to visualize for each agent
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plots showing the evolution of AMBIGUITY across episodes for two or more experiments
    """

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)
    # Determine the number of subplots based on number of data files
    num_datasets = len(file_data_path)
    # Figure for RISK
    fig, axes = plt.subplots(
        1,
        num_datasets,
        figsize=(18, 6),
        # gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )

    for num_data, d in enumerate(file_data_path):
        # Select correct policies from the list params["policies_to_vis"]
        curr_policies_to_vis = policies_to_vis[
            num_data * num_policies_vis : (num_data * num_policies_vis)
            + num_policies_vis
        ]

        # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
        data = np.load(d, allow_pickle=True).item()
        exp_name = data["exp_name"]
        num_runs = data["num_runs"]
        num_episodes = data["num_episodes"]
        num_policies = data["num_policies"]
        num_steps = data["num_steps"]

        # Take care of the fact that policies are created and saved differently in the two types of agents
        if "paths" or "au" in exp_name:
            policies = data["policies"]
        elif "plans" or "aa" in exp_name:
            policies = data["ordered_policies"][0, 0, 0, :, :]
        else:
            raise ValueError("exp_name is not an accepted name for the experiment.")

        # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
        # was passed through the command line
        if select_run != -1:
            pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
            selected_runs = (pi_runs > 0.5).nonzero()[0]
            num_runs = len(selected_runs)
            efe = data["expected_free_energies"][selected_runs]
            efe_ambiguity = data["efe_ambiguity"][selected_runs]
        else:
            efe = data["expected_free_energies"]
            efe_ambiguity = data["efe_ambiguity"]

        # Averaging the expected free energies and the RISK component over the runs
        avg_efe = np.mean(efe, axis=0)  # .squeeze()
        avg_efe_ambiguity = np.mean(efe_ambiguity, axis=0)  # .squeeze()
        std_efe_ambiguity = np.std(efe_ambiguity, axis=0)  # .squeeze()

        assert avg_efe.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"
        assert avg_efe_ambiguity.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"

        if len(policies_to_vis) == 0:

            for p in range(NUM_POLICIES_VIS):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps}"
                    x = np.arange(1, num_episodes + 1)
                    # Ambiguity
                    y_efea = avg_efe_ambiguity[:, p, num_tsteps].flatten()
                    stdy_efea = std_efe_ambiguity[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # Ambiguity
                    y_efea = avg_efe_ambiguity[:, p, :].flatten()
                    stdy_efea = std_efe_ambiguity[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efea,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )
        else:
            for i, p in enumerate(curr_policies_to_vis):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps}"
                    x = np.arange(1, num_episodes + 1)
                    # Ambiguity
                    y_efea = avg_efe_ambiguity[:, p, num_tsteps].flatten()
                    stdy_efea = std_efe_ambiguity[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # Ambiguity
                    y_efea = avg_efe_ambiguity[:, p, :].flatten()
                    stdy_efea = std_efe_ambiguity[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efea,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
                )
        axes[num_data].set_xlabel(x_label)
        axes[num_data].set_ylabel("Ambiguity", rotation=90)
        # axes_1.set_ylim(2, 6)  # Uncomment for Tmaze3 experiments, comment out for others
        # axes_1.set_ylim(4, 11)  # Uncomment for Tmaze4 experiments, comment out for others
        axes[num_data].set_ylim(y_limits[0], y_limits[1])

        if "paths" or "au" in exp_name:
            axes[num_data].set_title(f"Policy-as-path agent\n")
        elif "plans" or "aa" in exp_name:
            axes[num_data].set_title(f"Policy-as-plan agent\n")

    # Figure title
    plot_title = f"Ambiguity at {title_label}\n"
    fig.suptitle(f"{plot_title}\n", y=1.1)
    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Put the legend at the bottom center
    fig.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )
    # Save figures
    fig.savefig(
        save_dir + "/" + f"{env_layout}_efe_ambiguity.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig)


def plot_efe_anov_subplots(
    file_data_path,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    num_policies_vis,
    num_tsteps=None,
    policies_to_vis=[],
):
    """Function to plot the expected free energy component A-NOVELTY for all policies of an agent (averaged over
    all the runs/agents) for two or more experiments on different axes of the same figure.

    Inputs:
    - file_data_path (str): file path to the data file
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_policies_vis (int): number of policies to visualize for each agent
    - num_tsteps (int): timestep for which to plot the EFE component
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plots showing the evolution of A-NOVELTY across episodes for two or more experiments
    """

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)
    # Determine the number of subplots based on number of data files
    num_datasets = len(file_data_path)
    # Figure for RISK
    fig, axes = plt.subplots(
        1,
        num_datasets,
        figsize=(18, 6),
        # gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )

    for num_data, d in enumerate(file_data_path):
        # Select correct policies from the list params["policies_to_vis"]
        curr_policies_to_vis = policies_to_vis[
            num_data * num_policies_vis : (num_data * num_policies_vis)
            + num_policies_vis
        ]

        # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
        data = np.load(d, allow_pickle=True).item()
        exp_name = data["exp_name"]
        num_runs = data["num_runs"]
        num_episodes = data["num_episodes"]
        num_policies = data["num_policies"]
        num_steps = data["num_steps"]

        # Take care of the fact that policies are created and saved differently in the two types of agents
        if "paths" or "au" in exp_name:
            policies = data["policies"]
        elif "plans" or "aa" in exp_name:
            policies = data["ordered_policies"][0, 0, 0, :, :]
        else:
            raise ValueError("exp_name is not an accepted name for the experiment.")

        # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
        # was passed through the command line
        if select_run != -1:
            pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
            selected_runs = (pi_runs > 0.5).nonzero()[0]
            num_runs = len(selected_runs)
            efe = data["expected_free_energies"][selected_runs]
            efe_Anovelty = data["efe_Anovelty"][selected_runs]
        else:
            efe = data["expected_free_energies"]
            efe_Anovelty = data["efe_Anovelty"]

        # Averaging the expected free energies and the RISK component over the runs
        avg_efe = np.mean(efe, axis=0)  # .squeeze()
        avg_efe_Anovelty = np.mean(efe_Anovelty, axis=0)  # .squeeze()
        std_efe_Anovelty = np.std(efe_Anovelty, axis=0)  # .squeeze()

        assert avg_efe.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"
        assert avg_efe_Anovelty.shape == (
            num_episodes,
            num_policies,
            num_steps,
        ), "Wrong dimenions!"

        if len(policies_to_vis) == 0:

            for p in range(NUM_POLICIES_VIS):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps}"
                    x = np.arange(1, num_episodes + 1)
                    # A-novelty
                    y_efeA = avg_efe_Anovelty[:, p, num_tsteps].flatten()
                    stdy_efeA = std_efe_Anovelty[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # A-novelty
                    y_efeA = avg_efe_Anovelty[:, p, :].flatten()
                    stdy_efeA = std_efe_Anovelty[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efeA,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )
        else:
            for i, p in enumerate(curr_policies_to_vis):
                # Plotting all time steps unless a specific time step is provided
                if num_tsteps != None:
                    x_label = "Episode"
                    title_label = f"step {num_tsteps}"
                    x = np.arange(1, num_episodes + 1)
                    # A-novelty
                    y_efeA = avg_efe_Anovelty[:, p, num_tsteps].flatten()
                    stdy_efeA = std_efe_Anovelty[:, p, num_tsteps].flatten()
                else:
                    x_label = "Step"
                    title_label = "all steps"
                    x = np.arange(num_episodes * num_steps)
                    # A-novelty
                    y_efeA = avg_efe_Anovelty[:, p, :].flatten()
                    stdy_efeA = std_efe_Anovelty[:, p, :].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                # Plot on figures
                axes[num_data].plot(
                    x,
                    y_efeA,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
                )
        axes[num_data].set_xlabel(x_label)
        axes[num_data].set_ylabel("A-novelty", rotation=90)
        # axes_1.set_ylim(2, 6)  # Uncomment for Tmaze3 experiments, comment out for others
        # axes_1.set_ylim(4, 11)  # Uncomment for Tmaze4 experiments, comment out for others
        axes[num_data].set_ylim(y_limits[0], y_limits[1])

        if "paths" or "au" in exp_name:
            axes[num_data].set_title(f"Policy-as-path agent\n")
        elif "plans" or "aa" in exp_name:
            axes[num_data].set_title(f"Policy-as-plan agent\n")

    # Figure title
    plot_title = f"A-novelty at {title_label}\n"
    fig.suptitle(f"{plot_title}\n", y=1.1)
    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Put the legend at the bottom center
    fig.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )
    # Save figures
    fig.savefig(
        save_dir + "/" + f"{env_layout}_efe_anov.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig)


###########################################################################################################
##### Probabilities over policies
###########################################################################################################
def plot_pi_prob_first(
    file_data_path,
    x_ticks_estep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    policies_to_vis=[],
):
    """Function to plot the probability over policies, Q(pi), averaged over the runs at the first time
    step of each episode for one experiment.

    Inputs:
    - file_data_path (str): file path to stored data
    - x_ticks_tstep (int): step for the ticks in the x axis
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plot showing the evolution of Q(pi) across episodes.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_probabilities'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
    elif "plans" or "aa" in exp_name:
        policies = data["ordered_policies"][0, 0, 0, :, :]
    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Visualize default 16 policies, unless specified differently from CL
    if len(policies_to_vis) == 0:
        # Broadcasting comparison: compare each target with all rows
        matches = (
            policies[None, :, :] == POLICIES_TO_VIS_GRIDW9[:, None, :]
        )  # shape: (16, num_rows, 4)
        # Now reduce over last dimension to check full-row match
        row_matches = np.all(matches, axis=2)  # shape: (16, num_rows)
        # Each row in row_matches should contain exactly one True
        row_indices = np.argmax(row_matches, axis=1)  # shape: (16,)
        policies_to_vis = list(row_indices)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_prob = data["pi_probabilities"][selected_runs]
    else:
        pi_prob = data["pi_probabilities"]

    # Averaging the policies' probabilities over the runs for first step of each episode only
    avg_pi_prob = np.mean(pi_prob[:, :, :, 0], axis=0)  # .squeeze()
    std_pi_prob = np.std(pi_prob[:, :, :, 0], axis=0)  # .squeeze()
    # Making sure avg_pi_prob_ls has the right dimensions
    assert avg_pi_prob.shape == (num_episodes, num_policies), "Wrong dimenions!"
    # assert np.all(np.sum(avg_pi_prob_ls, axis=1)) == True, 'Probabilities do not sum to one!'

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    x = np.arange(1, num_episodes + 1)

    if len(policies_to_vis) == 0:

        for p in range(NUM_POLICIES_VIS):
            y = avg_pi_prob[:, p + POLICY_INDEX_OFFSET].flatten()
            std = std_pi_prob[:, p + POLICY_INDEX_OFFSET].flatten()
            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i]
                for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x,
                y,
                ".-",
                color=cmap(p),
                label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
            )

    else:

        # Create inset if needed
        # axins = None
        # if "paths" or "au" in exp_name:
        #     # Get current axes
        #     axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
        #     axins.set_ylim(0.0152, 0.0162)
        #     axins.set_xlim(0, 100)
        #     axins.tick_params(labelleft=True, labelbottom=True)

        for i, p in enumerate(policies_to_vis):
            y = avg_pi_prob[:, p].flatten()
            std = std_pi_prob[:, p].flatten()
            # int_vals = ",".join(str(int(x)) for x in policies[p])

            # Policy action sequence converted into string
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

            ax.plot(
                x,
                y,
                ".-",
                color=cmap(i),
                label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
            )

        #     if axins:
        #         axins.plot(
        #             x,
        #             y,
        #             ".-",
        #             color=cmap(i),
        #             label=f"$\\pi_{{{i}}}$: {policy_action_seq_leg}",
        #         )
        # # Add inset connection after all lines are drawn
        # if axins:
        #     mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Probability mass", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])

    title = "First-step policy probability\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)
    # Save figure and show
    plt.savefig(
        save_dir
        + "/"
        + f"{env_layout}_{exp_name}_first_pi_probs_offset{POLICY_INDEX_OFFSET}.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
    )
    # plt.show()
    plt.close()


def plot_pi_prob_first_subplots(
    file_data_path,
    x_ticks_estep,
    y_limits,
    select_run,
    save_dir,
    env_layout,
    num_policies_vis,
    policies_to_vis=[],
):
    """Function to plot the probability over policies, Q(pi), averaged over the runs at the first time
    step of each episode for one or more experiments on different axes of the same figure.

    Inputs:
    - file_data_path (str): file path to stored data
    - x_ticks_tstep (int): step for the ticks in the x axis
    - y_limits (list): list with lower and upper value for the y axis
    - select_run (int): index to exclude some run based on the final policies' probability
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - num_policies_vis (int): number of policies to visualize for each agent
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plots showing the evolution of Q(pi) across episodes.
    """

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", NUM_POLICIES_VIS)
    # Determine the number of subplots based on number of data files
    num_datasets = len(file_data_path)
    fig, ax = plt.subplots(1, num_datasets, figsize=(18, 6))

    for num_data, d in enumerate(file_data_path):
        # Select correct policies from the list params["policies_to_vis"]
        curr_policies_to_vis = policies_to_vis[
            num_data * num_policies_vis : (num_data * num_policies_vis)
            + num_policies_vis
        ]

        # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_probabilities'
        data = np.load(d, allow_pickle=True).item()
        exp_name = data["exp_name"]
        num_runs = data["num_runs"]
        num_episodes = data["num_episodes"]
        num_policies = data["num_policies"]

        # Take care of the fact that policies are created and saved differently in the two types of agents
        if "paths" or "au" in exp_name:
            policies = data["policies"]
        elif "plans" or "aa" in exp_name:
            policies = data["ordered_policies"][0, 0, 0, :, :]
        else:
            raise ValueError("exp_name is not an accepted name for the experiment.")

        # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
        # was passed through the command line
        if select_run != -1:
            pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
            selected_runs = (pi_runs > 0.5).nonzero()[0]
            pi_prob = data["pi_probabilities"][selected_runs]
        else:
            pi_prob = data["pi_probabilities"]

        # Averaging the policies' probabilities over the runs for first step of each episode only
        avg_pi_prob = np.mean(pi_prob[:, :, :, 0], axis=0)  # .squeeze()
        std_pi_prob = np.std(pi_prob[:, :, :, 0], axis=0)  # .squeeze()
        # Making sure avg_pi_prob_ls has the right dimensions
        assert avg_pi_prob.shape == (num_episodes, num_policies), "Wrong dimenions!"
        # assert np.all(np.sum(avg_pi_prob_ls, axis=1)) == True, 'Probabilities do not sum to one!'

        x = np.arange(1, num_episodes + 1)

        if len(policies_to_vis) == 0:

            for p in range(NUM_POLICIES_VIS):
                y = avg_pi_prob[:, p + POLICY_INDEX_OFFSET].flatten()
                std = std_pi_prob[:, p + POLICY_INDEX_OFFSET].flatten()
                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i]
                    for i in list(policies[p + POLICY_INDEX_OFFSET].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                ax[num_data].plot(
                    x,
                    y,
                    ".-",
                    color=cmap(p),
                    label=f"$\\pi_{{{p + POLICY_INDEX_OFFSET}}}$: {policy_action_seq_leg}",
                )

        else:

            # Title
            title = "First-step policy probability\n"
            title += (
                "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
            )
            ax[num_data].set_title(title + "\n")

            # Create inset if needed
            # axins = None
            # if "paths" or "au" in exp_name:
            #     axins = inset_axes(
            #         ax[num_data], width="40%", height="40%", loc="upper right"
            #     )
            #     axins.set_ylim(0.015, 0.016)
            #     axins.set_xlim(0, 100)
            #     axins.tick_params(labelleft=True, labelbottom=True)

            for i, p in enumerate(curr_policies_to_vis):
                y = avg_pi_prob[:, p].flatten()
                # std = std_pi_prob[:, p].flatten()

                # Policy action sequence converted into string
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_leg = f"{', '.join(map(str, policy_action_arrows))}"

                ax[num_data].plot(
                    x,
                    y,
                    ".-",
                    color=cmap(i),
                    label=f"$\\pi_{{{i + 1}}}$: {policy_action_seq_leg}",
                )

            #     if axins:
            #         axins.plot(
            #             x,
            #             y,
            #             ".-",
            #             color=cmap(i),
            #             label=f"$\\pi_{{{i + 1}}}$: {policy_action_seq_leg}",
            #         )

            # # Add inset connection after all lines are drawn
            # if axins:
            #     mark_inset(ax[num_data], axins, loc1=2, loc2=4, fc="none", ec="0.5")

        ax[num_data].set_xticks(
            np.arange(x_ticks_estep, num_episodes + 1, step=x_ticks_estep)
        )
        ax[num_data].set_xlabel("Episode")
        ax[num_data].set_ylabel("Probability mass", rotation=90)
        # ax[num_data].ylim(0, 0.45)  # Uncomment for Tmaze3 experiments, comment out for others
        # ax[num_data].ylim(0, 0.09)  # Uncomment for Tmaze4 experiments, comment out for others
        ax[num_data].set_ylim(
            y_limits[0], y_limits[1]
        )  # Uncomment for Tmaze4 experiments, comment out for others

    # fig.suptitle("First-step policy probability\n", y=1.1)
    # Create shared legend (both axes plot the SAME policies and in the same order)
    # so we can retrieve and use the handles/labels from one of them
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Policies",
        ncol=4,
        title_fontsize=16,
        handlelength=2,  # shrink the line handle
        columnspacing=0.5,  # space between columns
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
    )

    # Save figure and show
    fig.savefig(
        save_dir + "/" + f"{env_layout}_first_pi_probs_offset{POLICY_INDEX_OFFSET}.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


###########################################################################################################
#### Policy-conditioned state probabilities
###########################################################################################################
def plot_Qs_pi_first(file_data_path, select_run, episode, save_dir, env_layout):
    """
    Function to visualize the Q(S_i|pi) for each policy at the first time step of one or more episodes,
    Note thatthe Q(S_i|pi) are categorical distributions telling you the state beliefs the agent has
    at that time step.

    Inputs:
    - file_data_path (str): file path to saved data
    - select_run (int): index to exclude some run based on the final policies' probability
    - episode (list): list of indices to select one or more episodes
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment

    Outputs:
    - heatmap showing the Q(S_i|pi) for each policy at the first step of one or more episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        Qs_pi_prob = data["policy_state_prob_first"][selected_runs]
    else:
        Qs_pi_prob = data["policy_state_prob_first"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
        # Averaging the Q(S|pi) over the runs
        avg_Qspi = np.mean(Qs_pi_prob, axis=0)  # .squeeze()
        # Selecting the probabilities for the last episode only
        last_episode_Qspi = avg_Qspi[episode, :, :, :]
        # print(last_episode_Qspi.shape)

    elif "plans" or "aa" in exp_name:
        num_policies = data["num_policies"]
        policies = data["ordered_policies"][0, 0, 0, :, :]
        # Retrieve the Qs to concatenate the first step of each episode to the remaining future steps
        Qs = data["states_beliefs"]
        # Averaging the Q(S) over the runs
        avg_Qs = np.mean(Qs, axis=0)
        # Selecting the probabilities for the last episode only
        first_step_Qs = np.tile(
            avg_Qs[episode, :, 0][:, np.newaxis, :, np.newaxis], (1, num_policies, 1, 1)
        )
        # Averaging the Q(S|pi) over the runs
        avg_Qspi = np.mean(Qs_pi_prob, axis=0)
        # Selecting the probabilities for one episode
        last_episode_Qspi = avg_Qspi[episode, :, :, :]
        # Concatenate with first step beliefs
        last_episode_Qspi = np.concatenate((first_step_Qs, last_episode_Qspi), axis=3)

    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")

    # Heatmap of the Q(s|pi) for every policy at the end of the experiment (after last episode)
    for p in range(last_episode_Qspi.shape[1]):

        # Creating figure and producing heatmap for policy p
        fig, ax = plt.subplots(
            1, len(episode), figsize=(22, 6)
        )  # constrained_layout=True)
        plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing

        # fig.set_figwidth(20)
        # fig.set_figheight(6)
        ims = []

        for e in range(last_episode_Qspi.shape[0]):

            X, Y = np.meshgrid(
                np.arange(1, num_steps + 1), np.arange(1, num_states + 1)
            )
            im = ax[e].pcolormesh(
                X, Y, last_episode_Qspi[e, p, :, :].squeeze(), shading="auto"
            )
            ims.append(im)

            # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
            qspi_labels = []
            for s in range(num_steps):
                # qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
                qspi_labels.append(rf"$Q(s_{{{s + 1}}}|\pi_{{{p}}})$")

            ax[e].set_xticks(np.arange(1, num_steps + 1) - 0.5, minor=True)
            ax[e].set_xticklabels(qspi_labels, minor=True)
            ax[e].tick_params(
                which="minor",
                top=True,
                bottom=False,
                labeltop=True,
                labelbottom=False,
                labelsize=20,
            )
            ax[e].grid(which="minor", color="w", linestyle="-", linewidth=3)
            plt.setp(ax[e].get_xticklabels(minor=True), ha="left", rotation=30)

            # Loop over data dimensions and create text annotations.
            # Note 1: i, j are inverted in ax[e].text() because row-column coordinates in a matrix correspond
            # to y-x Cartesian coordinates
            for i in range(num_states):
                for j in range(num_steps):
                    text = ax[e].text(
                        j + 1,
                        i + 1,
                        f"{last_episode_Qspi[e, p, i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="m",
                        fontsize="18",
                    )

            ax[e].set_xticks(np.arange(1, num_steps + 1))
            ax[e].set_xlabel("Time Step", fontsize=20)
            ax[e].invert_yaxis()
            ax[e].set_yticks(np.arange(1, num_states + 1))
            ax[e].set_ylabel("State", rotation=90, fontsize=20)

            # ax[e].set_title(
            #     f"First-step state beliefs for $\\pi_{{{p}}}$: [{policy_action_seq_figtitle}] in episode {episode}",
            #     pad=20,
            # )
            ax[e].set_title(
                f"Episode {episode[e] + 1}",
                pad=20,
            )

        # Policy action sequence converted into string
        policy_action_seq_filetitle = f"{''.join(map(str, policies[p].astype(int)))}"
        policy_action_arrows = [actions_map[i] for i in list(policies[p].astype(int))]
        policy_action_seq_figtitle = f"{', '.join(map(str, policy_action_arrows))}"

        # Create a colorbar using the last image, shared across all axes
        cbar = fig.colorbar(
            ims[-1], ax=ax.ravel().tolist(), orientation="vertical", pad=0.03
        )
        cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom", fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        # Figure title
        plot_title = f"First-step state probabilities for $\\pi_{{{p}}}$: {policy_action_seq_figtitle}\n"
        fig.suptitle(f"{plot_title}\n", y=1.3)

        # Save figure and show
        plt.savefig(
            save_dir
            + "/"
            + f"{env_layout}_{exp_name}_Qs_pi{p}_a{policy_action_seq_filetitle}_fs_ep{episode}.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.3,
        )
        # plt.show()
        plt.close(fig)


def plot_Qs_pi_last(file_data_path, select_run, episode, save_dir, env_layout):
    """
    Function to visualize the Q(S_i|pi) for each policy at the last time step of one or more episodes,
    Note thatthe Q(S_i|pi) are categorical distributions telling you the state beliefs the agent has
    at that time step.

    Inputs:
    - file_data_path (str): file path to saved data
    - select_run (int): index to exclude some run based on the final policies' probability
    - episode (list): list of indices to select one or more episodes
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment

    Outputs:
    - heatmap showing the Q(S_i|pi) for each policy at the first step of one or more episodes
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:
        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        Qs_pi_prob = data["policy_state_prob"][selected_runs]
    else:
        Qs_pi_prob = data["policy_state_prob"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
        # Averaging the Q(S|pi) over the runs
        avg_Qspi = np.mean(Qs_pi_prob, axis=0)  # .squeeze()
        # Selecting the probabilities for the last episode only
        last_episode_Qspi = avg_Qspi[episode, :, :, :]
        # print(last_episode_Qspi.shape)

        # Averaging the Q(S|pi) over the runs
        avg_Qspi = np.mean(Qs_pi_prob, axis=0)  # .squeeze()
        # Selecting the probabilities for the last episode only
        last_episode_Qspi = avg_Qspi[episode, :, :, :]

        # Heatmap of the Q(s|pi) for every policy at the end of the experiment (after last episode)
        for p in range(last_episode_Qspi.shape[1]):

            # Creating figure and producing heatmap for policy p
            fig, ax = plt.subplots(
                1, len(episode), figsize=(22, 6)
            )  # constrained_layout=True)
            plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing

            # fig.set_figwidth(20)
            # fig.set_figheight(6)
            ims = []

            for e in range(last_episode_Qspi.shape[0]):

                X, Y = np.meshgrid(
                    np.arange(1, num_steps + 1), np.arange(1, num_states + 1)
                )
                im = ax[e].pcolormesh(
                    X, Y, last_episode_Qspi[e, p, :, :].squeeze(), shading="auto"
                )
                ims.append(im)

                # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
                qspi_labels = []
                for s in range(num_steps):
                    # qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
                    qspi_labels.append(rf"$Q(s_{{{s + 1}}}|\pi_{{{p}}})$")

                ax[e].set_xticks(np.arange(1, num_steps + 1) - 0.5, minor=True)
                ax[e].set_xticklabels(qspi_labels, minor=True)
                ax[e].tick_params(
                    which="minor",
                    top=True,
                    bottom=False,
                    labeltop=True,
                    labelbottom=False,
                    labelsize=20,
                )
                ax[e].grid(which="minor", color="w", linestyle="-", linewidth=3)

                plt.setp(ax[e].get_xticklabels(minor=True), ha="left", rotation=30)

                # Loop over data dimensions and create text annotations.
                # Note 1: i, j are inverted in ax[e].text() because row-column coordinates in a matrix correspond
                # to y-x Cartesian coordinates
                for i in range(num_states):
                    for j in range(num_steps):
                        text = ax[e].text(
                            j + 1,
                            i + 1,
                            f"{last_episode_Qspi[e, p, i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="m",
                            fontsize="18",
                        )

                ax[e].set_xticks(np.arange(1, num_steps + 1))
                ax[e].set_xlabel("Time Step", fontsize=20)
                ax[e].invert_yaxis()
                ax[e].set_yticks(np.arange(1, num_states + 1))
                ax[e].set_ylabel("State", rotation=90, fontsize=20)

                # ax[e].set_title(
                #     f"First-step state beliefs for $\\pi_{{{p}}}$: [{policy_action_seq_figtitle}] in episode {episode}",
                #     pad=20,
                # )
                ax[e].set_title(
                    f"Episode {episode[e] + 1}",
                    pad=20,
                )

            # Policy action sequence converted into string
            policy_action_seq_filetitle = (
                f"{''.join(map(str, policies[p].astype(int)))}"
            )
            policy_action_arrows = [
                actions_map[i] for i in list(policies[p].astype(int))
            ]
            policy_action_seq_figtitle = f"{', '.join(map(str, policy_action_arrows))}"

            # Create a colorbar using the last image, shared across all axes
            cbar = fig.colorbar(
                ims[-1], ax=ax.ravel().tolist(), orientation="vertical", pad=0.03
            )
            cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom", fontsize=20)
            cbar.ax.tick_params(labelsize=20)

            # Figure title
            plot_title = f"Last-step state probabilities for $\\pi_{{{p}}}$: {policy_action_seq_figtitle}\n"
            fig.suptitle(f"{plot_title}\n", y=1.3)

            # Save figure and show
            plt.savefig(
                save_dir
                + "/"
                + f"{env_layout}_{exp_name}_Qs_pi{p}_a{policy_action_seq_filetitle}_ls_ep{episode}.jpg",
                format="jpg",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            # plt.show()
            plt.close()

    elif "plans" or "aa" in exp_name:
        # Retrieve the Qs to concatenate the first step of each episode to the remaining future steps
        Qs = data["states_beliefs"]
        # Averaging the Q(S) over the runs
        avg_Qs = np.mean(Qs, axis=0).squeeze()
        # Selecting the probabilities for the last episode only
        last_episode_Qs = avg_Qs[episode, :, :]

        # Creating figure and producing heatmap for policy p
        fig, ax = plt.subplots(
            1, len(episode), figsize=(22, 6)
        )  # constrained_layout=True)
        plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing

        ims = []

        for e in range(last_episode_Qs.shape[0]):

            X, Y = np.meshgrid(np.arange(num_steps), np.arange(num_states))
            im = ax[e].pcolormesh(X, Y, last_episode_Qs[e, :, :], shading="auto")
            ims.append(im)

            # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
            qspi_labels = []
            for s in range(num_steps):
                # qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
                qspi_labels.append(rf"$Q(s_{{{s}}})$")

            ax[e].set_xticks(np.arange(num_steps) - 0.5, minor=True)
            ax[e].set_xticklabels(qspi_labels, minor=True)
            ax[e].tick_params(
                which="minor",
                top=True,
                bottom=False,
                labeltop=True,
                labelbottom=False,
                labelsize=20,
            )
            ax[e].grid(which="minor", color="w", linestyle="-", linewidth=3)
            plt.setp(ax[e].get_xticklabels(minor=True), ha="left", rotation=30)

            # Loop over data dimensions and create text annotations.
            # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond
            # to y-x Cartesian coordinates
            for i in range(num_states):
                for j in range(num_steps):
                    text = ax[e].text(
                        j,
                        i,
                        f"{last_episode_Qs[e, i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="m",
                        fontsize="18",
                    )

            ax[e].set_xticks(np.arange(num_steps))
            ax[e].set_xlabel("Time Step")
            ax[e].invert_yaxis()
            ax[e].set_yticks(np.arange(num_states))
            ax[e].set_ylabel("State", rotation=90)
            ax[e].set_title(
                f"Episode {episode[e] + 1}",
                pad=20,
            )

        # Create a colorbar using the last image, shared across all axes
        cbar = fig.colorbar(
            ims[-1], ax=ax.ravel().tolist(), orientation="vertical", pad=0.03
        )
        cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom", fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        # Create colorbar
        # cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")

        fig.suptitle(f"Last-step state beliefs", y=1.3)
        # Save figure and show
        plt.savefig(
            save_dir + "/" + f"{env_layout}_{exp_name}_Qs_ls_ep{episode}.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.3,
        )
        # plt.show()
        plt.close()

    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")


def plot_Qs_pi_all(file_data_path, select_pol, episode, save_dir, env_layout):
    """
    Function to visualize the Q(S_i|pi) for each policy at ALL time steps of one episode,

    Note that the Q(S_i|pi) are categorical distributions telling you the state beliefs the agent has
    at that time step, in this case these beliefs involve the past, the present, and the future.

    Inputs:
    - file_data_path (str): file path to saved data
    - select_run (int): index to exclude some run based on the final policies' probability
    - episode (list): list of indices to select one episode
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment

    Outputs:
    - heatmap showing the Q(S_i|pi) for each policy at ALL time steps of one episode
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_pol != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_pol, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        Qs_pi_prob = data["every_tstep_prob"][selected_runs]
    else:
        Qs_pi_prob = data["every_tstep_prob"]

    # Take care of the fact that policies are created and saved differently in the two types of agents
    if "paths" or "au" in exp_name:
        policies = data["policies"]
        # Averaging the Q(S|pi) over the runs
        avg_Qspi = np.mean(Qs_pi_prob, axis=0)  # .squeeze()
        # Selecting the probabilities for one episode only and swapping relevant axes for plotting below
        all_Qspi = np.swapaxes(avg_Qspi[episode, :, :, :, :], 1, -1)
        # print(last_episode_Qspi.shape)

        for e in range(all_Qspi.shape[0]):

            ep_all_Qspi = all_Qspi[e]

            # Heatmap of the Q(s|pi) for every policy at the end of the experiment (after last episode)
            for p in range(ep_all_Qspi.shape[1]):

                # Creating figure and producing heatmap for policy p
                fig, ax = plt.subplots(
                    1, num_steps, figsize=(22, 6)
                )  # constrained_layout=True)
                plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing

                # fig.set_figwidth(20)
                # fig.set_figheight(6)
                ims = []

                for t in range(ep_all_Qspi.shape[0]):

                    X, Y = np.meshgrid(
                        np.arange(1, num_steps + 1), np.arange(1, num_states + 1)
                    )
                    im = ax[t].pcolormesh(
                        X, Y, ep_all_Qspi[t, p, :, :].squeeze(), shading="auto"
                    )
                    ims.append(im)

                    # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
                    qspi_labels = []
                    for s in range(num_steps):
                        # qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
                        qspi_labels.append(rf"$Q(s_{{{s + 1}}}|\pi_{{{p}}})$")

                    ax[t].set_xticks(np.arange(1, num_steps + 1) - 0.5, minor=True)
                    ax[t].set_xticklabels(qspi_labels, minor=True)
                    ax[t].tick_params(
                        which="minor",
                        top=True,
                        bottom=False,
                        labeltop=True,
                        labelbottom=False,
                        labelsize=20,
                    )
                    ax[t].grid(which="minor", color="w", linestyle="-", linewidth=3)
                    plt.setp(ax[t].get_xticklabels(minor=True), ha="left", rotation=30)

                    # Loop over data dimensions and create text annotations.
                    # Note 1: i, j are inverted in ax[e].text() because row-column coordinates in a matrix correspond
                    # to y-x Cartesian coordinates
                    for i in range(num_states):
                        for j in range(num_steps):
                            text = ax[t].text(
                                j + 1,
                                i + 1,
                                f"{ep_all_Qspi[t, p, i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="m",
                                fontsize="18",
                            )

                    ax[t].set_xticks(np.arange(1, num_steps + 1))
                    ax[t].set_xlabel("Time Step", fontsize=20)
                    ax[t].invert_yaxis()
                    ax[t].set_yticks(np.arange(1, num_states + 1))
                    ax[t].set_ylabel("State", rotation=90, fontsize=20)

                    # ax[e].set_title(
                    #     f"First-step state beliefs for $\\pi_{{{p}}}$: [{policy_action_seq_figtitle}] in episode {episode}",
                    #     pad=20,
                    # )
                    ax[t].set_title(
                        f"Step {t + 1}",
                        pad=20,
                    )

                # Policy action sequence converted into string
                policy_action_seq_filetitle = (
                    f"{''.join(map(str, policies[p].astype(int)))}"
                )
                policy_action_arrows = [
                    actions_map[i] for i in list(policies[p].astype(int))
                ]
                policy_action_seq_figtitle = (
                    f"{', '.join(map(str, policy_action_arrows))}"
                )

                # Create a colorbar using the last image, shared across all axes
                cbar = fig.colorbar(
                    ims[-1], ax=ax.ravel().tolist(), orientation="vertical", pad=0.03
                )
                cbar.ax.set_ylabel(
                    "Probability", rotation=-90, va="bottom", fontsize=20
                )
                cbar.ax.tick_params(labelsize=20)

                # Figure title
                plot_title = f"State probabilities for $\\pi_{{{p}}}$ in episode {episode[e] + 1}: {policy_action_seq_figtitle}\n"
                fig.suptitle(f"{plot_title}\n", y=1.3)

                # Save figure and show
                plt.savefig(
                    save_dir
                    + "/"
                    + f"{env_layout}_{exp_name}_Qs_pi{p}_a{policy_action_seq_filetitle}_all_ep{episode[e]}.jpg",
                    format="jpg",
                    bbox_inches="tight",
                    pad_inches=0.3,
                )
                # plt.show()
                plt.close(fig)

    elif "plans" or "aa" in exp_name:

        pass

    else:
        raise ValueError("exp_name is not an accepted name for the experiment.")


#########################################################################################################
##### State-observation matrices, state-transitions matrices, state visits (heatmaps)
#########################################################################################################
def plot_matrix_A(
    file_data_path,
    x_ticks_estep,
    state_A,
    select_policy,
    save_dir,
    env_layout,
):
    """
    Function to plot state-observation mappings, i.e., matrix A of size (num_states, num_states),
    averaged over the runs.

    The columns of A are categorical distributions so their elements must sum to one, e.g., column 0 (zero)
    tells you the probability of the agent believing to be in a certain state when it is in state 0 (zero),
    e.g. P(O=0|S=0). If the agent has sound beliefs, then in state 0 (zero) it should believe to be in state 0.
    In other words, matrix A should approximate an identity matrix.

    Inputs:
    - file_data_path (str): file path to the saved data
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number;
    - state_A (int): index i to slice a certain column of A to represent the evolution of Q(O|S_i);
    - select_policy (int): exclude some runs from visualization based on policies probabilities
    - save_dir (str): directory where to save the images
    - env_layout (str): layout of the environment

    Outputs:
    - plot showing the emission probabilities for a specific state, S_i (i.e., a column of A), over the entire
      experiment;
    - heatmap showing matrix A at the end of the experiment to see what the agent learned.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'so_mappings'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        so_mappings = data["so_mappings"][selected_runs]
    else:
        so_mappings = data["so_mappings"]

    # Making sure state_A is one of the of the columns of matrix A (i.e. one of the state for which
    # we want to see the the state-observation mapping)
    assert state_A >= 0 and state_A <= num_states - 1, "Invalid state."

    # Computing the mean (avg) and std of the emission probabilities over the runs
    # Note 1: so_mapping is of shape (num_runs, num_episodes, num_states, num_states)
    avg_som = np.mean(so_mappings, axis=0)  # .squeeze()
    std_som = np.std(so_mappings, axis=0)  # .squeeze()
    # Selecting a specific state-observation mapping, i.e., the emission probabilities when in state
    # state_A, stored in the corresponding column of A
    # Note 2: we are basically singling out a column of A to see how it changes episode after episode
    # (due to the agent learning)
    s = state_A
    avg_som_state = avg_som[:, :, s]  # .squeeze()
    std_som_state = std_som[:, :, s]  # .squeeze()
    # Plotting the state-observation mapping from state s throughout the experiment
    x = np.arange(1, num_episodes + 1)
    y_data = avg_som_state[:, :]

    fig1, ax1 = plt.subplots()

    # Setting up plotting markers and a counter to cycle through them
    markers = [".", "+", "x"]
    counter = 0
    # Looping over every element of the selected column of A to plot how that value (probability)
    # changes episode after episode
    # Note 3: we are plotting how every single emission probability from state S_i changes during
    # the experiment. The number of those probabilities is given by avg_som_state.shape[1]
    # (the second dimension of avg_som_state).
    for c in range(avg_som_state.shape[1]):
        counter += 1
        m = None
        if counter <= 10:
            m = 0
        elif counter > 10 and counter <= 20:
            m = 1
        else:
            m = 2
        # Selecting a single emission probability from state S_i
        y = y_data[:, c]
        # TODO: the proper marker does not seem to be selected and used in the plot
        # (yes, because with 8 states you have m < 10 so they all get the same marker...)
        ax1.plot(x, y, marker=markers[m], linestyle="-", label=f"$P(O={c}|S={s})$")
        # ax1.fill_between(x, y-std_som_state[:, c], y+std_som_state[:, c], alpha=0.3)
        ax1.fill_between(
            x,
            y - (1.96 * std_som_state[:, c] / np.sqrt(num_runs)),
            y + (1.96 * std_som_state[:, c] / np.sqrt(num_runs)),
            alpha=0.3,
        )

    ax1.set_xticks(np.arange(x_ticks_estep, num_episodes + 1, step=x_ticks_estep))
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Probability Mass", rotation=90)
    # ax1.legend([f'$P(O={o}|S={s})$' for o in range(num_states)], loc='upper right')
    ax1.legend(loc="upper right")
    ax1.set_title(f"Emission Probabilities from State {s}\n")
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_emis_prob_state{s}_path.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()

    # Heatmap of the emission probabilites from all states at the end of the experiment
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(avg_som[-1, :, :].squeeze())

    ax2.set_yticks(np.arange(num_states))
    ax2.set_yticklabels(np.arange(num_states))
    ax2.set_xticks(np.arange(num_states))
    ax2.set_xticklabels(np.arange(num_states))

    # Loop over data dimensions and create text annotations.
    # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond to y-x Cartesian coordinates
    # for i in range(num_states):
    #     for j in range(num_states):
    #         text = ax2.text(j, i, f'{avg_som[-1, i, j]:.3f}', ha="center", va="center", color="w", fontsize='small')

    # Create colorbar
    cbar = ax2.figure.colorbar(im, ax=ax2)
    cbar.ax.set_ylabel("Probability Mass", rotation=-90, va="bottom")

    ax2.set_xlabel("States")
    ax2.set_ylabel("States", rotation=90)
    ax2.set_title(f"State-observation Matrix")

    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_matrix_A.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_matrix_B_kl(
    file_data_path,
    x_ticks_estep,
    y_lims,
    state_B,
    action_B,
    select_run,
    save_dir,
    env_layout,
):
    """
    Function to plot the sum of KL divergences of transition probabilities for each action across the episodes.
    Inputs:

    - file_data_path (str): file path to the saved data
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - state_B (int): index i to slice a certain column of B to represent the evolution of P(S_j|S_i, a)
    - action_B (int): index to select the action for which to represent the transition probabilities
    - select_run (int): exclude some runs from visualization based on policies probabilities
    - save_dir (str): directory where to save the images
    - env_layout (str): layout of the environment

    Outputs:
    - plot showing the evolution of the sum of KL divergences
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'transition_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        transitions_prob = data["transition_prob"][selected_runs]
    else:
        transitions_prob = data["transition_prob"]

    # Computing the mean (avg) and std of the transition probabilities over the runs
    avg_transitions_prob = np.mean(transitions_prob, axis=0)  # .squeeze()
    std_transitions_prob = np.std(transitions_prob, axis=0)  # .squeeze()

    # Making sure state_B is a valid value to select a matrix B and slice it
    assert state_B >= 0 and state_B <= num_states - 1, "Invalid state index."

    # Conmpute ground-truth B matrices for Tmaze4
    # NOTE: -1 in the y direction, from an external observer this would correspond to "up", in the
    # Gymnasium grid coordinate system the negative and positive y axes are swapped
    B_params = np.zeros((4, num_states, num_states))

    if "tmaze4" in env_layout:
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
    elif "ymaze4" in env_layout:

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
    elif "gridw9" in env_layout:

        # Down action: 3
        B_params[3, :, :] = np.array(
            [
                [1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        # Left action: 2
        B_params[2, :, :] = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        # Up action: 1
        B_params[1, :, :] = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        # Right action: 0
        B_params[0, :, :] = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
            ],
            dtype=np.float64,
        )

    rng = np.random.default_rng(seed=33)
    B = np.zeros((4, num_states, num_states))
    B_params = (B_params * 200) + 1

    for a in range(4):
        for s in range(num_states):
            B[a, :, s] = rng.dirichlet(B_params[a, :, s], size=1)

    x = np.arange(1, num_episodes + 1)
    action_errors = {}

    # Plotting sum of KL divergences for B matrices
    fig1, ax1 = plt.subplots(figsize=(5, 4), tight_layout=True)
    # Selecting the avg and std transition probabilities for a specific action (a=0, a=1, a=2, a=3)
    # throughout the experiment, i.e. for B_a
    for a in action_B:
        # Making sure action_B is a valid value to select a matrix B and slice it
        assert a >= 0 and a <= 3, "Invalid action index."
        transition_prob_action = avg_transitions_prob[:, a, :, :]  # .squeeze()
        y = []
        # std_tpa = std_transitions_prob[:, a, :, :]  # .squeeze()
        for t in range(transition_prob_action.shape[0]):
            kls = cat_KL(transition_prob_action[t], B[a])
            y.append(np.sum(kls))

        action_errors[a] = np.array(y)

        ax1.plot(
            x,
            y,
            linestyle="-",
            label=f"{actions_map[a]}",
        )

    ax1.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Nats", rotation=90)

    if "tmaze4" in env_layout:
        ax1.set_ylim(0, 20)
    elif "ymaze4" in env_layout:
        ax1.set_ylim(y_lims[0], y_lims[1])
    else:
        ax1.set_ylim(0, 32)

    # title = f"Transition Probabilities from State {s + 1} for Action {actions_map[a]}"
    # title += "\n(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        stitle = "(hard with goal shaping)"
    elif "softgs" in exp_name:
        stitle = "(soft with goal shaping)"
    elif "hard" in exp_name:
        stitle = "(hard without goal shaping)"
    elif "soft" in exp_name:
        stitle = "(soft without goal shaping)"
    else:
        stitle = "(preferences)"

    title = f"Sum of KL divergences for each action\n" f"{stitle}"

    # title = (
    #     f"Sum of KL divergences for each action\n"
    #     f"{'(action-unaware)' if 'paths' in exp_name else '(action-aware)'}"
    # )
    ax1.set_title(title, pad=15)

    # Create a separate figure for the legend
    fig_legend = plt.figure(figsize=(8, 2))
    # Use the same handles and labels
    handles, labels = ax1.get_legend_handles_labels()
    fig_legend.legend(
        handles,
        labels,
        title="Actions",
        ncol=4,
        title_fontsize=12,
        handlelength=2,  # shrink the line handle
        columnspacing=1.5,  # space between columns
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        fancybox=True,
    )
    # Save the legend figure separately
    fig_legend.savefig(
        save_dir + "/" + f"{env_layout}_actions_legend.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
    )

    plt.close(fig_legend)

    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_matrix_B_kl.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.1,
    )
    # plt.show()
    plt.close()

    # Plotting sum of KL divergences for B matrices
    fig2, ax2 = plt.subplots(figsize=(5, 4), tight_layout=True)
    # Select specific episodes
    selected = [0, 24, 49, 74, 99]  # Python indices for episodes 1, 25, 50, 75, 100
    x_labels = [1, 25, 50, 75, 100]
    # Data for inset
    a0 = action_errors[0][selected]
    a1 = action_errors[1][selected]
    a2 = action_errors[2][selected]
    a3 = action_errors[3][selected]
    # Plot stacked bars
    width = 15
    bars1 = ax2.bar(x_labels, a0, width=width)
    bars2 = ax2.bar(x_labels, a1, width=width, bottom=a0)
    bars3 = ax2.bar(x_labels, a2, width=width, bottom=a0 + a1)
    bars4 = ax2.bar(x_labels, a3, width=width, bottom=a0 + a1 + a2)

    # Formatting
    title = (
        f"Total KL divergence in selected episodes\n"
        f"{'(action-unaware)' if 'paths' in exp_name else '(action-aware)'}"
    )
    ax2.set_title(title, pad=15)
    ax2.set_xticks(x_labels)
    ax2.set_ylabel("Nats")
    ax2.set_xlabel("Episodes")

    # Function to label bars
    def annotate_bars(bars, heights):
        for bar, h in zip(bars, heights):
            if h > 0.5:  # avoid cluttering small segments
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + h / 2,
                    f"{h:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

    annotate_bars(bars1, action_errors[0][selected])
    annotate_bars(bars2, action_errors[1][selected])
    annotate_bars(bars3, action_errors[2][selected])
    annotate_bars(bars4, action_errors[3][selected])

    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_matrix_B_kl_selep.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
        # pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_matrix_B(
    file_data_path,
    x_ticks_estep,
    state_B,
    action_B,
    select_run,
    save_dir,
    env_layout,
):
    """
    Function to plot transition probabilities, i.e., matrices B (one for each available action) of size
    (num_states, num_states), averaged over the runs.

    The columns of a B matrix are categorical distributions so their elements must sum to one, e.g.,
    column 0 (zero) of B_up (the transition matrix for action up) gives the agent the probabilities
    of landing in the various states by going up from state 0 (zero). If the agent has learned correct
    transitions (and the environment is not stochastic), then going up from state 0 (zero) should lead
    to a specific state. In other words, columns of matrices B should have all values close to 0 except
    for one close to 1.

    Inputs:

    - file_data_path (str): file path to the saved data
    - x_ticks_estep (int): step for the ticks in the x axis when plotting as a function of episode number
    - state_B (int): index i to slice a certain column of B to represent the evolution of P(S_j|S_i, a)
    - action_B (int): index to select the action for which to represent the transition probabilities
    - select_run (int): exclude some runs from visualization based on policies probabilities
    - save_dir (str): directory where to save the images
    - env_layout (str): layout of the environment

    Outputs:
    - plot showing the transitions probabilities for a specific state and action (i.e., a column of a B matrix)
      over the entire experiment;
    - heatmap showing matrix B for a certain action at the end of the experiment to see what the agent learned.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'transition_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_run != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_run, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        transitions_prob = data["transition_prob"][selected_runs]
    else:
        transitions_prob = data["transition_prob"]

    # Computing the mean (avg) and std of the transition probabilities over the runs
    avg_transitions_prob = np.mean(transitions_prob, axis=0)  # .squeeze()
    std_transitions_prob = np.std(transitions_prob, axis=0)  # .squeeze()

    # Making sure state_B is a valid value to select a matrix B and slice it
    assert state_B >= 0 and state_B <= num_states - 1, "Invalid state index."

    # Selecting the avg and std transition probabilities for a specific action (a=0, a=1, a=2, a=3)
    # throughout the experiment, i.e. for B_a
    for a in action_B:
        # Making sure action_B is a valid value to select a matrix B and slice it
        assert a >= 0 and a <= 3, "Invalid action index."
        transition_prob_action = avg_transitions_prob[:, a, :, :]  # .squeeze()
        std_tpa = std_transitions_prob[:, a, :, :]  # .squeeze()
        # Selecting a specific state transition throughout the experiment
        s = state_B
        transition_state = transition_prob_action[:, :, s]  # .squeeze()
        std_transition_state = std_tpa[:, :, s]  # .squeeze()

        # Plotting the transition probabilites from state s for action a throughout the experiment.
        # For example, we could plot the transition probability from the state just before the goal for
        # the action that would bring the agent there to see if the agent learned the way to get to the
        # goal state.
        x = np.arange(1, num_episodes + 1)
        y_data = transition_state[:, :]

        fig1, ax1 = plt.subplots(figsize=(5, 4), tight_layout=True)
        # Setting up plotting markers and a counter to cycle through them
        markers = [".", "+", "x"]
        counter = 0
        # Looping over every element of the selected column of B to plot how that value (probability) changes
        # episode after episode
        # Note 1: we are plotting how every single transition probability from state s changes during the
        # experiment. The number of those probabilities is given by transition_state.shape[1] (the second
        # dimension of transition_state).
        for c in range(transition_state.shape[1]):
            counter += 1
            m = None
            if counter <= 10:
                m = 0
            elif counter > 10 and counter <= 20:
                m = 1
            else:
                m = 2
            y = y_data[:, c]
            ax1.plot(
                x,
                y,
                marker=markers[m],
                linestyle="-",
                label=f"$P(S_{{t+1}}={c+1}|S_t={s+1}, \\pi_t=${actions_map[a]}$)$",
            )
            # ax1.fill_between(x, y-std_transition_state[:, c], y+std_transition_state[:, c], alpha=0.3)
            ax1.fill_between(
                x,
                y - (1.96 * std_transition_state[:, c] / np.sqrt(num_runs)),
                y + (1.96 * std_transition_state[:, c] / np.sqrt(num_runs)),
                alpha=0.3,
            )

        ax1.set_xticks(
            [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
        )
        ax1.set_xlabel("Episode")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Probability mass", rotation=90)

        # title = f"Transition Probabilities from State {s + 1} for Action {actions_map[a]}"
        # title += "\n(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
        if "hardgs" in exp_name:
            stitle = "(hard with goal shaping)"
        elif "soft" in exp_name:
            stitle = "(soft without goal shaping)"
        elif "hard2" in exp_name:
            stitle = "(sparse with hard goal)"
        elif "hard" in exp_name:
            stitle = "(hard without goal shaping)"
        else:
            stitle = "(preferences)"

        title = (
            f"Transition probabilities from state {s + 1} for action {actions_map[a]}\n"
            f"{stitle}"
        )

        # title = (
        #     f"Transition probabilities from state {s + 1} for action {actions_map[a]}\n"
        #     f"{'(action-unaware)' if 'paths' in exp_name else '(action-aware)'}"
        # )

        ax1.set_title(title, pad=15)

        # Create a separate figure for the legend
        fig_legend = plt.figure(figsize=(10, 2))
        # Use the same handles and labels
        handles, labels = ax1.get_legend_handles_labels()
        fig_legend.legend(
            handles,
            labels,
            title="Transition probabilities",
            ncol=1,
            title_fontsize=12,
            handlelength=2,  # shrink the line handle
            # columnspacing=1.5,  # space between columns
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            fancybox=True,
        )
        # Save the legend figure separately
        fig_legend.savefig(
            save_dir + "/" + f"{env_layout}_transition_probs_s{s + 1}_a{a}_legend.pdf",
            format="pdf",
            dpi=200,
            bbox_inches=None,
        )

        plt.close(fig_legend)

        plt.savefig(
            save_dir + "/" + f"{env_layout}_{exp_name}_matrix_B_state{s}_action{a}.pdf",
            format="pdf",
            dpi=200,
            bbox_inches=None,
            # pad_inches=0.1,
        )
        # plt.show()
        plt.close()

    # Heatmap of the transition probabilites from all states for action all the actions at the end
    # of the experiment; the actions range from 0 to 3 (included)
    for a in range(0, 4):
        fig2, ax2 = plt.subplots(figsize=(5, 4), tight_layout=True)
        im = ax2.imshow(avg_transitions_prob[-1, a, :, :].squeeze(), vmin=0, vmax=1)

        # Major Ticks for states
        state_labels = np.arange(1, num_states + 1)
        ax2.set_yticks(np.arange(num_states))
        ax2.set_yticklabels(state_labels)
        ax2.set_xticks(np.arange(num_states))
        ax2.set_xticklabels(state_labels)
        # Minor ticks for grid lines
        ax2.set_yticks(np.arange(0.5, num_states - 0.5), minor=True)
        ax2.set_xticks(np.arange(0.5, num_states - 0.5), minor=True)
        # Retrieve major grid style
        major_gridline = ax2.get_xgridlines()[0]
        grid_style = {
            "linestyle": major_gridline.get_linestyle(),
            "linewidth": major_gridline.get_linewidth(),
            "color": major_gridline.get_color(),
            "alpha": major_gridline.get_alpha(),
        }

        # Apply to minor grid
        ax2.grid(True, which="minor", **grid_style)
        ax2.grid(False, which="major")

        # Loop over data dimensions and create text annotations.
        # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond to y-x Cartesian coordinates
        # for i in range(num_states):
        #     for j in range(num_states):
        #         text = ax2.text(j, i, f'{transition_prob_action[-1, i, j]:.3f}', ha="center", va="center", color="w", fontsize='medium')

        # Create colorbar
        cbar = ax2.figure.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")
        cbar.ax.set_ylim(0, 1)

        ax2.set_xlabel("States")
        ax2.set_ylabel("States", rotation=90)
        title = f"Transition matrix for action {actions_map[a]}\n"
        title += (
            "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
        )
        ax2.set_title(f"{title}", pad=15)

        # Save figure and show
        plt.savefig(
            save_dir + "/" + f"{env_layout}_{exp_name}_matrix_B_a{a}.pdf",
            format="pdf",
            dpi=200,
            bbox_inches=None,
        )
        # plt.show()
        plt.close()


def plot_state_visits(file_path, v_len, h_len, select_policy, save_dir, env_layout):
    """
    Plotting the state visits heatmap showing the frequency with which the agent has visited the
    maze's tiles.

    Inputs:

    - file_path: the file path where the cumulative reward data was stored while running the experiment
    - v_len (int): vertical length of the environment
    - h_len (int): horizontal length of the environment
    - select_run (int): exclude some runs from visualization based on policies probabilities
    - save_dir (str): directory where to save the images
    - env_layout (str): layout of the environment

    Outputs:
    - heatmap showing the frequency with which maze's tiles have been visited.

    Note 1: ax.imshow() is used to produce the heatmap. Because of the way plt.colormesh works, the data
    has to be reshaped, transposed and rotated 90 degrees.

    Detailed explanation/example:

    Consider for example the top-left tile in a 6-by-9 maze, that tile corresponds to
    state 0 and is the first value in the vector 'average_state_visits'. When you reshape that array,
    you get a matrix of shape (6,9) and the state 0 average is at location [0,0] in the matrix.
    In other words, the matrix is isomorphic to the pictorial representation of the maze.
    However, plt.colormesh takes the values in the first row of the matrix (which is the top row in the maze)
    and by default plots them using their matrix coordinates to access some graph coordinates in X and Y.
    These are arrays of shape (7,10) that by default in this example look like
    X = [[0, 1,..,7,8,9],..,[0, 1,..,7,8,9]] and Y = [[0,..0,],..[6,..6]];
    these graph coordinates are used to draw the corners of the bounding squares in the colormesh grid of
    the chart.
    So, the values of the top row in the maze end up at the bottom of the chart (if nothing is done).
    The optional array arguments X and Y can be manually set (if you don't want to reshape the data).
    For more info, see the matplotlib documentation.
    """

    # Retrieving state visits data and computing average over runs, and over runs and episodes
    data = np.load(file_path, allow_pickle=True).item()
    exp_name = data["exp_name"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        state_visits = data["state_visits"][selected_runs]
    else:
        state_visits = data["state_visits"]

    # Average over runs
    run_avg_sv = np.mean(state_visits, axis=0)
    # Total visitation counts
    tot_sv = np.sum(run_avg_sv, axis=0)
    # Total number of steps
    total_steps = np.sum(tot_sv)

    if env_layout == "tmaze3":
        env_matrix = np.zeros((v_len, h_len))
        env_matrix[0, :] = tot_sv[:-1]
        env_matrix[1, 1] = tot_sv[-1]

    elif env_layout == "tmaze4":
        env_matrix = np.zeros((v_len, h_len))
        env_matrix[0, :] = tot_sv[:-2]
        env_matrix[1, 1] = tot_sv[-2]
        env_matrix[2, 1] = tot_sv[-1]
    elif env_layout == "ymaze4":
        env_matrix = np.zeros((v_len, h_len))
        env_matrix[0, 0] = tot_sv[0]
        env_matrix[0, 2] = tot_sv[1]
        env_matrix[1, :] = tot_sv[2:5]
        env_matrix[2, 1] = tot_sv[-1]
    else:
        # Reshaping the state counts vector into a matrix so as to visualise the maze
        env_matrix = np.reshape(tot_sv, (v_len, h_len))

    # Heatmap of the state counts over all the experiment's episodes
    percentage_sv = env_matrix / total_steps * 100
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    im = ax.imshow(percentage_sv, vmin=0, vmax=100)

    # Setting x and y ticks to display grid correctly, then removing all ticks and labels.
    ax.set_xticks((np.arange(percentage_sv.shape[1]) + 1) - 0.5)
    ax.set_yticks((np.arange(percentage_sv.shape[0]) + 1) - 0.5)
    ax.tick_params(
        which="major", left=False, bottom=False, labelleft=False, labelbottom=False
    )
    ax.grid(which="major", color="grey", linestyle="-", linewidth=3)

    # Loop over data dimensions and create text annotations.
    # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond
    # to y-x Cartesian coordinates
    for i in range(env_matrix.shape[0]):
        for j in range(env_matrix.shape[1]):
            text = ax.text(
                j,
                i,
                f"{percentage_sv[i, j]:.1f}",
                ha="center",
                va="center",
                color="w",
                fontsize=16,
            )

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Percentage of time steps", rotation=-90, va="bottom")
    cbar.ax.set_ylim(0, 100)
    # Format color bar as percentages
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    title = "State-access frequency\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)

    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_sv.pdf",
        format="pdf",
        bbox_inches=None,
    )
    # plt.show()
    plt.close()


###########################################################################################################
##### Action sequence for a selected agent/run
###########################################################################################################


def plot_action_probs(
    file_data_path,
    x_ticks_estep,
    y_limits,
    step,
    save_dir,
    env_layout,
):
    """Function to plot the (un)normalized probabilities for each action at one or every step across episodes,
    averaged over the runs at the first time step of each episode for one experiment.

    Inputs:
    - file_data_path (str): file path to stored data
    - x_ticks_tstep (int): step for the ticks in the x axis
    - y_limits (list): list with lower and upper value for the y axis
    - step (int): index to identify the step for which to plot the action probabilities
    - save_dir (str): directory where to save the images
    - env_layout (str): layoiut of the environment
    - policies_to_vis (list): list of policies' indices to visualize a subset of the policies for each run/agent

    Outputs:
    - line plot showing the evolution of Q(pi) across episodes.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_probabilities'
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    action_probs = data["action_probs"]

    # Averaging the probabilities over the runs for a single step of each episode
    # avg_action_prob = np.mean(action_probs[:, :, :, step], axis=0).T
    # std_action_prob = np.std(action_probs[:, :, :, step], axis=0).T

    ### DEBUG ###
    # Select only one run
    avg_action_prob = action_probs[1, :, :, step].T
    # reshape((4, num_episodes))  # .squeeze()
    std_action_prob = action_probs[1, :, :, step].T

    # Making sure avg_pi_prob_ls has the right dimensions
    # print(avg_action_prob.shape)
    # assert avg_action_prob.shape == (num_episodes, 4), print(
    #     f"Shape {avg_action_prob.shape}"
    # )
    # assert np.all(np.sum(avg_pi_prob_ls, axis=1)) == True, 'Probabilities do not sum to one!'

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    x = np.arange(1, num_episodes + 1)
    y = avg_action_prob

    for a in range(4):

        ax.plot(
            x,
            y[a],
            ".-",
        )

    ax.set_xticks(
        [1] + list(np.arange(x_ticks_estep, (num_episodes) + 1, step=x_ticks_estep))
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Probability mass", rotation=90)
    ax.set_ylim(y_limits[0], y_limits[1])

    title = f"Action probabilities at step {step+1}\n"
    # title += "(action-unaware)" if "paths" or "au" in exp_name else " (action-aware)"
    if "hardgs" in exp_name:
        title += "(hard with goal shaping)"
    elif "softgs" in exp_name:
        title += "(soft with goal shaping)"
    elif "hard" in exp_name:
        title += "(hard without goal shaping)"
    elif "soft" in exp_name:
        title += "(soft without goal shaping)"
    else:
        title += "(preferences)"

    ax.set_title(title, pad=15)
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_{exp_name}_s{step+1}_action_probs.pdf",
        format="pdf",
        dpi=200,
        bbox_inches=None,
    )
    plt.show()
    plt.close()


def plot_action_seq(
    file_data_path,
    x_ticks_estep,
    policy_horizon,
    run_index,
    save_dir,
    env_layout,
):
    """
    Function to plot the sequence of actions realized by an agent/run in the experiment.
    """

    # Retrieving the data dictionary and extracting the content of various keys
    data = np.load(file_data_path, allow_pickle=True).item()
    exp_name = data["exp_name"]
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    reward_counts = data["reward_counts"]
    action_seqs = data["actual_action_sequence"]

    for i, r in enumerate(run_index):
        # Data for a specific run/agent (use index according to desired run/agent)
        run_actions = action_seqs[r]
        # print("Shape of run_actions")
        # print(run_actions.shape)
        # Create episode index repeated for each action (x axis)
        episodes = np.repeat(np.arange(1, num_episodes + 1), policy_horizon)
        # Create time steps, e.g. 0 and 1 for each episode if policy_horizon = 2 (y axis)
        timesteps_array = np.arange(1, policy_horizon + 1)
        timesteps = np.tile(
            timesteps_array, num_episodes
        )  # Alternating time steps 0 and 1
        # Define colors based on action values (0=right, 1=up, 2=left, 3=down)
        action_colors = {0: "blue", 1: "red", 2: "green", 3: "orange"}
        actions_map = {
            0: "$\\rightarrow$",
            1: "$\\downarrow$",
            2: "$\\leftarrow$",
            3: "$\\uparrow$",
        }
        colors = [action_colors[a] for a in run_actions.flatten()]

        # Plotting action sequences for each episode for run/agent 3
        plt.figure(figsize=(20, 3))
        plt.scatter(episodes, timesteps, c=colors, s=100, edgecolors="black")

        # plt.scatter(episodes, action_seqs[3], alpha=0.7, marker="o")

        plt.xlabel("Episode")
        plt.ylabel("Time step")
        plt.title(f"Actions executed by agent {r} in each episode ")

        plt.xticks(np.arange(1, num_episodes + 1, step=4))
        plt.ylim(0.5, 3.5)
        plt.yticks(np.arange(1, num_steps))
        plt.grid(True, linestyle="--", alpha=0.5)

        # Create a legend
        legend_patches = [
            Patch(color=color, label=f"Action {actions_map[action]}")
            for action, color in action_colors.items()
        ]
        plt.legend(handles=legend_patches, title="Actions")

        plt.savefig(
            save_dir + "/" + f"{env_layout}_{exp_name}_aseq_run{r}.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.show()
        plt.close()


#######################################################################################################
#### NOT USED (revision required)
#######################################################################################################
def plot_efe_Bcomps(file_data_path, select_policy, save_dir):
    """Plotting the expected free energy B-novelty component for a given policy at the first episode step
    averaged over the runs.

    Inputs:

    - file_data_path (string): file path where the total free energy data was stored (i.e. where log_data
      was saved)
    - save_dir (string): directory where to save the images.

    Outputs:

    - plot showing evolution of the expected free energy.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        num_runs = len(selected_runs)
        efe_Bnovelty_t = data["efe_Bnovelty_t"][selected_runs]
    else:
        efe_Bnovelty_t = data["efe_Bnovelty_t"]

    # Averaging the expected free energies and their components over the runs
    avg_efe_Bnovelty_t = np.mean(efe_Bnovelty_t, axis=0)  # .squeeze()
    std_efe_Bnovelty_t = np.std(efe_Bnovelty_t, axis=0)  # .squeeze()
    # Making sure efe has the right dimensions
    assert avg_efe_Bnovelty_t.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"

    plt.figure()

    for p in range(num_policies):

        # Plotting all the time steps
        # x = np.arange(num_episodes*num_steps)
        # y = np.reshape(-avg_efe[:, p, :], (num_episodes*num_steps))

        # Plotting a subset of the time steps (i.e. only the first for every episode)
        x = np.arange(1, num_episodes + 1)
        # x = np.arange(1*(num_steps-1))
        y_efeB_t1 = avg_efe_Bnovelty_t[:, p, 1].flatten()
        stdy_efeB_t1 = std_efe_Bnovelty_t[:, p, 1].flatten()
        y_efeB_t2 = avg_efe_Bnovelty_t[:, p, 2].flatten()
        stdy_efeB_t2 = std_efe_Bnovelty_t[:, p, 2].flatten()
        y_efeB_t3 = avg_efe_Bnovelty_t[:, p, 3].flatten()
        stdy_efeB_t3 = std_efe_Bnovelty_t[:, p, 3].flatten()
        # y = np.reshape(-avg_efe[2, p, 0:-1], (1*(num_steps-1)))

        plt.plot(x, y_efeB_t1, ".-", label=f"B-novelty at $t=1$ for $\\pi_{p}$")
        plt.plot(x, y_efeB_t2, ".-", label=f"B-novelty at $t=2$ for $\\pi_{p}$")
        plt.plot(x, y_efeB_t3, ".-", label=f"B-novelty at $t=3$ for $\\pi_{p}$")

        plt.fill_between(
            x,
            y_efeB_t1 - (1.96 * stdy_efeB_t1 / np.sqrt(num_runs)),
            y_efeB_t1 + (1.96 * stdy_efeB_t1 / np.sqrt(num_runs)),
            alpha=0.3,
        )

        plt.fill_between(
            x,
            y_efeB_t2 - (1.96 * stdy_efeB_t2 / np.sqrt(num_runs)),
            y_efeB_t2 + (1.96 * stdy_efeB_t2 / np.sqrt(num_runs)),
            alpha=0.3,
        )
        plt.fill_between(
            x,
            y_efeB_t3 - (1.96 * stdy_efeB_t3 / np.sqrt(num_runs)),
            y_efeB_t3 + (1.96 * stdy_efeB_t3 / np.sqrt(num_runs)),
            alpha=0.3,
        )

    plt.xlabel("Step")
    plt.ylabel("Value", rotation=90)
    plt.legend(loc="upper left")
    plt.title("B-novelty Components at the First Step\n")
    # Save figure and show
    plt.savefig(
        save_dir + "/" + "b_novelty_comps.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_pi_prob(file_data_path, x_ticks_tstep, select_policy, save_dir, env_layout):
    """Function to plot the probability over policies, Q(pi), averaged over the runs at every time step during
    the experiment.

    Inputs:

    - file_data_path (string): file path where the data was stored (i.e. where log_data was saved);
    - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps;
    - save_dir (string): directory where to save the images.

    Outputs:

    - plot showing the evolution of Q(pi) as the agent goes through the episodes.

    Note 1: the probability mass should concentrate on the most favoured policy as the agent experiences
    the maze (episode after episode).
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_probabilities'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        pi_prob = data["pi_probabilities"][selected_runs]
    else:
        pi_prob = data["pi_probabilities"]

    # Averaging the policies' probabilities over the runs
    avg_pi_prob = np.mean(pi_prob, axis=0)  # .squeeze()
    # Making sure avg_pi_prob_ls has the right dimensions
    assert avg_pi_prob.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"
    # assert np.all(np.sum(avg_pi_prob_ls, axis=1)) == True, 'Probabilities do not sum to one!'

    plt.figure()
    x = np.arange(num_episodes * num_steps)

    for p in range(num_policies):
        y = avg_pi_prob[:, p, :].flatten()
        plt.plot(x, y, ".-", label=f"$\\pi_{{{p + 1}}}$")

    plt.xticks(
        np.arange(x_ticks_estep, (num_episodes * num_steps) + 1, step=x_ticks_tstep)
    )
    plt.xlabel("Step")
    plt.ylabel("Probability Mass", rotation=90)
    # plt.legend(['Policy 1', 'Policy 2', 'Policy 3', 'Policy 4', 'Policy 5'], loc='upper right')
    plt.legend(
        title="Policies",
        ncol=4,
        title_fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
    )
    plt.ticklabel_format(style="plain")
    plt.ticklabel_format(useOffset=False)
    plt.title("Probability over Policies\n")
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_pi_probs_path_every_step.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()


def plot_Qs_pi_prob(
    file_data_path,
    x_ticks_estep,
    index_Si,
    value_Si,
    select_policy,
    save_dir,
    env_layout,
):
    """Plotting policies' beliefs over states at a certain time step, i.e. Q(S_t = s|pi),
    over the episodes (averaged over the runs).

    More specifically, we pick one of the random variables (r.v.) S_i, where i is in [0,..., num_steps-1],
    and we plot the probability of one realization of that r.v. conditioned on the policy.
    For example, for S_f = g, where f is the final time step and g is the goal state, we are interested
    in seeing whether Q(S_f=g|pi) is high, meaning the agent learned to predict that following policy pi
    leads to the goal state at the end of the episode.

    Inputs:

    - file_data_path (string): file path where the data was stored (i.e. where log_data was saved);
    - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    - index_Si (integer): index i in S_i to select the random variable we are interested in;
    - value_Si (integer): value of S_i we care about;
    - save_dir (string): directory where to save the images.

    Outputs:

    - plot showing the evolution of Q(s|pi) for one s (or more) as the agent goes through the episodes.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    # Note 1: policy_state_prob is a numpy array of shape:
    # (num_runs, num_episodes, num_policies, num_states, num_max_steps).
    data = np.load(file_data_path, allow_pickle=True).item()
    num_states = data["num_states"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        policy_state_prob = data["policy_state_prob"][selected_runs]
    else:
        policy_state_prob = data["policy_state_prob"]

    # Averaging the state probabilities conditioned on policies over the runs
    avg_prob = np.mean(policy_state_prob, axis=0)  # .squeeze()

    # Retrieving the number of episodes
    num_episodes = avg_prob.shape[0]

    # Checking that index_Si is within an episode and that the typed-in value of S_i is one of the
    # legitimate values
    assert index_Si >= 0 and index_Si <= (
        num_steps - 1
    ), "Invalid random variable index."
    assert value_Si >= 0 and value_Si <= (
        avg_prob.shape[2] - 1
    ), "Invalid random variable value."

    # Selecting each policy data in turn and create the corresponding figure
    plt.figure()

    for p in range(num_policies):

        x = np.arange(1, num_episodes + 1)  # *num_steps)
        # Selecting the realization of S_i, e.g. the goal state, for which to represent the changes in
        # probability; we can decide whether to represent the change in probability for the selected
        # state at every time step for every episode (done in the next function) or just at the time
        # step i for which we would expect the probability to be high (still for every episode),
        # e.g. for the goal state that would be the final step in the episode (this is what is done here).
        # In other words, we are interested in looking at whether the agent infers to be in the right
        # state in the present moment.
        for s in range(num_states):
            # Setting the realization of S_i
            if s == value_Si:
                # Setting the time step, i.e. the i in S_i
                r_tstep = index_Si
                y = avg_prob[:, p, s, r_tstep].flatten()
                plt.plot(x, y, ".-", label=f"$Q(S_{r_tstep}={s}|\\pi_{p})$")

    plt.xticks(np.arange(x_ticks_estep, num_episodes + 1, step=x_ticks_estep))
    plt.xlabel("Episode")
    plt.ylabel("Probability", rotation=90)
    plt.legend(loc="upper right")
    plt.title("State Belief\n")
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_Qs_pi_prob_path.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_Qt_pi_prob(
    file_data_path,
    x_ticks_tstep,
    index_tSi,
    value_tSi,
    select_policy,
    save_dir,
    env_layout,
):
    """Plotting beliefs over states at a certain time step for every policy, i.e. Q(s|pi), averaged over
    the runs *and* as a function of the experiment steps.

    More specifically, we are retrieving the data related to the random variables (r.v.) S_i, where i is
    equal to the final step, and we plot the probability of one realization of that r.v. (goal state)
    conditioned on the policy and as a function of the experiment steps.
    For example, for S_f = g, where f is the final time step and g is the goal state, we are interested in
    seeing how Q(S_f=g|pi) changes during the experiment at *every* time step (as opposed to at just one
    time step as done in the previous function).

    Inputs:

    - file_data_path (string): file path where the data was stored (i.e. where log_data was saved);
    - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps;
    - index_tSi (integer): index i in S_i to select the random variable we are interested in;
    - value_tSi (integer): value of S_i we care about;
    - save_dir (string): directory where to save the images.

    Outputs:

    - plot showing the evolution of Q(s|pi) for one s (or more) as the agent goes through the episodes.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'last_tstep_prob'
    # Note 1: last_tstep_prob is a numpy array of shape:
    # (num_runs, num_episodes, num_policies, num_states, num_max_steps).
    data = np.load(file_data_path, allow_pickle=True).item()
    num_states = data["num_states"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        every_tstep_prob = data["every_tstep_prob"][selected_runs]
    else:
        every_tstep_prob = data["every_tstep_prob"]

        # Averaging the state probabilities conditioned on policies over the runs and episodes
    avg_prob = np.mean(every_tstep_prob, axis=0)  # .squeeze()
    # avg_prob = np.mean(avg_prob, axis=0).squeeze()
    # assert avg_prob.shape == (num_policies, num_states, num_steps)

    # Checking that index_Si is within an episode and that the typed-in value of S_i is one of the
    # legitimate values
    assert index_tSi >= 0 and index_tSi <= (
        num_steps - 1
    ), "Invalid random variable index."
    assert value_tSi >= 0 and value_tSi <= (
        avg_prob.shape[3] - 1
    ), "Invalid random variable value."

    # Retrieving the number of episodes
    num_episodes = avg_prob.shape[0]

    # Selecting each policy data in turn and create the corresponding figure
    for p in range(num_policies):

        plt.figure()
        x = np.arange(num_episodes * num_steps)
        # Selecting the realization g of S_i for which to represent the change in probability at *every*
        # step during the experiment.
        # Note 1: If g and i are the goal state and the final time step respectively, we are visualizing how the
        # agent's belief about where it will be at the last time step in an episode change/is updated
        # throughout the experiment.
        for s in range(num_states):
            if s == value_tSi:
                r_tstep = index_tSi
                y = avg_prob[:, r_tstep, p, s, :].flatten()
                plt.plot(x, y, ".-", label=f"$Q(S_{r_tstep}={s}|\\pi_{p})$")

                plt.xticks(
                    np.arange(0, (num_episodes * num_steps) + 1, step=x_ticks_tstep)
                )

                # tSi_ticks = np.arange(2, (num_episodes * num_steps) + 1, step=x_ticks_tstep)
                # for t in tSi_ticks:
                #     plt.axvline(x=t)

                plt.vlines(
                    x=np.arange(2, (num_episodes * num_steps) + 1, step=x_ticks_tstep),
                    ymin=0,
                    ymax=1.0,
                    colors="purple",
                    ls=":",
                    lw=0.5,
                    label="t = 2",
                )

                plt.xlabel("Step")
                plt.ylabel("Probability Mass", rotation=90)
                plt.legend(loc="upper right")
                plt.title(f"State Belief at Every Step\n")

        # Save figure and show
        plt.savefig(
            save_dir + "/" + f"{env_layout}_Qt_pi{p}_prob_path.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.show()
        plt.close()


def plot_oa_sequence(file_data_path, num_episodes, num_steps):
    """Plotting the sequences of observations and actions for one or more runs and one or more episodes.
    Inputs:
        - data_path (string): file path where observations and actions sequences were stored (i.e. where log_data was saved)
    Outputs:
        - plot showing....

    NOTE: THIS FUNCTION IS NOT USED, AND SHOULD BE REVISED
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'observations' and 'actual_action_sequence'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    obs = data["observations"]
    actions = data["actual_action_sequence"]
    # Selecting one or more runs and one or more episode to print/plot the observation and action sequences
    # Selecting run 0 and episode 30 (the last)
    obs_sequence = obs[3, 24, :, :].squeeze()
    print(obs_sequence)
    print(np.argmax(obs_sequence, axis=0))

    actions_sequence = actions[3, 24, :].squeeze()
    print(actions_sequence)
