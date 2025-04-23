"""
Definition of function(s) for plotting and saving data

Created on Wed Aug 5 16:16:00 2020
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

plt.rc("axes", titlesize=20, labelsize=18)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("legend", fontsize=16)
plt.rc("figure", titlesize=20)


def plot_action_seq(
    file_data_path,
    x_ticks_estep,
    policy_horizon,
    run_index,
    save_dir,
    env_layout,
):
    """
    Function to plot action sequences across episodes.
    """

    # Retrieving the data dictionary and extracting the content of various keys
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    reward_counts = data["reward_counts"]
    action_seqs = data["actual_action_sequence"]
    # Data for a specific run/agent (use index according to desired run/agent)
    run_actions = action_seqs[run_index]
    # Create episode index repeated for each action (x axis)
    episodes = np.repeat(np.arange(num_episodes), policy_horizon)
    # Create time steps, e.g. 0 and 1 for each episode if policy_horizon = 2 (y axis)
    timesteps_array = np.arange(policy_horizon)
    timesteps = np.tile(timesteps_array, num_episodes)  # Alternating time steps 0 and 1
    # Define colors based on action values (0=right, 1=up, 2=left, 3=down)
    action_colors = {0: "blue", 1: "red", 2: "green", 3: "orange"}
    colors = [action_colors[a] for a in run_actions.flatten()]

    # Plotting action sequences for each episode for run/agent 3
    plt.figure(figsize=(20, 3))
    plt.scatter(episodes, timesteps, c=colors, s=100, edgecolors="black")

    # plt.scatter(episodes, action_seqs[3], alpha=0.7, marker="o")

    plt.xlabel("Episode")
    plt.ylabel("Time step")
    plt.title("Actions sequences per episode")

    plt.xticks(np.arange(0, num_episodes, step=5))
    plt.yticks([1, 2, 3])  # Since we only have 2 time steps per episode
    plt.grid(True, linestyle="--", alpha=0.5)

    # Create a legend
    legend_patches = [
        Patch(color=color, label=f"Action {action}")
        for action, color in action_colors.items()
    ]
    plt.legend(handles=legend_patches, title="Actions")

    plt.savefig(
        save_dir + "/" + f"{env_layout}_actions_path_run{run_index}.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_reward_counts(file_data_path, x_ticks_estep, save_dir, env_layout):
    """
    Function to plot the average number of agents reaching the goal state in each episode.

    Inputs:

    - file_data_path (string): file path where all metrics have been stored;
    - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    - save_dir (string): directory where to save the images.
    """

    # Retrieving the data dictionary and extracting the content of various keys
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    reward_counts = data["reward_counts"]

    avg_rewards = np.mean(reward_counts, axis=0)
    std_rewards = np.std(reward_counts, axis=0)

    plt.figure()
    plt.plot(
        np.arange(num_episodes),
        avg_rewards,
        ".-",
        label="",
    )
    plt.xticks(np.arange(0, (num_episodes) + 1, step=x_ticks_estep))
    # Get current axis and set y-axis to percentages
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xlabel("Episode")
    # set y-axis from 0% to 100% (adding 0.5 to have more space at the top)
    plt.ylim(0, 1.05)
    plt.ylabel("Percentage of agents", rotation=90)
    # plt.legend(loc="upper right") # not needed for single line
    plt.title("Agents solving the task\n")

    # Confidence interval (if needed, uncomment the following lines)
    # plt.fill_between(
    #     np.arange(num_episodes),
    #     avg_rewards - (1.96 * std_rewards / np.sqrt(num_runs)),
    #     avg_rewards + (1.96 * std_rewards / np.sqrt(num_runs)),
    #     alpha=0.3,
    # )

    plt.savefig(
        save_dir + "/" + f"{env_layout}_path_agents_goal.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_pi_fe(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    select_policy,
    save_dir,
):
    """Plotting the free energy conditioned on a specific policy, F_pi, averaged over the runs.

    Inputs:

    - file_data_path (string): file path where the policy free energy data was stored
      (i.e. where log_data was saved);
    - step_fe_pi (integer): timestep used to plot the free energy;
    - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps;
    - save_dir (string): directory where to save the images.

    Outputs:

    - scatter plot of F_pi, showing its evolution as a function of the episodes' steps;
    - plot showing how F_pi at the last time step changes as the agent learns about the maze
      (episode after episode).
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]
    # print(data["pi_probabilities"].shape)

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        print(
            f"Number of runs with policy {select_policy} more probable: {len(selected_runs)}/{len(pi_runs)}"
        )
        pi_fe = data["pi_free_energies"][selected_runs]
    else:
        pi_fe = data["pi_free_energies"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, "Invalid step number."

    # Looping over the policies
    for p in range(num_policies):

        # Computing the mean (average) and std of one policy's free energies over the runs
        # TODO: handle rare case in which you train only for one episode, in that case squeeze()
        # will raise the exception
        avg_pi_fe = np.mean(pi_fe[:, :, p, :], axis=0)  # .squeeze()
        std_pi_fe = np.std(pi_fe[:, :, p, :], axis=0)  # .squeeze()
        # Making sure avg_pi_fe has the right dimensions
        assert avg_pi_fe.shape == (
            num_episodes,
            num_steps,
        ), f"Wrong dimenions. avg_pi_fe is of shape {avg_pi_fe.shape} but should be {(num_episodes, num_steps)}"
        # Plotting the free energy for every time step
        x1 = np.arange(num_episodes * num_steps)
        y1 = avg_pi_fe.flatten()

        plt.figure(figsize=(8, 24))
        plt.plot(x1, y1, ".-", label=f"Policy $\\pi_{p}$")
        plt.xticks(np.arange(0, (num_episodes * num_steps) + 1, step=x_ticks_tstep))
        plt.xlabel("Step")
        # plt.ylabel('Free Energy', rotation=90)
        plt.legend(loc="upper right")
        plt.title("Free Energy at Every Step\n")
        plt.savefig(
            save_dir + "/" + f"pi{p}_fes.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.show()
        plt.close()

        # Plotting the free energy at the last time step of every episode for all episodes
        # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
        x2 = np.arange(num_episodes)
        y2 = avg_pi_fe[:, step_fe_pi]

        fig, ax = plt.subplots()
        ax.plot(x2, y2, ".-", label=f"Policy $\\pi_{p}$")
        ax.set_xticks(np.arange(0, num_episodes + 1, step=x_ticks_estep))
        ax.set_xlabel("Episode")
        # ax.set_ylabel('Free Energy', rotation=90)
        ax.legend(loc="upper right")
        ax.set_title("Last Step Free Energy\n")
        ax.fill_between(
            x2,
            y2 - (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
            y2 + (1.96 * std_pi_fe[:, -1] / np.sqrt(num_runs)),
            alpha=0.3,
        )
        # Save figure and show
        plt.savefig(
            save_dir + "/" + f"pi{p}_fes_path_last_step.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.show()
        plt.close()


def plot_pi_fe_compare(
    file_data_path,
    step_fe_pi,
    x_ticks_estep,
    x_ticks_tstep,
    select_policy,
    save_dir,
    env_layout,
):
    """Function similar to plot_pi_fe() with the only difference that all policy-conditioned free energies,
    F_pi are plotted on the same figure for comparison (ideal when there aren't many runs or policies).

    Inputs:

    - file_data_path (string): file path where the policy free energy data was stored (i.e. where log_data was saved);
    - step_fe_pi (integer): timestep used to plot the free energy;
    - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    - x_ticks_tstep (integer): step for the ticks in the x axis when plotting as a function of total timesteps;
    - save_dir (string): directory where to save the images.

    Outputs:

    - scatter plot of F_pi, showing its evolution as a function of the episodes' steps;
    - plot showing how F_pi at the last time step changes as the agent learns about the maze
      (episode after episode).
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'pi_free_energies'
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
        pi_fe = data["pi_free_energies"][selected_runs]
    else:
        pi_fe = data["pi_free_energies"]

    # Checking that the step_fe_pi is within an episode
    assert (
        step_fe_pi >= 0 and step_fe_pi <= num_steps - 1
    ) or step_fe_pi == -1, "Invalid step number."

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", 16)

    fig, ax = plt.subplots(figsize=(24, 8))
    # Looping over the policies for Figure 1
    for p in range(num_policies):

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

        ax.plot(x1, y1, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")

    # Completing drawing axes for Figure 1
    ax.set_xticks(np.arange(0, (num_episodes * num_steps) + 1, step=x_ticks_tstep))
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
        save_dir + "/" + f"{env_layout}_policies_fe_path_every_step.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()
    # Looping over the policies for Figure 2
    policy_index_offset = 48
    for p in range(16):

        # Computing the mean (average) and std of one policy's free energies over the runs
        # TODO: handle rare case in which you train only for one episode, in that case squeeze()
        # will raise the exception
        avg_pi_fe = np.mean(
            pi_fe[:, :, p + policy_index_offset, :], axis=0
        )  # .squeeze()
        std_pi_fe = np.std(
            pi_fe[:, :, p + policy_index_offset, :], axis=0
        )  # .squeeze()
        # Making sure avg_pi_fe has the right dimensions
        # print(avg_pi_fe.shape)
        # print((num_episodes, num_steps))
        assert avg_pi_fe.shape == (num_episodes, num_steps), "Wrong dimenions!"

        # Plotting the free energy at the last time step of every episode for all episodes
        # Note 1: another time step can be chosen by changing the index number, i, in avg_pi_fe[:, i]
        x2 = np.arange(num_episodes)
        y2 = avg_pi_fe[:, step_fe_pi]

        ax.plot(
            x2, y2, ".-", color=cmap(p), label=f"$\\pi_{{{p + policy_index_offset}}}$"
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
    ax.set_xticks(np.arange(0, num_episodes + 1, step=x_ticks_estep))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Free energy", rotation=90)
    ax.set_ylim(2, 8)
    ax.legend(
        title="Policies",
        ncol=4,
        title_fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
    )

    if step_fe_pi != -1:
        step_num = f"{step_fe_pi + 1}"
    else:
        step_num = f"{num_steps}"

    ax.set_title(f"Policy-conditioned free energy at step {step_num}\n")

    fig.savefig(
        save_dir + "/" + f"{env_layout}_policies_fe_path_step{step_num}.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_total_fe(
    file_data_path,
    step_fe,
    x_ticks_estep,
    x_ticks_tstep,
    select_policy,
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
    - save_dir (string): directory where to save the images.

    Outputs:

    - scatter plot of F, showing its evolution as a function of the episodes' steps;
    - plot showing how F at the last time step changes as the agent learns about the maze (episode after episode).
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
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
    plt.xticks(np.arange(0, (num_episodes * num_steps) + 1, step=x_ticks_tstep))
    plt.xlabel("Step")
    plt.ylabel("Total free energy", rotation=90)
    # plt.legend(loc="upper right") # No need for legend
    plt.title("Every-step total free energy across episodes\n")
    # plt.show()
    plt.close()

    # Plotting the total free energy at the last time step of every episode
    # Note 1: another time step can be chosen by changing the index number, i, in avg_total_fe[:, i]
    x2 = np.arange(num_episodes)
    y2 = avg_total_fe[:, step_fe]
    if step_fe != -1:
        step_num = f"{step_fe + 1}"
    else:
        step_num = f"{num_steps}"

    fig, ax = plt.subplots()
    ax.plot(x2, y2, ".-", label="free energy")
    ax.set_xticks(np.arange(0, num_episodes + 1, step=x_ticks_estep))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Free energy", rotation=90)
    ax.set_ylim(0, 10)
    # ax.legend(loc="upper right") # No need for legend
    ax.set_title(f"Free energy at step {step_num}\n")
    ax.fill_between(
        x2,
        y2 - (1.96 * std_total_fe[:, -1] / np.sqrt(num_runs)),
        y2 + (1.96 * std_total_fe[:, -1] / np.sqrt(num_runs)),
        alpha=0.3,
    )
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_total_fe_path_step{step_num}.jpg",
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
        plt.plot(x, y, ".-", label=f"$\\pi_{{{p}}}$")

    plt.xticks(np.arange(0, (num_episodes * num_steps) + 1, step=x_ticks_tstep))
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
    plt.show()


def plot_pi_prob_first(
    file_data_path,
    x_ticks_estep,
    x_ticks_tstep,
    select_policy,
    save_dir,
    env_layout,
):
    """Function to plot the probability over policies, Q(pi), averaged over the runs at the first time
    step of each episode during the experiment.

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
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
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
    cmap = plt.cm.get_cmap("tab20", 16)

    plt.figure()
    x = np.arange(num_episodes)

    policy_index_offset = 16
    for p in range(16):
        y = avg_pi_prob[:, p + policy_index_offset].flatten()
        std = std_pi_prob[:, p + policy_index_offset].flatten()
        plt.plot(
            x, y, ".-", color=cmap(p), label=f"$\\pi_{{{p + policy_index_offset}}}$"
        )

        # plt.fill_between(
        #     x,
        #     y - (1.96 * std / np.sqrt(num_runs)),
        #     y + (1.96 * std / np.sqrt(num_runs)),
        #     alpha=0.3,
        # )

    plt.xticks(np.arange(0, num_episodes + 1, step=x_ticks_estep))
    plt.xlabel("Episode")
    plt.ylabel("Probability mass", rotation=90)
    # plt.ylim(0.02, 0.15)
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
    plt.title("First-step policy probability\n")
    # Save figure and show
    plt.savefig(
        save_dir
        + "/"
        + f"{env_layout}_first_pi_probs_path_offset{policy_index_offset}.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_efe(file_data_path, select_policy, save_dir, env_layout, select_step=None):
    """Function to plot the expected free energy, EFE, for a given policy over all the steps
    averaged over the runs.

    Inputs:

    - file_data_path (string): file path where the total free energy data was stored (i.e. where log_data
      was saved)
    - select_policy (integer): index to include/exclude some policies from data plotting;
    - select_step (integer): index to select same step at each episode to plot corresponding EFE value;
    - save_dir (string): directory where to save the images.

    Outputs:

    - plot showing evolution of the expected free energy
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'total_free_energies'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data["num_episodes"]
    num_policies = data["num_policies"]
    num_steps = data["num_steps"]

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

    plt.figure()
    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", num_policies)

    if select_step == None:
        x_label = "Step"
        plot_title = "Expected free energy at every step"
        # Plotting all the time steps across episodes
        for p in range(num_policies):
            x = np.arange(num_episodes * num_steps)
            # x = np.arange(1*(num_steps-1))
            y = avg_efe[:, p, :].flatten()
            # y = np.reshape(-avg_efe[2, p, 0:-1], (1*(num_steps-1)))
            plt.plot(x, y, ".-", color=cmap(p), label=f"Policy $\\pi_{{{p}}}$")

    else:
        x_label = "Episode"
        plot_title = f"Expected free energy at step {select_step}"
        # Plotting EFE at single time step for each episode
        for p in range(num_policies):
            x = np.arange(num_episodes)
            y = avg_efe[:, p, select_step].flatten()
            plt.plot(x, y, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")

    plt.xlabel(f"{x_label}")
    plt.ylabel("Expected free energy", rotation=90)
    # plt.ylim(3, 6)
    plt.legend(
        title="Policies",
        title_fontsize=16,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
    )
    # plt.legend(loc="upper right")
    plt.title(f"{plot_title}\n")
    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_efe_path.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_efe_comps(
    file_data_path, select_policy, save_dir, env_layout, num_tsteps=None
):
    """Plotting the expected free energy components (ambiguity, risk and novelty) for a given policy over
    all the steps averaged over the runs.

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
        efe = data["expected_free_energies"][selected_runs]
        efe_ambiguity = data["efe_ambiguity"][selected_runs]
        efe_risk = data["efe_risk"][selected_runs]
        efe_Anovelty = data["efe_Anovelty"][selected_runs]
        efe_Bnovelty = data["efe_Bnovelty"][selected_runs]
    else:
        efe = data["expected_free_energies"]
        efe_ambiguity = data["efe_ambiguity"]
        efe_risk = data["efe_risk"]
        efe_Anovelty = data["efe_Anovelty"]
        efe_Bnovelty = data["efe_Bnovelty"]

    # Averaging the expected free energies and their components over the runs
    avg_efe = np.mean(efe, axis=0)  # .squeeze()
    avg_efe_ambiguity = np.mean(efe_ambiguity, axis=0)  # .squeeze()
    std_efe_ambiguity = np.std(efe_ambiguity, axis=0)  # .squeeze()
    avg_efe_risk = np.mean(efe_risk, axis=0)  # .squeeze()
    std_efe_risk = np.std(efe_risk, axis=0)  # .squeeze()
    avg_efe_Anovelty = np.mean(efe_Anovelty, axis=0)  # .squeeze()
    std_efe_Anovelty = np.std(efe_Anovelty, axis=0)  # .squeeze()
    avg_efe_Bnovelty = np.mean(efe_Bnovelty, axis=0)  # .squeeze()
    std_efe_Bnovelty = np.std(efe_Bnovelty, axis=0)  # .squeeze()
    # Making sure efe has the right dimensions
    assert avg_efe.shape == (num_episodes, num_policies, num_steps), "Wrong dimenions!"
    assert avg_efe_ambiguity.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"
    assert avg_efe_risk.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"
    assert avg_efe_Anovelty.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"
    assert avg_efe_Bnovelty.shape == (
        num_episodes,
        num_policies,
        num_steps,
    ), "Wrong dimenions!"

    # Pre-generate distinct colors
    cmap = plt.cm.get_cmap("tab20", num_policies)

    ### Plotting risk and B-novelty in separate figures

    # Figure for RISK
    fig_1, axes_1 = plt.subplots(
        1,
        1,
        figsize=(6, 5),
        gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )  # 1 row, 1 column
    # Figure for B-novelty
    fig_2, axes_2 = plt.subplots(
        1,
        1,
        figsize=(6, 5),
        gridspec_kw={"wspace": 0.4},  # Set figure size and horizontal spacing
    )  # 1 row, 1 column

    for p in range(num_policies):

        # Plotting all time steps unless a specific time step is provided
        if num_tsteps != None:
            x_label = "Episode"
            title_label = f"step {num_tsteps}"
            x = np.arange(num_episodes)
            # Risk
            y_efer = avg_efe_risk[:, p, num_tsteps].flatten()
            stdy_efer = std_efe_risk[:, p, num_tsteps].flatten()
            # B-novelty
            y_efeB = avg_efe_Bnovelty[:, p, num_tsteps].flatten()
            stdy_efeB = std_efe_Bnovelty[:, p, num_tsteps].flatten()
        else:
            x_label = "Step"
            title_label = "all steps"
            x = np.arange(num_episodes * num_steps)
            # Risk
            y_efer = avg_efe_risk[:, p, :].flatten()
            stdy_efer = std_efe_risk[:, p, :].flatten()
            # Ambiguity
            y_efeB = avg_efe_Bnovelty[:, p, :].flatten()
            stdy_efeB = std_efe_Bnovelty[:, p, :].flatten()

        # Plot on figures
        axes_1.plot(x, y_efer, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")
        axes_2.plot(x, y_efeB, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")
        # Create confidence intervals for Risk and B-novelty (if needed, uncomment following lines)
        # axes_1[0].fill_between(
        #     x,
        #     y_efer - (1.96 * stdy_efer / np.sqrt(num_runs)),
        #     y_efer + (1.96 * stdy_efer / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

        # axes_1[1].fill_between(
        #     x,
        #     y_efeB - (1.96 * stdy_efeB / np.sqrt(num_runs)),
        #     y_efeB + (1.96 * stdy_efeB / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

    axes_1.set_xlabel(x_label)
    axes_2.set_xlabel(x_label)
    axes_1.set_ylabel("Risk", rotation=90)
    # axes_1.set_ylim(3, 6)
    axes_2.set_ylabel("B-novelty", rotation=90)
    # axes_2.set_ylim(0.3, 0.5)

    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes_1.get_legend_handles_labels()
    # Put the legend at the bottom center
    fig_1.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )

    # Gather handles and labels from one of the axes to create common legend
    handles, labels = axes_2.get_legend_handles_labels()
    # Put the legend at the bottom center
    fig_2.legend(
        handles,
        labels,
        title="Polices",
        title_fontsize=16,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
    )

    axes_1.set_title(f"Risk at {title_label}\n")
    axes_2.set_title(f"B-novelty at {title_label}\n")
    # Save figures
    fig_1.savefig(
        save_dir + "/" + f"{env_layout}_efe_path_risk.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig_2.savefig(
        save_dir + "/" + f"{env_layout}_efe_path_bnov.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig_1)
    plt.close(fig_2)

    # Plotting ambiguity and A-novelty in separate figures
    fig_3, axes_3 = plt.subplots(1, 1, figsize=(6, 5))  # 1 row, 1 column
    fig_4, axes_4 = plt.subplots(1, 1, figsize=(6, 5))  # 1 row, 1 column

    for p in range(num_policies):
        # Plotting all time steps unless a specific time step is provided
        if num_tsteps != None:
            x_label = "Episode"
            title_label = f"step {num_tsteps}"
            x = np.arange(num_episodes)
            # Ambiguity
            y_efea = avg_efe_ambiguity[:, p, num_tsteps].flatten()
            stdy_efea = std_efe_ambiguity[:, p, num_tsteps].flatten()
            # A-novelty
            y_efeA = avg_efe_Anovelty[:, p, num_tsteps].flatten()
            stdy_efeA = std_efe_Anovelty[:, p, num_tsteps].flatten()
        else:
            x_label = "Step"
            title_label = "all steps"
            x = np.arange(num_episodes * num_steps)
            # Ambiguity
            y_efea = avg_efe_ambiguity[:, p, :].flatten()
            stdy_efea = std_efe_ambiguity[:, p, :].flatten()
            # A-novelty
            y_efeA = avg_efe_Anovelty[:, p, :].flatten()
            stdy_efeA = std_efe_Anovelty[:, p, :].flatten()

        # Plot subplots
        axes_3.plot(x, y_efea, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")
        axes_4.plot(x, y_efeA, ".-", color=cmap(p), label=f"$\\pi_{{{p}}}$")
        # Confidence intervals for ambiguity and A-novelty
        # axes_2[0].fill_between(
        #     x,
        #     y_efea - (1.96 * stdy_efea / np.sqrt(num_runs)),
        #     y_efea + (1.96 * stdy_efea / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )
        # axes_2[1].fill_between(
        #     x,
        #     y_efeA - (1.96 * stdy_efeA / np.sqrt(num_runs)),
        #     y_efeA + (1.96 * stdy_efeA / np.sqrt(num_runs)),
        #     color=cmap(p),
        #     alpha=0.3,
        # )

    axes_3.set_xlabel(x_label)
    axes_4.set_xlabel(x_label)
    axes_3.set_ylabel("Ambiguity", rotation=90)
    axes_4.set_ylabel("A-novelty", rotation=90)
    axes_3.legend(
        title="Policies",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
    )
    axes_4.legend(
        title="Policies",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
    )
    axes_3.set_title(f"Ambiguity at {title_label} across episodes\n")
    axes_4.set_title(f"A-novelty at {title_label} across episodes\n")
    # Save figure
    fig_3.savefig(
        save_dir + "/" + f"{env_layout}_efe_path_amb.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig_4.savefig(
        save_dir + "/" + f"{env_layout}_efe_path_anov.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close(fig_3)
    plt.close(fig_4)


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
        x = np.arange(num_episodes)
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

        x = np.arange(num_episodes)  # *num_steps)
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

    plt.xticks(np.arange(0, num_episodes + 1, step=x_ticks_estep))
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


def plot_so_mapping(
    file_data_path,
    x_ticks_estep,
    state_A,
    select_policy,
    save_dir,
    env_layout,
):
    """Plotting state-observation mappings (emission probabilities), i.e., matrix A of size
    (num_states, num_states), averaged over the runs.

    The columns of A are categorical distributions so their elements must sum to one, e.g., column 0 (zero)
    tells you the probability of the agent believing to be in a certain state when it is in state 0 (zero),
    e.g. P(O=0|S=0). If the agent has sound beliefs, then in state 0 (zero) it should believe to be in state 0.
    In other words, matrix A should approximate an identity matrix.

    Inputs:

    - file_data_path (string): file path where the data was stored (i.e. where log_data was saved);
    - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    - state_A (integer): index i to slice a certain column of A to represent the evolution of Q(O|S_i);
    - save_dir (string): directory where to save the images.

    Outputs:

    - plot showing the emission probabilities for a specific state, S_i (i.e., a column of A), over the entire
      experiment;
    - heatmap showing matrix A at the end of the experiment to see what the agent learned.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'so_mappings'
    data = np.load(file_data_path, allow_pickle=True).item()
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
    x = np.arange(num_episodes)
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

    ax1.set_xticks(np.arange(0, num_episodes + 1, step=x_ticks_estep))
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
    plt.show()

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
        save_dir + "/" + f"{env_layout}_so_map_path.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()


def plot_transitions(
    file_data_path,
    x_ticks_estep,
    state_B,
    action_B,
    select_policy,
    save_dir,
    env_layout,
):
    """Plotting transition probabilities, i.e., matrices B (one for each available action) of size
    (num_states, num_states), averaged over the runs.

    The columns of a B matrix are categorical distributions so their elements must sum to one, e.g.,
    column 0 (zero) of B_up (the transition matrix for action up) gives the agent the probabilities
    of landing in the various states by going up from state 0 (zero). If the agent has learned correct
    transitions (and the environment is not stochastic), then going up from state 0 (zero) should lead
    to a specific state. In other words, columns of matrices B should have all values close to 0 except
    for one close to 1.

    Inputs:

    - file_data_path (string): file path where the data was stored (i.e. where log_data was saved);
    - x_ticks_estep (integer): step for the ticks in the x axis when plotting as a function of episode number;
    - action_B (integer): index to select the action for which to represent the transition probabilities;
    - save_dir (string): directory where to save the images;

    Outputs:

    - plot showing the transitions probabilities for a specific state and action (i.e., a column of a B matrix)
      over the entire experiment;
    - heatmap showing matrix B for a certain action at the end of the experiment to see what the agent learned.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'transition_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_runs = data["num_runs"]
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        transitions_prob = data["transition_prob"][selected_runs]
    else:
        transitions_prob = data["transition_prob"]

    # Computing the mean (avg) and std of the transition probabilities over the runs
    avg_transitions_prob = np.mean(transitions_prob, axis=0)  # .squeeze()
    std_transitions_prob = np.std(transitions_prob, axis=0)  # .squeeze()

    # Making sure state_B and action_B are valid values to select a matrix B and slice it
    assert action_B >= 0 and action_B <= 3, "Invalid action index."
    assert state_B >= 0 and state_B <= num_states - 1, "Invalid state index."
    # Selecting the avg and std transition probabilities for a specific action (a=0, a=1, a=2, a=3)
    # throughout the experiment, i.e. for B_a
    a = action_B
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
    x = np.arange(num_episodes)
    y_data = transition_state[:, :]

    fig1, ax1 = plt.subplots()
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
            label=f"$P(S_{{t+1}}={c}|S_t={s}, \\pi_t={a})$",
        )
        # ax1.fill_between(x, y-std_transition_state[:, c], y+std_transition_state[:, c], alpha=0.3)
        ax1.fill_between(
            x,
            y - (1.96 * std_transition_state[:, c] / np.sqrt(num_runs)),
            y + (1.96 * std_transition_state[:, c] / np.sqrt(num_runs)),
            alpha=0.3,
        )

    ax1.set_xticks(np.arange(0, num_episodes + 1, step=x_ticks_estep))
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Probability Mass", rotation=90)
    # ax1.legend([f'$P(S_{{t+1}}={i}|S_t={s}, \\pi_t={a})$' for i in range(num_states)], loc='upper right')
    ax1.legend(loc="upper right")
    ax1.set_title(f"Transition Probabilities from State {s} for Action {a}\n")
    plt.savefig(
        save_dir + "/" + f"{env_layout}_tr_probs_state{s}_action{a}_path.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()

    # Heatmap of the transition probabilites from all states for action all the actions at the end
    # of the experiment; the actions range from 0 to 3 (included)
    for a in range(0, 4):
        fig2, ax2 = plt.subplots()
        im = ax2.imshow(avg_transitions_prob[-1, a, :, :].squeeze(), vmin=0, vmax=1)

        ax2.set_yticks(np.arange(num_states))
        ax2.set_yticklabels(np.arange(num_states))
        ax2.set_xticks(np.arange(num_states))
        ax2.set_xticklabels(np.arange(num_states))

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
        ax2.set_title(f"Transition Matrix for Action {a}")

        # Save figure and show
        plt.savefig(
            save_dir + "/" + f"{env_layout}_transmap_path_a{a}.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.show()
        plt.close()


def plot_Qs_pi_final(file_data_path, select_policy, save_dir, env_layout):
    """Visualising the Q(S_i|pi) for each policy at the end of the experiment, where i is in
    [0,...,num_steps-1] and indicates the time step during an episode. Note that the the Q(S_i|pi)
    are categorical distributions telling you the state beliefs the agent has for each episode's time step.

    Inputs:

    - file_data_path (string): file path where transition probabilities were stored
      (i.e. where log_data was saved);
    - save_dir (string): directory where to save the images;

    Outputs:

    - heatmap showing the Q(S_i|pi) for each policy at the end of the experiment.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]
    policies = data["policies"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        Qs_pi_prob = data["policy_state_prob"][selected_runs]
    else:
        Qs_pi_prob = data["policy_state_prob"]

    # Averaging the Q(S|pi) over the runs
    avg_Qspi = np.mean(Qs_pi_prob, axis=0)  # .squeeze()
    # Selecting the probabilities for the last episode only
    last_episode_Qspi = avg_Qspi[-1, :, :, :]

    # Heatmap of the Q(s|pi) for every policy at the end of the experiment (after last episode)
    for p in range(last_episode_Qspi.shape[0]):

        # Creating figure and producing heatmap for policy p
        fig, ax = plt.subplots()
        fig.set_figwidth(5)
        fig.set_figheight(6)
        X, Y = np.meshgrid(np.arange(num_steps), np.arange(num_states))
        im = ax.pcolormesh(X, Y, last_episode_Qspi[p, :, :].squeeze(), shading="auto")

        # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
        qspi_labels = []
        for s in range(num_steps):
            # qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
            qspi_labels.append(rf"$Q(s_{{{s}}}|\pi_{{{p}}})$")

        ax.set_xticks(np.arange(num_steps) - 0.5, minor=True)
        ax.set_xticklabels(qspi_labels, minor=True)
        ax.tick_params(
            which="minor", top=True, bottom=False, labeltop=True, labelbottom=False
        )
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)

        plt.setp(ax.get_xticklabels(minor=True), ha="left", rotation=30)

        # Loop over data dimensions and create text annotations.
        # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond
        # to y-x Cartesian coordinates
        for i in range(num_states):
            for j in range(num_steps):
                text = ax.text(
                    j,
                    i,
                    f"{last_episode_Qspi[p, i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="m",
                    fontsize="medium",
                )

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(num_steps))
        ax.set_xlabel("Time Step")
        ax.invert_yaxis()
        ax.set_yticks(np.arange(num_states))
        ax.set_ylabel("State", rotation=90)
        ax.set_title(
            f"Last-step State Beliefs for Policy $\\pi_{{{p}}}$: [{', '.join(map(str, policies[p]))}]"
        )

        # Save figure and show
        plt.savefig(
            save_dir + "/" + f"{env_layout}_Qs_pi{p}_final_path.jpg",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # plt.show()
        plt.close()


def plot_Qs_pi_first(file_data_path, select_policy, save_dir, env_layout):
    """Visualising the Q(S_i|pi) for each policy at the first time time step of the last episode, where i
    is in [0,...,num_steps-1] and indicates the time step during an episode. Note that the the Q(S_i|pi)
    are categorical distributions telling you the state beliefs the agent has for each episode's time step.

    Inputs:

    - file_data_path (string): file path where transition probabilities were stored
      (i.e. where log_data was saved);
    - save_dir (string): directory where to save the images;

    Outputs:

    - heatmap showing the Q(S_i|pi) for each policy at the end of the experiment.
    """

    # Retrieving the data dictionary and extracting the content of required keys, e.g. 'policy_state_prob'
    data = np.load(file_data_path, allow_pickle=True).item()
    num_episodes = data["num_episodes"]
    num_steps = data["num_steps"]
    num_states = data["num_states"]
    policies = data["policies"]

    # Ignoring certain runs depending on the final probability of a certain policy, if corresponding argument
    # was passed through the command line
    if select_policy != -1:

        pi_runs = data["pi_probabilities"][:, -1, select_policy, -1]
        selected_runs = (pi_runs > 0.5).nonzero()[0]
        Qs_pi_prob = data["policy_state_prob_first"][selected_runs]
    else:
        Qs_pi_prob = data["policy_state_prob_first"]

    # Averaging the Q(S|pi) over the runs
    avg_Qspi = np.mean(Qs_pi_prob, axis=0)  # .squeeze()
    # Selecting the probabilities for the last episode only
    last_episode_Qspi = avg_Qspi[-1, :, :, :]

    # Heatmap of the Q(s|pi) for every policy at the end of the experiment (after last episode)
    for p in range(last_episode_Qspi.shape[0]):

        # Creating figure and producing heatmap for policy p
        fig, ax = plt.subplots()
        fig.set_figwidth(5)
        fig.set_figheight(6)
        X, Y = np.meshgrid(np.arange(num_steps), np.arange(num_states))
        im = ax.pcolormesh(X, Y, last_episode_Qspi[p, :, :].squeeze(), shading="auto")

        # Setting top minor ticks to separate the different Q(s|pi) and adding corresponding labels
        qspi_labels = []
        for s in range(num_steps):
            # qspi_labels = [r'$Q(s_{0}|\pi)$', r'$Q(s_{1}|\pi)$', r'$Q(s_{2}|\pi)$', r'$Q(s_{3}|\pi)$', r'$Q(s_{4}|\pi)$', r'$Q(s_{5}|\pi)$', r'$Q(s_{6}|\pi)$']
            qspi_labels.append(rf"$Q(s_{{{s}}}|\pi_{{{p}}})$")

        ax.set_xticks(np.arange(num_steps) - 0.5, minor=True)
        ax.set_xticklabels(qspi_labels, minor=True)
        ax.tick_params(
            which="minor", top=True, bottom=False, labeltop=True, labelbottom=False
        )
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)

        plt.setp(ax.get_xticklabels(minor=True), ha="left", rotation=30)

        # Loop over data dimensions and create text annotations.
        # Note 1: i, j are inverted in ax.text() because row-column coordinates in a matrix correspond
        # to y-x Cartesian coordinates
        for i in range(num_states):
            for j in range(num_steps):
                text = ax.text(
                    j,
                    i,
                    f"{last_episode_Qspi[p, i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="m",
                    fontsize="medium",
                )

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(num_steps))
        ax.set_xlabel("Time Step")
        ax.invert_yaxis()
        ax.set_yticks(np.arange(num_states))
        ax.set_ylabel("State", rotation=90)
        ax.set_title(
            f"First-step State Beliefs for Policy $\\pi_{{{p}}}$: [{', '.join(map(str, policies[p]))}]"
        )

        # Save figure and show
        plt.savefig(
            save_dir + "/" + f"{env_layout}_Qs_pi{p}_start_path.jpg",
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


def plot_state_visits(file_path, v_len, h_len, select_policy, save_dir, env_layout):
    """Plotting the state visits heatmap showing the frequency with which the agent has visited the
    maze's tiles.

    Inputs:

    - file_path: the file path where the cumulative reward data was stored while running the experiment;
    - v_len (integer): vertical length of the environment;
    - h_len (integer): horizontal length of the environment.

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

    # Reshaping the state counts vector into a matrix so as to visualise the maze
    env_matrix = np.reshape(tot_sv, (v_len, h_len))

    # Heatmap of the state counts over all the experiment's episodes
    percentage_sv = env_matrix / total_steps * 100
    fig, ax = plt.subplots()
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

    ax.set_title(f"State-access frequency in the experiment", pad=20)

    # Save figure and show
    plt.savefig(
        save_dir + "/" + f"{env_layout}_sv_path.jpg",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()
