```{=org}
#+STARTUP: overview indent
```
# CleanAIF

The code in this repository can be used to train agents with active
inference in discrete state-spaces, e.g., a grid world, and visualize
various learning metrics (e.g., free energy and expected free energy).

# Overview

Active inference is a computational framework for adaptive behaviour
according to which intelligent agents follow an imperative to minimize
variational and expected free energies given a generative model of the
environment. In the active inference algorithm, minimising variational
free energy corresponds to perception, i.e., inferring latent states
from observations, whereas minimising expected free energy implements
planning, i.e., inferring the best sequence of action (policy) to
pursue.

In other words, the framework provides a recipe to solve a partially
observable Markov decision process (POMDP). The repository currently
offers the implementation of two main inference agents, each with its
own main script, utility functions scripts, and configuration scripts
(loosely inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl)).
These agents can be trained in a custom
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment
(for episodic tasks).

## Agents

The implemented agents can be described and distinguished as follow:

1.  Action-aware agents

    - these agents have access to the sequence of actions they performed
      in the past, e.g., $(a_{1:\tau-1})$ where $\tau$ represent the
      present time step
    - perceptual inference corresponds to inferring the divergent future
      state-trajectory afforded by each policy $\pi_{i}$
    - policy inference involves updating the probability over policies
      by differentiating them only with respect to their future
      consequences (as the agent does not have uncertainty about past
      motor actions)

2.  Action-unaware agents

    - these agents *lack* access to the sequence of actions they
      performed in the past
    - perceptual inference involves inferring how consistent the past
      state-trajectory of each policy is with the collected
      observations, in addition to inferring the future state-trajectory
      afforded by each policy $\pi_{i}$
    - policy inference combines the evidence for each policy with the
      expected free energy to derive an update of the policy
      probabilities, guiding then action selection

## Environments

The agents can be trained in different grid worlds by selecting the
corresponding configuration file (see examples of command-line
instructions below). The available environments include:

- a T-maze, either with 4 or 5 states/tiles
- a Y-maze with 6 states/tiles
- and a square grid world, either with 9 or 16 states/tiles.

The configuration files can be modified to set the agent\'s goal in the
environment, the type of preferences, the length of a policy, and the
number of policies an agent will consider (see configuration files in
`src/clean_aif/config_agents`{.verbatim} folder).

# Installation

The code in this repo allows you to train an active inference agent in
various grid world environments based on
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

This guide assumes that the user has installed
[Git](https://git-scm.com/downloads) and Python (through
[Anaconda](https://www.anaconda.com/download) or similar distributions)
into their system. Also, the package depends on a custom Gymnasium
environment that needs to be installed separately, follow the
instructions at this
[link](https://github.com/FilConscious/cust-gridworlds) before you do
anything else.

After installing the Gymnasium environment:

1.  Open a terminal and move into the local folder/working directory
    where you installed the custom Gymnasium environment:

    ``` bash
    cd /home/working-dir
    ```

2.  Clone the Github repository either with:

    ``` bash
    git clone https://github.com/FilConscious/cleanAIF.git
    ```

    if you do not have a SSH key, or with:

    ``` bash
    git clone git@github.com:FilConscious/cleanAIF.git
    ```

if you have one (see the [GitHub SSH
docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
for info on SSH keys). After cloning the repo, you will have the folder
`./cleanAIF`{.verbatim} in your working directory which you can rename
as you see fit. Inside the folder you can find the source code, the
`README.org` and other files.

1.  Create a Python virtual environment or conda environment with a
    recent version of Python, e.g.:

    ``` bash
    conda create --name my-env python=3.12
    ```

2.  Activate the environment:

    ``` bash
    conda activate my-env
    ```

3.  Move inside the local repo folder `./cleanAIF`{.verbatim} (if not
    done already):

    ``` bash
    cd /home/working-dir/cleanAIF
    ```

4.  Install the package with the following command:

    ``` bash
    pip install --editable .
    ```

The last step installs the package, together with other required
libraries/packages, in editable mode. This means that it is possible to
modify your working/local copy of the algorithm and used it immediately,
without going through the whole process of building the package and
installing it again.

# How to run an experiment

1.  Move into the local repo directory:

    ``` bash
    cd /home/working-dir/cleanAIF
    ```

2.  Activate the conda environment:

    ``` bash
    conda activate myenv
    ```

3.  For training an agent in the T-maze (with 4 states) you can execute
    the following commands from the terminal:

    - for the action-unaware agent:

      ``` bash
      main_aif_au --exp_name aif_au --gym_id gridworld-v1 --env_layout tmaze3 --num_runs 10 --num_episodes 100 --num_steps 3 --inf_steps 10 --action_selection kd -lB --num_policies 16 --pref_loc all_goal
      ```

    - for the action-aware agent:

      ``` bash
      main_aif_au --exp_name aif_aa --gym_id gridworld-v1 --env_layout tmaze3 --num_runs 10 --num_episodes 100 --num_steps 3 --inf_steps 10 --action_selection kd -lB --num_policies 16 --pref_loc all_goal
      ```

4.  For visualising metrics of one experiment:

    ``` bash
    vis_aif -gid gridworld-v1 -el tmaze3 -nexp 1 -rdir episodic_e100_pol16 -fpi 0 1 2 -i 4 -v 8 -ti 4 -tv 8 -vl 3 -hl 3 -xtes 20 -ph 2 -selrun 0 -npv 16 -sb 4 -ab 0 1 2 3
    ```

A detailed explanation of each command-line argument can be found in the
main script for each agent, e.g.
`src/clean_aif/agents/aif_au.py`{.verbatim}, and in the visualisation
script, i.e., `src/clean_aif/vis_plots`{.verbatim}.

# How to reproduce results in (Torresan et al. 2025) {#how-to-reproduce-results-in-citetorresan2025a}

We include below the command-line instructions to run the experiments
and obtain the plots discussed in (Torresan et al. 2025). To obtain the
same results, it is crucial to specify the configuration files for each
agent in a way that matches the experiments\' task. For this and further
details on the theory and algorithmic implementations supporting the
experiments, please see (Torresan et al. 2025).

Note: the command line instruction `main_aif_aa_pi_cutoff` (see below)
is used to train a variation of the action-aware agent that does not
plan beyond the length of an episode, and allows for a fairer comparison
with the action-unaware agent.

## Experiment 1: 4-step t-maze

For the action-unaware agent, execute:

``` bash
main_aif_au --exp_name aif_au --gym_id gridworld-v1 --env_layout tmaze4 --num_runs 10 --num_episodes 100 --num_steps 4 --inf_steps 10 --action_selection kd -lB --num_policies 64 --pref_loc all_goal
```

For the action-aware agent, execute:

``` bash
main_aif_aa_pi_cutoff --exp_name aif_aa --gym_id gridworld-v1 --env_layout tmaze4 --num_runs 10 --num_episodes 100 --num_steps 4 --inf_steps 10 --action_selection kd -lB --num_policies 64 --pref_loc all_goal
```

To visualise and compare metrics of the two agents, execute:

``` bash
vis_aif -gid gridworld-v1 -el tmaze4 -nexp 2 -rdir episodic_e100_pol16_maxinf10_learnB -fpi 0 1 2 3 -i 4 -v 8 -ti 4 -tv 8 -vl 3 -hl 3 -xtes 20 -ph 3 -selrun 0 -selep 24 49 74 99 -npv 16 -sb 4 -ab 0 1 2 3
```

## Experiment 2: 5-step grid world

For the action-unaware agent, execute:

``` bash
main_aif_au --exp_name aif_au --gym_id gridworld-v1 --env_layout gridw9 --num_runs 10 --num_episodes 180 --num_steps 5 --inf_steps 10 --action_selection kd -lB --num_policies 256 --pref_loc all_goal
```

For the action-aware agent, execute:

``` bash
main_aif_aa_pi_cutoff --exp_name aif_aa --gym_id gridworld-v1 --env_layout gridw9 --num_runs 10 --num_episodes 180 --num_steps 5 --inf_steps 10 --action_selection kd -lB --num_policies 256 --pref_loc all_goal
```

To visualise and compare metrics of the two agents, execute:

``` bash
vis_aif -gid gridworld-v1 -el gridw9 -nexp 2 -rdir episodic_e180_pol16_maxinf10_learnB -fpi 0 1 2 3 4 -i 4 -v 8 -ti 4 -tv 8 -vl 3 -hl 3 -xtes 20 -ph 4 -selrun 0 -selep 24 49 74 99 -npv 16 -sb 4 -ab 0 1 2 3
```

# References {#references .unnumbered}

:::: {#refs .references .csl-bib-body .hanging-indent entry-spacing="0"}
::: {#ref-Torresan2025a .csl-entry}
Torresan, Filippo, Keisuke Suzuki, Ryota Kanai, and Manuel Baltieri.
2025. "Active Inference for Action-Unaware Agents." arXiv.
<https://doi.org/10.48550/arXiv.2508.12027>.
:::
::::
