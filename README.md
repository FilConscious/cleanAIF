
# Table of Contents

1.  [CleanAIF](#org2849e67)
2.  [Overview](#orgece0bf2)
    1.  [Policies as paths](#orgff6e691)
    2.  [Policies as plans](#org4ab679b)
    3.  [Features of the three agents](#org8adc89b)
3.  [Installation](#org7d4f427)
4.  [Run an experiment](#org8393def)
5.  [References](#org1db9791)



<a id="org2849e67"></a>

# CleanAIF

The code in this repository can be used to train different (discrete) active inference agents in custom Gymnasium environments and visualize various metrics.


<a id="orgece0bf2"></a>

# Overview

Active inference is a computational framework for adaptive behaviour inspired by neuroscientific considerations whose main tenet is that sentient agents follow an imperative to minimize prediction error. Formally, prediction error is quantified in terms of variational free energy, given a generative model, and expected free energy, given a set of policies (sequences of actions) the agent can pick from to act in the environment.

In a nutshell, the framework provides a recipe to solve a partially observable Markov decision process (POMDP) and has mainly dealt with discrete action and states spaces. The repository currently offers the implementation of three active inference agents, each with its own main script, utility functions, and plotting scripts.

The main difference between these agents consists in how policies are interpreted, which in turn has some implications for how to run experiments and understand the results. This distinction involving policies is explained below. After that the three agents currently implemented are described in more detail.


<a id="orgff6e691"></a>

## Policies as paths

In this type of agent, the policies are sequences of actions that specify a *full trajectory* in the environment, involving a fixed number of time steps that corresponds to the duration of each episode.

At each time step, the agent collects an observation and tries to infer its past, present, and future locations, based on probabilistic beliefs about the most probable location at each time step through the “path” afforded by a policy (i.e., categorical probability distributions conditioned on a policy). This is the process of state-estimation via free energy minimization.

Then, at the planning stage, the agent picks a policy after evaluating its expected free energy, quantifying whether for the remaining time steps that policy is likely to bring it at a specified goal state. The action the agent performs is the action the chosen policy dictates for that time step.

This perception-planning-action loop is repeated at each time step for a set number of episodes, and at the end of each episode the agent can update parameters related to either the state-observation mapping or the transition probabilities in the environment.


<a id="org4ab679b"></a>

## Policies as plans

In this type of agent, the policies are sequences of actions of a fixed length&#x2014;also known as the policy horizon&#x2014;that specify a *future motor plan* to be executed in the environment. The first action of a policy is supposed to be performed at the present time step, the second one at the next time step, and so on. Crucially, these action plans are not carried out in full in this implementation as it is made clear below.

At each time step, the agent again collects an observation and tries to infer its past, present, and future locations. However, this time inference about the past and present relies on *policy-independent* categorical probability distributions, computed at each past time step after having received the corresponding observation while inference about future locations makes use again of policy-dependent probabilistic beliefs.

At the planning stage, policies are gain evaluated based on their expected free energy. The action the agent performs is the action the chosen policy dictates for that time step whereas all the other actions in the motor plan are discarded, i.e. they are not going to be executed at the followint time step in the order the policy dictates.

This perception-planning-action loop is repeated at each time step for a set number of episodes, each potentially lasting a varying number of time steps but not exceeding a maximum number, also know as the truncation point. An agent that correctly learns about the environment will be able to reach the goal state before the truncation point, causing the termination of that episode. At the end of each episode the agent can update parameters related to either the state-observation mapping or the transition probabilities in the environment.


<a id="org8adc89b"></a>

## Features of the three agents

1.  **Episodic policy-as-path agent**
    
    This is the original agent I implemented based on my initial understanding/interpretation of (Da Costa et al. 2020). The key features/aspects are:
    
    -   the environment is episodic, i.e., it resets after $T$ time steps (truncation point) by bringing the agent back to a fixed (or random), initial state
    
    -   the agent interacts with the environment for a total of $N$ episodes (each lasting $T$ time steps), set by the researcher/modeller
    
    -   the agent is tailored for this environment, meaning that:
        -   the agent is aware of its starting location, i.e., the initial state of each episode (this condition can be relaxed)
        
        -   the agent has a set of preferences that dictate what its preferred location is at each time step, note that this could be uniform except for the final state
        
        -   each policy the agent considers is of length $T$, so the agent interacts with the environment for a number of steps equal to the episode duration
        
        -   the agent&rsquo;s goal is to be in a particular location at *end* of the episode, i.e. when the each episode terminates, another way of saying this is that *the time step at which the terminal state occurs is equal to the truncation point of the environment*, i.e., $t_{\text{termination}} = T$
        
        -   each policy can be interpreted as a *path* to the goal state/location
        
        -   the agent has policy-conditioned probabilistic beliefs for each time step in an episode, i.e., the agent considers random variables $S_{0}, \dots, S_{T-1}$ that refers to a time-dependent state of the environment in an episode
        
        -   the agent updates its key parameters at the end of each episode
    
    -   each episode represents the same POMDP both for the agent and the environment

2.  **Episodic policy-as-plan agent**
    
    This is the agent implemented by (Heins et al. 2022). The key features/aspects are:
    
    -   the environment is episodic, i.e., it resets after a maximum number $T$ of time steps (truncation point) by bringing the agent back to a fixed (or random), initial state
    
    -   the agent interacts with the environment for a total of $N$ episodes but each can last a variable number $t$ of time steps, depending on whether the agent arrives at the goal state sooner or later, this is a crucial difference with agent (1):
        -   if $t_{\text{terminal}}$ is used to denote the time step at which the goal/termination state is reached, then $t_{\text{terminal}} \leq T$, i.e., *the terminal state can be reached at a time step that is less than or equal to the truncation point of the environment*
    
    -   the agent is again tailored for this environment, meaning that:
        -   the agent is aware of its starting location, i.e., the initial state of each episode (this condition can be relaxed)
        
        -   differently from agent (1) the length $h$ of each policy is such that $h \ll T$ (much shorter than the total number of steps in an episode) and $h \leq t_{\text{termination}} < h$, i.e., the policy horizon could exceed or be shorter than the minimum number of time steps required for reaching the goal/terminal state
        
        -   each policy can be interpreted as a *plan* the agent evaluates at each time step from which to perform an action that could bring it closer to the goal/terminal state
        
        -   the agent has a set of preferences that dictate what its preferred location is no matter the time step considered, the agent&rsquo;s goal is to be in that particular location *at some point*
        
        -   the agent has $h$ *policy-dependent* probabilistic beliefs for the corresponding future time steps each policy envisions, i.e.,  $S_{t + 1}, \dots, S_{h}$ where $t$ is the current/present time step, and *policy-independent* probabilistic beliefs that are accumulated as the trajectory in an environment grows, i.e., $S_{0}, \dots, S_{t}$, with $t = T$ when the maximum number of step has been reached in an episode
        
        -   the agent updates its key parameters at the end of each episode
    
    -   each episode represents a POMDP for the agent but one in which *the policies are not modelled as part of the POMDP*, they are the result of a canny trick to ensure the agent will reach the goal state at some point (note: this interpretation need to be double-checked)

3.  **Continuing policy-as-path agent**
    
    This is the new agent whose existence I became aware of after further thinking on how it would make sense to compare agent (1) and (2) above. It shares most of the key features/aspects with agent (1) but:
    
    -   the environment is *not* episodic, this means that the agent is provided with a continuous stream of experience, e.g., observations
    
    -   of course, that cannot be endless so we set a truncation point as for the setup of agent (2), say a large number of time steps $T$ that could represent the entire “life” of the agent
    
    -   the agent still “thinks” in terms of episodes with policies as paths (as agent (1)), while knowing that after $h$ time steps, the length of an episode/policy, learning should occur
    
    -   note that in this setup the environment does not reset after each episode because the environment is not episodic, the episodes are in the agent&rsquo;s mind
    
    -   one implication of this is that potentially the agent will start each episode in a different environmental location and the optimal policy will be every time different

Consider that both agent (1) and agent (2) can be made closer to each other by simply making the truncation point different from the terminal state (for agent 1) or by equating truncation point and time step of terminal state (for agent 2). However, the crucial difference involving the policies remains.


<a id="org7d4f427"></a>

# Installation

The code in this repo allows you to train an active inference agent in different discrete, grid-world, custom environments, based on Gymnasium (<https://gymnasium.farama.org/>).

This guide assumes that the user has installed [Git](https://git-scm.com/downloads) and Python (through [Anaconda](https://www.anaconda.com/download) or similar distributions) into their system. Also, the package depends on a custom Gymnasium environment that needs to be installed separately, follow the instructions at this [link](https://github.com/FilConscious/cust-gridworlds) before you do anything else.

After installing the Gymnasium environment:

1.  Open a terminal and move into the local folder/working directory where you installed the custom Gymnasium environment:
    
        cd /home/working-dir

2.  Clone the Github repository either with:
    
        git clone https://github.com/FilConscious/cleanAIF.git
    
    if you do not have a SSH key, or with:
    
        git clone git@github.com:FilConscious/cleanAIF.git

if you have one (see the [GitHub SSH docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) for info on SSH keys). After cloning the repo, you will have the folder `./cleanAIF` in your working directory which you can rename as you see fit. Inside the folder you can find the source code, the `README.org` and other files.

1.  Create a Python virtual environment or conda environment with a recent version of Python, e.g.:
    
        conda create --name my-env python=3.12

2.  Activate the environment:
    
        conda activate my-env

3.  Move inside the local repo folder `./cleanAIF` (if not done already):
    
        cd /home/working-dir/cleanAIF

4.  Install the package with the following command:
    
        pip install --editable .

The latter step installs the package, together with other required libraries/packages, in editable mode. This means that it is possible to modify your working/local copy of the algorithm and used it immediately, without going through the whole process of building the package and installing it again.


<a id="org8393def"></a>

# Run an experiment

1.  Move into the local repo directory (if not already there):
    
        cd /home/working-dir/cleanAIF

2.  Activate the conda environment (if not already done):
    
        conda activate myenv

3.  For training an agent in a simple T-maze you can execute the following commands from the terminal:
    -   for the episodic policy-as-path agent:
        
            main_aif_paths_cont --exp_name "aif-paths-cont" --gym_id "GridWorld-v1" --num_runs 10 --num_episodes 100 --num_steps 3 --inf_steps 5 --action_selection kd -lB --task_type "continuing"
    
    -   for the continuing policy-as-path agent:
        
            main_aif_paths_cont --exp_name "aif-paths-cont" --gym_id "GridWorld-v1" --num_runs 10 --num_episodes 100 --num_steps 3 --inf_steps 5 --action_selection kd -lB --task_type "continuing"
    
    -   for the episodic policy-as-plans agent:
        
        COMING SOON

4.  For visualizing some metrics you can execute the following:
    
        vis_aif_paths_cont -i 4 -v 8 -ti 4 -tv 8 -vl 3 -hl 3 -xtes 10


<a id="org1db9791"></a>

# References

Da Costa, Lancelot, Thomas Parr, Noor Sajid, Sebastijan Veselic, Victorita Neacsu, and Karl Friston. 2020. “Active Inference on Discrete State-Spaces: A Synthesis.” Journal of Mathematical Psychology 99 (December): 102447. <doi:10.1016/j.jmp.2020.102447>.

Heins, Conor, Beren Millidge, Daphne Demekas, Brennan Klein, Karl Friston, Iain D. Couzin, and Alexander Tschantz. 2022. “Pymdp: A Python Library for Active Inference in Discrete State Spaces.” Journal of Open Source Software 7 (73). The Open Journal: 4098. <doi:10.21105/joss.04098>.

