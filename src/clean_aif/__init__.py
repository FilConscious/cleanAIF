"""
Main function to train the agent (used as entry point, see the pyproject.toml).

Created on 11/12/2024
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import argparse
import importlib
from pathlib import Path
from datetime import datetime

# __all__ = ["bforest_env", "omaze_env", "grid_env", "maze_env", "environment"]
VERSION = "0.1.0"


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
        default="aif-pi-plans",
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
        default=5,
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
    params = vars(args)

    ##################################
    ### 2. CREATE DIRECTORY FOR LOGGING
    ##################################

    # Datetime object containing current date and time
    now = datetime.now()
    # Converting data-time in an appropriate string: '_dd.mm.YYYY_H.M.S'
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S_")

    # Retrieving current working directory and creating folders where to store the data collected
    # from one experiment.
    # Note 1: a data path is created so that running this file multiple times produces aptly named folders
    # (e.g. for trying different hyperparameters values).
    saving_directory = Path.cwd().joinpath("results")
    data_path = saving_directory.joinpath(
        dt_string
        + f'{params["env_name"]}r{params["num_runs"]}e{params["num_episodes"]}prF{params["pref_type"]}AS{params["action_selection"]}lA{str(params["learn_A"])[0]}lB{str(params["learn_B"])[0]}lD{str(params["learn_D"])[0]}'
    )

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    ###################
    ### 3. RUN TRAINING
    ###################

    # We use the task's name provided through the command line (e.g., task1) to import the Python module of
    # the same name, used to start training the agent in the corresponding task; if the task does not exist
    # an exception is raised.
    task_module_name = params["task_name"]
    try:
        task_module = importlib.import_module(".tasks." + task_module_name, "aifgym")
    except Exception as error:
        print(f"Something went wrong with the import of the task module: ", error)
        print(f"{task_module_name} module could not be imported.")
    else:
        # Calling the train function from the imported task module to start agent's training
        task_module.train(params, data_path, "aif_exp" + dt_string)


# if __name__ == "__main__":
# main()
