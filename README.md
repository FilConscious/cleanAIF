

# CleanAIF

Train active inference agents in custom Gymnasium environments and visualize their behaviour.


# Overview

The project started during my PhD as a personal attempt to understand active inference, a computational framework for adaptive behaviour inspired by neuroscientific considerations. Understanding meant going through several key active inference papers in the literature, to grasp and re-derive the fundamental mathematical and computational components of the framework, and implementing from scratch the main algorithm.

After several failed attempts, everything clicked into place when I read the synthesis provided by (Da Costa et al. 2020). The code of this repo is more or less an implementation of active inference that closely follow the presentation of that paper.

Since its inception, the project has been kept private, and many excellent projects on active inference have been released in the meantime (see [Resources](docs/aif-gym-docs.md) in the docs). Nonetheless, I decided to make the repo public at the end of my PhD journey in the hope that it may offer another didactic version of active inference (and for closure).


# Quick Install

1.  Open a terminal and move into your preferred local folder or working directory:
    
    `cd /home/working-dir`

2.  Clone the Github repository (or download it into the same folder):
    
    `git clone https://github.com/FilConscious/cleanAIF.git`

3.  Create Python virtual environment or conda environment with a recent version of Python (>=3.10), e.g.:
    
    `conda create --name myenv python=3.12`

4.  Activate the environment:
    
    `conda activate myenv`

5.  Install the package:
    
    `pip install --editable .`

For detailed instructions, see [Installation](docs/aif-gym-docs.md).


# Run an experiment

1.  Move into the local repo directory
    
    `cd home/././name-of-repo/`

2.  Activate conda environment
    
    `conda activate myenv`

3.  Execute a Python script for training
    
    `main_aif_paths --exp_name "aif-paths" --gym_id "GridWorld-v1" --num_runs 100 --num_episodes 100 --num_steps 5 --inf_steps 5 --pref_type states --action_selection kd`

4.  Execute Python script for data visualization
    
    `plot_aif_paths -i 4 -v 8 -ti 4 -tv 8 -vl 3 -hl 3`

For more detailed instructions, see [How to Run an Experiment](docs/aif-gym-docs.md).


# References

Da Costa, Lancelot, Thomas Parr, Noor Sajid, Sebastijan Veselic, Victorita Neacsu, and Karl Friston. 2020. “Active Inference on Discrete State-Spaces: A Synthesis.” Journal of Mathematical Psychology 99 (December): 102447. <10.1016/j.jmp.2020.102447>.

