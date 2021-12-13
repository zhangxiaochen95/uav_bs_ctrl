# uav_bs_ctrl
Code implementation of paper ``Cooperative Flight Control of Multiple UAV Base Stations with Heterogeneous Graph Neural Networks''. 

This work is currently under review and therefore no license is provided for reuse/distribution of codes in any form. Uploaded version is a demo of main components. Full version would be added soon after review.

Table of contents:
- `main_example.py` is the main file which trains 'GVis&Comm' for multi-UAV-BS flight control.
- `tasks/` gives the system model and the problem formulation into RL environment.
- `mods.py` defines the model 'GVis&Comm' based on PyTorch and DGL.
- `drqn.py` includes core components and training algorithm of multi-agent DRQN.
- `utils/` provides tools for logging/plotting results.
