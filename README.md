# Cooperative Trajectory Design of Multiple UAV Base Stations with Heterogeneous Graph Neural Networks

This is an improved python3 implmentation of our paper [Cooperative Trajectory Design of Multiple UAV Base Stations with Heterogeneous Graph Neural Networks](https://ieeexplore.ieee.org/document/9892688) on *IEEE Transactions on Wireless Communications*. 

Multi-agent reinforcement learning (MARL) is leveraged to train distrbuted policies for unmanned arial vehicle base stations (UBSs) providing wireless coverage for ground terminals (GTs). Particularly, graph neural networks (GNNs) are used to enbale efficient local observation encoding and multi-agent communication. 

Our code is based on [PyTorch](https://pytorch.org) and [DGL](https://www.dgl.ai). In addition, Log/plot/run utilities are taken from [Spinning Up](https://github.com/openai/spinningup) by OpenAI. 

## Prerequisites

The `requirements.txt` file shows installed packages to run the code. Note that it is based on osx-arm64 platform. 

## Running Experiments

3 experiments are included:
* `run_exp1.py`: Investigate the effectiveness of graph observation encoder.
* `run_exp2.py`: Demonstrate the improvement of graph communication over independent agents and [QMIX](https://arxiv.org/abs/1803.11485). Further, [TarMAC](https://arxiv.org/abs/1810.11187) is added as an optional communication protocol.
* `run_exp3.py`: Combination of graph observation and communication.

`plot_results.py` is used to illustrate curves of logged metrics. Then, `test_policies.py` is used to evaluate trained agents. We also provide `collect_curves.py` to summarize and output results held in multiple directories.

## Citing Our Work

If you find our work helpful or use this code in your research, please cite our work. BibTeX format is 

```tex
@ARTICLE{9892688,
  author={Zhang, Xiaochen and Zhao, Haitao and Wei, Jibo and Yan, Chao and Xiong, Jun and Liu, Xiaoran},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Cooperative Trajectory Design of Multiple UAV Base Stations With Heterogeneous Graph Neural Networks}, 
  year={2023},
  volume={22},
  number={3},
  pages={1495-1509},
  doi={10.1109/TWC.2022.3204794}}
```
Thank you very much!

## Contact

Please contact zhangxiaochen14@nudt.edu.cn if you have any issue. Any suggestion or discussion is also welcomed.
