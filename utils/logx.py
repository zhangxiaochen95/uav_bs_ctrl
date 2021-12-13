r"""Toolkits to log data during training"""
import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.plot import plot_trajectory


def makedir(dir_name):
    # Create the directory if it doesn't exist.
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory `{dir_name}` is created.")
    else:
        print(f"Directory `{dir_name}` already exists.")
    return dir_name


def mov_avg(x, window_size):
    """Smooths the data by moving-average."""
    if len(x) < window_size:
        return x
    else:
        y = np.ones(window_size)
        x = np.asarray(x)
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        return smoothed_x


class MetricLogger(object):
    def __init__(self, save_dir=None):
        self.save_dir = save_dir  # Directory to save data
        # Data to be stored
        self.env_info = {}
        self.agent_info = {}
        self.train = []  # Each entry is a dict recording one training episode
        self.test = []  # Each entry is a list holding dicts of test episodes
        self.curr_epi = {}  # Contents of latest episode

    def log_env_info(self, **kwargs):
        """Record info of environment."""
        for k, v in kwargs.items():
            self.env_info[k] = v

    def log_agent_info(self, **kwargs):
        """Record info of agent."""
        for k, v in kwargs.items():
            self.agent_info[k] = v

    def init_episode(self):
        """Initializes a new episode."""
        self.curr_epi = {}

    def log_step(self, **kwargs):
        """Records a time step."""
        for k, v in kwargs.items():
            if not k in self.curr_epi.keys():
                self.curr_epi[k] = []
            self.curr_epi[k].append(v)

    def finish_train_episode(self):
        """Stores data held by curr_epi as a training episode."""
        self.train.append(self.curr_epi.copy())

    def init_test(self):
        """Adds a test including multiple episodes"""
        self.test.append([])

    def finish_test_episode(self):
        """Stores data held by curr_epi as a test episode."""
        self.test[-1].append(self.curr_epi.copy())

    def save(self):
        """Write data to given directory."""
        path = self.save_dir + 'log.pickle'
        file = open(path, 'wb')
        pickle.dump({'env_info': self.env_info, 'agent_info': self.agent_info,
                     'train': self.train, 'test': self.test}, file)
        file.close()
        print(f"Data saved as {path}")

    def load(self, target_dir):
        """Loads data from target directory."""
        with open(target_dir + 'log.pickle', 'rb') as file:
            data = pickle.load(file)
        self.env_info, self.train, self.test = data['env_info'], data['train'], data['test']
        self.save_dir = target_dir

    def display_metrics(self, names, mode, aggr_type='mean', inr=10, win_size=10):
        """Extracts, analyzes and show metrics of overall training and test."""
        def display_one_metric(name):
        
            def aggr_data(x):
                if aggr_type == 'mean':
                    return np.stack(x).mean(0)
                elif aggr_type == 'sum':
                    return np.stack(x).sum(0)
                else:
                    raise ValueError("Invalid aggregation type.")

            # Each value is the aggregated result of one training episode.
            if mode == 'train':
                y = []
                for epi in range(len(self.train)):
                    if name in self.train[epi].keys():
                        y.append(aggr_data(self.train[epi][name]))
                y = np.stack(y)

            # Each Value is the aggregated result of one test including multiple test episodes.
            elif mode == 'test':
                y = []
                for t in range(len(self.test)):
                    y.append([])
                    for epi in range(len(self.test[t])):
                        y[t].append(aggr_data(self.test[t][epi][name]))
                y = np.stack(y).mean(1)

            else:
                raise ValueError("Invalid mode.")

            # Compute x and moving average of original y.
            x = inr * np.arange(y.size)
            y_smooth = mov_avg(y, win_size)

            # Export data as '.csv' file.
            df = pd.DataFrame({'original': pd.Series(y, index=x),
                               'mov_avg': pd.Series(y_smooth, index=x)})
            df.to_csv(self.save_dir + mode + '_' + name + ".csv")

            # Export data as plot.
            fig, ax = plt.subplots()
            ax.plot(x, y, color='tab:blue', alpha=0.5, label='original data')
            ax.plot(x, y_smooth, color='tab:blue', label='moving average')

            ax.set_xlabel('episode')
            ax.set_ylabel(name)
            plt.legend(loc='upper right')
            plt.title("{} vs. episodes".format(name))
            plt.savefig(self.save_dir + mode + '_' + name + '.png')
            plt.close()

        # Consecutively process each metric.
        for name in names:
            display_one_metric(name)

    def display_test(self, i_test):
        for epi in range(len(self.test[i_test])):
            plot_trajectory(self.test[i_test][epi], self.env_info, self.save_dir + "test_{}_epi_{}".format(i_test, epi))

    def display_info(self):
        print("env info:")
        for k, v in self.env_info.items():
            print(f"{k}: {v}")

        print("agent info:")
        for k, v in self.agent_info.items():
            print(f"{k}: {v}")

