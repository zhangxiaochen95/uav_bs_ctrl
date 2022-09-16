r"""Tools for visualization"""
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from envs.common import *


class Recorder(object):
    def __init__(self, env):
        self.env = env
        self.film = dict(pos_ubs=None, global_reward=None, fair_idx=None)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def reload(self):
        self.film = dict(pos_ubs=[self.pos_ubs.copy()],
                         reward=[],
                         fair_idx=[])

    def click(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.film) and isinstance(self.film[k], list)
            self.film[k].append(v)

    def replay(self, annotate=True, show_img=False, save_dir=None):

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        path_ubs = np.stack(self.film.get('pos_ubs'))
        init_pos_ubs, final_pos_ubs = path_ubs[0], path_ubs[-1]

        if self.t > 0:
            ax.scatter(path_ubs[0, :, 0], path_ubs[0, :, 1], marker='s', color='r')
            for i in range(self.n_ubs):
                ax.plot(path_ubs[:, i, 0], path_ubs[:, i, 1], linestyle='dashed', color='r', linewidth=0.5)

        # Plot the positions of UBSs/GTs.
        ax.scatter(final_pos_ubs[:, 0], final_pos_ubs[:, 1], marker='o', s=75, color='r', label='UBSs')
        ax.scatter(self.pos_gts[:, 0], self.pos_gts[:, 1], marker='o', color='b', label='GTs')

        # Plot the boundary of coverage/sensing relations.
        for m in range(self.n_ubs):
            if self.r_cov < np.inf:
                bound_cov_x, bound_cov_y = plot_circ(final_pos_ubs[m, 0], final_pos_ubs[m, 1], self.r_cov)
                ax.plot(bound_cov_x, bound_cov_y, linestyle='dashed', color='black')

            if self.r_sns < np.inf:
                bound_s_x, bound_s_y = plot_circ(final_pos_ubs[m, 0], final_pos_ubs[m, 1], self.r_sns)
                ax.plot(bound_s_x, bound_s_y, linestyle='dashed', color='b', alpha=0.25, linewidth=0.5)

            if self.r_comm < np.inf:
                bound_s_x, bound_s_y = plot_circ(final_pos_ubs[m, 0], final_pos_ubs[m, 1], self.r_comm)
                ax.plot(bound_s_x, bound_s_y, linestyle='dashed', color='r', alpha=0.25, linewidth=0.5)

        # Plot the boundary of legal region.
        ax.plot(*plot_line(np.array([0, 0]), np.array([self.range_pos, 0])), color='black')
        ax.plot(*plot_line(np.array([self.range_pos, 0]), np.array([self.range_pos, self.range_pos])), color='black')
        ax.plot(*plot_line(np.array([self.range_pos, self.range_pos]), np.array([0, self.range_pos])), color='black')
        ax.plot(*plot_line(np.array([0, 0]), np.array([0, self.range_pos])), color='black')

        # Annotate the indices of UBSs/GTs.
        if annotate:
            offset = np.array([0, 5])
            for m in range(self.n_ubs):
                ax.annotate("UBS-{}".format(m), xy=final_pos_ubs[m], xycoords='data',
                            xytext=offset, textcoords='offset points', size='medium')
            # for i in range(recorder.n_gts):
            #     ax.annotate("GT-{}".format(i), xy=recorder.pos_gts[i], xycoords='data',
            #                 xytext=offset, textcoords='offset points', size='medium')

        ax.axis([-0.1 * self.range_pos, 1.1 * self.range_pos, -0.1 * self.range_pos, 1.1 * self.range_pos])
        ax.legend(loc='lower right')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.title("Trajectories")

        if show_img:
            plt.show()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            write_to_disk(save_dir, path_ubs, pos_gts=self.pos_gts)
            fig_path = osp.join(save_dir, 'trajectories.png')
            plt.savefig(fig_path)

        plt.close()
