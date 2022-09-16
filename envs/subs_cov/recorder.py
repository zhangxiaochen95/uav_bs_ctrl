r"""Tools for visualization"""
import itertools

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.common import *


class Recorder(object):
    def __init__(self, env):
        self.env = env
        self.film = dict(pos_ubs=[], global_utility=[], reward=[],
                         total_throughput=[], fair_idx=[], velocity=[], rate_per_gt=[])

    def __getattr__(self, item):
        return getattr(self.env, item)

    def reload(self):
        self.film = dict(pos_ubs=[self.pos_ubs.copy()], global_utility=[], reward=[],
                         total_throughput=[], fair_idx=[], velocity=[], rate_per_gt=[])

    def click(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.film) and isinstance(self.film[k], list)
            self.film[k].append(v)

    def replay(self, annotate=True, show_img=False, save_dir=None):
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(2, 4)

        # ============================================================================================================ #
        ax = fig.add_subplot(gs[:, 0:2])
        ax.set_aspect('equal')

        path_ubs = np.stack(self.film.get('pos_ubs'))
        init_pos_ubs, final_pos_ubs = path_ubs[0], path_ubs[-1]

        # Plot the positions of UBSs/GTs.
        ax.scatter(final_pos_ubs[0], final_pos_ubs[1], marker='o', s=75, color='r', label='UBS')
        ax.scatter(self.pos_gts[:, 0], self.pos_gts[:, 1], marker='o', color='b', label='GTs')

        if path_ubs.shape[0] > 1:
            ax.scatter(path_ubs[0, 0], path_ubs[0, 1], marker='s', color='r')
            ax.plot(path_ubs[:, 0], path_ubs[:, 1],
                    linestyle='dashed', color='r', linewidth=0.5)

        # Plot the boundary of coverage/relations.
        if self.r_cov < np.inf:
            bound_cov_x, bound_cov_y = plot_circ(final_pos_ubs[0], final_pos_ubs[1], self.r_cov)
            ax.plot(bound_cov_x, bound_cov_y, linestyle='dashed', color='black')

        # Plot the boundary of legal region.
        ax.plot(*plot_line(np.array([0, 0]), np.array([self.range_pos, 0])), color='black')
        ax.plot(*plot_line(np.array([self.range_pos, 0]), np.array([self.range_pos, self.range_pos])), color='black')
        ax.plot(*plot_line(np.array([self.range_pos, self.range_pos]), np.array([0, self.range_pos])), color='black')
        ax.plot(*plot_line(np.array([0, 0]), np.array([0, self.range_pos])), color='black')

        # Annotate the indices of UBSs/GTs.
        if annotate:
            offset = np.array([0, 5])
            ax.annotate("UBS", xy=final_pos_ubs, xycoords='data',
                        xytext=offset, textcoords='offset points', size='medium')

            for i in range(self.n_gts):
                ax.annotate("GT-{}".format(i), xy=self.pos_gts[i], xycoords='data',
                            xytext=offset, textcoords='offset points', size='medium')

        ax.axis([-0.1 * self.range_pos, 1.1 * self.range_pos, -0.1 * self.range_pos, 1.1 * self.range_pos])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.legend(loc='lower right')

        # ============================================================================================================ #
        ax = fig.add_subplot(gs[:, 2:4])
        ax.set_xlabel('Timestep')

        ax.set_box_aspect(1)
        color = 'tab:red'
        fair_idx = np.array(self.film['fair_idx'])
        ax.plot(fair_idx, color=color)
        ax.set_ylabel("Jain's Fairness Index", color=color)
        ax.tick_params(axis='y', labelcolor=color)

        ax = ax.twinx()
        ax.set_box_aspect(1)
        color = 'tab:blue'
        reward = np.array(self.film['reward'])
        ax.plot(reward, color=color)
        ax.set_ylabel("Reward", color=color)
        ax.tick_params(axis='y', labelcolor=color)

        # ============================================================================================================ #
        # ax = fig.add_subplot(gs[:, 4:6])
        # ax.set_xlabel('Timestep')
        # ax.set_box_aspect(1)
        # rate_per_gt = np.stack(self.film['rate_per_gt']).T
        # for m in range(self.n_gts):
        #     ax.plot(rate_per_gt[m], label=f"GT-{m}")
        # ax.set_ylabel("Rate (Mbps)")

        if show_img:
            plt.show()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            write_to_disk(save_dir, path_ubs, pos_gts=self.pos_gts, fair_idx=fair_idx, reward=reward)
            fig_path = osp.join(save_dir, 'trajectories.png')
            plt.savefig(fig_path)

        plt.close()
