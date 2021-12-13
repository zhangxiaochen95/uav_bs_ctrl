r"""Tools for advanced visualization"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
          'b', 'c', 'g', 'm', 'r', 'y']


def plot_circ(x_o, y_o, r):
    """Plots a circle centered at given origin."""
    t = np.linspace(0, 2 * np.pi, 100)
    x_data, y_data = r * np.cos(t), r * np.sin(t)
    return x_o + x_data, y_o + y_data


def plot_topology(env, is_annotate=True):
    """Plots the static topology from given environment."""
    fig, ax = plt.subplots()
    # Plot the positions of UAV-BSs/GTs.
    ax.scatter(env.p_uavs[:, 0], env.p_uavs[:, 1], marker='*', color='tab:blue', label='UAV-BSs')
    ax.scatter(env.p_gts[:, 0], env.p_gts[:, 1], marker='x', color='tab:red', label='GTs')

    # Plot the boundary of coverage/relations.
    for m in range(env.n_uavs):
        if env.r_cov < np.inf:
            bound_cov_x, bound_cov_y = plot_circ(env.p_uavs[m, 0], env.p_uavs[m, 1], env.r_cov)
            ax.plot(bound_cov_x, bound_cov_y, linestyle='solid', color='black')
        if env.e_thres['gt'] < np.inf:
            bound_vis_x, bound_vis_y = plot_circ(env.p_uavs[m, 0], env.p_uavs[m, 1], env.e_thres['gt'])
            ax.plot(bound_vis_x, bound_vis_y, linestyle='dashed', color='black')
        if env.e_thres['uav'] < np.inf:
            bound_vis_x, bound_vis_y = plot_circ(env.p_uavs[m, 0], env.p_uavs[m, 1], env.e_thres['uav'])
            ax.plot(bound_vis_x, bound_vis_y, linestyle='dotted', color='black')
        if env.e_thres['comm'] < np.inf:
            bound_vis_x, bound_vis_y = plot_circ(env.p_uavs[m, 0], env.p_uavs[m, 1], env.e_thres['comm'])
            ax.plot(bound_vis_x, bound_vis_y, linestyle='dashdot', color='black')

    # Plot the boundary of legal region.
    ax.plot(np.zeros(100), np.linspace(0, env.len_area, 100), color='black')
    ax.plot(env.len_area * np.ones(100), np.linspace(0, env.len_area, 100), color='black')
    ax.plot(np.linspace(0, env.len_area, 100), np.zeros(100), color='black')
    ax.plot(np.linspace(0, env.len_area, 100), env.len_area * np.ones(100), color='black')

    # Annotate the indices of UAV-BSs/GTs.
    if is_annotate:
        offset = np.array([0, 5])
        for m in range(env.n_uavs):
            ax.annotate("UAV-{}".format(m), xy=env.p_uavs[m], xycoords='data',
                        xytext=offset, textcoords='offset points', size='small')
        for i in range(env.n_gts):
            ax.annotate("GT-{}".format(i), xy=env.p_gts[i], xycoords='data',
                        xytext=offset, textcoords='offset points', size='small')

    ax.axis([-0.1 * env.len_area, 1.1 * env.len_area, -0.1 * env.len_area, 1.1 * env.len_area])
    ax.legend(loc='upper right')
    plt.title("Topology")
    plt.show()


def plot_trajectory(data, env_info, save_path):
    r_glo = np.stack(data['r_glo'])
    tot_p = np.stack(data['tot_p'])
    fair_idx = np.stack(data['fair_idx'])
    n_cov = np.stack(data['n_cov'])
    p_gts, p_uavs = np.stack(data['p_gts']), np.stack(data['p_uavs'])

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    # subplot (0, 0): rewards and penalty
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(r_glo, color='black', label='r_glo', marker='.', markevery=max(1, int(r_glo.size / 10)))
    ax.set_xlabel('step')
    ax.set_ylabel('reward')
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.plot(tot_p, color='tab:red', label='tot_p', marker='x', markevery=max(1, int(tot_p.size / 8)))
    ax2.set_ylabel('tot_p')
    ax2.legend(loc='upper right')
    ax.set_title("reward & penalty")

    # subplot (0, 1): flight trajectory
    ax = fig.add_subplot(gs[0, 1])
    for i in range(p_gts.shape[1]):
        ax.plot(p_gts[:, i, 0], p_gts[:, i, 1], color='black')
        ax.scatter(p_gts[-1, i, 0], p_gts[-1, i, 1],
                   color='black', marker='x')
    for m in range(env_info['n_uavs']):
        ax.plot(p_uavs[:, m, 0], p_uavs[:, m, 1], color=colors[m], label="uav_{}".format(m))
        ax.scatter(p_uavs[0, m, 0], p_uavs[0, m, 1],
                   color=colors[m], marker='o', label='p_start_{}'.format(m))
        ax.scatter(p_uavs[-1, m, 0], p_uavs[-1, m, 1],
                   color=colors[m], marker='*', label='p_end_{}'.format(m))

        # Plot the range of coverage/vision/communication if provided.
        if env_info['r_cov'] < np.inf:
            x, y = plot_circ(p_uavs[-1, m, 0], p_uavs[-1, m, 1], env_info['r_cov'])
            ax.plot(x, y, color=colors[m], linestyle='solid', linewidth=0.5)
        if env_info['e_thres']['gt'] < np.inf:
            x, y = plot_circ(p_uavs[-1, m, 0], p_uavs[-1, m, 1], env_info['e_thres']['gt'])
            ax.plot(x, y, color=colors[m], linestyle='dashed', linewidth=0.5)
        if env_info['e_thres']['uav'] < np.inf:
            x, y = plot_circ(p_uavs[-1, m, 0], p_uavs[-1, m, 1], env_info['e_thres']['uav'])
            ax.plot(x, y, color=colors[m], linestyle='dotted', linewidth=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis([0 - 0.1 * env_info['len_area'], 1.1 * env_info['len_area'],
             0 - 0.1 * env_info['len_area'], 1.1 * env_info['len_area']])
    ax.set_title("trajectories")

    # subplot (1, 0): fairness and number of covered GTs
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(fair_idx, color='black', label='fair_idx', marker='.', markevery=max(1, int(fair_idx.size / 10)))
    ax.set_ylim(bottom=-0.1, top=1.1)
    ax.set_xlabel('step')
    ax.set_ylabel('fair_idx')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.step(np.arange(n_cov.size), n_cov, color='tab:red', label='n_cov', marker='x', markevery=int(n_cov.size / 8))
    # ax2.fill_between(np.arange(n_cov.size), 0, n_cov, color='tab:red', alpha=.5, linewidth=0)
    ax.set_ylim(bottom=-0.1)
    ax2.set_ylabel('n_cov')
    ax2.legend(loc='upper right')
    ax.set_title("fairness index & number of covered GTs")

    plt.savefig(save_path + ".png")
    plt.close()