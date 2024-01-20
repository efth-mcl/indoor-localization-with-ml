from itertools import product
import numpy as np
import matplotlib.pyplot as plt


def udp_figure(udpresults_dict, figsize=(16, 8), ylim_max=23, legend_fontsize=20, legend_loc='upper left',
               axes_label_fontsize=20, ticks_fontsize=16, plot_icons=['*-', 'o-', '^-'], markersize=10, save_obj=None):
    plt.figure(figsize=figsize)
    for (key, value), plot_icon in zip(udpresults_dict.items(), plot_icons):
        udptime = [t for t in range(1, len(value) + 1)]
        plt.plot(udptime, value, plot_icon, markersize=markersize, label=key)

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.xlabel(r'$time \, (s)$', fontsize=axes_label_fontsize)
    plt.ylabel(r'$UDP \, Throughput \, (Mbps)$', fontsize=axes_label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0, ylim_max)

    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj), format='eps')


def mean_udp_per_dBm_bar_figure(avgsdBm_df, figsize=(16, 8), ylim_max=23,
                                legend_fontsize=20, legend_loc='upper left', axes_label_fontsize=20,
                                ticks_fontsize=20, save_obj=None):
    plt.figure(figsize=figsize)
    index = np.arange(3)
    bar_width = 0.15
    for i, key in enumerate(avgsdBm_df.keys()):
        plt.bar(index + bar_width * i, avgsdBm_df[key], bar_width, label=key)

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.ylabel(r'$Mean \, UDP \, Throughput \, (Mbps)$',
               fontsize=axes_label_fontsize)

    plt.xticks(index + bar_width, [r'${}$'.format(idx) for idx in avgsdBm_df.index],
               fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.tight_layout()

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0, ylim_max)

    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj), format='eps')


def model_feagure(means_vals, model_vals, figsize=(16, 8), ylim_max=23, legend_fontsize=20,
                  legend_loc='upper left', axes_label_fontsize=20, ticks_fontsize=16, plot_icons=['*-', 'o-'],
                  markersize=10, save_obj=None):
    plt.figure(figsize=figsize)
    hopsaxes = [h for h in range(1, len(means_vals) + 1)]

    plt.plot(hopsaxes, means_vals, plot_icons[0], markersize=markersize, label=r'$Mean \, UDP$')
    plt.plot(hopsaxes, model_vals, plot_icons[1], markersize=markersize, label=r'$Model \, UDP$')

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.xlabel(r'$hops \, \#$', fontsize=axes_label_fontsize)
    plt.ylabel(r'$Mean \, UDP \, Throughput \, (Mbps)$', fontsize=axes_label_fontsize)
    plt.xticks([1, 2, 3], fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0, ylim_max)

    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj), format='eps')


def mean_udp_figure(avgsdBm_df, figsize=(16, 8), ylim_max=23, legend_fontsize=20,
                    legend_loc='upper left', axes_label_fontsize=20, ticks_fontsize=20, save_obj=None):
    plt.figure(figsize=figsize)
    index = np.arange(3)
    bar_width = 0.15
    for i, key in enumerate(avgsdBm_df.keys()):
        plt.bar(index + bar_width * i, avgsdBm_df[key], bar_width, label=key)

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.ylabel(r'$Mean \, UDP \, Throughput \, (Mbps)$',
               fontsize=axes_label_fontsize)

    plt.xticks(index + bar_width, [r'${}$'.format(idx) for idx in avgsdBm_df.index],
               fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.tight_layout()

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0, ylim_max)

    if save_obj is not None:
        plt.savefig('{}/eps_pics/{}/{}.eps'.format(*save_obj), format='eps')


def mean_udp_bar3d_figure(complite_means_per_txpower, line_means_per_txpower, babeld_means_per_txpower,
                          figsize=(8, 8), legend_fontsize=20, legend_loc='upper right', axes_label_fontsize=12,
                          ticks_fontsize=12, save_obj=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    xy = list(product([1, 2, 3], [1, 2, 3]))
    x = np.array([x[0] for x in xy])
    y = np.array([y[1] for y in xy])
    complite = complite_means_per_txpower.flatten()
    line = line_means_per_txpower.flatten()
    babeld = babeld_means_per_txpower.flatten()

    ax.bar3d(x, y, 0, [0.1] * 9, [0.1] * 9, complite.reshape(9), color="tab:blue", label="Complite Graph")
    cl_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:blue")
    ax.bar3d(x + 0.1, y, 0, [0.1] * 9, [0.1] * 9, line.reshape(9), color="tab:orange", label="Line Graph")
    ln_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:orange")
    ax.bar3d(x + 0.2, y, 0, [0.1] * 9, [0.1] * 9, babeld.reshape(9), color="tab:green", label="Babeld")
    ba_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:green")
    ax.set_zlabel(r'$Mean \, UDP \, Throughput \, (Mbps)$')

    ax.view_init(45, 40)
    ticks = [1, 2, 3]

    plt.xticks(ticks, ['10 dBm', '20 dBm', '31 dBm'], fontsize=ticks_fontsize)
    plt.yticks(ticks, ['RPI 0 (c) -- RPI 1 (s)', 'RPI 0 (c) -- RPI 2 (s)', 'RPI 0 (c) -- RPI 3 (s)'],
               fontsize=ticks_fontsize)
    ax.legend([cl_proxy, ln_proxy, ba_proxy], ['Complite Graph', 'Line Graph', 'Babeld'])
    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj), format='eps')
