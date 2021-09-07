import matplotlib.pyplot as plt

def position_figure(df, ap_points_with_index, save_obj=None):
    plt.figure(figsize=(16,8))
    plt.plot(df["GroundTruthPositionX[m]"], df["GroundTruthPositionY[m]"], 'o', label="moving device position")
    plt.plot(ap_points_with_index[:,1], ap_points_with_index[:,2], '^', markersize=12, label="AP position")
    for inx, x, y in ap_points_with_index:
        plt.text(x,y, int(inx), fontsize=20)

    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_obj is None:
        plt.xticks(color='red')
        plt.yticks(color='red')
        plt.xlabel(r'$x \, axis$', fontsize=20, color='red')
        plt.ylabel(r'$y \, axis$', fontsize=20, color='red')
    else:
        plt.xlabel(r'$x \, axis$', fontsize=20)
        plt.ylabel(r'$y \, axis$', fontsize=20)
        plt.savefig('{}/DataFigures/{}/{}.svg'.format(*save_obj))


def timestamp_figure(data, ylabel, save_obj=None):
    plt.figure(figsize=(16, 8))
    plt.plot(data)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_obj is None:
        plt.xticks(color='red')
        plt.yticks(color='red')
        plt.xlabel(r'$\# \, of \, samples$', fontsize=20, color='red')
        plt.ylabel(ylabel, fontsize=20, color='red')
    else:
        plt.xlabel(r'$\# \, of \, samples$', fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.savefig('{}/DataFigures/{}/{}.svg'.format(*save_obj))


def timestamp_grad_figure(data, ylabel, up_thres, save_obj=None):
    plt.figure(figsize=(16, 8))
    plt.plot(data, label=r'$\nabla \, timestamp$')
    plt.plot([0, len(data)], [up_thres, up_thres], label='upper threshold ' + r'$({})$'.format(up_thres))

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20, loc="lower right")
    if save_obj is None:
        plt.xticks(color='red')
        plt.yticks(color='red')
        plt.xlabel(r'$\# \, of \, samples$', fontsize=20, color='red')
        plt.ylabel(ylabel, fontsize=20, color='red')
    else:
        plt.xlabel(r'$\# \, of \, samples$', fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.savefig('{}/DataFigures/{}/{}.svg'.format(*save_obj))

def splited_timestamp_figure(data, new_m_ind, ylabel, save_obj=None):
    def sub_figure(split_data, si):
        plt.figure(figsize=(16,8))
        plt.plot(split_data)
        if save_obj is None:
            plt.xticks(color='red')
            plt.yticks(color='red')
            plt.xlabel(r'$\# \, of \, samples$', fontsize=20, color='red')
            plt.ylabel(ylabel, fontsize=20, color='red')
        else:
            plt.xlabel(r'$\# \, of \, samples$', fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.savefig('{}/DataFigures/{}/{}-{}.svg'.format(*save_obj, si))
    n_subplots = len(new_m_ind) - 1
    nx_subplots = n_subplots // 2
    ny_subplots = n_subplots - nx_subplots

    for si in range(n_subplots):
        # plt.subplot(nx_subplots, ny_subplots, si + 1)
        sub_figure(data[new_m_ind[si]+1:new_m_ind[si+1]].to_numpy(), si)

