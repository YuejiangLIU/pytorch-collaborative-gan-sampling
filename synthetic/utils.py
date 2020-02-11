import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_sample(model_batch, real_batch, scale, fname, color=None):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)
    if real_batch is not None:
        ax.scatter(real_batch[:, 0], real_batch[:, 1], s=100, c='g', alpha=0.1)
    if color is None:
        color = 'b'
    if model_batch is not None:
        ax.scatter(model_batch[:, 0], model_batch[:, 1], s=100, c=color, alpha=0.1)
    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def draw_kde(samps, scale, fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    from scipy.stats import kde
    nbins = 100
    x = samps[:, 0]
    y = samps[:, 1]
    k = kde.gaussian_kde([x, y])
    k.set_bandwidth(bw_method=k.factor/2.)
    xi, yi = np.mgrid[-scale:scale:nbins*1j, -scale:scale:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    vmax_factor = 0.2
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuPu, vmin=np.min(zi), vmax=max(np.max(zi)*vmax_factor, np.min(zi)))

    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
