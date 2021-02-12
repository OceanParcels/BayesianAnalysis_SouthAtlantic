import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm  as cm

def custom_cmap(color_list):
    """
        custom_cmap(color_list)
    It creates a color map from a list of colors Returns a colormap.
    Be careful with non-monochromatic cmaps.
    E.g.
        mycmap = custom_cmap(['blue', 'cyan', '#f0f0f0'])
    """
    return LinearSegmentedColormap.from_list('mycmap', color_list)

def display_cmap(cmap):
    """
        display_cmap(cmap)
    It plots a colormap in a simple way.
    To create a custom cmap:
        from matplotlib.colors import LinearSegmentedColormap
        basic_cols=['blue', ... ,'red']
        my_cmap=LinearSegmentedColormap.from_list('mycmap', basic_cols)
    """
    plt.imshow(np.linspace(0, 100, 256)[None, :],  aspect=25,    interpolation='nearest', cmap=cmap)
    plt.axis('off')

def ridge_plot(data, xlabel ='', title='', bins=128, h_space=-0.5, alpha=1, figsize=(8,8), cmap='tab10'):
    """
        ridge_plot(data, xlabel, bins=128, h_space=-0.5, alpha=1, figsize=(8,6))
    Plots a comparison of kernel density estimates (KDE) for a diferent groups of data.
    data : a dictionary with a 1D series per key/set (unlimited number of keys/sets).
    xlabel : the string that contains the label of the plot.
    title : the title for the plot.
    bins : the number of points to plot the computed KDE.
    h_space : the separation between distributions, it should be negative for them to overlap.
    alpha : the transparency.
    figsize : the figure size.
    cmap : the colormap. Use the predefined Matplotlib colormaps.
    """
    nrows = len(data.keys())
    labels = list(data.keys())
    x_colors = np.linspace(0,1, nrows)
    colors = cm.get_cmap(cmap)(x_colors)
    fig, axes = plt.subplots(nrows,  sharex=True, figsize=figsize)
    min_glob = 999

    for i, key in enumerate(data.keys()):
        val_min = data[key][~np.isnan(data[key])].min()
        val_max = data[key][~np.isnan(data[key])].max()

        if val_min < min_glob:
            min_glob = val_min

        x_values = np.linspace(val_min, val_max, bins)
        c = colors[i]
        kernel = stats.gaussian_kde(data[key][~np.isnan(data[key])])
        kde = kernel(x_values)
        axes[i].plot(x_values, kde, color="#f0f0f0", lw=1)
        axes[i].fill_between(x_values, kde, color=c, alpha=alpha)
        rect = axes[i].patch
        rect.set_alpha(0)
        axes[i].tick_params(left=False, labelleft=False)
        axes[0].set_title(title, fontweight="bold", fontsize=14)

        if i == len(data.keys())-1:
            axes[i].tick_params(bottom=True, left=False, labelleft=False)
            spines = ["top","right","left"]
            #axes[i].set_ylim(-0.05,)
            axes[i].set_xlim(min_glob,)
            axes[i].set_xlabel(xlabel, fontsize=14)

        else:
            axes[i].tick_params(bottom=False, left=False, labelleft=False)
            spines = ["top","right","left","bottom"]

        for s in spines:
            axes[i].spines[s].set_visible(False)

        depth_label = str(key)

    for j,l in enumerate(data.keys()):
        axes[j].text(min_glob, 0., labels[j], fontweight="bold", fontsize=13, ha="right")

    plt.subplots_adjust(hspace=h_space)
    return fig, axes

def waterfall_plot(data, bins=64, dist_height=30, alpha=1, figsize=(6,7), cmap='tab10', border=False):
    """
        waterfall_plot(data, bins=64, dist_height=30, alpha=1, figsize=(6,7), cmap='tab10', border=False)
    Guillaume's waterfall plot.
    data : a dictionary with a 1D series per key/set (unlimited number of keys/sets).
    bins : the number of points to plot the computed KDE.
    dist_height : the height of the distributions. It must be tweaked manually.
    alpha : the transparency.
    figsize : the figure size.
    cmap : is the colormap, use the predefined cmaps.
    borde : if True it plots a black line around the distributions.
    You can add the labels and title and keep editing the plot.
    E.g.
        plot_recipes.waterfall_plot(data)
        plt.xlabel(..)
        plt.title(..)
        plt.grid(..)
    """
    nrows = len(data.keys())
    levels = np.array(list(data.keys()))
    means = [data[i][~np.isnan(data[i])].mean() for i in data.keys()]
    x_colors = np.linspace(0,1, nrows)
    colors = cm.get_cmap(cmap)(x_colors)
    plt.figure(figsize=figsize)
    plt.plot(means, -levels, 'ko--', label='Mean')

    for i, key in enumerate(levels):
        c = colors[i]
        val_min = data[key][~np.isnan(data[key])].min()
        val_max = data[key][~np.isnan(data[key])].max()
        x_values = np.linspace(val_min, val_max, bins)
        base_line = np.zeros_like(x_values) - levels[i]
        kernel = stats.gaussian_kde(data[key][~np.isnan(data[key])])
        kde = kernel(x_values)*dist_height - levels[i]
        plt.fill_between(x_values, base_line, kde, alpha=alpha, color=c)
        if border == True:
            plt.plot(x_values, kde, color='k', lw=1)

    plt.legend(fontsize=12, shadow=True, loc='upper left')