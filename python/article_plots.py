'''
This script plots the article shown in the article, except for the map with the
clusters. This script requires the already processed data obtained by running
`compute_probability.py` and `beached_probability.py`.

'''


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mtick

river_sources = np.load('../river_sources.npy', allow_pickle=True).item()

ordered_labels = ['Recife',
                  'Salvador',
                  'Paraiba',
                  'Rio-de-Janeiro',
                  'Congo',
                  'Santos',
                  'Itajai',
                  'Porto-Alegre',
                  'Rio-de-la-Plata',
                  'Cape-Town']

###############################################################################
# parameters
###############################################################################
series = 6
average_window = 1234
compute_time_series = True
output_path = f'../article_figs/sa-s{series:02d}/'

min_particle_cond = 10

plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'sans-serif'
###############################################################################
# Prabability maps
###############################################################################
posterior = xr.load_dataset(
    f'../data/analysis/sa-s{series:02d}/posterior_sa-s{series:02d}' +
    f'_aw{average_window}.nc')
likelihood = xr.load_dataset(
    f'../data/analysis/sa-s{series:02d}/likelihood_sa-s{series:02d}' +
    f'_aw{average_window}.nc')

labels = list(posterior.keys())
y, x = np.meshgrid(posterior['lat'], posterior['lon'])

# likelihood
t = 0
fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(10, 4),
                       subplot_kw={'projection': ccrs.PlateCarree()},
                       sharey=True, constrained_layout=True)
ax = ax.reshape(15)

for k, loc in enumerate(ordered_labels):
    z = likelihood[loc][t]
    ax[k].set_extent([-73.0, 25, -60, 0], crs=ccrs.PlateCarree())
    # ax[k].add_feature(cfeature.OCEAN)
    ax[k].add_feature(cfeature.LAND, zorder=2, facecolor='#808080')
    ax[k].add_feature(cfeature.RIVERS)
    ax[k].set_title(loc)
    im = ax[k].pcolormesh(x, y, z, cmap='viridis', vmax=0.0003)
    gl = ax[k].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5)

    if k in [1, 2, 3, 4, 6, 7, 8, 9]:
        gl.left_labels = False

    if k in [0, 1, 2, 3, 4]:
        gl.bottom_labels = False

    gl.top_labels = False
    gl.right_labels = False
    h = ax[k].scatter(river_sources[loc][1], river_sources[loc][0],
                      s=20, marker='o', color='red', edgecolors='k',
                      zorder=3, label='Release locations')

for k in range(10, 15):
    ax[k].axis('off')

ax[10].legend(handles=[h], loc='upper center', shadow=True)
bar_ax = fig.add_axes([0.3, 0.21, 0.4, 0.05])
cbar = fig.colorbar(im, cax=bar_ax, orientation='horizontal', extend='max')
cbar.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

plt.savefig(output_path + 'likelihood_4y_average.pdf', format='pdf')
plt.close()

# posterior
t = 0

fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(10, 4),
                       subplot_kw={'projection': ccrs.PlateCarree()},
                       sharey=True, constrained_layout=True)
ax = ax.reshape(15)

for k, loc in enumerate(ordered_labels):
    z = posterior[loc][t]
    ax[k].set_extent([-73.0, 25, -60, 0],
                     crs=ccrs.PlateCarree())
    # ax[k].add_feature(cfeature.OCEAN)
    ax[k].add_feature(cfeature.LAND, zorder=1, facecolor='#808080')
    # ax[k].add_feature(cfeature.RIVERS)
    ax[k].set_title(loc)
    im = ax[k].pcolormesh(x, y, z, cmap='plasma', vmax=1)
    gl = ax[k].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5)

    if k in [1, 2, 3, 4, 6, 7, 8, 9]:
        gl.left_labels = False

    if k in [0, 1, 2, 3, 4]:
        gl.bottom_labels = False

    gl.top_labels = False
    gl.right_labels = False
    h = ax[k].scatter(river_sources[loc][1], river_sources[loc][0],
                      s=20, marker='o', color='red', edgecolors='k',
                      zorder=3, label='Release locations')

for k in range(10, 15):
    ax[k].axis('off')

ax[10].legend(handles=[h], loc='upper center', shadow=True)
bar_ax = fig.add_axes([0.3, 0.21, 0.4, 0.05])
cbar = fig.colorbar(im, cax=bar_ax, orientation='horizontal')

plt.savefig(output_path + 'posterior_4y_average.pdf', format='pdf')
plt.close()

###############################################################################
# time series plot cropped
###############################################################################
plt.rcParams['font.size'] = 8
if compute_time_series:
    posterior30 = xr.load_dataset(
        f'../data/analysis/sa-s{series:02d}/posterior_' +
        f'sa-s{series:02d}_aw30.nc')
    likelihood30 = xr.load_dataset(
        f'../data/analysis/sa-s{series:02d}/likelihood_' +
        f'sa-s{series:02d}_aw30.nc')

    A = (35, 47)
    B = (78, 47)
    C = (59, 60)
    time = np.linspace(1, 53, 53)*30/365

    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, wspace=0.05, height_ratios=[0.2]+[0.8/3]*3)

    ##
    ax00 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    gl = ax00.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=0.5, color='black', alpha=0.5,
                        linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax00.set_extent([-73.0, 24.916666, -60.916664, -5.0833335],
                    crs=ccrs.PlateCarree())

    ax00.add_feature(cfeature.OCEAN)
    ax00.add_feature(cfeature.LAND, zorder=1)
    ax00.add_feature(cfeature.COASTLINE)

    ilons = [A[0], B[0], C[0]]
    ilats = [A[1], B[1], C[1]]
    labels = ['A', 'B', 'C']

    print('################################')
    for i in range(3):
        lon_coord = posterior30['lon'][ilons[i]].values
        lat_coord = posterior30['lat'][ilats[i]].values
        ax00.scatter(lon_coord, lat_coord,
                     s=60, marker='o', color='red', edgecolors='k')
        ax00.text(posterior30['lon'][ilons[i]]+2, posterior30['lat']
                  [ilats[i]]+2, labels[i], fontsize=12)

        print(f'Point {labels[i]} coords: {lat_coord} lat, {lon_coord} lon')
    print('################################')

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.axis('off')
    ax11 = fig.add_subplot(gs[1, :])
    ax21 = fig.add_subplot(gs[2, :], sharex=ax11)
    ax31 = fig.add_subplot(gs[3, :], sharex=ax11)
    plt.setp(ax11.get_xticklabels(), visible=False)
    plt.setp(ax21.get_xticklabels(), visible=False)
    handles = []

    for k, loc in enumerate(ordered_labels):
        a = posterior30[loc][:, A[0], A[1]].where(
            posterior30['counts'][:, A[0], A[1]] >= min_particle_cond)
        b = posterior30[loc][:, B[0], B[1]].where(
            posterior30['counts'][:, B[0], B[1]] >= min_particle_cond)
        c = posterior30[loc][:, C[0], C[1]].where(
            posterior30['counts'][:, C[0], C[1]] >= min_particle_cond)

        hdl = ax11.plot(time, a, '.-', label=loc, color=f'C{k}')
        ax21.plot(time, b, '.-', label=loc, color=f'C{k}')
        ax31.plot(time, c, '.-', label=loc, color=f'C{k}')
        handles.append(hdl[0])

    ax11_t = ax11.twinx()
    ax21_t = ax21.twinx()
    ax31_t = ax31.twinx()

    hdl_twin = ax11_t.plot(time, posterior30['counts'][:, A[0], A[1]],
                           '--', label='Number of particles', c='k')
    handles = handles + hdl_twin
    ax21_t.plot(time, posterior30['counts'][:,  B[0], B[1]], '--',
                label=loc, c='k')
    ax31_t.plot(time, posterior30['counts'][:,  C[0], C[1]], '--',
                label=loc, c='k')
    up_lim = 200
    ax11_t.set_ylim(0, up_lim)
    ax21_t.set_ylim(0, up_lim)
    ax31_t.set_ylim(0, up_lim)
    ax11_t.set_xlim(0, 3.4)
    ax21_t.set_xlim(0, 3.4)
    ax31_t.set_xlim(0, 3.4)
    ax21_t.set_ylabel('Number of particles', fontsize=10, labelpad=10)
    ax21.set_ylabel('Posterior Probability', fontsize=10)
    ax11.grid()
    ax21.grid()
    ax31.grid()
    ax11.set_ylim(0, 1)
    ax21.set_ylim(0, 1)
    ax31.set_ylim(0, 1)
    ax11.text(0.1, 0.85, 'A', fontsize=12)
    ax21.text(0.1, 0.85, 'B', fontsize=12)
    ax31.text(0.1, 0.85, 'C', fontsize=12)
    ax31.set_xlabel('Particle age (years)', fontsize=10)

    ax01.legend(handles=handles, loc='lower center', ncol=2)
    plt.savefig(output_path + 'time_series_cropped.pdf', format='pdf')
    plt.close()


###############################################################################
# Beaching probability plot
###############################################################################
america = xr.load_dataset(
    f'../data/analysis/sa-s{series:02d}/beach_posterior_America_' +
    f'sa-s{series:02d}_average{average_window}.nc')
africa = xr.load_dataset(
    f'../data/analysis/sa-s{series:02d}/beach_posterior_Africa_' +
    f'sa-s{series:02d}_average{average_window}.nc')

african_sources = ['Congo', 'Cape-Town']
american_sources = ['Paraiba', 'Itajai', 'Rio-de-la-Plata', 'Rio-de-Janeiro',
                    'Porto-Alegre', 'Santos', 'Recife', 'Salvador']

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, wspace=0.1, height_ratios=[0.9, 0.1])
ax = gs.subplots(sharey=True)
lower_margin_am = 0
lower_margin_af = 0

for k, loc in enumerate(ordered_labels):
    ax[0, 0].barh(america['lat'], america[loc][0], label=loc, height=1.02,
                  left=lower_margin_am, color=f'C{k}', align='center')
    lower_margin_am += np.nan_to_num(america[loc][0])

    ax[0, 1].barh(africa['lat'], africa[loc][0], height=1.02,
                  left=lower_margin_af, color=f'C{k}', align='center')
    lower_margin_af += np.nan_to_num(africa[loc][0])

    if loc in african_sources:
        ax[0, 1].scatter(0.04, river_sources[loc][0], color=f'C{k}',
                         edgecolor='k', zorder=3, s=100, linewidths=2)

    elif loc in american_sources:
        ax[0, 0].scatter(0.04, river_sources[loc][0], color=f'C{k}',
                         edgecolor='k', zorder=3, s=100, linewidths=2)

my_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
ax[0, 0].set_xticklabels(my_ticks)
ax[0, 1].set_xticklabels(my_ticks)
ax[0, 0].set_ylim(-40, -5)
ax[0, 0].set_xlim(0, 1)
ax[0, 0].legend(bbox_to_anchor=(1, -0.15), loc='center', ncol=4)
ax[0, 0].set_title('American coast', fontsize=10)

ax[0, 1].set_title('African coast', fontsize=10)
ax[0, 0].set_ylabel('Latitude', fontsize=10)
ax[0, 0].grid(color='k', linestyle='--', alpha=0.5)
ax[0, 1].grid(color='k', linestyle='--', alpha=0.5)

ax[1, 0].axis('off')
ax[1, 1].axis('off')

plt.savefig(output_path + 'beached_posterior_4y_average.pdf', format='pdf')
plt.close()

###############################################################################
# crop images
###############################################################################
