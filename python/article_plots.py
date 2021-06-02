import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from matplotlib.lines import Line2D
# import matplotlib.cm as cm
# import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

river_sources = np.load('../river_sources.npy', allow_pickle=True).item()

posterior_avg_1500 = np.load(
    '../data/analysis/sa-S03/posterior_sa-S03_average1500.npy',
    allow_pickle=True).item()
params = np.load('../data/analysis/sa-S03/params_sa-S03_average1500.npy',
                 allow_pickle=True).item()
likelihood_avg_1500 = np.load(
    '../data/analysis/sa-S03/likelihood_sa-S03_average1500.npy',
    allow_pickle=True).item()


ordered_labels = ['Rio-de-Janeiro',
                  'Santos',
                  'Itajai',
                  'Rio-de-la-Plata',
                  'Porto-Alegre',
                  'Cape-Town',
                  'Recife',
                  'Salvador',
                  'Congo']

labels = list(posterior_avg_1500.keys())

# # likelihood 4 years
y, x = np.meshgrid(params['lat_range'], params['lon_range'])
t = 0

fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(10, 3),
                       subplot_kw={'projection': ccrs.PlateCarree()},
                       sharey=True)

ax = ax.reshape(10)

for k, loc in enumerate(ordered_labels):

    # np.ma.masked_array(prob[loc][t], mask=prob['dimensions']['mask'])
    z = likelihood_avg_1500[loc][t]

    ax[k].set_extent([-73.0, 24.916666, -60.916664, -5.0833335],
                     crs=ccrs.PlateCarree())
    # ax[k].add_feature(cfeature.OCEAN)
    ax[k].add_feature(cfeature.LAND, zorder=0)
    ax[k].add_feature(cfeature.COASTLINE)
#     ax[k].add_feature(cfeature.RIVERS)
    ax[k].set_title(loc)
    im = ax[k].pcolormesh(x, y, z, cmap='plasma', vmax=0.0001)
    gl = ax[k].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5)

    if k in [1, 2, 3, 4, 6, 7, 8]:
        gl.left_labels = False

    if k in [0, 1, 2, 3, 4]:
        gl.bottom_labels = False

    gl.top_labels = False
    gl.right_labels = False
#     ax[k].scatter(river_sources[loc][1], river_sources[loc][0],
#                s=50, marker='o', color='red', edgecolors='k', zorder=3)

bar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=bar_ax, orientation='vertical', extend='max')
# cbar.set_label('Likelihood', rotation=270, labelpad=15, fontsize=13)
# cbar.ax.ticklabel_format()
cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
# fig.text(0.1,0.93,f'Average particle age = {t:0.0f} months')
# fig.text(0.1,0.93,f'4.1 years average')
plt.savefig('../article_figs/likelihood_4y_average', dpi=200)


# posterior 4 years
y, x = np.meshgrid(params['lat_range'], params['lon_range'])
t = 0

fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(10, 3),
                       subplot_kw={'projection': ccrs.PlateCarree()},
                       sharey=True)

ax = ax.reshape(10)

for k, loc in enumerate(ordered_labels):

    # np.ma.masked_array(prob[loc][t], mask=prob['dimensions']['mask'])
    z = posterior_avg_1500[loc][t]

    ax[k].set_extent([-73.0, 24.916666, -60.916664, -5.0833335],
                     crs=ccrs.PlateCarree())
    # ax[k].add_feature(cfeature.OCEAN)
    ax[k].add_feature(cfeature.LAND, zorder=1)
    ax[k].add_feature(cfeature.COASTLINE)
    ax[k].add_feature(cfeature.RIVERS)
    ax[k].set_title(loc)
    im = ax[k].pcolormesh(x, y, z, cmap='plasma', vmax=1)
    gl = ax[k].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5)

    if k in [1, 2, 3, 4, 6, 7, 8]:
        gl.left_labels = False

    if k in [0, 1, 2, 3, 4]:
        gl.bottom_labels = False

    gl.top_labels = False
    gl.right_labels = False
#     ax[k].scatter(river_sources[loc][1], river_sources[loc][0],
#                s=50, marker='o', color='red', edgecolors='k', zorder=3)

bar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=bar_ax, orientation='vertical')
# cbar.set_label('Posterior Probability', rotation=270, labelpad=15, fontsize=13)
# fig.text(0.1,0.93,f'Average particle age = {t:0.0f} months')
# fig.text(0.1,0.93,f'4.1 years average')
plt.savefig('../article_figs/posterior_4y_average', dpi=200)

# time series
fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(4, 2, wspace=0.1, height_ratios=[0.2]+[0.8/3]*3)

##
ax00 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
gl = ax00.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.5, color='black', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax00.set_extent([-73.0, 24.916666, -60.916664, -5.0833335],
                crs=ccrs.PlateCarree())

ax00.add_feature(cfeature.OCEAN)
ax00.add_feature(cfeature.LAND, zorder=1)
ax00.add_feature(cfeature.COASTLINE)

ilons = [50, 90, 70]
ilats = [50, 50, 70]
labels = ['A', 'B', 'C']


for i in range(3):
    ax00.scatter(params['lon_range'][ilons[i]], params['lat_range'][ilats[i]],
                 s=60, marker='o', color='red', edgecolors='k')
    ax00.text(params['lon_range'][ilons[i]]+2, params['lat_range']
              [ilats[i]]+2, labels[i], fontsize=14)

##
ax01 = fig.add_subplot(gs[0, 1])
ax01.axis('off')

##
ax11 = fig.add_subplot(gs[1, :])
ax21 = fig.add_subplot(gs[2, :], sharex=ax11)
ax31 = fig.add_subplot(gs[3, :], sharex=ax11)
plt.setp(ax11.get_xticklabels(), visible=False)
plt.setp(ax21.get_xticklabels(), visible=False)


handles = []
for k, loc in enumerate(ordered_labels):

    hdl = ax11.plot(time, posterior_avg[loc][:, 60, 50], '-',
                    label=loc, color=f'C{k}')
    ax21.plot(time, posterior_avg[loc][:, 90, 50], '-',
              label=loc, color=f'C{k}')
    ax31.plot(time, posterior_avg[loc][:, 70, 70], '-',
              label=loc, color=f'C{k}')

    handles.append(hdl[0])

ax11.set_xlim(0, 50)
ax21.set_xlim(0, 50)
ax31.set_xlim(0, 50)

ax11_t = ax11.twinx()
ax21_t = ax21.twinx()
ax31_t = ax31.twinx()

hdl_twin = ax11_t.plot(time, counts_avg[:, 60, 50], '--',
                       label='Number of particles', c='k')
handles = handles + hdl_twin
ax21_t.plot(time, counts_avg[:, 90, 50], '--', label=loc, c='k')
ax31_t.plot(time, counts_avg[:, 70, 70], '--', label=loc, c='k')
ax11_t.set_ylim(0, 150)
ax21_t.set_ylim(0, 150)
ax31_t.set_ylim(0, 150)

ax21_t.set_ylabel('Average number of particles', fontsize=14, labelpad=10)
ax21.set_ylabel('Posterior Probability', fontsize=14)

ax11.grid()
ax21.grid()
ax31.grid()
ax11.set_ylim(-0.05, 1.05)
ax21.set_ylim(-0.05, 1.05)
ax31.set_ylim(-0.05, 1.05)

ax11.text(0.2, 0.85, 'A', fontsize=14)
ax21.text(0.2, 0.85, 'B', fontsize=14)
ax31.text(0.2, 0.85, 'C', fontsize=14)
ax31.set_xlabel('Particle age (months)', fontsize=14)

ax01.legend(handles=handles, bbox_to_anchor=(0.5, 0.5), loc='center', ncol=2)
plt.savefig('../article_figs/time_series_map', dpi=200)


# Beached probability
post_beach = np.load(
    '../data/analysis/sa-S03/beached_posterior_sa-S03_average1500.npy',
    allow_pickle=True).item()
params_beach = np.load(
    '../data/analysis/sa-S03/beached_params_sa-S03_average1500.npy',
    allow_pickle=True).item()

african_sources = ['Luanda', 'Cuvo', 'Chiloango-Congo', 'Cape-Town']
american_sources = ['Paraiba', 'Itajai', 'Rio-de-la-Plata',
                    'Rio-de-Janeiro', 'Porto-Alegre', 'Santos']

# x_colors = np.linspace(0,1, 9)
# colors = cm.get_cmap('tab10')(x_colors)

fig = plt.figure(figsize=(6, 8))
gs = fig.add_gridspec(2, 2, wspace=0.1, height_ratios=[0.9, 0.1])
ax = gs.subplots(sharey=True)
lower_margin_am = 0
lower_margin_af = 0

for k, loc in enumerate(ordered_labels):
    ax[0, 0].barh(params_beach['lat_range_america'],
                  post_beach['America'][loc][0], label=loc, height=1,
                  left=lower_margin_am, color=f'C{k}')
    lower_margin_am += np.nan_to_num(post_beach['America'][loc][0])

    ax[0, 1].barh(params_beach['lat_range_america'],
                  post_beach['Africa'][loc][0], height=1,
                  left=lower_margin_af, color=f'C{k}')
    lower_margin_af += np.nan_to_num(post_beach['Africa'][loc][0])

    if loc in african_sources:
        ax[0, 1].scatter(0.96, river_sources[loc][0], color=f'C{k}',
                         edgecolor='k', zorder=3, s=100,
                         linewidths=2)
    elif loc in american_sources:
        ax[0, 0].scatter(0.04, river_sources[loc][0], color=f'C{k}',
                         edgecolor='k', zorder=3, s=100,
                         linewidths=2)

my_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
ax[0, 0].set_xticklabels(my_ticks)
ax[0, 1].set_xticklabels(my_ticks)
ax[0, 0].set_ylim(-40, -5)
ax[0, 0].set_xlim(0, 1)
ax[0, 0].legend(bbox_to_anchor=(1, -0.15), loc='center', ncol=3)
ax[0, 0].set_title('America', fontsize=14)

ax[0, 1].set_title('Africa', fontsize=14)
ax[0, 0].set_ylabel('Latitude', fontsize=13)
# ax[0,0].set_xlabel('Probability', fontsize=13)
# ax[0,1].set_xlabel('Probability', fontsize=13)
ax[0, 0].grid(color='k', linestyle='--', alpha=0.5)
ax[0, 1].grid(color='k', linestyle='--', alpha=0.5)

ax[1, 0].axis('off')
ax[1, 1].axis('off')
plt.savefig('../article_figs/beached_posterior_4y_average', dpi=200)
