
# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = '/Users/6525954/Library/CloudStorage/OneDrive-UniversiteitUtrecht/BayesianAnalysis_SouthAtlantic/priors_river_inputs.csv'

df = pd.read_csv(data, index_col=0)

# %% plot a pie chart of the river inputs prior

x_colors = np.linspace(0, 1, 13)
cs = cm.get_cmap('tab20')(x_colors)
explode = [0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.15, 0.15]

plt.figure(figsize=(7,6.5))
wedgeprops = {'fontsize': 9}

fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
ax.pie(df['prior'], labels=df.index,
       colors=cs, autopct='%.1f%%',
       shadow=True, explode=explode,
       rotatelabels=True, startangle=15,textprops=wedgeprops)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
fig.savefig('/Users/6525954/Library/CloudStorage/OneDrive-UniversiteitUtrecht/BayesianAnalysis_SouthAtlantic/piechart_river_inputs.png', 
            dpi=300, facecolor=(1, 0, 0, 0))
# %%
