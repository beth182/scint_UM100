# script to plot just the grids present for a given hours SA

# imports
import rasterio  # the GEOS-based raster package
import matplotlib.pyplot as plt  # the visualization package
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import geopandas

import warnings
warnings.filterwarnings("ignore")

# SA location

current_path = os.getcwd().replace('\\', '/') + '/'

# model = '100m'
model = '300m'

path = 'BCT_IMU'

sa_dir = current_path + '../../SA_134/'
grid_dir = current_path + '../grid_coord_lookup/grid_polygons/UM' + model.split('m')[0] + '_shapes/'

hour_choice = 12

# ToDo: make this flexable
# one do do with temp grids
sa_file = 'BCT_IMU_15000_2016_134_' + str(hour_choice) + '_00.tif'

raster_path = sa_dir + sa_file

csv_location = current_path + path + '_SA_UM' + model.split('m')[0] + '_grid_percentages.csv'

# read SA

raster = rasterio.open(raster_path)
SA_data = raster.read(1)
raster_extent = np.asarray(raster.bounds)[[0, 2, 1, 3]]

cmap = mpl.cm.jet
cmap.set_bad('white', 1.)
plt.imshow(SA_data, interpolation='none', cmap=cmap, extent=raster_extent)

# read existing csv
existing_df = pd.read_csv(csv_location)
existing_df.index = existing_df['Unnamed: 0']
existing_df = existing_df.drop(columns=['Unnamed: 0'])
existing_df.index.name = 'grid'

all_hour_df = existing_df[str(hour_choice)]
hour_df = all_hour_df.iloc[np.where(all_hour_df >0)[0]]

hour_grids = hour_df.index.to_list()

for grid in hour_grids:
    print(grid)

    grid_file_path = grid_dir + str(grid) + '.gpkg'

    assert os.path.isfile(grid_file_path)

    # plot grid
    grid_gpkg = geopandas.read_file(grid_file_path)
    grid_gpkg.boundary.plot(ax=plt.gca(), color='skyblue')

print('end')
