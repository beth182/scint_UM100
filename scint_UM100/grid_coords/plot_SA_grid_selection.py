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
sa_dir = 'D:/Documents/scint_UM100/scint_UM100/SA_134/'
grid_dir = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/UM100_shapes/'

hour_choice = 12

# ToDo: make this flexable
# one do do with temp grids
sa_file = 'BCT_IMU_15000_2016_134_' + str(hour_choice) + '_00.tif'

raster_path = sa_dir + sa_file

csv_location = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/test.csv'

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

hour_grids = existing_df[str(hour_choice)].index.to_list()

for grid in hour_grids:
    print(grid)

    grid_file_path = grid_dir + str(grid) + '.gpkg'

    assert os.path.isfile(grid_file_path)

    # plot grid
    grid_gpkg = geopandas.read_file(grid_file_path)
    grid_gpkg.boundary.plot(ax=plt.gca(), color='skyblue')

print('end')