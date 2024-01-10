import os
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# user inputs
path = 'BCT_IMU'

target_DOY = 2016134
target_hour = 12
small_model = '100m'
large_model = '300m'

threshold_value = 0.0

sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/' + path + '_SA_UM' + \
                      large_model.split('m')[0] + '_grid_percentages.csv'

assert os.path.isfile(sa_grids_lookup_csv)

sa_grids_df = pd.read_csv(sa_grids_lookup_csv)

sa_grids_df.index = sa_grids_df['Unnamed: 0']
sa_grids_df = sa_grids_df.drop(columns=['Unnamed: 0'])
sa_grids_df.index.name = 'grid'

hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour)] > threshold_value]

target_grid_list = hour_grid_df.index.to_list()

# plot the grid boundry box polygons
grid_dir = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_polygons/UM' + large_model.split('m')[0] + '_shapes/'

large_model_dfs = []
for grid in target_grid_list:
    grid_file_path = grid_dir + str(grid) + '.gpkg'
    try:
        assert os.path.isfile(grid_file_path)
    except:
        print('end')

    grid_gpkg = geopandas.read_file(grid_file_path)
    large_model_dfs.append(grid_gpkg)

# read in all model grids


grid_dir_model = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_polygons/UM' + \
                 small_model.split('m')[0] + '_shapes/'

if small_model == '100m':
    if path == 'BCT_IMU':
        start_grid = 15000
        stop_grid = 25000
    else:
        print('end')
elif small_model == '300m':
    start_grid = 1000
    stop_grid = 2000
else:
    print('end')

model_dfs = []
model_grid_names = []

for i in range(start_grid, stop_grid):

    print(i)

    grid_file_path = grid_dir_model + str(i) + '.gpkg'
    try:
        assert os.path.isfile(grid_file_path)
    except:
        print('end')

    grid_gpkg = geopandas.read_file(grid_file_path)
    model_dfs.append(grid_gpkg)
    model_grid_names.append(i)

# plot them out
overlap_grids = []

for grid_name, grid_model in zip(model_grid_names, model_dfs):

    for grid_large_model in large_model_dfs:

        overlap_df = grid_model.loc[grid_model.intersects(grid_large_model.unary_union)].reset_index(drop=True)

        if len(overlap_df) > 0:
            assert len(overlap_df) == 1

            # overlap has happened
            # save overlap model grid
            overlap_grids.append(grid_name)

# save overlap grids to a csv file
# create a df
dict = {'hour': np.ones(len(overlap_grids)) * target_hour, 'grids': overlap_grids}

overlap_df = pd.DataFrame.from_dict(dict)

save_path = os.getcwd().replace('\\', '/') + '/'
overlap_df.to_csv(save_path + small_model + '_overlap_with_' + large_model + '_at_' + str(target_hour) + '.csv')

print('end')
