import os
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

path = 'BCT_IMU'
target_hour = 12

large_model = '300m'
small_model = '100m'


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


# plot the data in real world coords
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=ccrs.epsg(32631))


for grid in target_grid_list:
    grid_file_path = grid_dir + str(grid) + '.gpkg'
    try:
        assert os.path.isfile(grid_file_path)
    except:
        print('end')

    grid_gpkg = geopandas.read_file(grid_file_path)
    grid_gpkg.boundary.plot(ax=ax, color='red')




# read in csv files for overlap

save_path = os.getcwd().replace('\\', '/') + '/'

csv_path = save_path + small_model + '_overlap_with_' + large_model + '_at_12.csv'
df = pd.read_csv(csv_path)
grid_list = df.grids.to_list()


# create plot
small_grid_dir = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_polygons/UM' + small_model.split('m')[0] + '_shapes/'
for grid in grid_list:

    grid_file_path = small_grid_dir + str(grid) + '.gpkg'

    try:
        assert os.path.isfile(grid_file_path)
    except:
        print('end')

    grid_gpkg = geopandas.read_file(grid_file_path)
    grid_gpkg.boundary.plot(ax=ax, color='skyblue')


print('end')