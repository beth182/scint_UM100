# imports
import os
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import iris
import cartopy.crs as ccrs
from pyproj import Proj, transform

import warnings

warnings.filterwarnings("ignore")

from scint_UM100.data_retreval import retrieve_data_funs

# user inputs
target_DOY = 134
target_hour = 12
variable_name = 'upward_heat_flux_in_air'
model = '100m'
run = '20160512T1200Z'
levels = True

# ToDo: move this to a lookup
if levels == True:
    target_filetype = 'pexptb'
else:
    target_filetype = 'pvera'

# first ouput timestamp is 1300 on the day before (DOY 133). So add 11 hours to get to midnight of target day (134)
file_index_hour = 11 + target_hour

main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

# construct file name
target_file_name = 'umnsaa_' + target_filetype + str(file_index_hour).zfill(3) + '.nc'

# construct total path and check the file exists
target_file_path = netcdf_dir + target_file_name
assert os.path.isfile(target_file_path)

# read file
nc_file = nc.Dataset(target_file_path)

# look up grids for this hour
sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/SA_UM100_grid_percentages.csv'

sa_grids_df = pd.read_csv(sa_grids_lookup_csv)

sa_grids_df.index = sa_grids_df['Unnamed: 0']
sa_grids_df = sa_grids_df.drop(columns=['Unnamed: 0'])
sa_grids_df.index.name = 'grid'

# select grids with values bigger than 0 for this time
hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour)] > 0]

target_grid_list = hour_grid_df.index.to_list()

# look up the coords of these grids from the coord lookup
coord_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_coords.csv'

coord_lookup_df = pd.read_csv(coord_lookup_csv)

lf_df_list = []

for target_grid in target_grid_list:
    print(target_grid)

    # subset the coord df for just the current grid
    grid_coords = coord_lookup_df[coord_lookup_df.grid == target_grid][coord_lookup_df.descrip == 'BL']
    lf_df_list.append(grid_coords)

target_grid_coords = pd.concat(lf_df_list)

# temp read in cube
pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/100m/umnsaa_pexptb023.pp'
assert os.path.isfile(pp_file_path)

cube = iris.load(pp_file_path, variable_name)[0]

rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

ll = ccrs.Geodetic()

# convert coords
inProj = Proj(init='epsg:32631')
outProj = Proj(init='epsg:4326')

lat_inds = []
lon_inds = []

for index, row in target_grid_coords.iterrows():
    print(index)

    # get coord
    x = row.x
    y = row.y

    point = transform(inProj, outProj, x, y)

    point_x = point[0]

    if point_x < 0:
        point_x = 360 + point_x

    point_y = point[1]

    target_xy = rot_pole.transform_point(point_x, point_y, ll)  # lower left corner

    x_new = target_xy[0]
    y_new = target_xy[1]

    latitudes = cube[0].coord('grid_latitude')
    longitudes = cube[0].coord('grid_longitude')

    nearest_lat = latitudes.nearest_neighbour_index(y_new)
    nearest_lon = longitudes.nearest_neighbour_index(x_new)

    lat_inds.append(nearest_lat)
    lon_inds.append(nearest_lon)

    # checks to see how close the converted coord back is to the orig x y in epsg 32631
    """
    lat_value = latitudes.cell(nearest_lat)
    lon_value = longitudes.cell(nearest_lon)
    real_world_xy = ll.transform_point(lon_value[0], lat_value[0], rot_pole)
    coords = transform(outProj, inProj, real_world_xy[0], real_world_xy[1])
    """

def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

assert latitudes.shape[0] == longitudes.shape[0]
lat_lon_tuples = merge(lat_inds, lon_inds)

var_array = nc_file.variables[variable_name][0,:,:]

# start with an array full of nans
a = np.full((800,800), np.nan)

# lats
for i in range (0, 800):
    # lons
    for j in range(0, 800):
        if (i, j) in lat_lon_tuples:
            a[i, j] = var_array[i,j]








import rasterio
import rasterio.plot
import geopandas

# plot the data in real world coords
# from stackoverflow issue: https://stackoverflow.com/questions/62346854/how-to-convert-projection-x-and-y-coordinate-in-netcdf-iris-cube-to-lat-lon
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection=ccrs.epsg(32631))

# open SA file
sa_file = 'D:/Documents/scint_UM100/scint_UM100/SA_134/BCT_IMU_15000_2016_134_12_00.tif'
raster = rasterio.open(sa_file)
raster_array = raster.read()
# make all 0 vals in array nan
raster_array[raster_array == 0.0] = np.nan
# force non-zero vals to be 1
bool_arr = np.ones(raster_array.shape)
# remove nans in bool array
nan_index = np.where(np.isnan(raster_array))
bool_arr[nan_index] = 0.0

rasterio.plot.show(bool_arr, contour=True, transform=raster.transform, contour_label_kws={}, ax=ax, zorder=10)

grid_dir = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_polygons/UM100_shapes/'
for grid in target_grid_coords.grid.to_list():
    grid_file_path = grid_dir + str(grid) + '.gpkg'
    assert os.path.isfile(grid_file_path)
    grid_gpkg = geopandas.read_file(grid_file_path)
    grid_gpkg.boundary.plot(ax=ax, color='skyblue')



# get UM coords
proj_x = cube.coord("grid_longitude").points
proj_y = cube.coord("grid_latitude").points

# get UM coord systems
cs_nat = cube.coord_system()
cs_nat_cart = cs_nat.as_cartopy_projection()

im = ax.pcolormesh(proj_x,
                   proj_y,
                   a,
                   transform=cs_nat_cart,
                   cmap='jet')


print('end')





print('end')














# handle time
run_times = retrieve_data_funs.handle_time(nc_file)
assert run_times[0].hour == target_hour
assert run_times[0].strftime('%j') == str(target_DOY)

level_height = nc_file.variables['level_height'][:]

# handle how to target grids
# nc_file.variables['upward_heat_flux_in_air'][:,400,400]
