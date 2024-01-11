# imports
import os
import iris
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd

from model_eval_tools.retrieve_UKV import read_premade_model_files

import scint_fp.functions.plot_functions.plot_sa_lines.sa_lines_funs as sa_lines

# user choices
# model = '100m'
# model = '300m'
model = 'ukv'

DOY = 2016134
run = '20160512T1200Z'

threshold_value = 0

# QH on the hour
target_filetype = 'pexptb'

# variable_name = 'upward_heat_flux_in_air'
# variable_name = 'air_temperature'
# variable_name = 'upward_air_velocity'
variable_name = 'cloud_volume_fraction_in_atmosphere_layer'

colour_dict = {'BCT_IMU': 'red', 'SCT_SWT': 'mediumorchid', 'IMU_BTT': 'green', 'BTT_BCT': 'blue'}

save_path = os.getcwd().replace('\\', '/') + '/'

target_hour = 12

# surface = True
surface = False


# source area location on cluster
sa_base_dir = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/PycharmProjects/scintillometer_footprints/scint_fp/test_outputs/10_mins_ending/'
scint_shp_dir = 'D:/Documents/scint_plots/scint_plots/sa_position_and_lc_fraction/scint_path_shp/'

# data location
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'
pp_dir = main_dir + 'pp/' + run + '/' + model + '/'

# Model files
# first ouput timestamp is 1300 on the day before (DOY 133). So add 11 hours to get to midnight of target day (134)
file_index_hour = 11 + target_hour

# construct file name
target_file_name = 'umnsaa_' + target_filetype + str(file_index_hour).zfill(3)

target_file_name_nc = target_file_name + '.nc'
target_file_name_pp = target_file_name + '.pp'

# construct total path and check the file exists
target_file_path_nc = netcdf_dir + target_file_name_nc
target_file_path_pp = pp_dir + target_file_name_pp

assert os.path.isfile(target_file_path_nc)
assert os.path.isfile(target_file_path_pp)

# load one file into iris first
# load iris cube from pp file
cube = iris.load(target_file_path_pp, variable_name)[0]
# get UM coords
proj_x = cube.coord("grid_longitude").points
proj_y = cube.coord("grid_latitude").points

# get UM coord systems
cs_nat = cube.coord_system()
cs_nat_cart = cs_nat.as_cartopy_projection()

########################################################################################################################


# read corresponding nc file
nc_file = nc.Dataset(target_file_path_nc)
QH = nc_file.variables[variable_name]

# find the SAs for this day
# sa_dir = sa_base_dir + str(DOY) + '/'
# sa_files = sa_lines.find_SA_rasters(sa_dir)
sa_file = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/PycharmProjects/scintillometer_footprints/scint_fp/test_outputs/10_mins_ending/2016134/BCT_IMU_15000_2016_134_12_00.tif'


if variable_name != 'cloud_volume_fraction_in_atmosphere_layer':
    if surface:
        model_level_ind = 0
    else:
        if model == 'ukv':
            model_level_ind = 3
        else:
            model_level_ind = 6

    QH_vals = QH[model_level_ind, :, :]

else:
    assert surface == False
    QH_vals = np.sum(QH, axis=0)


# handle SA
raster = rasterio.open(sa_file)
raster_array = raster.read()

# make all 0 vals in array nan
raster_array[raster_array == 0.0] = np.nan

# force non-zero vals to be 1
bool_arr = np.ones(raster_array.shape)

# remove nans in bool array
nan_index = np.where(np.isnan(raster_array))
bool_arr[nan_index] = 0.0

path_here = sa_file.split('/')[-1].split('_')[0] + '_' + sa_file.split('/')[-1].split('_')[1]








if variable_name == 'upward_heat_flux_in_air':
    vmin = -110
    vmax = 600

elif variable_name == 'air_temperature':
    vmin = 287
    vmax = 295
elif variable_name == 'upward_air_velocity':
    vmin = -1.9
    vmax = 1.8

else:
    assert variable_name == 'cloud_volume_fraction_in_atmosphere_layer'
    vmin = 0
    vmax = 10

if surface:
    if variable_name == 'upward_heat_flux_in_air':
        variable_name = 'surface_QH'
    elif variable_name == 'air_temperature':
        variable_name = 'surface_T'
    else:
        assert variable_name == 'upward_air_velocity'
        variable_name = 'surface_W'

# plot the data in real world coords
# from stackoverflow issue: https://stackoverflow.com/questions/62346854/how-to-convert-projection-x-and-y-coordinate-in-netcdf-iris-cube-to-lat-lon
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection=ccrs.epsg(32631))
im = ax.pcolormesh(proj_x,
                   proj_y,
                   QH_vals,
                   vmin=vmin, vmax=vmax,
                   transform=cs_nat_cart,
                   cmap='jet')

ax.set_ylim(5699098.929806716, 5724057.205820773)
ax.set_xlim(273544.6001669762, 298502.87618103356)

plt.colorbar(im, fraction=0.046, pad=0.04)

# plot paths
df_path = gpd.read_file(scint_shp_dir + path_here + '.shp')

if variable_name == 'cloud_volume_fraction_in_atmosphere_layer':
    path_c = 'white'
else:
    path_c = 'black'

df_path.plot(edgecolor=path_c, ax=ax, linewidth=3.0)

rasterio.plot.show(bool_arr, contour=True, transform=raster.transform, contour_label_kws={}, ax=ax,
                   colors=path_c, zorder=10)

current_path = os.getcwd().replace('\\', '/') + '/plots/fields/'

plt.savefig(current_path + '/' + model + '_' + variable_name + '_' + str(target_hour).zfill(2) + '.png',
            bbox_inches='tight',
            dpi=300)

print('end')
