# imports
from os import listdir
from os.path import isfile, join
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os

from model_eval_tools.retrieve_UKV import read_premade_model_files

# user choices
# model = '100m'
# model = '300m'
model = 'ukv'

run = '20160512T1200Z'

# QH on the hour
target_filetype = 'pexptb'

variable_name = 'upward_air_velocity'

surface = True
# surface = False

target_hour = 12

save_path = os.getcwd().replace('\\', '/') + '/'



# Model files
# first ouput timestamp is 1300 on the day before (DOY 133). So add 11 hours to get to midnight of target day (134)
file_index_hour = 11 + target_hour

# find model file
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

# construct file name
target_file_name = 'umnsaa_' + target_filetype + str(file_index_hour).zfill(3) + '.nc'

# construct total path and check the file exists
target_file_path = netcdf_dir + target_file_name
assert os.path.isfile(target_file_path)




# read data
nc_file = nc.Dataset(target_file_path)
QH = nc_file.variables[variable_name]

lat = nc_file['grid_latitude']

if model == 'ukv':
    lon = nc_file['grid_longitude_0']
    lons = lon[:] - 360
else:
    lon = nc_file['grid_longitude']
    lons = lon[:]

lats = lat[:]

max_lat = -0.7278
min_lat = -1.1499

min_lon = 1.2918999
max_lon = 1.714

start_ind_lat = np.where(abs(lats - min_lat) == min(abs(lats - min_lat)))[0][0]
stop_ind_lat = np.where(abs(lats - max_lat) == min(abs(lats - max_lat)))[0][0]
start_ind_lon = np.where(abs(lons - min_lon) == min(abs(lons - min_lon)))[0][0]
stop_ind_lon = np.where(abs(lons - max_lon) == min(abs(lons - max_lon)))[0][0]

if surface:
    model_level_ind = 0
else:
    if model == 'ukv':
        model_level_ind = 3
    else:
        model_level_ind = 6

QH_vals = QH[model_level_ind, :, :]

QH_vals_constrained = QH_vals[start_ind_lat:stop_ind_lat, start_ind_lon:stop_ind_lon]

if variable_name == 'upward_heat_flux_in_air':
    vmin = -110
    vmax = 600

elif variable_name == 'air_temperature':
    vmin = 287
    vmax = 295
else:
    assert variable_name == 'upward_air_velocity'
    vmin = -1.9
    vmax = 1.8

if surface:
    if variable_name == 'upward_heat_flux_in_air':
        variable_name = 'surface_QH'
    elif variable_name == 'air_temperature':
        variable_name = 'surface_T'
    else:
        assert variable_name == 'upward_air_velocity'
        variable_name = 'surface_W'

plt.figure(figsize=(10,10))
im = plt.imshow(QH_vals_constrained,
                vmin=vmin, vmax=vmax,
                cmap='jet', origin='lower')

plt.colorbar(im, fraction=0.046, pad=0.01)
plt.xticks([])
plt.yticks([])




current_path = os.getcwd().replace('\\', '/') + '/plots/fields/'

plt.savefig(current_path + '/' + model + '_' + variable_name + '_' + str(target_hour).zfill(2) + '.png', bbox_inches='tight',
            dpi=300)



# plt.show()


print('end')
