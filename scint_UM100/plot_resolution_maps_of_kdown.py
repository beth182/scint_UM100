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
model = '300m'
# model = 'ukv'

run = '20160512T1200Z'

# QH on the hour
target_filetype = 'pvera'

variable_name = 'surface_downwelling_shortwave_flux_in_air'

save_path = os.getcwd().replace('\\', '/') + '/'

# data location
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

file_list = [f for f in listdir(netcdf_dir) if isfile(join(netcdf_dir, f))]

# find just the files of the target stash
target_files = []

for file in file_list:

    if target_filetype in file:
        target_files.append(netcdf_dir + file)

max_val = 900
min_val = 0

for file_path in target_files:

    # if file_path != '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/netcdf/20160512T1200Z/300m/umnsaa_pvera036.nc':
    #     continue

    # read data
    nc_file = nc.Dataset(file_path)

    QH = nc_file.variables[variable_name]

    lat = nc_file['grid_latitude']

    if model == 'ukv':
        # lon = (nc_file['grid_latitude_bnds'][:, 0] + nc_file['grid_latitude_bnds'][:, 1]) / 2
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

    # reads in time
    # get time units for time conversion and start time
    unit_start_time = nc_file.variables['time'].units

    # Read in minutes since the start time and add it on
    # Note: time_to_datetime needs time_since to be a list. Hence put value inside a single element list first
    time_since_start = [np.squeeze(nc_file.variables['forecast_reference_time'])]

    run_start_time = read_premade_model_files.time_to_datetime(unit_start_time, time_since_start)[0]

    # get number of forecast hours to add onto time_start
    run_len_hours = np.squeeze(nc_file.variables['forecast_period'][:]).tolist()

    if type(run_len_hours) == float:
        run_len_hours = [run_len_hours]

    run_times = [run_start_time + dt.timedelta(seconds=hr * 3600) for hr in run_len_hours]

    if len(QH.shape) == 2:
        len_times = 1
    else:
        len_times = QH.shape[0]

    if len(run_times) - len_times == 1:
        run_times = [run_start_time + dt.timedelta(seconds=hr * 3600) for hr in run_len_hours][:-1]

    assert len(run_times) == len_times

    for i in range(0, len_times):

        time_here = run_times[i]

        if len_times == 1:
            QH_vals = QH[:, :]
        else:
            QH_vals = QH[i, :, :]

        QH_vals_constrained = QH_vals[start_ind_lat:stop_ind_lat, start_ind_lon:stop_ind_lon]
        # QH_vals_constrained = QH_vals

        plt.figure(figsize=(10,10))
        im = plt.imshow(QH_vals_constrained, vmin=min_val, vmax=max_val, cmap='jet', origin='lower')
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.title(time_here.strftime('%j %H'))
        # plt.show()
        plt.savefig(save_path + '/plots/' + model + '_' + time_here.strftime('%j_%H') + '.png', bbox_inches='tight',
                    dpi=300)

        print(save_path + '/plots/' + model + '_' + time_here.strftime('%j_%H') + '.png')

print('end')
