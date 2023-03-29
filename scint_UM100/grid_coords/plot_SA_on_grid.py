# imports
from os import listdir
from os.path import isfile, join
import os
import iris
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import rasterio
import geopandas as gpd

from model_eval_tools.retrieve_UKV import read_premade_model_files

import scint_fp.functions.plot_functions.plot_sa_lines.sa_lines_funs as sa_lines

# user choices
model = '100m'
# model = '300m'
# model = 'ukv'

run = '20160512T1200Z'

# QH on the hour
target_filetype = 'pvera'

variable_name = 'surface_upward_sensible_heat_flux'

colour_dict = {'BCT_IMU': 'red', 'SCT_SWT': 'mediumorchid', 'IMU_BTT': 'green', 'BTT_BCT': 'blue'}

save_path = os.getcwd().replace('\\', '/') + '/'

# source area location on cluster
sa_base_dir = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/PycharmProjects/scintillometer_footprints/scint_fp/test_outputs/10_mins_ending/'
scint_shp_dir = 'D:/Documents/scint_plots/scint_plots/sa_position_and_lc_fraction/scint_path_shp/'

# data location
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

pp_dir = main_dir + 'pp/' + run + '/' + model + '/'

file_list_pp = [f for f in listdir(pp_dir) if isfile(join(pp_dir, f))]

target_files_pp = []
target_files_nc = []

for file in file_list_pp:

    if target_filetype in file:
        target_files_pp.append(pp_dir + file)
        target_files_nc.append(netcdf_dir + file.split('.')[0] + '.nc')

# load one file into iris first
# load iris cube from pp file
cube = iris.load(target_files_pp[0], variable_name)[0]
# get UM coords
proj_x = cube.coord("grid_longitude").points
proj_y = cube.coord("grid_latitude").points

# get UM coord systems
cs_nat = cube.coord_system()
cs_nat_cart = cs_nat.as_cartopy_projection()

########################################################################################################################

for target_nc_file in target_files_nc:

    assert os.path.isfile(target_nc_file)
    print(target_nc_file)

    # read corresponding nc file
    nc_file = nc.Dataset(target_nc_file)

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

    if run_times[0].strftime('%j') != '134':
        if run_times[-1].strftime('%j') != '134':
            continue

    QH = nc_file.variables[variable_name]

    if len(QH.shape) == 2:
        len_times = 1
    else:
        len_times = QH.shape[0]

    if len(run_times) - len_times == 1:
        run_times = [run_start_time + dt.timedelta(seconds=hr * 3600) for hr in run_len_hours][:-1]

    assert len(run_times) == len_times

    for i in range(0, len_times):

        time_here = run_times[i]

        if time_here.strftime('%j') != '134':
            continue

        # find the SAs for this day
        doy_choice = time_here.strftime('%Y%j')
        sa_dir = sa_base_dir + doy_choice + '/'
        sa_files = sa_lines.find_SA_rasters(sa_dir)

        sa_dict = {}
        for file in sa_files:
            sa_hour = int(file.split('_')[-2])
            sa_dict[sa_hour] = []

        for file in sa_files:
            sa_hour = int(file.split('_')[-2])
            sa_dict[sa_hour].append(file)

        if time_here.hour not in sa_dict.keys():
            continue

        print(time_here.strftime('%Y %j %H'))

        # if hour is included for the obs:
        if len_times == 1:
            QH_vals = QH[:, :]
        else:
            QH_vals = QH[i, :, :]

        max_val = 740
        min_val = -100

        # plot the data in real world coords
        # from stackoverflow issue: https://stackoverflow.com/questions/62346854/how-to-convert-projection-x-and-y-coordinate-in-netcdf-iris-cube-to-lat-lon
        fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax = fig.add_subplot(111, projection=ccrs.epsg(32631))
        ax.set_title(time_here.strftime('%j %H'))
        im = ax.pcolormesh(proj_x,
                           proj_y,
                           QH_vals,
                           transform=cs_nat_cart,
                           vmin=min_val, vmax=max_val,
                           cmap='jet')

        ax.set_ylim(5700701.754871839, 5722454.38075565)
        ax.set_xlim(273544.6001669762, 298502.87618103356)

        plt.colorbar(im, fraction=0.046, pad=0.04)

        # get SA file list for this hour
        sa_hour_files = sa_dict[time_here.hour]

        for sa_file in sa_hour_files:
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

            # plot paths
            df_path = gpd.read_file(scint_shp_dir + path_here + '.shp')
            df_path.plot(edgecolor=colour_dict[path_here], ax=ax, linewidth=3.0)

            rasterio.plot.show(bool_arr, contour=True, transform=raster.transform, contour_label_kws={}, ax=ax,
                               colors=[colour_dict[path_here]], zorder=10)

        plt.savefig(save_path + '../plots/' + model + '_' + time_here.strftime('%j_%H') + '.png', bbox_inches='tight',
                    dpi=300)

        print(save_path + '../plots/' + model + '_' + time_here.strftime('%j_%H') + '.png')

print('end')
