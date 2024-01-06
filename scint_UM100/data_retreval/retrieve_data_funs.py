import numpy as np
import datetime as dt
import os
import pandas as pd

from model_eval_tools.retrieve_UKV import read_premade_model_files
from scint_flux import look_up


def handle_time(nc_file):
    """
    Function to handle model time and convert to a list of datetimes
    :return:
    """
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

    return run_times


def merge(list1, list2):
    """
    makes tuples from 2 lists (of coords)
    :param list1:
    :param list2:
    :return:
    """
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


def grab_obs_z_f_vals(target_DOY, target_hour, path,
                      main_dir='D:/Documents/scint_UM100/scint_UM100/data_retreval/z_f_csvs/'):
    """

    :return:
    """

    path_num = dict((v, k) for k, v in look_up.scint_path_numbers.items())[path]

    file_name = 'z_f_' + str(path_num) + '_' + str(target_DOY) + '.csv'
    file_path = main_dir + file_name
    assert os.path.isfile(file_path)

    # pandas read file
    df = pd.read_csv(file_path)
    df.index = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    z_f = df.iloc[np.where(df.index.hour == target_hour)[0][0]].z_f

    return z_f


def grab_model_altitude(model,
                        grid_num,
                        main_dir='D:/Documents/scint_UM100/scint_UM100/land_cover/'):
    """

    :return:
    """

    file_name = model + '_all_grids_altitude.csv'
    file_path = main_dir + file_name
    assert os.path.isfile(file_path)

    # pandas read file
    df = pd.read_csv(file_path)
    altitude = df.iloc[np.where(df['Unnamed: 0'] == grid_num)[0][0]].altitude

    return altitude


def save_model_stash_to_csv(model, target_hour, grid_nums, grid_vals_QH, grid_vals_W, grid_vals_T, grid_vals_rho,
                            grid_levels,
                            main_dir='D:/Documents/scint_UM100/scint_UM100/data_retreval/stash_data/'):
    """

    :return:
    """

    # form a df
    # create the dataframe
    dict = {'grid': grid_nums, 'level_IND': grid_levels, 'upward_heat_flux_in_air': grid_vals_QH,
            'upward_air_velocity': grid_vals_W, 'air_temperature': grid_vals_T, 'm01s00i253': grid_vals_rho}

    df = pd.DataFrame.from_dict(dict)
    df.index = df.grid
    df = df.drop(columns=['grid'])

    # form csv filename
    filename = 'stash_data_' + model + '_' + str(target_hour).zfill(2) + '.csv'
    filepath = main_dir + filename

    # check to see if this file already exists
    if os.path.isfile(filepath):

        # read in the df
        existing_df = pd.read_csv(filepath)
        existing_df.index = existing_df.grid
        existing_df = existing_df.drop(columns=['grid'])

        # check if index matches
        if (df.index == existing_df.index).all():
            pass
        else:
            print('end')

        # check if level indexes match
        if (df.level_IND == existing_df.level_IND).all():
            pass
        else:
            print('end')

        # check if the stash code already exists
        if np.isclose(df['upward_heat_flux_in_air'], existing_df['upward_heat_flux_in_air']).all():
            pass
        else:
            print('end')

        # then save
        existing_df.to_csv(filepath)

    else:
        # save
        df.to_csv(filepath)

    print('end')


# threshold plots
# add into loop of hour
"""

threshold_list = np.arange(0.0001, 1, 0.0001).tolist()

w_av_threshold_list = []
len_grids_threshold_list = []

print(' ')
for i in threshold_list:
    print(i)

    # loop through different thresholds
    temp_sa = np.ma.masked_where(sa_a <= i, sa_a)
    temp_a = np.ma.masked_where(sa_a <= i, a)

    weighted_temp_a = (temp_sa / np.nansum(temp_sa)) * temp_a

    weighted_av_temp_a = np.nansum(weighted_temp_a)
    w_av_threshold_list.append(weighted_av_temp_a)

    len_grids_temp = temp_sa.count() + np.count_nonzero(~np.isnan(temp_a.data)) - np.count_nonzero(
        ~np.isnan(temp_a.mask))
    len_grids_threshold_list.append(len_grids_temp)



fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

plt.title(str(target_DOY) + ' ' + str(target_hour).zfill(2))

ax1.scatter(threshold_list, w_av_threshold_list, c='r', marker='.')

ax1.set_ylabel('Weighted Average UM100 $Q_{H}$ ($W$ $m^{2}$)')

ax2.set_ylabel('# of UM100 grids')

ax1.set_xlabel('SA Weight Threshold')

ax2.spines['left'].set_color('r')

ax2.spines['right'].set_color('b')

ax2.tick_params(which='both', color='blue')
ax1.tick_params(axis='y', which='both', color='red')
ax1.yaxis.label.set_color('r')
ax2.yaxis.label.set_color('b')

ax1.tick_params(axis='y', colors='r')
ax2.tick_params(axis='y', colors='b')

ax2.scatter(threshold_list, len_grids_threshold_list, c='b', marker='.')

current_path = os.getcwd().replace('\\', '/') + '/plots/changing_thresholds/' + model + '/'
plt.savefig(current_path + path + '_' + str(target_hour).zfill(2) + '.png', layout='tight', dpi=300)



print('end')

"""

# other various plotting stuff

"""
plt.figure()
plt.title('UM100 $Q_H$')
# plt.imshow(a, vmin=150, vmax=500)
plt.imshow(a)
plt.colorbar()
plt.xlim(390, 420)
plt.ylim(445, 400)

plt.figure()
plt.title('UM100 Grid: SA %')
plt.imshow(sa_a)
plt.colorbar()
plt.xlim(390, 420)
plt.ylim(445, 400)

sa_a_norm = (sa_a - np.nanmin(sa_a)) / (np.nanmax(sa_a) - np.nanmin(sa_a))
sa_a_norm[np.isnan(sa_a_norm)] = 0

plt.figure()
plt.title('weighted UM100 $Q_H$')
# plt.imshow(a, alpha=sa_a_norm, vmin=150, vmax=500)
plt.imshow(a, alpha=sa_a_norm)
plt.colorbar()
plt.xlim(390, 420)
plt.ylim(445, 400)
plt.annotate(str(round(weighted_av_a, 1)), xy=(0.05, 0.95), xycoords='axes fraction')
"""

# no map no alpha real world coords

"""

# plot the data in real world coords
# from stackoverflow issue: https://stackoverflow.com/questions/62346854/how-to-convert-projection-x-and-y-coordinate-in-netcdf-iris-cube-to-lat-lon
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection=ccrs.epsg(32631))

# open SA file
sa_file = 'D:/Documents/scint_UM100/scint_UM100/SA_134/BCT_IMU_15000_2016_134_' + str(target_hour).zfill(2) + '_00.tif'
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

grid_dir = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_polygons/UM' + model.split('m')[0] + '_shapes/'
for grid in target_grid_coords.grid.to_list():
    grid_file_path = grid_dir + str(int(grid)) + '.gpkg'


    if os.path.isfile(grid_file_path):
        pass
    else:
        print(grid_file_path)

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
"""
