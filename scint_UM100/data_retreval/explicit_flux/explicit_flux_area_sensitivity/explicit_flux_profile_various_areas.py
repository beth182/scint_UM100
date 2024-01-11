# imports
import os
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.plot
import geopandas
import matplotlib.colors as colors

import iris
import cartopy.crs as ccrs
from pyproj import Proj, transform

import warnings

warnings.filterwarnings("ignore")

from scint_UM100.data_retreval import retrieve_data_funs
from scint_flux import constants


def grab_explicit_flux_profile_grids(save_path, small_model, large_model, target_hour, coord_lookup_df):
    """

    :return:
    """

    # read in area csv with the grids

    overlap_csv_path = save_path + small_model + '_overlap_with_' + large_model + '_at_' + str(target_hour) + '.csv'
    assert os.path.isfile(overlap_csv_path)

    # read this in
    overlap_df = pd.read_csv(overlap_csv_path)

    # remove duplicates
    overlap_df = overlap_df.drop_duplicates(subset='grids', keep="last")

    target_grid_list_e = overlap_df.grids.to_list()

    lf_df_list_e = []

    for target_grid in target_grid_list_e:
        # subset the coord df for just the current grid
        grid_coords = coord_lookup_df[coord_lookup_df.grid == target_grid][coord_lookup_df.descrip == 'MID']
        lf_df_list_e.append(grid_coords)

    target_grid_coords_e = pd.concat(lf_df_list_e)
    target_grid_coords_e = target_grid_coords_e.drop(columns=['Unnamed: 0'])
    target_grid_coords_e.index = target_grid_coords_e.grid
    target_grid_coords_e.index = target_grid_coords_e.index.astype(int)

    return target_grid_coords_e


def calculate_explicit_flux(target_DOY, target_hour, path, rho_grid_arrays, T_grid_arrays, W_grid_arrays):
    """

    :return:
    """

    # calculate explicit flux
    ########################################################################################################################
    # find the effective measurement height of the observation for this hour
    # z_f csvs are screated within scint_fp package
    z_f = retrieve_data_funs.grab_obs_z_f_vals(target_DOY, target_hour, path)

    rho_grid_av = np.array(rho_grid_arrays).mean(axis=0)
    rho = rho_grid_av / (constants.radius_of_earth + z_f) ** 2

    rho_cp = constants.cp * rho

    # explicit flux
    T_av = np.array(T_grid_arrays).mean(axis=0)
    W_av = np.array(W_grid_arrays).mean(axis=0)

    T_primes = []
    W_primes = []
    T_prime_W_primes = []

    for T_grid, W_grid in zip(T_grid_arrays, W_grid_arrays):
        T_prime = T_av - T_grid
        W_prime = W_av - W_grid

        T_prime_W_prime = T_prime * W_prime

        T_primes.append(T_prime)
        W_primes.append(W_prime)
        T_prime_W_primes.append(T_prime_W_prime)

    explicit = rho_cp * np.array(T_prime_W_primes).mean(axis=0)

    return explicit





# user inputs
########################################################################################################################

path = 'BCT_IMU'
target_DOY = 2016134

target_hour = 12

# small_model = '100m'
small_model = '300m'




# faff
########################################################################################################################
if target_DOY == 2016134:
    run = '20160512T1200Z'
else:
    assert TypeError('Run cannot be done for this day')

levels = True

# threshold_value = 1.0
threshold_value = 0.0

# ToDo: move this to a lookup
if levels == True:
    target_filetype = 'pexptb'
else:
    target_filetype = 'pvera'

save_path = os.getcwd().replace('\\', '/') + '/'


# find files
########################################################################################################################

# Model files
# first ouput timestamp is 1300 on the day before (DOY 133). So add 11 hours to get to midnight of target day (134)
file_index_hour = 11 + target_hour

# find model file
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + small_model + '/'

# construct file name
target_file_name = 'umnsaa_' + target_filetype + str(file_index_hour).zfill(3) + '.nc'

# construct total path and check the file exists
target_file_path = netcdf_dir + target_file_name
assert os.path.isfile(target_file_path)

# read model file
nc_file = nc.Dataset(target_file_path)

# checks that this is the correct hour
run_times = retrieve_data_funs.handle_time(nc_file)
assert run_times[0].hour == target_hour
assert run_times[0].strftime('%j') == str(target_DOY)[-3:]



# SA stuff FOR PARAMETERIZED FLUX
########################################################################################################################
# look up grids for this hour
sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/' + path + '_SA_UM' + \
                      small_model.split('m')[0] + '_grid_percentages.csv'
sa_grids_df = pd.read_csv(sa_grids_lookup_csv)
sa_grids_df.index = sa_grids_df['Unnamed: 0']
sa_grids_df = sa_grids_df.drop(columns=['Unnamed: 0'])
sa_grids_df.index.name = 'grid'

# select grids with values bigger than 0 for this time
if small_model == '100m':
    hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour)] > threshold_value]
elif small_model == '300m':
    hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour).zfill(2)] > threshold_value]
elif small_model == 'ukv':
    hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour).zfill(2)] > threshold_value]
else:
    print('end')

target_grid_list_p = hour_grid_df.index.to_list()

# look up the coords of these grids from the coord lookup
coord_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_coords_' + small_model + '.csv'
coord_lookup_df = pd.read_csv(coord_lookup_csv)

lf_df_list_p = []

for target_grid in target_grid_list_p:
    print(target_grid)
    # subset the coord df for just the current grid
    grid_coords = coord_lookup_df[coord_lookup_df.grid == target_grid][coord_lookup_df.descrip == 'MID']
    lf_df_list_p.append(grid_coords)

target_grid_coords_p = pd.concat(lf_df_list_p)
target_grid_coords_p = target_grid_coords_p.drop(columns=['Unnamed: 0'])

target_grid_coords_p.index = target_grid_coords_p.grid
target_grid_coords_p.index = target_grid_coords_p.index.astype(int)















# FOR EXPLICIT FLUX
########################################################################################################################


target_grid_coords_e_ukv = grab_explicit_flux_profile_grids(save_path, small_model, 'ukv', target_hour, coord_lookup_df)

if small_model == '100m':
    target_grid_coords_e_300 = grab_explicit_flux_profile_grids(save_path, small_model, '300m', target_hour, coord_lookup_df)








# combine
########################################################################################################################
target_grid_coords_e_ukv = target_grid_coords_e_ukv.add_suffix('_e_ukv')


if small_model == '100m':
    target_grid_coords_e_300 = target_grid_coords_e_300.add_suffix('_e_300')
    target_grid_coords_e = target_grid_coords_e_ukv.join(target_grid_coords_e_300)

    # combine parameterized and explicit df
    target_grid_coords_all = target_grid_coords_e.join(target_grid_coords_p)

else:
    # combine parameterized and explicit df
    target_grid_coords_all = target_grid_coords_e_ukv.join(target_grid_coords_p)




# combine SA weight and coord df
target_grid_coords_all = target_grid_coords_all.join(sa_grids_df)




# temp read in cube
########################################################################################################################
pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/' + small_model.split('m')[0] + 'm/umnsaa_pexptb023.pp'
assert os.path.isfile(pp_file_path)

# takes a long time
cube = iris.load(pp_file_path, 'upward_heat_flux_in_air')[0]

rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

ll = ccrs.Geodetic()

# convert coords
inProj = Proj(init='epsg:32631')
outProj = Proj(init='epsg:4326')




# find all lat/lons
########################################################################################################################
lat_lon_tuples_ukv = []
lat_lon_tuples_300 = []
lat_lon_tuples_p = []

for index, row in target_grid_coords_all.iterrows():

    print(index)

    # get coord
    x = row.x_e_ukv
    y = row.y_e_ukv

    point = transform(inProj, outProj, x, y)

    point_x = point[0]

    if point_x < 0:
        point_x = 360 + point_x

    point_y = point[1]

    target_xy = rot_pole.transform_point(point_x, point_y, ll)  # mid point

    x_new = target_xy[0]
    y_new = target_xy[1]

    latitudes = cube[0].coord('grid_latitude')
    longitudes = cube[0].coord('grid_longitude')

    if small_model == 'ukv':
        longitudes = longitudes - 360

    nearest_lat = latitudes.nearest_neighbour_index(y_new)
    nearest_lon = longitudes.nearest_neighbour_index(x_new)

    # temp solution here to tuple repeat issue
    coord_tuple = (nearest_lat, nearest_lon)
    temp_tuple_list = lat_lon_tuples_ukv.copy()
    temp_tuple_list.append(coord_tuple)

    # check for dups
    if len(temp_tuple_list) > 1:
        dup = {x for x in temp_tuple_list if temp_tuple_list.count(x) > 1}
        if len(dup) > 0:
            nearest_lat = latitudes.nearest_neighbour_index(y_new)

            new_coord_tuple = (nearest_lat, nearest_lon)

            if new_coord_tuple in lat_lon_tuples_ukv:
                print('PROBLEM')
            else:
                coord_tuple = new_coord_tuple

    else:
        pass


    if small_model == '100m':
        if not np.isnan(row.grid_e_300):
            lat_lon_tuples_300.append(coord_tuple)

    if not np.isnan(row.grid):
        lat_lon_tuples_p.append(coord_tuple)

    lat_lon_tuples_ukv.append(coord_tuple)













# build arrays
########################################################################################################################
# read nc files
QH_array = nc_file.variables['upward_heat_flux_in_air'][:]
W_array = nc_file.variables['upward_air_velocity'][:]
T_array = nc_file.variables['air_temperature'][:]
rho_array = nc_file.variables['m01s00i253'][:]

model_level_heights = nc_file.variables['level_height'][:]

if levels == True:
    i_ind = 1
    j_ind = 2
else:
    i_ind = 0
    j_ind = 1

# check if the model domain is a square

if small_model == 'ukv':
    array_size_i = QH_array.shape[i_ind]
    array_size_j = QH_array.shape[j_ind]

else:
    assert latitudes.shape[0] == longitudes.shape[0]
    assert QH_array.shape[i_ind] == QH_array.shape[j_ind]

    array_size_i = QH_array.shape[i_ind]
    array_size_j = QH_array.shape[i_ind]

QH_grid_arrays_p = []
T_grid_arrays_p = []
W_grid_arrays_p = []
rho_grid_arrays_p = []




T_grid_arrays_ukv = []
W_grid_arrays_ukv = []
rho_grid_arrays_ukv = []

T_grid_arrays_300 = []
W_grid_arrays_300 = []
rho_grid_arrays_300 = []


# lats
for i in range(0, array_size_i):
    # lons
    for j in range(0, array_size_j):
        if (i, j) in lat_lon_tuples_ukv:


            T_grid_array = T_array[:, i, j]
            W_grid_array = W_array[:, i, j]
            QH_grid_array = QH_array[:, i, j]
            rho_grid_array = rho_array[:, i, j]

            T_grid_arrays_ukv.append(T_grid_array)
            W_grid_arrays_ukv.append(W_grid_array)
            rho_grid_arrays_ukv.append(rho_grid_array)

            if (i, j) in lat_lon_tuples_300:
                T_grid_arrays_300.append(T_grid_array)
                W_grid_arrays_300.append(W_grid_array)
                rho_grid_arrays_300.append(rho_grid_array)

            if (i, j) in lat_lon_tuples_p:
                QH_grid_arrays_p.append(QH_grid_array)
                T_grid_arrays_p.append(T_grid_array)
                W_grid_arrays_p.append(W_grid_array)
                rho_grid_arrays_p.append(rho_grid_array)






# calculate parameterized flux
########################################################################################################################
parameterized = np.array(QH_grid_arrays_p).mean(axis=0)





# calculate explicit flux
########################################################################################################################

explicit_p = calculate_explicit_flux(target_DOY, target_hour, path, rho_grid_arrays_p, T_grid_arrays_p, W_grid_arrays_p)

explicit_ukv = calculate_explicit_flux(target_DOY, target_hour, path, rho_grid_arrays_ukv, T_grid_arrays_ukv, W_grid_arrays_ukv)

if small_model == '100m':
    explicit_300 = calculate_explicit_flux(target_DOY, target_hour, path, rho_grid_arrays_300, T_grid_arrays_300, W_grid_arrays_300)
    total_300 = explicit_300 + parameterized



# calculate total flux
########################################################################################################################
total_p = explicit_p + parameterized
total_ukv = explicit_ukv + parameterized





# plot
# match Humphrey
########################################################################################################################
plt.figure(figsize=(6, 10))

plt.axvline(x=0, color='k', alpha=0.3)


if small_model == '100m':
    min_model_level = 4
    max_model_level = 7

else:
    assert small_model == '300m'
    min_model_level = 5
    max_model_level = 7



plt.fill_between(x=[-1000, 1000], y1=model_level_heights[min_model_level], y2=model_level_heights[max_model_level], color='magenta', alpha=0.3)



# plt.axhline(y=z_f, color='magenta', alpha=0.3)

plt.plot(parameterized, model_level_heights, label='Parametrized', linestyle='--', c='green')

plt.plot(explicit_ukv, model_level_heights, label='Explicit: Area UKV', c='r', linestyle=':')
plt.plot(total_ukv, model_level_heights, label='Total: Area UKV', c='r')

if small_model == '300m':

    plt.plot(explicit_p, model_level_heights, label='Explicit: Area UM300', c='b', linestyle=':')
    plt.plot(total_p, model_level_heights, label='Total: Area UM300', c='b')



if small_model == '100m':

    plt.plot(explicit_p, model_level_heights, label='Explicit: Area UM100', c='orange', linestyle=':')
    plt.plot(total_p, model_level_heights, label='Total: Area UM100', c='orange')


    plt.plot(explicit_300, model_level_heights, label='Explicit: Area UM300', c='b', linestyle=':')
    plt.plot(total_300, model_level_heights, label='Total: Area UM300', c='b')



plt.ylim(0, 1500)
plt.xlim(-300, 400)
plt.legend()
plt.ylabel('Height above $z_{ES}$ (m)')
plt.xlabel('$Q_{H}$ W $m^{-2}$')
plt.title('UM' + small_model[:-1])


plt.savefig(small_model + '_flux_profile_multiple.png', bbox_inches='tight', dpi=300)

print('end')



# plot
# Zoom in
########################################################################################################################
plt.figure(figsize=(6, 6))

plt.axvline(x=0, color='k', alpha=0.3)

plt.fill_between(x=[-1000, 1000], y1=model_level_heights[min_model_level], y2=model_level_heights[max_model_level], color='magenta', alpha=0.3)

plt.plot(parameterized, model_level_heights, label='Parametrized', linestyle='--', c='green')

plt.plot(explicit_ukv, model_level_heights, label='Explicit: Area UKV', c='r', linestyle=':')
plt.plot(total_ukv, model_level_heights, label='Total: Area UKV', c='r')

if small_model == '300m':

    plt.plot(explicit_p, model_level_heights, label='Explicit: Area UM300', c='b', linestyle=':')
    plt.plot(total_p, model_level_heights, label='Total: Area UM300', c='b')



if small_model == '100m':

    plt.plot(explicit_p, model_level_heights, label='Explicit: Area UM100', c='orange', linestyle=':')
    plt.plot(total_p, model_level_heights, label='Total: Area UM100', c='orange')


    plt.plot(explicit_300, model_level_heights, label='Explicit: Area UM300', c='b', linestyle=':')
    plt.plot(total_300, model_level_heights, label='Total: Area UM300', c='b')


plt.ylim(0, 100)
plt.xlim(-300, 400)
plt.ylabel('Height above $z_{ES}$ (m)')
plt.xlabel('$Q_{H}$ W $m^{-2}$')
plt.title('UM' + small_model[:-1])


plt.savefig(small_model + '_flux_profile_multiple_zoom.png', bbox_inches='tight', dpi=300)
