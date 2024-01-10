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

# user inputs
path = 'BCT_IMU'
target_DOY = 2016134

# target_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
target_hours = [12]

# variable_name = 'upward_heat_flux_in_air'
# variable_name = 'upward_air_velocity'
# variable_name = 'air_temperature'

# model = '100m'
model = '300m'

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

csv_name = path + '_' + str(target_DOY)[-3:].zfill(3) + '_UM100_QH_' + model + '.csv'
csv_path = os.getcwd().replace('\\', '/') + '/' + csv_name

# check to see if the index exists


for target_hour in target_hours:

    if os.path.isfile(csv_path):
        existing_df = pd.read_csv(csv_path)

        existing_df.index = existing_df.hour
        existing_df = existing_df.drop(columns=['hour'])

        if target_hour in existing_df.index:

            continue
            # print('end')
        else:
            pass
    else:
        pass

    # find the effective measurement height of the observation for this hour
    # z_f csvs are screated within scint_fp package
    z_f = retrieve_data_funs.grab_obs_z_f_vals(target_DOY, target_hour, path)

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

    # read model file
    nc_file = nc.Dataset(target_file_path)

    # checks that this is the correct hour
    run_times = retrieve_data_funs.handle_time(nc_file)
    assert run_times[0].hour == target_hour
    assert run_times[0].strftime('%j') == str(target_DOY)[-3:]

    # look up grids for this hour
    # sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/SA_UM100_grid_percentages.csv'
    # sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/' + path + '_SA_UM100_grid_percentages.csv'

    sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/' + path + '_SA_UM' + \
                          model.split('m')[0] + '_grid_percentages.csv'

    sa_grids_df = pd.read_csv(sa_grids_lookup_csv)

    sa_grids_df.index = sa_grids_df['Unnamed: 0']
    sa_grids_df = sa_grids_df.drop(columns=['Unnamed: 0'])
    sa_grids_df.index.name = 'grid'

    # select grids with values bigger than 0 for this time
    if model == '100m':
        hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour)] > threshold_value]
    elif model == '300m':
        hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour).zfill(2)] > threshold_value]
    elif model == 'ukv':
        hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour).zfill(2)] > threshold_value]
    else:
        print('end')

    target_grid_list = hour_grid_df.index.to_list()

    # look up the coords of these grids from the coord lookup
    coord_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_coords_' + model + '.csv'

    coord_lookup_df = pd.read_csv(coord_lookup_csv)

    lf_df_list = []

    for target_grid in target_grid_list:
        print(target_grid)

        # subset the coord df for just the current grid
        # grid_coords = coord_lookup_df[coord_lookup_df.grid == target_grid][coord_lookup_df.descrip == 'BL']
        grid_coords = coord_lookup_df[coord_lookup_df.grid == target_grid][coord_lookup_df.descrip == 'MID']
        lf_df_list.append(grid_coords)

    target_grid_coords = pd.concat(lf_df_list)

    # combine SA weight and coord df
    target_grid_coords = target_grid_coords.join(sa_grids_df, on='grid')

    # temp read in cube
    if model == 'ukv':
        pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/' + \
                       model.split('m')[0] + '/umnsaa_pexptb023.pp'
    else:
        pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/' + \
                       model.split('m')[0] + 'm/umnsaa_pexptb023.pp'

    assert os.path.isfile(pp_file_path)

    # takes a long time
    cube = iris.load(pp_file_path, 'upward_heat_flux_in_air')[0]

    rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    ll = ccrs.Geodetic()

    # convert coords
    inProj = Proj(init='epsg:32631')
    outProj = Proj(init='epsg:4326')

    lat_inds = []
    lon_inds = []

    tuple_list = []

    altitude_list = []

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

        target_xy = rot_pole.transform_point(point_x, point_y, ll)  # mid point

        x_new = target_xy[0]
        y_new = target_xy[1]

        latitudes = cube[0].coord('grid_latitude')
        longitudes = cube[0].coord('grid_longitude')

        if model == 'ukv':
            longitudes = longitudes - 360

        # not fully understanding why +1's are needed here
        # nearest_lat = latitudes.nearest_neighbour_index(y_new) + 1
        # nearest_lon = longitudes.nearest_neighbour_index(x_new) + 1

        nearest_lat = latitudes.nearest_neighbour_index(y_new)
        nearest_lon = longitudes.nearest_neighbour_index(x_new)

        # temp solution here to tuple repeat issue

        coord_tuple = (nearest_lat, nearest_lon)
        temp_tuple_list = tuple_list.copy()
        temp_tuple_list.append(coord_tuple)

        # check for dups
        if len(temp_tuple_list) > 1:
            dup = {x for x in temp_tuple_list if temp_tuple_list.count(x) > 1}
            if len(dup) > 0:
                nearest_lat = latitudes.nearest_neighbour_index(y_new)

                new_coord_tuple = (nearest_lat, nearest_lon)

                if new_coord_tuple in tuple_list:
                    print('PROBLEM')
                else:
                    coord_tuple = new_coord_tuple

        else:
            pass

        lat_inds.append(nearest_lat)
        lon_inds.append(nearest_lon)

        tuple_list.append(coord_tuple)

        # checks to see how close the converted coord back is to the orig x y in epsg 32631
        """
        lat_value = latitudes.cell(nearest_lat)
        lon_value = longitudes.cell(nearest_lon)
        real_world_xy = ll.transform_point(lon_value[0], lat_value[0], rot_pole)
        coords = transform(outProj, inProj, real_world_xy[0], real_world_xy[1])
        """

    lat_lon_tuples = retrieve_data_funs.merge(lat_inds, lon_inds)

    target_grid_coords['lat_inds'] = lat_inds
    target_grid_coords['lon_inds'] = lon_inds
    target_grid_coords['ind_tuples'] = lat_lon_tuples

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

    if model == 'ukv':
        array_size_i = QH_array.shape[i_ind]
        array_size_j = QH_array.shape[j_ind]

    else:
        assert latitudes.shape[0] == longitudes.shape[0]
        assert QH_array.shape[i_ind] == QH_array.shape[j_ind]

        array_size_i = QH_array.shape[i_ind]
        array_size_j = QH_array.shape[i_ind]

    QH_grid_arrays = []
    T_grid_arrays = []
    W_grid_arrays = []
    rho_grid_arrays = []

    # lats
    for i in range(0, array_size_i):
        # lons
        for j in range(0, array_size_j):
            if (i, j) in lat_lon_tuples:
                T_grid_array = T_array[:, i, j]
                W_grid_array = W_array[:, i, j]
                QH_grid_array = QH_array[:, i, j]
                rho_grid_array = rho_array[:, i, j]

                QH_grid_arrays.append(QH_grid_array)
                T_grid_arrays.append(T_grid_array)
                W_grid_arrays.append(W_grid_array)
                rho_grid_arrays.append(rho_grid_array)

    # parameterized flux
    parameterized = np.array(QH_grid_arrays).mean(axis=0)

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

    total = explicit + parameterized

    # plot
    plt.figure(figsize=(6, 10))

    plt.plot(parameterized, model_level_heights, label='Parametrized')
    plt.plot(explicit, model_level_heights, label='Explicit')
    plt.plot(total, model_level_heights, label='Total')

    plt.ylim(0, 1500)
    plt.xlim(-300, 400)
    plt.legend()
    plt.ylabel('Height above $z_{ES}$ (m)')
    plt.xlabel('$Q_{H}$ W $m^{-2}$')
    plt.title('UM' + model[:-1])

    plt.axvline(x=0, color='k', alpha=0.5, linestyle='--')

    save_path = os.getcwd().replace('\\', '/') + '/'
    plt.savefig(model + '_flux_profile.png', bbox_inches='tight', dpi=300)

    print('end')

print('end')
