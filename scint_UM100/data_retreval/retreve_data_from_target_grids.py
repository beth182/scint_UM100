# imports
import os
import netCDF4 as nc
import pandas as pd
import numpy as np

import iris
import cartopy.crs as ccrs
from pyproj import Proj, transform

import warnings

warnings.filterwarnings("ignore")

from scint_UM100.data_retreval import retrieve_data_funs
from scint_UM100.data_retreval import QH_alpha_plot
from scint_UM100.data_retreval.explicit_flux import csv_explicit_flux

from scint_UM100.data_retreval.explicit_flux.explicit_flux_area_sensitivity import explicit_flux_profile_various_areas


def grab_model_data(path, target_DOY, target_hour, model,
                    levels=True):
    """

    :param path:
    :param target_DOY:
    :param target_hours:
    :param variable_name:
    :param model:
    :param levels:
    :return:
    """

    # faff
    ####################################################################################################################

    if target_DOY == 2016134:
        run = '20160512T1200Z'
    else:
        assert TypeError('Run cannot be done for this day')

    # threshold_value = 1.0
    threshold_value = 0.0

    # ToDo: move this to a lookup
    if levels == True:
        target_filetype = 'pexptb'
    else:
        target_filetype = 'pvera'

    save_path = os.getcwd().replace('\\', '/') + '/'












    csv_name = path + '_' + str(target_DOY)[-3:].zfill(3) + '_UM100_QH_' + model + '.csv'
    csv_path = os.getcwd().replace('\\', '/') + '/' + csv_name

    # find the effective measurement height of the observation for this hour
    # z_f csvs are screated within scint_fp package
    z_f = retrieve_data_funs.grab_obs_z_f_vals(target_DOY, target_hour, path)











    # find files
    ####################################################################################################################
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




    # SA stuff FOR PARAMETERIZED FLUX
    ####################################################################################################################

    # look up grids for this hour
    sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/' + path + '_SA_UM' + \
                          model.split('m')[0] + '_grid_percentages.csv'

    sa_grids_df = pd.read_csv(sa_grids_lookup_csv)

    sa_grids_df.index = sa_grids_df['Unnamed: 0']
    sa_grids_df = sa_grids_df.drop(columns=['Unnamed: 0'])
    sa_grids_df.index.name = 'grid'

    # select grids with values bigger than 0 for this time
    if model == '100m':
        hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour)] > threshold_value]
    else:
        hour_grid_df = sa_grids_df[sa_grids_df[str(target_hour).zfill(2)] > threshold_value]

    target_grid_list_p = hour_grid_df.index.to_list()

    # look up the coords of these grids from the coord lookup
    coord_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_coords_' + model + '.csv'

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
    ####################################################################################################################


    target_grid_coords_e = explicit_flux_profile_various_areas.grab_explicit_flux_profile_grids(save_path + 'explicit_flux/explicit_flux_area_sensitivity/', model, 'ukv', target_hour, coord_lookup_df)
    target_grid_coords_e = target_grid_coords_e.add_suffix('_e')




    # COMBINE
    ####################################################################################################################

    target_grid_coords_all = target_grid_coords_e.join(target_grid_coords_p)

    # combine SA weight and coord df
    target_grid_coords_all = target_grid_coords_all.join(sa_grids_df)




    # temp read in cube
    ####################################################################################################################
    if model == 'ukv':
        pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/' + \
                       model.split('m')[0] + '/umnsaa_pexptb' + str(file_index_hour).zfill(3) + '.pp'
    else:
        pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/' + \
                       model.split('m')[0] + 'm/umnsaa_pexptb' + str(file_index_hour).zfill(3) +'.pp'

    assert os.path.isfile(pp_file_path)

    # takes a long time
    cube = iris.load(pp_file_path, 'upward_heat_flux_in_air')[0]

    rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    ll = ccrs.Geodetic()

    # convert coords
    inProj = Proj(init='epsg:32631')
    outProj = Proj(init='epsg:4326')







    # find all lat/lons
    ####################################################################################################################

    lat_lon_tuples_e = []
    lat_lon_tuples_p = []



    altitude_list = []

    for index, row in target_grid_coords_all.iterrows():

        print(index)

        # Look up the altitude
        if model != 'ukv':
            altitude = retrieve_data_funs.grab_model_altitude(model, row.grid_e)
        else:
            altitude = 0

        # get coord
        x = row.x_e
        y = row.y_e

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

        nearest_lat = latitudes.nearest_neighbour_index(y_new)
        nearest_lon = longitudes.nearest_neighbour_index(x_new)

        # temp solution here to tuple repeat issue

        coord_tuple = (nearest_lat, nearest_lon)
        temp_tuple_list = lat_lon_tuples_e.copy()
        temp_tuple_list.append(coord_tuple)

        # check for dups
        if len(temp_tuple_list) > 1:
            dup = {x for x in temp_tuple_list if temp_tuple_list.count(x) > 1}
            if len(dup) > 0:
                nearest_lat = latitudes.nearest_neighbour_index(y_new)

                new_coord_tuple = (nearest_lat, nearest_lon)

                if new_coord_tuple in lat_lon_tuples_e:
                    print('PROBLEM')
                else:
                    coord_tuple = new_coord_tuple

        else:
            pass

        altitude_list.append(altitude)

        lat_lon_tuples_e.append(coord_tuple)



        if not np.isnan(row.grid):
            lat_lon_tuples_p.append(coord_tuple)

















    target_grid_coords_all['altitudes'] = altitude_list
    target_grid_coords_all['ind_tuples'] = lat_lon_tuples_e








    # build arrays
    ########################################################################################################################

    QH_array = nc_file.variables['upward_heat_flux_in_air'][:]
    W_array = nc_file.variables['upward_air_velocity'][:]
    T_array = nc_file.variables['air_temperature'][:]
    rho_array = nc_file.variables['m01s00i253'][:]

    # define level heights
    # ToDo: this will break if Levels = False
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



    # start with an array full of nans
    a = np.full((array_size_i, array_size_j), np.nan)

    sa_a = np.full((array_size_i, array_size_j), np.nan)

    grid_nums = []
    grid_levels = []

    QH_grid_arrays_p = []

    T_grid_arrays_e = []
    W_grid_arrays_e = []
    rho_grid_arrays_e = []


    # lats
    for i in range(0, array_size_i):
        # lons
        for j in range(0, array_size_j):
            if (i, j) in lat_lon_tuples_e:
                # match the model heights
                # get altitude for this grid
                grid_altitude = target_grid_coords_all.loc[target_grid_coords_all['ind_tuples'] == (i, j)].altitudes
                model_heights = model_level_heights + float(grid_altitude)

                if model != 'ukv':
                    height_index = np.abs(model_heights - z_f).argmin()
                else:
                    height_index = 3

                grid_nums.append(int(target_grid_coords_all.loc[target_grid_coords_all['ind_tuples'] == (i, j)].grid_e))
                grid_levels.append(height_index)

                T_grid_arrays_e.append(T_array[height_index, i, j])
                W_grid_arrays_e.append(W_array[height_index, i, j])
                rho_grid_arrays_e.append(rho_array[height_index, i, j])


                if (i, j) in lat_lon_tuples_p:
                    QH_grid_arrays_p.append(QH_array[height_index, i, j])
                    a[i, j] = QH_array[height_index, i, j]

                    if model == '100m':
                        sa_val = float(target_grid_coords_all.loc[target_grid_coords_all['ind_tuples'] == (i, j)][str(target_hour)])
                    else:
                        sa_val = float(target_grid_coords_all.loc[target_grid_coords_all['ind_tuples'] == (i, j)][str(target_hour).zfill(2)])

                    sa_a[i, j] = sa_val

                else:
                    QH_grid_arrays_p.append(np.nan)

    if np.nansum(sa_a) > 99.99:
        pass
    else:
        print('end')

    # save to a CSV here!
    retrieve_data_funs.save_model_stash_to_csv(model, target_hour, grid_nums, QH_grid_arrays_p, W_grid_arrays_e, T_grid_arrays_e, rho_grid_arrays_e, grid_levels)










    # read the explicit flux
    explicit_dir = 'D:/Documents/scint_UM100/scint_UM100/data_retreval/stash_data/'
    explicit_filename = 'stash_data_' + model + '_' + str(target_hour).zfill(2) + '.csv'
    explicit_filepath = explicit_dir + explicit_filename
    assert os.path.isfile(explicit_filepath)

    csv_explicit_flux.calculate_explicit_flux(model, target_hour, target_DOY, path)

    # read file
    explicit_df = pd.read_csv(explicit_filepath)
    # make sure total flux is a column
    assert 'total_flux' in explicit_df.columns
    explicit_flux = explicit_df.total_flux - explicit_df.upward_heat_flux_in_air

    # make sure they're all the same val
    assert np.isclose(explicit_flux.dropna().iloc[0], explicit_flux.dropna()).all()
    explicit_val = explicit_flux.dropna().iloc[0]

    total_flux_a = a + explicit_val
    weighted_a = (sa_a / np.nansum(sa_a)) * total_flux_a
    weighted_av_a = np.nansum(weighted_a)
    a_weighted_percent = (weighted_a / np.nansum(weighted_a)) * 100

    # create or write to csv file of weighted average values
    out_df = pd.DataFrame(
        {'hour': [target_hour], 'weighted_av_a': [weighted_av_a],
         'av_a': [np.nanmean(total_flux_a)], 'len_grids': [len(target_grid_list_p)], 'explicit': [explicit_val]})

    out_df.index = out_df.hour
    out_df = out_df.drop(columns=['hour'])

    if os.path.isfile(csv_path):

        existing_df = pd.read_csv(csv_path)

        existing_df.index = existing_df.hour
        existing_df = existing_df.drop(columns=['hour'])

        new_df = pd.concat([existing_df, out_df])
        new_df.to_csv(csv_path)


    else:
        out_df.to_csv(csv_path)

    # alpha plot
    QH_alpha_plot.QH_alpha_plot(target_hour, model, path, target_grid_coords_all, target_grid_list_p, cube, sa_a,
                                total_flux_a,
                                weighted_av_a)

    print('end')
    print(' ')
