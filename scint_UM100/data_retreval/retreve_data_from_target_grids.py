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

# user inputs
path = 'BCT_IMU'
target_DOY = 134

# target_hour = 12
# target_hour = 6

target_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

variable_name = 'upward_heat_flux_in_air'
# model = '100m'
# model = '300m'
model = 'ukv'

run = '20160512T1200Z'
levels = True

# threshold_value = 1.0
threshold_value = 0.0

# ToDo: move this to a lookup
if levels == True:
    target_filetype = 'pexptb'
else:
    target_filetype = 'pvera'

csv_name = path + '_' + str(target_DOY).zfill(3) + '_UM100_QH_' + model + '.csv'
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

    print('end')

    # checks that this is the correct hour
    run_times = retrieve_data_funs.handle_time(nc_file)
    assert run_times[0].hour == target_hour
    assert run_times[0].strftime('%j') == str(target_DOY)

    # look up grids for this hour
    # sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/SA_UM100_grid_percentages.csv'
    # sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/' + path + '_SA_UM100_grid_percentages.csv'
    sa_grids_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/SA_grid_overlap/' + path + '_SA_UM' + model.split('m')[0] + '_grid_percentages.csv'

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

    print('end')

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

    print('end')

    # temp read in cube

    if model == 'ukv':
        pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/' + model.split('m')[0] + '/umnsaa_pexptb023.pp'
    else:
        pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/pp/20160512T1200Z/' + model.split('m')[0] + 'm/umnsaa_pexptb023.pp'

    assert os.path.isfile(pp_file_path)

    # takes a long time
    cube = iris.load(pp_file_path, variable_name)[0]

    rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

    ll = ccrs.Geodetic()

    # convert coords
    inProj = Proj(init='epsg:32631')
    outProj = Proj(init='epsg:4326')

    lat_inds = []
    lon_inds = []

    tuple_list = []

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

    print('end')

    target_grid_coords['lat_inds'] = lat_inds
    target_grid_coords['lon_inds'] = lon_inds
    target_grid_coords['ind_tuples'] = lat_lon_tuples

    var_array = nc_file.variables[variable_name][0, :, :]

    # check if the model domain is a square

    if model == 'ukv':
        array_size_i = var_array.shape[0]
        array_size_j = var_array.shape[1]

    else:
        assert latitudes.shape[0] == longitudes.shape[0]
        assert var_array.shape[0] == var_array.shape[1]

        array_size_i = var_array.shape[0]
        array_size_j = var_array.shape[0]

    # start with an array full of nans
    a = np.full((array_size_i, array_size_j), np.nan)
    sa_a = np.full((array_size_i, array_size_j), np.nan)

    # lats
    for i in range(0, array_size_i):
        # lons
        for j in range(0, array_size_j):
            if (i, j) in lat_lon_tuples:
                a[i, j] = var_array[i, j]

                sa_val = float(target_grid_coords.loc[target_grid_coords['ind_tuples'] == (i, j)][str(target_hour).zfill(2)])
                sa_a[i, j] = sa_val

    if np.isclose(100, np.nansum(sa_a)):
        pass
    else:
        print('end')

    weighted_a = (sa_a / np.nansum(sa_a)) * a
    weighted_av_a = np.nansum(weighted_a)
    a_weighted_percent = (weighted_a / np.nansum(weighted_a)) * 100

    # threshold_list = np.arange(0.0001, 1, 0.0001).tolist()
    #
    # w_av_threshold_list = []
    # len_grids_threshold_list = []
    #
    # print(' ')
    # for i in threshold_list:
    #     print(i)
    #
    #     # loop through different thresholds
    #     temp_sa = np.ma.masked_where(sa_a <= i, sa_a)
    #     temp_a = np.ma.masked_where(sa_a <= i, a)
    #
    #     weighted_temp_a = (temp_sa / np.nansum(temp_sa)) * temp_a
    #
    #     weighted_av_temp_a = np.nansum(weighted_temp_a)
    #     w_av_threshold_list.append(weighted_av_temp_a)
    #
    #     len_grids_temp = temp_sa.count() + np.count_nonzero(~np.isnan(temp_a.data)) - np.count_nonzero(
    #         ~np.isnan(temp_a.mask))
    #     len_grids_threshold_list.append(len_grids_temp)
    #
    # # """
    #
    # fig, ax1 = plt.subplots(figsize=(10,6))
    # ax2 = ax1.twinx()
    #
    # plt.title(str(target_DOY) + ' ' + str(target_hour).zfill(2))
    #
    # ax1.scatter(threshold_list, w_av_threshold_list, c='r', marker='.')
    #
    # ax1.set_ylabel('Weighted Average UM100 $Q_{H}$ ($W$ $m^{2}$)')
    #
    # ax2.set_ylabel('# of UM100 grids')
    #
    # ax1.set_xlabel('SA Weight Threshold')
    #
    # ax2.spines['left'].set_color('r')
    #
    # ax2.spines['right'].set_color('b')
    #
    # ax2.tick_params(which='both', color='blue')
    # ax1.tick_params(axis='y', which='both', color='red')
    # ax1.yaxis.label.set_color('r')
    # ax2.yaxis.label.set_color('b')
    #
    # ax1.tick_params(axis='y', colors='r')
    # ax2.tick_params(axis='y', colors='b')
    #
    # ax2.scatter(threshold_list, len_grids_threshold_list, c='b', marker='.')
    #
    # current_path = os.getcwd().replace('\\', '/') + '/plots/changing_thresholds/'
    # plt.savefig(current_path + path + '_' + str(target_hour).zfill(2) + '.png', layout='tight', dpi=300)
    #
    # # """
    #
    # print('end')


    # """
    # create or write to csv file of weighted average values
    out_df = pd.DataFrame({'hour': [target_hour], 'weighted_av_a': [weighted_av_a], 'av_a': [np.nanmean(a)],
                           'len_grids': [len(target_grid_list)]})
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

    
    # plot the data in real world coords
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection=ccrs.epsg(32631))

    # plot the land cover map
    landcover_raster_filepath = 'C:/Users/beths/OneDrive - University of Reading/Model_Eval/QGIS/Elliott/LandUseMM_7classes_32631.tif'
    landcover_raster = rasterio.open(landcover_raster_filepath)
    # color_list_lc = ["white", "dimgrey", "lightgrey", "deepskyblue", "lawngreen", "darkgreen", "limegreen", "olive"]
    color_list_lc = ["white", "black", "dimgrey", "white", "lightgrey", "lightgrey", "lightgrey", "lightgrey"]
    # make a color map of fixed colors
    cmap_lc = colors.ListedColormap(color_list_lc)
    bounds_lc = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm_lc = colors.BoundaryNorm(bounds_lc, cmap_lc.N)
    rasterio.plot.show(landcover_raster, ax=ax, cmap=cmap_lc, norm=norm_lc, interpolation='nearest', alpha=0.5)

    # plot paths
    scint_shp_file_path = 'D:/Documents/scint_plots/scint_plots/sa_position_and_lc_fraction/scint_path_shp/BCT_IMU.shp'
    df_12 = geopandas.read_file(scint_shp_file_path)
    df_12.plot(edgecolor='red', ax=ax, linewidth=3.0)

    # Plot observation SA
    # open SA file
    sa_file = 'D:/Documents/scint_UM100/scint_UM100/SA_134/BCT_IMU_15000_2016_134_' + str(target_hour).zfill(
        2) + '_00.tif'
    raster = rasterio.open(sa_file)
    raster_array = raster.read()
    # make all 0 vals in array nan
    raster_array[raster_array == 0.0] = np.nan
    # force non-zero vals to be 1
    bool_arr = np.ones(raster_array.shape)
    # remove nans in bool array
    nan_index = np.where(np.isnan(raster_array))
    bool_arr[nan_index] = 0.0
    # plot
    rasterio.plot.show(bool_arr, contour=True, transform=raster.transform, contour_label_kws={}, ax=ax, zorder=10)

    target_grid_coords['grid'] = target_grid_coords['grid'].astype(int)

    # plot the grid boundry box polygons
    grid_dir = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_polygons/UM' + model.split('m')[0] + '_shapes/'
    for grid in target_grid_coords.grid.to_list():
        grid_file_path = grid_dir + str(grid) + '.gpkg'

        try:
            assert os.path.isfile(grid_file_path)
        except:
            print('end')

        grid_gpkg = geopandas.read_file(grid_file_path)
        grid_gpkg.boundary.plot(ax=ax, color='skyblue')

        # ax.annotate(text=grid_file_path.split('.')[0].split('/')[-1], xy=grid_gpkg.iloc[0].geometry.centroid.coords[0])

    plt.figtext(0.2, 0.83, 'n grids = ' + str(len(target_grid_list)))
    plt.figtext(0.2, 0.8, 'W av = ' + str(round(weighted_av_a, 1)))

    # plot the UM data
    # from stackoverflow issue: https://stackoverflow.com/questions/62346854/how-to-convert-projection-x-and-y-coordinate-in-netcdf-iris-cube-to-lat-lon
    # get UM coords
    proj_x = cube.coord("grid_longitude").points
    proj_y = cube.coord("grid_latitude").points

    # get UM coord systems
    cs_nat = cube.coord_system()
    cs_nat_cart = cs_nat.as_cartopy_projection()

    sa_a_norm = (sa_a - np.nanmin(sa_a)) / (np.nanmax(sa_a) - np.nanmin(sa_a))
    sa_a_norm[np.isnan(sa_a_norm)] = 0

    im = ax.pcolormesh(proj_x,
                       proj_y,
                       a,
                       transform=cs_nat_cart,
                       cmap='jet',
                       alpha=sa_a_norm)

    plt.colorbar(im)

    # limits which stay constant between and which suit both day's SAs
    ax.set_xlim(283000, 287000)
    ax.set_ylim(5711000, 5718000)

    plt.title('DOY 134: ' + str(target_hour).zfill(2))

    # im = ax.pcolormesh(proj_x,
    #                    proj_y,
    #                    a,
    #                    transform=cs_nat_cart,
    #                    cmap='jet')

    current_path = os.getcwd().replace('\\', '/') + '/plots/'

    plt.savefig(current_path + model + '/' + path + '_' + str(target_hour).zfill(2) + '.png', layout='tight')

    # """


print('end')

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
