# imports
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio.plot
import geopandas
import matplotlib.colors as colors
import cartopy.crs as ccrs

import warnings


def QH_alpha_plot(target_hour, model, path, target_grid_coords, target_grid_list, cube, sa_a, a, weighted_av_a):
    """

    :return:
    """

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


    # plot the grid boundry box polygons
    grid_dir = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_polygons/UM' + model.split('m')[
        0] + '_shapes/'
    for grid in target_grid_coords.dropna().grid.to_list():
        grid = int(grid)
        grid_file_path = grid_dir + str(grid) + '.gpkg'

        try:
            assert os.path.isfile(grid_file_path)
        except:
            print('end')

        grid_gpkg = geopandas.read_file(grid_file_path)
        grid_gpkg.boundary.plot(ax=ax, color='skyblue')

        # ax.annotate(text=grid_file_path.split('.')[0].split('/')[-1], xy=grid_gpkg.iloc[0].geometry.centroid.coords[0])

    t = plt.figtext(0.35, 0.83,
                    'n grids = ' + str(len(target_grid_list)) + '\n' + 'W av = ' + str(round(weighted_av_a, 1)))
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=None))

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

    if target_hour == 12:

        im = ax.pcolormesh(proj_x,
                           proj_y,
                           a,
                           transform=cs_nat_cart,
                           cmap='jet',
                           alpha=sa_a_norm,
                           vmin=60,
                           vmax=425)

    else:

        im = ax.pcolormesh(proj_x,
                           proj_y,
                           a,
                           transform=cs_nat_cart,
                           cmap='jet',
                           alpha=sa_a_norm)

    plt.colorbar(im, pad=0.01)

    # limits which stay constant between and which suit both day's SAs
    ax.set_xlim(283000, 287000)
    ax.set_ylim(5711000, 5718000)

    plt.title('DOY 134: ' + str(target_hour).zfill(2))

    current_path = os.getcwd().replace('\\', '/') + '/plots/'

    plt.savefig(current_path + model + '/' + path + '_' + str(target_hour).zfill(2) + '.png', bbox_inches='tight',
                dpi=300)
