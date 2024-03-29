import rasterio  # the GEOS-based raster package
import geopandas  # the GEOS-based vector package
import matplotlib.pyplot as plt  # the visualization package
import numpy as np  # the array computation library
from rasterio.mask import mask
import matplotlib as mpl
import glob
import pandas as pd
import os

import warnings

warnings.filterwarnings("ignore")

# model = '100m'
# model = '300m'
model = 'ukv'


# SA location
sa_dir = os.getcwd().replace('\\', '/') + '/../../SA_134/'
# sa_dir = '/scint_UM100/SA_134/'

path = 'BCT_IMU'
# path = 'BTT_BCT'
# path = 'IMU_BTT'

# os.chdir(sa_dir)
hour_list = []
for file in glob.glob(sa_dir + path + "*.tif"):
    hour = file.split('.')[-2].split('_')[-2]
    hour_list.append(hour)

for hour in hour_list:

    print(hour)

    sa_file = path + '_15000_2016_134_' + hour + '_00.tif'

    raster_path = sa_dir + sa_file

    raster = rasterio.open(raster_path)
    SA_data = raster.read(1)

    # value of the total values across all of the raster
    total_SA_sum = np.nansum(SA_data)

    raster_extent = np.asarray(raster.bounds)[[0, 2, 1, 3]]

    # temp reduced grids
    # gpkg_dir_path = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/UM100_shapes_reduced/'

    gpkg_dir_path = os.getcwd().replace('\\', '/') + '/../grid_coord_lookup/grid_polygons/' + 'UM' + model.split('m')[0] + '_shapes/'

    # plotting all grids against raster data
    f = plt.figure(figsize=(20, 20))

    # SA_data = np.ma.masked_where(SA_data < (total_SA_sum / 10000), SA_data)
    cmap = mpl.cm.jet
    cmap.set_bad('white', 1.)
    plt.imshow(SA_data, interpolation='none', cmap=cmap, extent=raster_extent)

    grid_file_list = glob.glob(gpkg_dir_path + '*.gpkg')

    # make sure there is no QGIS nonsense with len files
    # assert len(grid_file_list) == 25600

    # makes list to see total value outside loop
    grid_vals = {}


    if model == '100m':
        if path == 'BCT_IMU':
            start_grid = 15000
            stop_grid = 20000
            # stop_grid = 15500
        elif path == 'BTT_BCT':
            start_grid = 11000
            stop_grid = 20000
        elif path == 'IMU_BTT':
            start_grid = 10500
            stop_grid = 17500
        else:
            print('end')
    elif model == '300m':
        start_grid = 1
        stop_grid = 2401
    elif model == 'ukv':
        start_grid = 1
        stop_grid = 122
    else:
        print('end')

    for i in range(start_grid, stop_grid):

        print(i)

        grid_num = str(i)
        gpkg_file_name = grid_num + '.gpkg'

        gpkg_file = gpkg_dir_path + gpkg_file_name
        grid_gpkg = geopandas.read_file(gpkg_file)

        # bounding box for our grid is the total_bounds of our grid dataframe:
        grid_bbox = grid_gpkg.total_bounds

        # Subsetting raster data
        # takes the coordinates from a bounding box and provides a rasterio.windows.Window object.
        # We can use it directly from our grid_bbox data using *, the unpack operator
        grid_window = raster.window(*grid_bbox)

        # Now, when we read() in our data, we can use this rasterio.windows.Window object to only read
        # the data we want to analyze: that within the bounds of the grid:
        grid_SA_data = raster.read(1, window=grid_window)

        # adding to the all grid plot
        grid_gpkg.boundary.plot(ax=plt.gca(), color='skyblue')

        # Summing the SA data within the grid box
        # implementing zonal statistics using rasterio mask
        grid_geometry = grid_gpkg.geometry

        # try except Value error here because ValueError occurs when the grid has no overlap of raster
        try:
            masked, mask_transform = mask(dataset=raster,
                                          shapes=grid_geometry, crop=True,
                                          all_touched=False,
                                          filled=False)

            # The mask function returns an array with the shape (n_bands, n_rows, n_columns).
            # Thus, when working with a single band, the mask function will output an array that has an 'extra'
            # dimension; the array will be three-dimensional despite the output only having two effective dimensions.
            # So, we must use squeeze() to remove extra dimensions that only have one element
            masked = masked.squeeze()

            # summing the values in this grid
            summed_val = np.nansum(masked)
            # as a percentage of the total raster, and rounded to 4 sig fig.
            grid_sum = round((summed_val / total_SA_sum) * 100, 4)
            print(grid_sum)

        except:
            # traceback.print_exc()
            grid_sum = np.nan

        grid_vals[i] = grid_sum

    calculated_sum_list = []
    for grid in sorted(grid_vals):
        calculated_sum_list.append(grid_vals[grid])

    # sum calculated by adding the sums of all individual grids
    calculated_sum = np.nansum(calculated_sum_list)

    df = pd.Series(grid_vals, index=grid_vals.keys())
    df = df.dropna()

    # in percentages of sum:
    percent = (df / df.sum()) * 100
    percent.name = sa_file.split('.')[0].split('_')[-2] + ' %'

    df.name = sa_file.split('.')[0].split('_')[-2]

    df_full = pd.concat([df, percent], axis=1)

    df_data = df_full.iloc[np.where(df_full[sa_file.split('.')[0].split('_')[-2] + ' %'] > 0)[0]]

    # pylab.savefig(save_path + 'raster_grids_' + time_string + '.png', bbox_inches='tight')

    if os.path.isfile(os.getcwd().replace('\\', '/') + '/' + path + '_SA_' + 'UM' + model.split('m')[0] + '_grid_percentages.csv'):
        # read existing csv
        existing_df = pd.read_csv(os.getcwd().replace('\\', '/') + '/' + path+ '_SA_' + 'UM' + model.split('m')[0] + '_grid_percentages.csv')

        existing_df.index = existing_df['Unnamed: 0']
        existing_df = existing_df.drop(columns=['Unnamed: 0'])
        existing_df.index.name = 'grid'

        all_df = pd.concat([df_data, existing_df], axis=1)
        all_df = all_df.fillna(0)

        all_df.to_csv(os.getcwd().replace('\\', '/') + '/' + path + '_SA_' + 'UM' + model.split('m')[0] + '_grid_percentages.csv')

    else:
        df_data.to_csv(os.getcwd().replace('\\', '/') + '/' + path + '_SA_' + 'UM' + model.split('m')[0] + '_grid_percentages.csv')

print('end')
