import rasterio  # the GEOS-based raster package
import numpy  # the array computation library
import geopandas  # the GEOS-based vector package
import matplotlib.pyplot as plt  # the visualization package
import numpy as np
from rasterio.mask import mask
import pylab
import matplotlib as mpl
import pandas as pd
from os import listdir
from os.path import isfile, join

import warnings
warnings.filterwarnings("ignore")

# SA location
sa_dir = 'D:/Documents/scint_UM100/scint_UM100/SA_134/'

# ToDo: make this flexable
sa_file = 'BCT_IMU_15000_2016_134_12_00.tif'


raster_path = sa_dir + sa_file



raster = rasterio.open(raster_path)
SA_data = raster.read(1)

# value of the total values across all of the raster
total_SA_sum = np.nansum(SA_data)

raster_extent = numpy.asarray(raster.bounds)[[0, 2, 1, 3]]

gpkg_dir_path = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/UM100_shapes/'

# plotting all grids against raster data
f = plt.figure(figsize=(20, 20))

# SA_data = np.ma.masked_where(SA_data < (total_SA_sum / 10000), SA_data)
cmap = mpl.cm.jet
cmap.set_bad('white', 1.)
plt.imshow(SA_data, interpolation='none', cmap=cmap, extent=raster_extent)


grid_file_list = [f for f in listdir(gpkg_dir_path) if isfile(join(gpkg_dir_path, f))]

# make sure there is no QGIS nonsense with len files
assert len(grid_file_list) == 25600

# makes list to see total value outside loop
grid_vals = {}

for i in range(1, len(grid_file_list) + 1):

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

    except ValueError:
        # traceback.print_exc()
        grid_sum = np.nan

    grid_vals[i] = grid_sum

print('end')