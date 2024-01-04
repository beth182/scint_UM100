# imports
import os
import iris
import cartopy.crs as ccrs
from pyproj import Proj, transform
import netCDF4 as nc
import pandas as pd
import numpy as np

# get the location of the target grid in rw coords
# convert to MO cords
# get the LC fractions for that grid
# save them as a file

model = '100m'
# model = '300m'
stash_code = 'surface_altitude'

# get the location of the target grid in rw coords
# existing csv:
coord_lookup_csv = 'D:/Documents/scint_UM100/scint_UM100/grid_coords/grid_coord_lookup/grid_coords_' + model + '.csv'

# read the csv
coord_lookup_df = pd.read_csv(coord_lookup_csv)

grid_coords = coord_lookup_df[coord_lookup_df.descrip == 'MID']

# convert to MO cords
# temp read in cube

pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/UM100_ancillaries/london_' + model + '/qrparm.orog.mn'

assert os.path.isfile(pp_file_path)

# takes a long time
cube = iris.load(pp_file_path)[0]

rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs()

ll = ccrs.Geodetic()

# convert coords
inProj = Proj(init='epsg:32631')
outProj = Proj(init='epsg:4326')

# read nc file
nc_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/netcdf/20160512T1200Z/UM' + \
               model.split('m')[0] + '_ancillaries/qrparm_orog_mn.nc'

assert os.path.isfile(nc_file_path)

nc_file = nc.Dataset(nc_file_path)

LC_array = nc_file.variables[stash_code]

assert LC_array.shape[0] == LC_array.shape[1]

lc_df_list = []

for index, row in grid_coords.iterrows():
    name = row.grid
    print(int(name), ' / ', len(grid_coords))

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

    latitudes = cube.coord('grid_latitude')
    longitudes = cube.coord('grid_longitude')

    assert latitudes.shape[0] == longitudes.shape[0]

    nearest_lat = latitudes.nearest_neighbour_index(y_new)
    nearest_lon = longitudes.nearest_neighbour_index(x_new)

    coord_tuple = (nearest_lat, nearest_lon)
    i = coord_tuple[0]
    j = coord_tuple[1]

    LC_grid = LC_array[i, j]

    # create a df with this info in
    lc_df = pd.DataFrame.from_dict({'altitude': [LC_grid]})
    lc_df.index = [name]
    lc_df_list.append(lc_df)

save_path = os.getcwd().replace('\\', '/') + '/'
all_grids_df = pd.concat(lc_df_list)
all_grids_df.to_csv(save_path + model + '_all_grids_altitude.csv')
print('CSV saved to: ' + save_path + model + '_all_grids_altitude.csv')

print('end')
