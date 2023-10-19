# imports
from os import listdir
from os.path import isfile, join
import os
import iris
import cartopy.crs as ccrs
from pyproj import Proj, transform
import pandas as pd

# user choices
# model = '100m'
# model = '300m'
model = 'ukv'

run = '20160512T1200Z'

# QH on the hour
target_filetype = 'pvera'

variable_name = 'surface_upward_sensible_heat_flux'

save_path = os.getcwd().replace('\\', '/') + '/'

# data location
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

pp_dir = main_dir + 'pp/' + run + '/' + model + '/'

file_list_nc = [f for f in listdir(netcdf_dir) if isfile(join(netcdf_dir, f))]
file_list_pp = [f for f in listdir(pp_dir) if isfile(join(pp_dir, f))]

# find just the files of the target stash
target_files_nc = []
target_files_pp = []

for file in file_list_pp:

    if target_filetype in file:
        target_files_pp.append(pp_dir + file)

target_pp_file = target_files_pp[0]



assert os.path.isfile(target_pp_file)

# LOAD AS A CUBE

cube = iris.load(target_pp_file, variable_name)[0]

rot_pole = cube.coord('grid_latitude').coord_system.as_cartopy_crs()
ll = ccrs.Geodetic()

# random coord in London - by IMU
# y_coord = 51.5260849
# x_coord = 359.89498901

# BTT coord
y_coord = 51.5214542
x_coord = 360 - 0.138843

target_xy = rot_pole.transform_point(x_coord, y_coord, ll)  # lower left corner INCORRECT: ACTUALLY MID POINT

x_new = target_xy[0]
y_new = target_xy[1]

latitudes = cube[0].coord('grid_latitude')
longitudes = cube[0].coord('grid_longitude')

if model == 'ukv':
    longitudes = longitudes - 360

nearest_lat_1 = latitudes.nearest_neighbour_index(y_new)
nearest_lon_1 = longitudes.nearest_neighbour_index(x_new)

nearest_lat_2 = nearest_lat_1 + 1
nearest_lat_0 = nearest_lat_1 - 1

nearest_lon_2 = nearest_lon_1 + 1
nearest_lon_0 = nearest_lon_1 - 1

lat_value_1 = latitudes.cell(nearest_lat_1)
lon_value_1 = longitudes.cell(nearest_lon_1)

lat_value_2 = latitudes.cell(nearest_lat_2)
lon_value_2 = longitudes.cell(nearest_lon_2)

lat_value_0 = latitudes.cell(nearest_lat_0)
lon_value_0 = longitudes.cell(nearest_lon_0)

# get coords of real-world back
real_world_xy_11 = ll.transform_point(lon_value_1[0], lat_value_1[0], rot_pole)

real_world_xy_22 = ll.transform_point(lon_value_2[0], lat_value_2[0], rot_pole)

real_world_xy_00 = ll.transform_point(lon_value_0[0], lat_value_0[0], rot_pole)

real_world_xy_10 = ll.transform_point(lon_value_1[0], lat_value_0[0], rot_pole)

real_world_xy_20 = ll.transform_point(lon_value_2[0], lat_value_0[0], rot_pole)

real_world_xy_01 = ll.transform_point(lon_value_0[0], lat_value_1[0], rot_pole)

real_world_xy_21 = ll.transform_point(lon_value_2[0], lat_value_1[0], rot_pole)

real_world_xy_02 = ll.transform_point(lon_value_0[0], lat_value_2[0], rot_pole)

real_world_xy_12 = ll.transform_point(lon_value_1[0], lat_value_2[0], rot_pole)

# convert coords
inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:32631')

p11 = transform(inProj, outProj, real_world_xy_11[0], real_world_xy_11[1])
p22 = transform(inProj, outProj, real_world_xy_22[0], real_world_xy_22[1])
p00 = transform(inProj, outProj, real_world_xy_00[0], real_world_xy_00[1])
p10 = transform(inProj, outProj, real_world_xy_10[0], real_world_xy_10[1])
p20 = transform(inProj, outProj, real_world_xy_20[0], real_world_xy_20[1])
p01 = transform(inProj, outProj, real_world_xy_01[0], real_world_xy_01[1])
p21 = transform(inProj, outProj, real_world_xy_21[0], real_world_xy_21[1])
p02 = transform(inProj, outProj, real_world_xy_02[0], real_world_xy_02[1])
p12 = transform(inProj, outProj, real_world_xy_12[0], real_world_xy_12[1])

point_list = ['p11', 'p22', 'p00', 'p10', 'p20', 'p01', 'p21', 'p02', 'p12']
x_list = [p11[0], p22[0], p00[0], p10[0], p20[0], p01[0], p21[0], p02[0], p12[0]]
y_list = [p11[1], p22[1], p00[1], p10[1], p20[1], p01[1], p21[1], p02[1], p12[1]]

df = pd.DataFrame.from_dict({'name': point_list, 'x': x_list, 'y': y_list})
df = df.set_index('name')

# df.to_csv(save_path + 'rotation_tests.csv')
# df.to_csv(save_path + 'rotation_tests_BTT.csv')
df.to_csv(save_path + 'rotation_tests_BTT_' + model + '.csv')

print('end')
