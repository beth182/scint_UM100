# imports
from os import listdir
from os.path import isfile, join
import os
import iris
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.pyplot as plt

# user choices
model = '100m'
# model = '300m'
# model = 'ukv'

run = '20160512T1200Z'

# QH on the hour
target_filetype = 'pvera'

variable_name = 'surface_upward_sensible_heat_flux'

save_path = os.getcwd().replace('\\', '/') + '/'

# data location
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

pp_dir = main_dir + 'pp/' + run + '/' + model + '/'

file_list_pp = [f for f in listdir(pp_dir) if isfile(join(pp_dir, f))]

target_files_pp = []
target_files_nc = []

for file in file_list_pp:

    if target_filetype in file:
        target_files_pp.append(pp_dir + file)
        target_files_nc.append(netcdf_dir + file.split('.')[0] + '.nc')

target_pp_file = target_files_pp[0]
target_nc_file = target_files_nc[0]

assert os.path.isfile(target_pp_file)
assert os.path.isfile(target_nc_file)

########################################################################################################################

# load iris cube from pp file
cube = iris.load(target_pp_file, variable_name)[0]

# get UM coords
proj_x = cube.coord("grid_longitude").points
proj_y = cube.coord("grid_latitude").points

# get UM coord systems
cs_nat = cube.coord_system()
cs_nat_cart = cs_nat.as_cartopy_projection()

# read corresponding nc file
nc_file = nc.Dataset(target_nc_file)

# first timestep
QH = nc_file[variable_name][:][0]

# plot the data in real world coords
# from stackoverflow issue: https://stackoverflow.com/questions/62346854/how-to-convert-projection-x-and-y-coordinate-in-netcdf-iris-cube-to-lat-lon
fig = plt.figure()
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.pcolormesh(proj_x,
              proj_y,
              QH,
              transform=cs_nat_cart)

ax.set_ylim(51.4, 51.59)
ax.set_xlim(-0.23, 0)

print('end')
