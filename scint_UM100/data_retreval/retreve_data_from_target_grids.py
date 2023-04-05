# imports
import os
import netCDF4 as nc


# user inputs
target_hour = 12
variable_name = 'surface_upward_sensible_heat_flux'
model = '100m'
run = '20160512T1200Z'
levels = True



# ToDo: move this to a lookup
if levels == True:
    target_filetype = 'pexptb'
else:
    target_filetype = 'pvera'


# first ouput timestamp is 1300 on the day before (DOY 133). So add 11 hours to get to midnight of target day (134)
file_index_hour = 11 + target_hour



main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

# construct file name
target_file_name = 'umnsaa_' + target_filetype + str(file_index_hour).zfill(3) + '.nc'

# construct total path and check the file exists
target_file_path = netcdf_dir + target_file_name
assert os.path.isfile(target_file_path)

# read file
nc_file = nc.Dataset(target_file_path)


print('end')
