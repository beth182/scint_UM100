# imports
import os
import netCDF4 as nc
import matplotlib.pyplot as plt

from scint_UM100.data_retreval import retrieve_data_funs


# user inputs
target_DOY = 134
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


# handle time
run_times = retrieve_data_funs.handle_time(nc_file)
assert run_times[0].hour == target_hour
assert run_times[0].strftime('%j') == str(target_DOY)


level_height = nc_file.variables['level_height'][:]


# handle how to target grids
# nc_file.variables['upward_heat_flux_in_air'][:,400,400]

print('end')
