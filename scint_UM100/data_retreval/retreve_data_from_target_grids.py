# imports
import os
import netCDF4 as nc
import numpy as np
import datetime as dt

from model_eval_tools.retrieve_UKV import read_premade_model_files


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



# reads in time
# get time units for time conversion and start time
unit_start_time = nc_file.variables['time'].units

# Read in minutes since the start time and add it on
# Note: time_to_datetime needs time_since to be a list. Hence put value inside a single element list first
time_since_start = [np.squeeze(nc_file.variables['forecast_reference_time'])]

run_start_time = read_premade_model_files.time_to_datetime(unit_start_time, time_since_start)[0]

# get number of forecast hours to add onto time_start
run_len_hours = np.squeeze(nc_file.variables['forecast_period'][:]).tolist()

if type(run_len_hours) == float:
    run_len_hours = [run_len_hours]

run_times = [run_start_time + dt.timedelta(seconds=hr * 3600) for hr in run_len_hours]



print('end')
