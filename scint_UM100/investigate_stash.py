# imports
from os import listdir
from os.path import isfile, join
import os
import iris
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import rasterio
import geopandas as gpd

from model_eval_tools.retrieve_UKV import read_premade_model_files

import scint_fp.functions.plot_functions.plot_sa_lines.sa_lines_funs as sa_lines

# user choices
model = '100m'
# model = '300m'
# model = 'ukv'

run = '20160512T1200Z'

# QH on the hour
# target_filetype = 'pvera'
target_filetype = 'pexptb'

# variable_name = 'surface_upward_sensible_heat_flux'
# variable_name = 'upward_air_velocity'
# variable_name = 'air_temperature'
variable_name = 'upward_heat_flux_in_air'

colour_dict = {'BCT_IMU': 'red', 'SCT_SWT': 'mediumorchid', 'IMU_BTT': 'green', 'BTT_BCT': 'blue'}

save_path = os.getcwd().replace('\\', '/') + '/'

# source area location on cluster
sa_base_dir = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/PycharmProjects/scintillometer_footprints/scint_fp/test_outputs/10_mins_ending/'
scint_shp_dir = 'D:/Documents/scint_plots/scint_plots/sa_position_and_lc_fraction/scint_path_shp/'

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


target_pp_file = target_files_pp[1]
target_nc_file = target_files_nc[1]

# load one file into iris first
# load iris cube from pp file
cube = iris.load(target_pp_file, variable_name)[0]

# cube.coord('time')
# cube.coord('time')[0]
# cube.coord('time')[0]._values

nc_file = nc.Dataset(target_nc_file)
# nc_file['time']
# nc_file['time'][0]
# nc_file[variable_name]



print('end')