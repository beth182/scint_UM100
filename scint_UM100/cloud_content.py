# imports
from os import listdir
from os.path import isfile, join
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os

from model_eval_tools.retrieve_UKV import read_premade_model_files

# user choices
# model = '100m'
# model = '300m'
model = 'ukv'

run = '20160512T1200Z'

# QH on the hour
target_filetype = 'pexptb'

variable_name = 'cloud_volume_fraction_in_atmosphere_layer'

save_path = os.getcwd().replace('\\', '/') + '/'

# data location
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
netcdf_dir = main_dir + 'netcdf/' + run + '/' + model + '/'

file_list = [f for f in listdir(netcdf_dir) if isfile(join(netcdf_dir, f))]

# find just the files of the target stash
target_files = []

for file in file_list:

    if target_filetype in file:
        target_files.append(netcdf_dir + file)

for file_path in target_files:


    # read data
    nc_file = nc.Dataset(file_path)

    print('end')