from os import listdir
from os.path import isfile, join
import os
import iris

# Actual version of this script is on the RACC

# model = '100m'
# model = '300m'
model = 'ukv'

run = '20160512T1200Z'

# main_dir = '/storage/basic/micromet/Tier_processing/rv006011/UM100/'
# main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/"
main_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/landuse/maggie_new/"

if model == 'ukv':

    pp_dir = main_dir

    netcdf_dir = "//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/netcdf/" + run + '/UM' + model + '_ancillaries/'


else:


    # pp_dir = main_dir + 'pp/' + run + '/' + model + '/'
    pp_dir = main_dir + 'UM100_ancillaries/london_' + model + '/'

    netcdf_dir = main_dir + 'netcdf/' + run + '/UM' + model.split('m')[0] + '_ancillaries/'

# check if the write out dir exists
if os.path.isdir(netcdf_dir) == False:
    os.makedirs(netcdf_dir)

file_list = [f for f in listdir(pp_dir) if isfile(join(pp_dir, f))]

for file_path in file_list:
    print(pp_dir + file_path)

    cubes = iris.load(pp_dir + file_path)
    iris.save(cubes, netcdf_dir + file_path.replace('.', '_') + '.nc')
