import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import iris
import cartopy.crs as ccrs

stash_code = 'm01s00i216'

pp_file_path_100 = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/UM100_ancillaries/london_' + '100m' + '/qrparm.veg.frac.urb2t'
pp_file_path_300 = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/UM100_ancillaries/london_' + '300m' + '/qrparm.veg.frac.urb2t'
pp_file_path_ukv = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/landuse/maggie_new/qrparm.veg.frac'

assert os.path.isfile(pp_file_path_100)
assert os.path.isfile(pp_file_path_300)
assert os.path.isfile(pp_file_path_ukv)

# load the UM cubes

cube_100 = iris.load(pp_file_path_100, stash_code)[0]
cube_300 = iris.load(pp_file_path_300, stash_code)[0]

cube_ukv = iris.load(pp_file_path_ukv, stash_code)[0]

# get UM coords
proj_x_100 = cube_100.coord("grid_longitude").points
proj_y_100 = cube_100.coord("grid_latitude").points

# get the max and min coords for the UM100
proj_x_100_max = proj_x_100.max()
proj_x_100_min = proj_x_100.min()
proj_y_100_max = proj_y_100.max()
proj_y_100_min = proj_y_100.min()

# create a constraint the to match 300 and UKV cube to 100's domain
gcon_300 = iris.Constraint(coord_values={'grid_latitude': lambda cell: proj_y_100_min < cell < proj_y_100_max,
                                         'grid_longitude': lambda cell: proj_x_100_min < cell < proj_x_100_max})

gcon_ukv = iris.Constraint(coord_values={'grid_latitude': lambda cell: proj_y_100_min < cell < proj_y_100_max,
                                         'grid_longitude': lambda
                                             cell: proj_x_100_min + 360 < cell < proj_x_100_max + 360})

# extract
extracted_300 = cube_300.extract(gcon_300)
extracted_ukv = cube_ukv.extract(gcon_ukv)

# get UM coords for extracted
proj_x_300 = extracted_300.coord("grid_longitude").points
proj_y_300 = extracted_300.coord("grid_latitude").points

proj_x_ukv = extracted_ukv.coord("grid_longitude").points
proj_y_ukv = extracted_ukv.coord("grid_latitude").points

# load the nc files

nc_file_path_100 = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/netcdf/20160512T1200Z/UM' + '100' + '_ancillaries/qrparm_veg_frac_urb2t.nc'
nc_file_path_300 = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/nc_ancils_crop/extracted_300.nc'
nc_file_path_ukv = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/nc_ancils_crop/extracted_ukv.nc'

assert os.path.isfile(nc_file_path_100)
assert os.path.isfile(nc_file_path_300)
assert os.path.isfile(nc_file_path_ukv)

# read the nc files for vals
nc_file_100 = nc.Dataset(nc_file_path_100)
urban_100 = nc_file_100.variables[stash_code][-1, :, :] + nc_file_100.variables[stash_code][-2, :, :]

nc_file_300 = nc.Dataset(nc_file_path_300)
urban_300 = nc_file_300.variables[stash_code][-1, :, :] + nc_file_300.variables[stash_code][-2, :, :]

nc_file_ukv = nc.Dataset(nc_file_path_ukv)
urban_ukv = nc_file_ukv.variables[stash_code][-1, :, :] + nc_file_ukv.variables[stash_code][-2, :, :]

print('end')

fig = plt.figure(figsize=(20, 7))
ax = fig.add_subplot(131)
im = ax.pcolormesh(proj_x_100,
                   proj_y_100,
                   urban_100,
                   vmin=0,
                   vmax=1,
                   cmap='jet')

ax2 = fig.add_subplot(132)
im2 = ax2.pcolormesh(proj_x_300,
                     proj_y_300,
                     urban_300,
                     vmin=0,
                     vmax=1,
                     cmap='jet')

ax3 = fig.add_subplot(133)
im3 = ax3.pcolormesh(proj_x_ukv,
                     proj_y_ukv,
                     urban_ukv,
                     vmin=0,
                     vmax=1,
                     cmap='jet')

ax.set_xticklabels([])
ax.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax3.set_xticklabels([])
ax3.set_yticklabels([])

ax.set_xticks([])
ax.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])


plt.subplots_adjust(wspace=0.01, hspace=0.01)

save_path = os.getcwd().replace('\\', '/') + '/'
plt.savefig(save_path + 'urban_frac.png', bbox_inches='tight', dpi=300)

print('end')
