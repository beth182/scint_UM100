import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import iris
import cartopy.crs as ccrs

model = '100m'
stash_code = 'm01s00i216'



# read in the ancil file for the model


if model == 'ukv':

    pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/landuse/maggie_new/qrparm.veg.frac'

else:

    pp_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/UM100_ancillaries/london_' + model + '/qrparm.veg.frac.urb2t'


nc_file_path = nc_file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/netcdf/20160512T1200Z/UM' + model.split('m')[0] + '_ancillaries/qrparm_veg_frac_urb2t.nc'



assert os.path.isfile(nc_file_path)
assert os.path.isfile(pp_file_path)


# locd the PP file to get the coord rotation
# load one file into iris first
# load iris cube from pp file
cube = iris.load(pp_file_path, stash_code)[0]
# get UM coords
proj_x = cube.coord("grid_longitude").points
proj_y = cube.coord("grid_latitude").points

# get UM coord systems
cs_nat = cube.coord_system()
cs_nat_cart = cs_nat.as_cartopy_projection()






# read the nc file for vals
nc_file = nc.Dataset(nc_file_path)
lc_fracs = nc_file.variables[stash_code]
urban = nc_file.variables[stash_code][-1, :, :] + nc_file.variables[stash_code][-2, :, :]


print('end')

# plot in read world coords
fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax = fig.add_subplot(111, projection=ccrs.epsg(32631))

im = ax.pcolormesh(proj_x,
                   proj_y,
                   urban,
                   transform=cs_nat_cart,
                   cmap='jet')

print('end')