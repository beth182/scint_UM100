import os
import netCDF4 as nc
import matplotlib.pyplot as plt

file_path = '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/LC08_L2SP_202024_20160513_LST.nc'

assert os.path.isfile(file_path)



nc_file = nc.Dataset(file_path)

ST = nc_file.variables['ST'][:]

plt.figure(figsize=(10, 10))
im = plt.imshow(ST, cmap='jet', origin='lower')
cbar = plt.colorbar(im, fraction=0.046, pad=0.01, orientation='horizontal')
cbar.set_label('Surface Temperature (K)')
plt.xticks([])
plt.yticks([])

save_path = os.getcwd().replace('\\', '/') + '/'
plt.savefig(save_path + 'ST_134.png', bbox_inches='tight', dpi=300)


print('end')