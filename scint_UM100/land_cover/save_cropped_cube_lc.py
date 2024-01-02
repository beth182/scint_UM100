import os
import iris

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

iris.save(extracted_300,
          '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/nc_ancils_crop/' + 'extracted_300.nc')
iris.save(extracted_ukv,
          '//rdg-home.ad.rdg.ac.uk/research-nfs/basic/micromet/Tier_processing/rv006011/UM100/nc_ancils_crop/' + 'extracted_ukv.nc')

print('end')
