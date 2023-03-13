#%%
import iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob
import datetime
import iris.analysis.cartography
import matplotlib.patches as mpatches

#%%
path='/data/users/lblunn/UM100/c3_QE_tests'

# %%
lons_100m = iris.load(f"{path}/umnsaa_pvera012_100m","m01s00i024")[0].coord('grid_longitude').points
lats_100m = iris.load(f"{path}/umnsaa_pvera012_100m","m01s00i024")[0].coord('grid_latitude').points
lon_min = np.array([np.min(lons_100m)-0.00001])
lon_max = np.array([np.max(lons_100m)+0.00001])
lat_min = np.array([np.min(lats_100m)-0.00001])
lat_max = np.array([np.max(lats_100m)+0.00001])
pole_lon = 177.5
pole_lat = 37.5

#%%
choice = "glm"

#%%

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def constrain_lat_lon(choice,var,lat_min,lat_max,lon_min,lon_max):
    if choice == "ukv":
        lon_min = lon_min + 360.
        lon_max = lon_max + 360.
        lat_constraint = iris.Constraint(grid_latitude=lambda cell: lat_min < cell < lat_max)
        lon_constraint = iris.Constraint(grid_longitude=lambda cell: lon_min < cell < lon_max)
        var = var.extract(lat_constraint & lon_constraint)
        var.coord('grid_longitude').points = var.coord('grid_longitude').points-360.
    elif choice == "glm":
        pass
    else:
        lat_constraint = iris.Constraint(grid_latitude=lambda cell: lat_min < cell < lat_max)
        lon_constraint = iris.Constraint(grid_longitude=lambda cell: lon_min < cell < lon_max)
        var = var.extract(lat_constraint & lon_constraint)
    return var

def lons_lats(choice,var):
    if choice in ["glm"]:
        lon, lat, lonm, latm = glm_centre_coords(var)
    else:
        lon = var.coord('grid_longitude').points
        lat = var.coord('grid_latitude').points
        lonm, latm = np.meshgrid(lon, lat, indexing='xy')
    return lon, lat, lonm, latm

def glm_centre_coords(var):
    lon = var.coord('longitude').points
    lat = var.coord('latitude').points
    idx = find_nearest(lon,-180.)
    lon = np.concatenate([lon[idx:],lon[:idx]+360.])
    lonm, latm = np.meshgrid(lon, lat, indexing='xy')
    return lon, lat, lonm, latm

def constrain_glm_data(choice,var,vard):
    if choice == 'glm':
        lon_min_glm, lon_max_glm, lat_min_glm, lat_max_glm = \
            unrotate(lon_min, lon_max, lat_min, lat_max)
        lon_min_idx = find_nearest(lon, lon_min_glm)
        lon_max_idx = find_nearest(lon, lon_max_glm)
        lat_min_idx = find_nearest(lat, lat_min_glm)
        lat_max_idx = find_nearest(lat, lat_max_glm)
        idx = find_nearest(var.coord('longitude').points,-180.)
        vard = np.concatenate([vard[:,idx:],vard[:,:idx]],axis=1)
        vard = vard[lat_min_idx:lat_max_idx+1,lon_min_idx:lon_max_idx+1]
    else:
        pass
    return vard

def unrotate(lon_min, lon_max, lat_min, lat_max):
    lats = np.zeros(4)
    lons = np.zeros(4)
    vals = iris.analysis.cartography.unrotate_pole(lon_min, lat_min, pole_lon, pole_lat)
    lons[0], lats[0] = vals[0], vals[1]
    vals = iris.analysis.cartography.unrotate_pole(lon_min, lat_max, pole_lon, pole_lat)
    lons[1], lats[1] = vals[0], vals[1]
    vals = iris.analysis.cartography.unrotate_pole(lon_max, lat_min, pole_lon, pole_lat)
    lons[2], lats[2] = vals[0], vals[1]
    vals = iris.analysis.cartography.unrotate_pole(lon_max, lat_max, pole_lon, pole_lat)
    lons[3], lats[3] = vals[0], vals[1]
    lon_min, lon_max, lat_min, lat_max = np.min(lons), np.max(lons), np.min(lats), np.max(lats)
    return lon_min, lon_max, lat_min, lat_max

def constrain_coords(choice,lonm,latm):
    if choice == 'glm':
        lon_min_glm, lon_max_glm, lat_min_glm, lat_max_glm = \
            unrotate(lon_min, lon_max, lat_min, lat_max)
        lon_min_idx = find_nearest(lon, lon_min_glm)
        lon_max_idx = find_nearest(lon, lon_max_glm)
        lat_min_idx = find_nearest(lat, lat_min_glm)
        lat_max_idx = find_nearest(lat, lat_max_glm)
        lonm = lonm[lat_min_idx:lat_max_idx+1,lon_min_idx:lon_max_idx+1]
        latm = latm[lat_min_idx:lat_max_idx+1,lon_min_idx:lon_max_idx+1]
    else:
        pass
    return lonm, latm

def fig_func(lonm,latm,data,label,vmin,vmax):
    ax = plt.subplot()
    cmap = plt.cm.get_cmap('hot_r')
    CS = ax.pcolormesh(lonm, latm, data, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(CS, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=f"{label}",size=16)

def load_output(choice):
    if choice == 'glm':
        string = 'pa*glm'
    else:
        string = f'*pvera*{choice}'
    i=0
    for filename in sorted(glob.glob(f'{path}/*{string}*', recursive=True)):
        print(filename)
        if i == 0:
            cube = iris.load(f"{filename}","m01s00i024")[0]
        else:
            p = iris.load(f"{filename}","m01s00i024")[0]
            cube_list = iris.cube.CubeList([cube, p])
            cube = cube_list.concatenate()[0]
        i+=1
    cube = constrain_lat_lon(choice,cube,lat_min,lat_max,lon_min,lon_max)
    if choice == 'glm':
        cube.coord('longitude').points = cube.coord('longitude').points-360.
    return cube

#%%
# plot surface temperature throughout the day

times = ['20180715T1200Z']

T_all = load_output(choice)

for t in times:
    d = datetime.datetime.strptime(t, '%Y%m%dT%H%MZ')
    hour = iris.Constraint(
        time=lambda cell: d <= cell.point <= d)
    lon, lat, lonm, latm = lons_lats(choice,T_all)
    T = T_all.extract(hour)
    T_d = T.data-273.15
    T_d = constrain_glm_data(choice,T,T_d)
    lonm, latm = constrain_coords(choice,lonm,latm)
    fig = plt.figure(figsize=(8,8))
    fig_func(lonm=lonm,latm=latm,data=T_d,label=r"Surface Temperature ($^\circ$C)",vmin=np.floor(np.min(T_d)),vmax=np.ceil(np.max(T_d)))
    plt.title(f'Time: {t[6:11]}',fontsize=16)
    fig.savefig(f'./figs/Tsurf_{t[6:11]}_{choice}.jpg', dpi=200, bbox_inches='tight')


# %%

# %%