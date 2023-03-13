# %%
import iris
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime
from datetime import timedelta
import iris.analysis.cartography
import matplotlib.patches as mpatches
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import os
import cartopy.crs as ccrs
from matplotlib import colors
import matplotlib
import matplotlib.dates as mdates

os.chdir("/net/home/h04/lblunn/Documents/Projects/UM100")
print(os.getcwd())

# %%
path = '/data/users/lblunn/UM100/case_20180715'
dirs = {"1500m": "1p5km_L70", "300m": "0p3km_L70", "100m": "0p1km_L140"}
res = "100m"
suffixes = ['000', '012', '024']


# %%
def load(path, dirs, lon_min, lon_max, lat_min, lat_max, res, file, var):
    dir = dirs[res]
    print(f"{path}/{dir}/{file}")
    cube = iris.load(f"{path}/{dir}/{file}", var)[0]
    print(cube)
    cube = constrain_lat_lon(cube, lon_min, lon_max, lat_min, lat_max)
    return cube


def constrain_lat_lon(cube, lon_min, lon_max, lat_min, lat_max):
    lat_constraint = iris.Constraint(grid_latitude=lambda cell: lat_min <= cell <= lat_max)
    lon_constraint = iris.Constraint(grid_longitude=lambda cell: lon_min <= cell <= lon_max)
    cube = cube.extract(lat_constraint & lon_constraint)
    return cube


def lons_lats(dataset):
    lon = dataset.coord('longitude').points
    lat = dataset.coord('latitude').points
    lonm, latm = np.meshgrid(lon, lat, indexing='xy')
    return lon, lat, lonm, latm


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# %%
###### Geometry ######
# %%

# %%
geom100 = iris.load(f"{path}/{dirs[res]}/umnsaa_pa000.pp", "m01s00i033")[0]
lon_min = np.array([np.min(geom100.coord('grid_longitude').points)])
lon_max = np.array([np.max(geom100.coord('grid_longitude').points)])
lat_min = np.array([np.min(geom100.coord('grid_latitude').points)])
lat_max = np.array([np.max(geom100.coord('grid_latitude').points)])

# %%
###### LOADING DATA ######
# %%

# %%
### land cover
lc = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
          lat_min=lat_min, lat_max=lat_max, res=res, \
          file=f"RA3_pack3_MORUSES_astart", var="m01s00i216")
print("lc:\n", lc)
# %%
### lai
lai = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
           lat_min=lat_min, lat_max=lat_max, res=res, \
           file=f"RA3_pack3_MORUSES_astart", var="m01s00i217")
print("lai:\n", lai)
# %%
### Tsurf
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_pvera{s}.pp", var="m01s00i024")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_pvera{s}.pp", var="m01s00i024")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
Tsurf = cube.copy()
print("Tsurf:\n", Tsurf)
# %%
### sensible heat flux
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i217")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i217")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
QH = cube.copy()
print("QH:", QH)
# %%
### latent heat flux
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i234")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i234")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
QE = cube.copy()
print("QE:", QE)
# %%
### ground flux
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i202")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i202")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
G = cube.copy()
print("G:", G)
# %%
### Lnet
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s02i201")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s02i201")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
Lnet = cube.copy()
print("Lnet:", Lnet)
# %%
### Ldown
suffixes = ['000', '012', '024']
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s02i207")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s02i207")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
Ldown = cube.copy()
print("Ldown:", Ldown)
# %%
### Lup
Lup = Ldown.copy()
Lup.data = Ldown.data - Lnet.data
# %%
### Knet
suffixes = ['000', '012', '024']
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s01i202")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s01i202")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
Knet = cube.copy()
print("Knet:", Knet)
# %%
### Kdown
suffixes = ['000', '012', '024']
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="c")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s01i235")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
Kdown = cube.copy()
print("Kdown:", Kdown)
# %%
### Kup
Kup = Kdown.copy()
Kup.data = Kdown.data - Knet.data
# %%
### latent heat flux on tiles
i = 0
for s in suffixes:
    if i == 0:
        cube = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                    lat_min=lat_min, lat_max=lat_max, \
                    res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i330")
    else:
        cube_temp = load(path=path, dirs=dirs, lon_min=lon_min, lon_max=lon_max, \
                         lat_min=lat_min, lat_max=lat_max, \
                         res=res, file=f"umnsaa_psurfa{s}.pp", var="m01s03i330")
        cube_list = iris.cube.CubeList([cube, cube_temp])
        cube = cube_list.concatenate()[0]
    i += 1
QE_tiles = cube.copy()
print("QE_tiles:", QE_tiles)


# %%
###### PLOTTING ######
# %%

# %%
# plot surface temperature

def lons_lats(choice, var):
    lon = var.coord('grid_longitude').points
    lat = var.coord('grid_latitude').points
    lonm, latm = np.meshgrid(lon, lat, indexing='xy')
    return lon, lat, lonm, latm


def fig_func(lonm, latm, data, label, vmin, vmax):
    ax = plt.subplot()
    cmap = plt.cm.get_cmap('hot_r')
    CS = ax.pcolormesh(lonm, latm, data, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(CS, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=f"{label}", size=16)


times = ['20180715T1200Z']

rp_cs = np.array([1.37, 1.37 + 0.04, -1.06, -1.06 + 0.04])
nd_cs = np.array([1.64, 1.64 + 0.04, -1.22, -1.22 + 0.04])
ld_cs = np.array([1.49, 1.49 + 0.04, -0.96, -0.96 + 0.04])
ht_cs = np.array([1.66, 1.66 + 0.04, -0.82, -0.82 + 0.04])
# ht_cs = np.array([1.69,1.69+0.01,-0.79,-0.79+0.01])

for t in times:
    d = datetime.datetime.strptime(t, '%Y%m%dT%H%MZ')
    hour = iris.Constraint(
        time=lambda cell: d <= cell.point <= d)
    lon, lat, lonm, latm = lons_lats(res, Tsurf)
    var = Tsurf.extract(hour)
    var_d = var.data - 273.15
    fig = plt.figure(figsize=(8, 8))
    fig_func(lonm=lonm, latm=latm, data=var_d, label=r"Surface Temperature ($^\circ$C)", vmin=np.floor(np.min(T_d)),
             vmax=np.ceil(np.max(var_d)))
    rect1 = mpatches.Rectangle((rp_cs[0], rp_cs[2]), rp_cs[1] - rp_cs[0], rp_cs[3] - rp_cs[2], \
                               fill=False, color="blue", linewidth=2)
    rect2 = mpatches.Rectangle((nd_cs[0], nd_cs[2]), nd_cs[1] - nd_cs[0], nd_cs[3] - nd_cs[2], \
                               fill=False, color="lime", linewidth=2)
    rect3 = mpatches.Rectangle((ld_cs[0], ld_cs[2]), ld_cs[1] - ld_cs[0], ld_cs[3] - ld_cs[2], \
                               fill=False, color="black", linewidth=2)
    rect4 = mpatches.Rectangle((ht_cs[0], ht_cs[2]), ht_cs[1] - ht_cs[0], ht_cs[3] - ht_cs[2], \
                               fill=False, color="magenta", linewidth=2)
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.gca().add_patch(rect3)
    plt.gca().add_patch(rect4)
    plt.title(f'Time: {t[5:11]}', fontsize=16)
    fig.savefig(f'./figs/LST_{t[6:11]}_{res}.jpg', dpi=200, bbox_inches='tight')


# %%
# plot QE

def lons_lats(choice, var):
    lon = var.coord('grid_longitude').points
    lat = var.coord('grid_latitude').points
    lonm, latm = np.meshgrid(lon, lat, indexing='xy')
    return lon, lat, lonm, latm


def fig_func(lonm, latm, data, label, vmin, vmax):
    ax = plt.subplot()
    cmap = plt.cm.get_cmap('hot_r')
    CS = ax.pcolormesh(lonm, latm, data, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(CS, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=f"{label}", size=16)


times = ['20180715T1230Z']

rp_cs = np.array([1.37, 1.37 + 0.04, -1.06, -1.06 + 0.04])
nd_cs = np.array([1.64, 1.64 + 0.04, -1.22, -1.22 + 0.04])
ld_cs = np.array([1.49, 1.49 + 0.04, -0.96, -0.96 + 0.04])
ht_cs = np.array([1.66, 1.66 + 0.04, -0.82, -0.82 + 0.04])
# ht_cs = np.array([1.69,1.69+0.01,-0.79,-0.79+0.01])

for t in times:
    d = datetime.datetime.strptime(t, '%Y%m%dT%H%MZ')
    hour = iris.Constraint(
        time=lambda cell: d <= cell.point <= d)
    lon, lat, lonm, latm = lons_lats(res, QE)
    var = QE.extract(hour)
    var_d = var.data
    fig = plt.figure(figsize=(8, 8))
    fig_func(lonm=lonm, latm=latm, data=var_d, label=r"$Q_E$ ($^\circ$C)", vmin=np.floor(np.min(var_d)),
             vmax=np.ceil(np.max(var_d)))
    rect1 = mpatches.Rectangle((rp_cs[0], rp_cs[2]), rp_cs[1] - rp_cs[0], rp_cs[3] - rp_cs[2], \
                               fill=False, color="blue", linewidth=2)
    rect2 = mpatches.Rectangle((nd_cs[0], nd_cs[2]), nd_cs[1] - nd_cs[0], nd_cs[3] - nd_cs[2], \
                               fill=False, color="lime", linewidth=2)
    rect3 = mpatches.Rectangle((ld_cs[0], ld_cs[2]), ld_cs[1] - ld_cs[0], ld_cs[3] - ld_cs[2], \
                               fill=False, color="black", linewidth=2)
    rect4 = mpatches.Rectangle((ht_cs[0], ht_cs[2]), ht_cs[1] - ht_cs[0], ht_cs[3] - ht_cs[2], \
                               fill=False, color="magenta", linewidth=2)
    plt.gca().add_patch(rect1)
    plt.gca().add_patch(rect2)
    plt.gca().add_patch(rect3)
    plt.gca().add_patch(rect4)
    plt.title(f'Time: {t[5:11]}', fontsize=16)
    fig.savefig(f'./figs/QE_{t[6:11]}_{res}.jpg', dpi=200, bbox_inches='tight')

# %%
# plot SEB

tcoord = QH.coord('time')
t = tcoord.units.num2date(tcoord.points)
latitudes = QH.coord('grid_latitude').points
longitudes = QH.coord('grid_longitude').points


def area_average(QH, QE, G, Lnet, Ldown, Lup, Knet, Kdown, Kup, lc, \
                 lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx):
    QHt = QH.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    QEt = QE.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Gt = G.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Lnett = Lnet.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Ldownt = Ldown.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Lupt = Lup.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Knett = Knet.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Kdownt = Kdown.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Kupt = Kup.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    Residual = -QHt - QEt - Gt + (Knett + Lnett)
    lc_av = np.zeros(len(lc.data[:, 0, 0]))
    for i in range(len(lc.data[:, 0, 0])):
        lc_av[i] = lc.data[i, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(0, 1))
    return QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av


def curves(ax, t, QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av):
    for i in range(len(t)):
        t[i] = datetime.datetime(t[i].year, t[i].month, t[i].day, t[i].hour, t[i].minute, t[i].second)
    t = t[5:]
    ax.plot_date(t, QHt[5:], color="orange", linestyle="solid", marker=None, label=r"$Q_H$")
    ax.plot_date(t, QEt[5:], color="blue", linestyle="solid", marker=None, label=r"$Q_E$")
    ax.plot_date(t, Gt[5:], color="brown", linestyle="solid", marker=None, label=r"$Q_G$")
    ax.plot_date(t, -Lnett[5:], color="gray", linestyle="dotted", marker=None, label=r"$L_N$")
    # ax.plot_date(t,Ldownt[5:],color="red",linestyle="dotted",label=r"$L_{down}$")
    # ax.plot_date(t,Lupt[5:],color="red",linestyle="dashed",label=r"$L_{up}$")
    ax.plot_date(t, Knett[5:], color="gray", linestyle="dashed", marker=None, label=r"$K_N$")
    # ax.plot_date(t,Kdownt[5:],color="blue",linestyle="dotted",label=r"$K_{down}$")
    # ax.plot_date(t,Kupt[5:],color="blue",linestyle="dashed",label=r"$K_{up}$")
    ax.plot_date(t, Residual[5:], color="green", linestyle="dashdot", marker=None, label=r"$R$")
    ax.plot_date(t, Knett[5:] + Lnett[5:], color="black", linestyle="solid", marker=None, label=r"$Q_N$")
    lc_av = np.concatenate(([lc_av[0] + lc_av[1]], [lc_av[2] + lc_av[3]], [lc_av[4]], \
                            [lc_av[5]], [lc_av[6]], [lc_av[8] + lc_av[9]]), axis=0)
    label = f'Tree: %1.2f\nGrass: %1.2f\nShrub: %1.2f\nIW: %1.2f\nSoil: %1.2f\nUrban: %1.2f' \
            % (lc_av[0], lc_av[1], lc_av[2], lc_av[3], lc_av[4], lc_av[5])
    ax.text(0.05, 0.95, label,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, color='black')
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
    ax.set_ylim(-150, 800)
    ax.axhline(0, color="black", linestyle="dotted")


rp_cs_idxs, nd_cs_idxs, ld_cs_idxs, ht_cs_idxs = \
    np.zeros(len(rp_cs), dtype=int), np.zeros(len(nd_cs), dtype=int), np.zeros(len(ld_cs), dtype=int), np.zeros(
        len(ht_cs), dtype=int)
for i in range(len(rp_cs)):
    if i == 0 or i == 1:
        coords = longitudes
    else:
        coords = latitudes
    rp_cs_idxs[i] = find_nearest(coords, rp_cs[i])[0]
    nd_cs_idxs[i] = find_nearest(coords, nd_cs[i])[0]
    ld_cs_idxs[i] = find_nearest(coords, ld_cs[i])[0]
    ht_cs_idxs[i] = find_nearest(coords, ht_cs[i])[0]

# x params
nx = 2
dsx = 0.1
dex = 0.1
dx = 0.01
dfx = (1 - dsx - dex - (nx - 1) * dx) / nx  # 1 = dsx + (nx-1)*dx + nx*dfx + dex

# y params
ny = 2
dsy = 0.1
dey = 0.1
dy = 0.06
dfy = (1 - dsy - dey - (ny - 1) * dy) / ny  # 1 = dsy + (ny-1)*dy + ny*dfy + dey

fig = plt.figure(figsize=(14, 8))

ax0 = fig.add_axes([dsx, 1 - dsy - dfy, dfx, dfy])
QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av = \
    area_average(QH, QE, G, Lnet, Ldown, Lup, Knet, Kdown, Kup, lc, \
                 rp_cs_idxs[2], rp_cs_idxs[3], rp_cs_idxs[0], rp_cs_idxs[1])
curves(ax0, t, QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av)
ax0.legend()
ax0.axes.xaxis.set_ticklabels([])
ax0.set_ylabel(r"SEB Component (W m$^{-2}$)")
ax0.set_title("Richmond Park")

ax1 = fig.add_axes([dsx + dfx + dx, 1 - dsy - dfy, dfx, dfy])
QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av = \
    area_average(QH, QE, G, Lnet, Ldown, Lup, Knet, Kdown, Kup, lc, \
                 nd_cs_idxs[2], nd_cs_idxs[3], nd_cs_idxs[0], nd_cs_idxs[1])
curves(ax1, t, QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av)
ax1.axes.xaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticklabels([])
ax1.set_title("North Downs")

ax2 = fig.add_axes([dsx, 1 - dsy - 2 * dfy - dy, dfx, dfy])
QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av = \
    area_average(QH, QE, G, Lnet, Ldown, Lup, Knet, Kdown, Kup, lc, \
                 ld_cs_idxs[2], ld_cs_idxs[3], ld_cs_idxs[0], ld_cs_idxs[1])
curves(ax2, t, QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av)
ax2.set_ylabel(r"SEB Component (W m$^{-2}$)")
ax2.set_xlabel(r"Time (UTC)")
ax2.set_title("Central London")

ax3 = fig.add_axes([dsx + dfx + dx, 1 - dsy - 2 * dfy - dy, dfx, dfy])
QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av = \
    area_average(QH, QE, G, Lnet, Ldown, Lup, Knet, Kdown, Kup, lc, \
                 ht_cs_idxs[2], ht_cs_idxs[3], ht_cs_idxs[0], ht_cs_idxs[1])
curves(ax3, t, QHt, QEt, Gt, Lnett, Ldownt, Lupt, Knett, Kdownt, Kupt, Residual, lc_av)
ax3.axes.yaxis.set_ticklabels([])
ax3.set_xlabel(r"Time (UTC)")
ax3.set_title("North East")

plt.savefig("./figs/SEB.jpg", dpi=200)


# %%
# plot QE tiles

def area_average_QEtiles(QE_tiles, lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx):
    QE_tiles_t = QE_tiles.data[:, lat_min_idx:lat_max_idx + 1, lon_min_idx:lon_max_idx + 1].mean(axis=(1, 2))
    return QE_tiles_t


def curves_QEtiles(ax, t, QEt, tiles, tile):
    for i in range(len(t)):
        t[i] = datetime.datetime(t[i].year, t[i].month, t[i].day, t[i].hour, t[i].minute, t[i].second)
    t = t[5:]
    ax.plot_date(t, QEt[5:], linestyle="solid", marker=None, label=f"{tiles[tile]}")  # ,color="blue"


# x params
nx = 1
dsx = 0.1
dex = 0.1
dx = 0.01
dfx = (1 - dsx - dex - (nx - 1) * dx) / nx  # 1 = dsx + (nx-1)*dx + nx*dfx + dex

# y params
ny = 1
dsy = 0.1
dey = 0.1
dy = 0.06
dfy = (1 - dsy - dey - (ny - 1) * dy) / ny  # 1 = dsy + (ny-1)*dy + ny*dfy + dey

tiles = ['Broad Leaf', 'Needle Leaf', 'C3  Grass', 'C4 Grass', \
         'Shrub', 'Inland Water', 'Soil', 'Ice', 'Urban Canyon', 'Urban Roof']

fig = plt.figure(figsize=(10, 6))

ax0 = fig.add_axes([dsx, 1 - dsy - dfy, dfx, dfy])
for tile in range(len(QE_tiles[:, 0, 0, 0].data)):
    QE_tiles_t = area_average_QEtiles(QE_tiles[tile, :, :, :], ht_cs_idxs[2], ht_cs_idxs[3], ht_cs_idxs[0],
                                      ht_cs_idxs[1])
    curves_QEtiles(ax0, t, QE_tiles_t, tiles, tile)
ax0.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
ax0.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))
ax0.set_ylim(0, 300)
ax0.set_xlabel(r"Time (UTC)")
ax0.set_ylabel(r"$Q_E$ (W m$^{-2}$)")
ax0.legend()

plt.savefig(f"./figs/QE_tiles.jpg", dpi=200)
