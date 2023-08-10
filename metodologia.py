from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
from mpl_toolkits.basemap import Basemap

from cpt_convert import loadCPT # Import the CPT convert function
from matplotlib.colors import LinearSegmentedColormap # Linear interpolation for color maps

from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,
                 cartopy_xlim, cartopy_ylim, CoordPair, vertcross, interpline,WrfProj)

import csv

import glob
import os

latestt=[]

lonestt=[]

n=["A535","A540","A612","A613","A614","A623","A506","A552","A447","A522","A532","A555","A556","A518","A607","A541","A455","A538"]

nn=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

for i in range(18):
    rows = []
    pathest=glob.glob(f'/home/allan/inmet/*{n[i]}*')
    #llxy = ll_to_xy(ncfile, EST[1,i], EST[2,i])
    with open(pathest[0], 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
            
    coluna_est_lat=''.join(rows[1])
    coluna_est_lon=''.join(rows[2])
    latestt=np.append(latestt,float(coluna_est_lat.split(" ")[1]))
    lonestt=np.append(lonestt,float(coluna_est_lon.split(" ")[1]))

#pegar a topografia q o wrf ta usando
ncfile = Dataset('/home/allan/MYNN_d01/wrfout_d01_2017-01-19_11:40:00')
topo=getvar(ncfile,'ter')
lats, lons = latlon_coords(topo)

    # Converts the CPT file to be used in Python
cptwnd = loadCPT('/home/allan/cpt/DEM_poster.cpt')
    # Makes a linear interpolation with the CPT file
cpt_wnd = LinearSegmentedColormap('cptwnd', cptwnd)

cart_proj = get_cartopy(topo)

fig = plt.figure(figsize=(12,9))
ax_wnd = fig.add_subplot(1,1,1, projection=crs.PlateCarree())
shapefile = list(shpreader.Reader('/home/allan/BR_UF_2022/BR_UF_2022.shp').geometries())
##shpES = list(shpreader.Reader('/home/allan/BR_UF_2022/ES_Municipios_2022.shp').geometries())
ax_wnd.add_geometries(shapefile, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)
##ax_wnd.add_geometries(shpES, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)
ax_wnd.coastlines(resolution='10m', color='black', linewidth=0.8)
ax_wnd.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
gl_wnd = ax_wnd.gridlines(crs=crs.PlateCarree(), color='black', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
gl_wnd.top_labels = False
gl_wnd.right_labels = False
gl_wnd.xlabel_style = dict(rotation=0, va='top', ha='center')
#


latlim1=-45.23
latlim2=-34.66
lonlim1=-23.84
lonlim2=-15.02

# #plot da linha do dominio da rodada
ax_wnd.plot((-45.23, -34.66),(-15.02,-15.02), color='k', linewidth=1.5)
ax_wnd.plot((-45.23, -34.66),(-23.84,-23.84), color='k', linewidth=1.5)
ax_wnd.plot((-45.23, -45.23),(-23.84,-15.02), color='k', linewidth=1.5)
ax_wnd.plot((-34.66, -34.66),(-23.84,-15.02), color='k', linewidth=1.5)

 #plot da linha do dominio da rodada do GABRIEL
#ax.plot((-41.47, -38.15),(-18.4,-18.4), color='k', linewidth=1.5, transform=crs.PlateCarree())
#ax.plot((-41.47, -38.15),(-20.62,-20.62), color='k', linewidth=1.5, transform=crs.PlateCarree())
#ax.plot((-41.47, -41.47),(-20.62,-18.4), color='k', linewidth=1.5, transform=crs.PlateCarree())
#ax.plot((-38.15, -38.15),(-20.62,-18.4), color='k', linewidth=1.5, transform=crs.PlateCarree())


#plt.scatter([lontest1,lontest2,lontest3,lontest4,lontest5,lontest6,lontest7,
#             lontest8,lontest9,lontest10,lontest11,lontest12,lontest13,lontest14,lontest15,
#             lontest16,lontest17,lontest18,[latest1,latest2,latest3,latest4,latest5,latest6,
#             latest7,latest8,latest9,latest10,latest11,latest12,
#             latest13,latest14,latest15,latest16,latest17,latest18],s=15,c='red',linewidths=0.7)



ax_wnd.stock_img()
plt.ylim(-34.43,-4.43)
plt.xlim(-54.945,-24.945)

plt.savefig(f'/home/allan/plotsTCC/mapadom_1', bbox_inches='tight', dpi=200)

plt.show()


#PLOT TOPOGRAFICO

n=["A534","A540","A612","A613","A614","A623","A506","A552","A447","A522","A532","A555",
   "A556","A518","A607","A541","A455","A538"]

nn=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]


ncfile = Dataset('/home/allan/MYNN_d01/wrfout_d01_2017-01-19_11:40:00')
topo=getvar(ncfile,'ter')
lats, lons = latlon_coords(topo)

    # Converts the CPT file to be used in Python
cptwnd = loadCPT('/home/allan/cpt/DEM_poster.cpt')
    # Makes a linear interpolation with the CPT file
cpt_wnd = LinearSegmentedColormap('cptwnd', cptwnd)

cart_proj = get_cartopy(topo)

# Create the figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)

#states = NaturalEarthFeature(category="cultural", scale="50m",
#                             facecolor="none",
#                             name="admin_1_states_provinces_shp")
#ax.add_feature(states, linewidth=0.5, edgecolor="black")
ax.coastlines('10m', linewidth=0.8)

# Set the map bounds
#ax.set_xlim(cartopy_xlim(topo))
#ax.set_ylim(cartopy_ylim(topo))

shapefile = list(shpreader.Reader('/home/allan/BR_UF_2022/BR_UF_2022.shp').geometries())

ax.add_geometries(shapefile, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)

#plot da linha do corte vertical
bbbbb=ax.plot((-43.5, -37.5),(-19.53,-19.53), color='b', linewidth=1.5, transform=crs.PlateCarree())
#ax_wnd.add_feature(shapefile, linewidth=.3, edgecolor="black", facecolor='none')
##ax_wnd.add_geometries(shpES, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)
ax.coastlines(resolution='10m', color='black', linewidth=0.8)
ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)


wnd_levels = np.arange(0.1, 3000, 10)
contours_wnd = ax.contourf(to_np(lons), to_np(lats), to_np(topo),
                         levels=wnd_levels, transform=crs.PlateCarree(), cmap=cpt_wnd, extend='max')
cb_wnd = fig.colorbar(contours_wnd, ax=ax, orientation="vertical", pad=.05)
cb_wnd.ax.tick_params(labelsize=5)

 #plot da linha do dominio da rodada do GABRIEL
#ax.plot((-41.47, -38.15),(-18.4,-18.4), color='k', linewidth=1.5, transform=crs.PlateCarree())
#ax.plot((-41.47, -38.15),(-20.62,-20.62), color='k', linewidth=1.5, transform=crs.PlateCarree())
#ax.plot((-41.47, -41.47),(-20.62,-18.4), color='k', linewidth=1.5, transform=crs.PlateCarree())
#ax.plot((-38.15, -38.15),(-20.62,-18.4), color='k', linewidth=1.5, transform=crs.PlateCarree())


#wnd_levels = np.arange(0)
#contours_wnd = ax.contourf(to_np(lons), to_np(lats), to_np(topo),
#                         levels=wnd_levels, transform=crs.PlateCarree(), cmap=cpt_wnd, color='paleturquoise')

ax.add_feature(cartopy.feature.OCEAN,facecolor='paleturquoise',alpha=0.4)

#ax.gridlines(color="black", linestyle="dotted")

#gl_wnd = ax.gridlines(crs=crs.PlateCarree(), color='black', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), linewidth=0.33, color='k',alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = dict(rotation=0, va='top', ha='center')






for x, y, z in zip(lonestt,latestt, nn):
    plt.text(x,y,str(z) ,c='white', transform=crs.PlateCarree(), horizontalalignment='center', verticalalignment='center')
    
plt.savefig(f'/home/allan/plotsTCC/mapadom_topograph', bbox_inches='tight', dpi=200)
