#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:39:46 2023

@author: allan
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader

from cpt_convert import loadCPT # Import the CPT convert function
from matplotlib.colors import LinearSegmentedColormap # Linear interpolation for color maps

from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,
                 cartopy_xlim, cartopy_ylim, CoordPair, vertcross, interpline,ll_to_xy, xy_to_ll)

import metpy.calc as mpcalc

import glob
import os

#import Ngl,Nio

np.set_printoptions(precision=2)

def get_height(pressure, temperature, geopotential_height, reference_pressure=1000.0):
    """
    Calculate the height of vertical levels using pressure, temperature, and geopotential height.
    
    Parameters:
        pressure (array-like): Pressure in Pa.
        temperature (array-like): Temperature in K.
        geopotential_height (array-like): Geopotential height in m.
        reference_pressure (float): Reference pressure in Pa. Default is 1000.0 Pa.
    
    Returns:
        height (array-like): Height in m.
    """
    Rd = 287.04  # Specific gas constant for dry air in J/kg/K
    g = 9.81  # Acceleration due to gravity in m/s^2
    
    # Calculate the scale height
    T0 = np.mean(temperature)  # Mean temperature in K
    H = Rd*T0/g
    
    # Calculate the pressure scale factor
    exponent = -1.0/H * (np.log(pressure/reference_pressure))
    psfc = np.power(pressure/reference_pressure, exponent)
    
    # Calculate the height
    height = geopotential_height + (Rd*T0/g)*(np.log(psfc))
    
    return height





caminho = 'wrf_arquivos1'

#vento
nome1='barbelas_vento_20m_'
nome2='componente_zonal_vento_20m'



MYNN="/home/allan/MYNN_d01/wrfout_d01_2017-01-19*23*"
TKE="/home/allan/3DTKE_d01/wrfout_d01_2017-01-19*23:00*"
SHH="/home/allan/SH_d01/wrfout_d01_2017-01-19*23:00*"

param=[MYNN]

#lonsestt=(lontest1,lontest2,lontest3,lontest4,lontest5,lontest6,lontest7, lontest8,lontest9,lontest10,lontest11,lontest12,lontest13,lontest14,lontest15, lontest16,lontest17,lontest18,lontest19,lontest20)
#latsestt=(latest1,latest2,latest3,latest4,latest5,latest6,latest7, latest8,latest9,latest10,latest11,latest12,latest13,latest14,latest15, latest16,latest17,latest18,latest19,latest20)
#for filename in sorted(glob.glob('/home/allan/wrf_arquivos1/wrf_trad_fields*')):
for i in param:
    for filename in sorted(glob.glob(i)):
    
        #print(filename)
        pr=filename.split('/')[3]
        #prr=filename[12:14]
        bolas=filename.split('/')[4]
        nn=[bolas.split('_')[2],bolas.split('_')[3]]
        nm='_'.join(nn)
        #Dataset do arquivo que leu
        ncfile = Dataset(filename)
        
        
        uvmet=getvar(ncfile,"uvmet")
        temperature = getvar(ncfile, "tc")
        tempK=getvar(ncfile,"temp", units='K')
        ter = getvar(ncfile,"ter",timeidx=-1)
#        PT = getvar(ncfile,"theta")
        pblh=getvar(ncfile,"PBLH")
        z = getvar(ncfile,"height",units="m")
        w = getvar(ncfile,"wa")
        pressao= getvar(ncfile,"pressure")
        qvapor = getvar(ncfile, 'QVAPOR')
        
        density = mpcalc.density(pressao[0:26],tempK[0:26],qvapor[0:26])
        
        cart_proj = get_cartopy(ter)
        
        #metpy.calc.density()
        
        temp = temperature[0,:,:]
        uv1 = uvmet[:,0,:,:]
        uz1 = uvmet[0,0,:,:]
        vz1 = uvmet[1,0,:,:]
#        wz1 = w[:,:,:]
        
        u = uvmet[0,:,:,:]
        
        teste = pblh + z
        
#        ptclp=PT[:,:,:]
    
        zztop=z[:,:,:]
        # Get the lat/lon coordinates (temp potencial)
        lats, lons = latlon_coords(temperature)
        
#        # Set the start point and end point for the cross section
#        cross_start = CoordPair(lat=-19.53, lon=-43.5)
#        cross_end = CoordPair(lat=-19.53, lon=-37.5)
#        
#        #interpolacao vertical temp pot
#        pt_cross = vertcross(ptclp[0:26], z[0:26], wrfin=ncfile, start_point=cross_start,
#                               end_point=cross_end, latlon=True, meta=True)
#        
#        #interpolacao vertical u
#        u_cross = vertcross(u[0:26], z[0:26], wrfin=ncfile, start_point=cross_start,
#                               end_point=cross_end, latlon=True, meta=True)
#            
#        ter_line = interpline(ter, wrfin=ncfile, start_point=cross_start,
#                              end_point=cross_end)
        
#        #interpolacao vertical w
#        w_cross = vertcross(w[0:26], z[0:26], wrfin=ncfile, start_point=cross_start,
#                               end_point=cross_end, latlon=True, meta=True)
#        
#        #interpolacao pressao
#        p_cross = vertcross(pressao[0:26], z[0:26], wrfin=ncfile, start_point=cross_start,
#                               end_point=cross_end, latlon=True, meta=True)
#        
#        #interpolacao densidade
#        den_cross = vertcross(density, z[0:26], wrfin=ncfile, start_point=cross_start,
#                               end_point=cross_end, latlon=True, meta=True)
    
    
        #tentando calcular a altura real
    
#        h200 = ter_line + 200
    
#        cart_proj = get_cartopy(PT)
        
        #Linha do corte vertical da estação.
        lon1 = -43.5
        lon2 = -37.5
        lat1 = -19.53
        lat2 = -19.53
        
        
        #modulo do vento
        #Mz1 = (uz1**2+vz1**2)**0.5
    
        # Get the lat/lon coordinates
        lats, lons = latlon_coords(uz1)
    
    
        # Get the map projection information
        cart_proj = get_cartopy(uz1)
       
        # Converts the CPT file to be used in Python
        cptwnd = loadCPT('/home/allan/cpt/GMT_polar.cpt')
        
        #cpt_file_1 = '/home/allan/cpt/GMT_polar.cpt'
        #cptwnd_1 = load_cpt(cpt_file_1)
        
        #cptwnd_1=loadCPT('/home/allan/cpt/BlWhRe.cpt')
        # Makes a linear interpolation with the CPT file
        cpt_wnd = LinearSegmentedColormap('cptwnd', cptwnd)
        
        #pra TEMP
        cpttemp = loadCPT('/home/allan/cpt/temp_19lev.cpt')
        cpt_temp = LinearSegmentedColormap('cpttemp', cpttemp)
            
        
        # Add a shapefile
        # https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2019/Brasil/BR/br_unidades_da_federacao.zip
        shapefile = list(shpreader.Reader('/home/allan/BR_UF_2022/BR_UF_2022.shp').geometries())
        shpES = list(shpreader.Reader('/home/allan/BR_UF_2022/ES_Municipios_2022.shp').geometries())
        
            # Create the figure
        fig = plt.figure(figsize=(12,9))
        ax_wnd = fig.add_subplot(1,1,1, projection=cart_proj)
    
    
    
        # Add coastlines, borders and gridlines
        ax_wnd.add_geometries(shapefile, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)
        ax_wnd.add_geometries(shpES, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)
        ax_wnd.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax_wnd.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
        gl_wnd = ax_wnd.gridlines(crs=crs.PlateCarree(), color='black', alpha=1.0, x_inline=False, y_inline=False, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
        gl_wnd.top_labels = False
        gl_wnd.right_labels = False
        gl_wnd.xlabel_style = dict(rotation=0, va='top', ha='center')
    
        
        
        #contorno da componente zonal do vento
        
        wnd_levels = np.arange(-10., 10., 0.5)
    #    streamlines_wnd =  plt.barbs(to_np(lons[::10,::10]), to_np(lats[::10,::10]), to_np(uz1[::10,::10]), to_np(vz1[::10,::10]),
    #                                     transform=crs.PlateCarree(), length=4)
        contours_wnd = ax_wnd.contourf(to_np(lons), to_np(lats), to_np(uz1),
                             levels=wnd_levels, transform=crs.PlateCarree(), cmap=cpt_wnd, extend='both')
        cb_wnd = fig.colorbar(contours_wnd, ax=ax_wnd, orientation="vertical", pad=.05)
        cb_wnd.ax.tick_params(labelsize=5)
        cb_wnd.ax.set_ylabel('U (m/s)')
        
        plt.savefig(f'/home/allan/plotsTCC/{pr}/u20/u20_{nm}', bbox_inches='tight', pad_inches=0, dpi=200)
    
        plt.show()
    
        
        #temp
        fig = plt.figure(figsize=(12,9))  
        ax_temp = fig.add_subplot(1,1,1, projection=cart_proj)
    
        #pra temp
        ax_temp.add_geometries(shapefile, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)
        ax_temp.add_geometries(shpES, crs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)
        ax_temp.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax_temp.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
        gl_temp = ax_temp.gridlines(crs=crs.PlateCarree(), color='black', alpha=1.0, x_inline=False, y_inline=False, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
        gl_temp.top_labels = False
        gl_temp.right_labels = False
        gl_temp.xlabel_style = {'rotation': 0}
        gl_temp.xlabel_style = dict(rotation=0, va='top', ha='center')
    
        ###temperatura
        temp_levels = np.arange(15., 34., 1.)
        contours_temp = ax_temp.contourf(to_np(lons), to_np(lats), to_np(temp),
                             levels=temp_levels, transform=crs.PlateCarree(), cmap=cpt_temp, extend='both')
        cb_temp = fig.colorbar(contours_temp, ax=ax_temp, orientation="vertical", pad=.05)
        cb_temp.ax.tick_params(labelsize=5)
        cb_temp.ax.set_ylabel('T (°C)')
        #ax_temp.set_ylabel("Altura do nível do mar (m)")
        
        plt.savefig(f'/home/allan/plotsTCC/{pr}/t20/t20_{nm}', bbox_inches='tight', pad_inches=0, dpi=200)
    
        plt.show()
#    
    
#        
#        
#        #PLOT TEMP VERTICAL
#        #niveis
#        niveis_pt= np.arange(295.,315.,1)
#        
#        fig = plt.figure(figsize=(12,9))
#        ax_cross= fig.add_subplot(1,1,1)
#        plt.ylim(0,3000)
    
        #NOVO
#        xs = np.arange(0, u_cross.shape[-1], 1)
#        ys = to_np(u_cross.coords["vertical"])
    
#        pt_contours = ax_cross.contourf(xs,
#                                          ys,
#                                          to_np(pt_cross),levels=niveis_pt,
#                                          cmap=get_cmap("jet"), extend='both')
#        
#    
#        #colorbar
#        cb_pt = fig.colorbar(pt_contours, ax=ax_cross)
#        cb_pt.ax.tick_params(labelsize=5)
#        cb_pt.ax.set_ylabel('TP (K)')
#        ax_cross.set_xticks(np.arange(0, 226, 25),["","-100","","-50","","0","","50","","100"])
#        ax_cross.set_ylabel("Altura do nível do mar (m)")
#        ax_cross.set_xlabel("Distancia da costa (km)")
#        
#        plt.savefig(f'/home/allan/plotsTCC/{pr}/vert_pt/vert_pt_{nm}', bbox_inches='tight', pad_inches=0, dpi=200)
#    
#        plt.show()
        
        
    
        #PLOT U CROSS
        niveis_u= np.arange(-10.,10.,1)
        niveis_p=np.arange(700,1000,10)
        niveis_d=np.arange(0.,3.20,0.01)
        fig = plt.figure(figsize=(12,9))    
        ax_ucross = fig.add_subplot(1,1,1)
        plt.ylim(0,3000)
        
        #NOVO
        #xs = np.arange(0, u_cross.shape[-1], 1)
        #ys = to_np(u_cross.coords["vertical"])
#        
#        ucross_contours = ax_ucross.contourf(xs,
#                                          ys,
#                                          to_np(u_cross),levels=niveis_u,
#                                          cmap=get_cmap("coolwarm"), extend='both')
#        
#        ucross_zero_contours = ax_ucross.contour(xs,
#                                          ys,
#                                          to_np(u_cross),[0.],
#                                          colors='k', linewidths=0.8)
#        
#        ucross_press_contours = ax_ucross.contour(xs,
#                                          ys,
#                                          to_np(p_cross),levels=niveis_p,
#                                          colors='k', linestyles='dashed', linewidths=0.8)  
#        
#        ucross_density_contours = ax_ucross.contour(xs,
#                                          ys,
#                                          to_np(den_cross),levels=niveis_d,
#                                          colors='red', linestyles='dashed', linewidths=0.8)
#        
#        #colorbar
#        
#        cb_ucross = fig.colorbar(ucross_contours, ax=ax_ucross)
#        cb_ucross.ax.tick_params(labelsize=5)
#        cb_ucross.ax.set_ylabel('U (m/s)')
#        ax_ucross.set_xticks(np.arange(0, 226, 25),["","-100","","-50","","0","","50","","100"])
#        ax_ucross.set_ylabel("Altura do nível do mar (m)")
#        ax_ucross.set_xlabel("Distancia da costa (km)")
#        
#        plt.savefig(f'/home/allan/plotsTCC/{pr}/vert_u/vert_u_{nm}', bbox_inches='tight', pad_inches=0, dpi=200)
#    
#        plt.show()
#        
#        #PLOT W CROSS
#        
#        xs = np.arange(0, w_cross.shape[-1], 1)
#        ys = to_np(w_cross.coords["vertical"])
#        
#        niveis_w = np.arange(-1.5,1.5,0.1)
#        
#        #NOVO
#        #xs = np.arange(0, u_cross.shape[-1], 1)
#        #ys = to_np(u_cross.coords["vertical"])
#        fig = plt.figure(figsize=(12,9))
#        ax_wcross = fig.add_subplot(1,1,1)
#        plt.ylim(0,3000)
#        
#        wcross_contours = ax_wcross.contourf(xs,
#                                          ys,
#                                          to_np(w_cross),levels=niveis_w,
#                                          cmap=get_cmap("coolwarm"), extend='both')
#        
#        
#        cb_wcross = fig.colorbar(wcross_contours, ax=ax_wcross)
#        cb_wcross.ax.tick_params(labelsize=5)
#        cb_wcross.ax.set_ylabel('W (m/s)')
#        #
#        #ax_wcross.xlabel("Distancia da Costa (km)")
#        ax_wcross.set_ylabel("Altura do nível do mar (m)")
#        ax_wcross.set_xlabel("Distancia da costa (km)")
#        #ax_ucross.tick_params(axis ='x', rotation = 45)
#        ax_wcross.set_xticks(np.arange(0, 226, 25),["","-100","","-50","","0","","50","","100"])
#    
#        #fig.supxlabel("Distancia da costa (km)")
#    
#     
#        
#        #ax_wnd.set_title(f'a)', {"fontsize" : 12}, loc='left')
#        #ax_temp.set_title(f'b)', {"fontsize" : 12},loc='left')
#        #ax_cross.set_title(f'c)', {"fontsize" : 12},loc='left')
#        #ax_ucross.set_title(f'd)', {"fontsize" : 12},loc='left')
#        #ax_wcross.set_title(f'e)', {"fontsize" : 12},loc='left')
#        
#        plt.savefig(f'/home/allan/plotsTCC/{pr}/vert_w/{nm}', bbox_inches='tight', pad_inches=0, dpi=200)
#    
#        plt.show()

llxy = ll_to_xy(ncfile, -19.31, -37.5)
xyll = xy_to_ll(ncfile, llxy[0], llxy[1])


