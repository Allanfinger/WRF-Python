import math


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
                 cartopy_xlim, cartopy_ylim, CoordPair, vertcross, interpline, ll_to_xy, xy_to_ll)

import glob
import os

import csv

from itertools import zip_longest

def dist_total(x,y):
    dist=np.sqrt(x**2+y**2)
    return dist

def calcular_distancia(lat1, lon1, lat2, lon2):
    # Raio médio da Terra em metros
    raio_terra = 6371000

    # Converter graus para radianos
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Diferença das coordenadas
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Cálculo da distância usando a fórmula de Haversine
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distancia = raio_terra * c

    # Calcular as distâncias em metros para as coordenadas x (longitude) e y (latitude)
    distancia_x = distancia * math.cos(lat1_rad)
    distancia_y = distancia

    return distancia_x, distancia_y

EST=np.array([["A535","A540","A612","A613","A614","A623","A506","A552","A447","A522","A532","A555","A556","A518","A607","A541","A455","A538"],
             [-19.533,-18.781,-20.271,-19.988,-19.357,-18.695,-16.686,-16.160,-16.088,-17.799,-18.830,-20.031,-21.770,-21.715,-17.706,-17.007,-18.748,-18.748],
             [-41.091,-40.987,-40.306,-40.580,-40.069,-40.391,-43.844,-42.310,-39.215,-40.250,-41.977,-44.011,-42.183,-43.364,-41.344,-42.390,-39.558,-44.454]])

ncfile = Dataset('/home/allan/MYNN_d01/wrfout_d01_2017-01-19_11:40:00')
ter = getvar(ncfile,"ter",timeidx=-1)

header = ['ESTACAO','LAT_EST','LON_EST','ALT_EST','LAT_WRF','LON_WRF','ALTURA_WRF','LAT_WRF-EST','LON_WRF-EST','ALTURA_WRF-EST','DISTANCIA_TOTAL','ALTURA_TOTAL']

with open(f'/home/allan/plotsTCC/estatistica/distancias.csv', 'w', encoding="UTF8", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(header)
myfile.close()

#temp = getvar(ncfile, "T2")
for estacao in EST[0]:
    rows = []
    pathest=glob.glob(f'/home/allan/inmet/*{estacao}*')
    #llxy = ll_to_xy(ncfile, EST[1,i], EST[2,i])
    with open(pathest[0], 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
            
    lat_est=[]
    lon_est=[]
    alt_est=[]
    nome_est=[]
    lat_wrf=[]
    lon_wrf=[]
    altura_wrff=[]
    
    #diferencas
    dif_lat=[]
    dif_lon=[]
    dif_alt=[]
    
    coluna_est_nome=''.join(rows[0])
    coluna_est_lat=''.join(rows[1])
    coluna_est_lon=''.join(rows[2])
    coluna_est_alt=''.join(rows[3])
    #nome_est=np.append(nome_est,coluna_est_nome.split(" ")[2])
    lat_estt=float(coluna_est_lat.split(" ")[1])
    lon_est=float(coluna_est_lon.split(" ")[1])
    alt_est=float(coluna_est_alt.split(" ")[1])
    
    llxy = ll_to_xy(ncfile, coluna_est_lat.split(" ")[1], coluna_est_lon.split(" ")[1])
    xyll = xy_to_ll(ncfile, llxy[0], llxy[1])
    
    lat_wrf=float(xyll[0])
    lon_wrf=float(xyll[1])
    
    altura_wrff=ter[llxy[1], llxy[0]]
    #altura_wrff=np.append(altura_wrff,float(altura_wrf))
    #wrf - altura da est
    dif_alt=float(altura_wrff - float(coluna_est_alt.split(" ")[1]))
    #dif_alt=np.append(dif_alt,float(altura_wrf - float(coluna_est_alt.split(" ")[1])))
    
    #wrf - lat/lon est
    dif_distancia=calcular_distancia(float(xyll[0]), float(xyll[1]), float(coluna_est_lat.split(" ")[1]), float(coluna_est_lon.split(" ")[1]))
    total_dist=dist_total(dif_distancia[0],dif_distancia[1])
    #
    
    dif_lon=dif_distancia[0]
    dif_lat=dif_distancia[1]
    
    alt_total=abs(dif_alt)
    #altura total
    
    
    d_SH = [estacao,str(float(lat_estt)),str(float(lon_est)), str(float(alt_est)),str(float(lat_wrf)),str(float(lon_wrf)),str(float(altura_wrff)),str(float(dif_lat)),str(float(dif_lon)),str(float(dif_alt)),str(float(total_dist)),str(float(alt_total))]
    
    with open(f'/home/allan/plotsTCC/estatistica/distancias.csv', 'a', encoding="UTF8", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(d_SH)
    myfile.close()
        #SH
#export_data = zip_longest(*d_SH, fillvalue = '')



#with open(f'/home/allan/plotsTCC/dist_ponto/{EST[0,i]}.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
#    wr = csv.writer(myfile, delimiter=',')
#    wr.writerow(d_zin)
#myfile.close()