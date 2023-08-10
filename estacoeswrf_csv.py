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
                 cartopy_xlim, cartopy_ylim, CoordPair, vertcross, interpline, ll_to_xy, g_uvmet, xy_to_ll)

from metpy.calc import wind_components
from metpy.units import units

from pint import UnitRegistry

import glob
import os

import csv

import pandas as pd

#funcao pra passar pra float com null values
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan
    
def ceil_without_nan(array):
    array = np.array(array)
    nan_mask = np.isnan(array)  # Cria uma máscara com True para os valores 'nan'
    ceil_values = np.ceil(array[~nan_mask])  # Arredonda para cima apenas os valores não 'nan'
    result = np.empty_like(array)
    result[~nan_mask] = ceil_values  # Preenche apenas os valores não 'nan' com os valores arredondados
    return result

#funcao componente u
def find_u(speed,degree):
    try:
        return wind_components(speed * units('m/s'), degree * units.deg)
    except ValueError:
        return np.nan
    #calculo pra dir do vento baseado em u e v, ao inves de pegar do wrf
def calcular_direcao_vento(u, v):
    # Converter para radianos e calcular o ângulo
    direcao_rad = np.arctan2(v, u)

    # Converter para graus
    direcao_deg = np.degrees(direcao_rad)

    # Ajustar a direção entre 0 e 360 graus
    direcao_deg = (direcao_deg + 360) % 360

    return direcao_deg
    



#pint
ureg = UnitRegistry()        
Q_ = ureg.Quantity
#import Ngl,Nio



EST=np.array([["A535","A540","A612","A613","A614","A623","A506","A552","A447","A522","A532","A555","A556","A518","A607","A541","A455","A538"],
             [-19.533,-18.781,-20.271,-19.988,-19.357,-18.695,-16.686,-16.160,-16.088,-17.799,-18.830,-20.031,-21.770,-21.715,-17.706,-17.007,-18.748,-18.748],
             [-41.091,-40.987,-40.306,-40.580,-40.069,-40.391,-43.844,-42.310,-39.215,-40.250,-41.977,-44.011,-42.183,-43.364,-41.344,-42.390,-39.558,-44.454]])

#MYNN="/home/allan/MYNN_d01/wrfout_d01*"
#TKE="/home/allan/3DTKE_d01/wrfout_d01*"
#SHH="/home/allan/SH_d01/wrfout_d01*"
    
#EST=[]

SHH = "/home/allan/SH_d01/csv/"
MYNN = "/home/allan/MYNN_d01/csv/"
TKE = "/home/allan/3DTKE_d01/csv/"

#loop estacao

for estacao in EST[0]:
    #    
    #temperaturas array vazios
    temp_SH=[]
    temp_MYNN=[]
    temp_TKE=[]
    #vento arrays vazios
    wnd_SH=[]
    wnd_MYNN=[]
    wnd_TKE=[]
    #dir do vento arrays vazios
    wnddir_SH=[]
    wnddir_MYNN=[]
    wnddir_TKE=[]
    #
    wnddir_calc_SH=[]
    wnddir_calc_MYNN=[]
    wnddir_calc_TKE=[]
    #componente u do vento
    uwind_SH=[]
    uwind_MYNN=[]
    uwind_TKE=[]
    #componente v do vento
    vwind_SH=[]
    vwind_MYNN=[]
    vwind_TKE=[]
    #pressao
    pressao_SH=[]
    pressao_MYNN=[]
    pressao_TKE=[]
    #temp ponto de orvalho
    td_SH=[]
    td_MYNN=[]
    td_TKE=[]
    #arrays vazios est
    temp_est=[]
    wnd_est=[]
    wnddir_est=[]
    uwind_est=[]
    vwind_est=[]
    td_est=[]
    #caminho da est
    pathest=glob.glob(f'/home/allan/inmet/*{estacao}*')
    path_SH=glob.glob(f'{SHH}*{estacao}*')
    path_MYNN=glob.glob(f'{MYNN}*{estacao}*')
    path_TKE=glob.glob(f'{TKE}*{estacao}*')

    rows = []
    rows_SH=[]
    rows_MYNN=[]
    rows_TKE=[]
    
    #abertura csv da est
    
    with open(pathest[0], 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    
    #linhas que importam
    #columnas=rows[31:104]
    columnas=rows[46:107]
    for f in columnas:
        coluna1=''.join(f)
        temp_est=np.append(temp_est,coluna1.split(";")[3])
        wnd_est=np.append(wnd_est,coluna1.split(";")[8])
        wnddir_est=np.append(wnddir_est,coluna1.split(";")[6])
        td_est=np.append(td_est,coluna1.split(";")[4])
        
        
    #passando pra float
    temp_est_float = [convert_to_float(value) for value in temp_est]
    wnd_est_float = [convert_to_float(value) for value in wnd_est] 
    wnddir_est_float = [convert_to_float(value) for value in wnddir_est]
    td_est_float = [convert_to_float(value) for value in td_est]
    
        #componente u e v estacao
    for g in range(0, len(wnd_est_float)):
       uvento=find_u(wnd_est_float[g],wnddir_est_float[g])
       #print(kkkkkk)
       uwind_est=np.append(uwind_est,uvento[0].magnitude)
       vwind_est=np.append(vwind_est,uvento[1].magnitude)
       


    
    #abertura csv SH
    with open(path_SH[0], 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows_SH.append(row)
            
    colunitas=rows_SH[12:73]
    for f in colunitas:
        coluna1=','.join(f)
        pressao_SH=np.append(pressao_SH,coluna1.split(",")[1])
        temp_SH=np.append(temp_SH,coluna1.split(",")[3])
        wnd_SH=np.append(wnd_SH,coluna1.split(",")[5])
        uwind_SH=np.append(uwind_SH,coluna1.split(",")[6])
        vwind_SH=np.append(vwind_SH,coluna1.split(",")[7])
        #calculo dir do vento
        wnddir=calcular_direcao_vento(float(coluna1.split(",")[6]), float(coluna1.split(",")[7]))
        wnddir_calc_SH=np.append(wnddir_calc_SH,wnddir)
        #wdir normal
        wnddir_SH=np.append(wnddir_SH,coluna1.split(",")[4])
        td_SH=np.append(td_SH,coluna1.split(",")[2])
        
        #passando pra float
    temp_SH_float = [convert_to_float(value) for value in temp_SH]
    wnd_SH_float = [convert_to_float(value) for value in wnd_SH] 
    wnddir_SH_float = [convert_to_float(value) for value in wnddir_SH]
    td_SH_float = [convert_to_float(value) for value in td_SH]
    uwind_SH_float = [convert_to_float(value) for value in uwind_SH]
    vwind_SH_float = [convert_to_float(value) for value in vwind_SH]
        
    #abertura csv MYNN
    with open(path_MYNN[0], 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows_MYNN.append(row)
            
    colunitas=rows_MYNN[12:73]
    for f in colunitas:
        coluna1=','.join(f)
        pressao_MYNN=np.append(pressao_MYNN,coluna1.split(",")[1])
        temp_MYNN=np.append(temp_MYNN,coluna1.split(",")[3])
        wnd_MYNN=np.append(wnd_MYNN,coluna1.split(",")[5])
        uwind_MYNN=np.append(uwind_MYNN,coluna1.split(",")[6])
        vwind_MYNN=np.append(vwind_MYNN,coluna1.split(",")[7])
        #calc wndir
        wnddir=calcular_direcao_vento(float(coluna1.split(",")[6]), float(coluna1.split(",")[7]))
        wnddir_calc_MYNN=np.append(wnddir_calc_MYNN,wnddir)
        
        wnddir_MYNN=np.append(wnddir_MYNN,coluna1.split(",")[4])
        td_MYNN=np.append(td_MYNN,coluna1.split(",")[2])
        
            #passando pra float
    temp_MYNN_float = [convert_to_float(value) for value in temp_MYNN]
    wnd_MYNN_float = [convert_to_float(value) for value in wnd_MYNN] 
    wnddir_MYNN_float = [convert_to_float(value) for value in wnddir_MYNN]
    td_MYNN_float = [convert_to_float(value) for value in td_MYNN]
    uwind_MYNN_float = [convert_to_float(value) for value in uwind_MYNN]
    vwind_MYNN_float = [convert_to_float(value) for value in vwind_MYNN]
        
#abertura csv 3DTKE
    with open(path_TKE[0], 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows_TKE.append(row)
            
    colunitas=rows_TKE[12:73]
    for f in colunitas:
        coluna1=','.join(f)
        pressao_TKE=np.append(pressao_TKE,coluna1.split(",")[1])
        temp_TKE=np.append(temp_TKE,coluna1.split(",")[3])
        wnd_TKE=np.append(wnd_TKE,coluna1.split(",")[5])
        uwind_TKE=np.append(uwind_TKE,coluna1.split(",")[6])
        vwind_TKE=np.append(vwind_TKE,coluna1.split(",")[7])
        #calc wnddir
        wnddir=calcular_direcao_vento(float(coluna1.split(",")[6]), float(coluna1.split(",")[7]))
        wnddir_calc_TKE=np.append(wnddir_calc_TKE,wnddir)
        
        wnddir_TKE=np.append(wnddir_TKE,coluna1.split(",")[4])
        td_TKE=np.append(td_TKE,coluna1.split(",")[2])
        
        
                #passando pra float
    temp_TKE_float = [convert_to_float(value) for value in temp_TKE]
    wnd_TKE_float = [convert_to_float(value) for value in wnd_TKE] 
    wnddir_TKE_float = [convert_to_float(value) for value in wnddir_TKE]
    td_TKE_float = [convert_to_float(value) for value in td_TKE]
    uwind_TKE_float = [convert_to_float(value) for value in uwind_TKE]
    vwind_TKE_float = [convert_to_float(value) for value in vwind_TKE]
    #vamos comecar o tal do plot. primeiro: temperatura.
    #um de cada vez em!
 
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(61), temp_TKE_float, 'b--', label=f'WRF TKE')
    ax1.plot(range(61), temp_SH_float, 'm-.', label=f'WRF SH')
    ax1.plot(range(61), temp_MYNN_float, 'r:', label=f'WRF MYNN')
    ax1.plot(range(len(temp_TKE_float)), temp_est_float, c='k', label=f'Observado em {estacao}')
    ax1.plot
    plt.legend(loc='upper left')

    plt.xlabel("Horario")
    plt.ylabel("Temperatura (Celsius)")
    plt.tick_params(axis ='x', rotation = 45)
    plt.xticks(np.arange(0, 73, 6),["18/01 00h","06h","12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])

    plt.savefig(f'/home/allan/plotsTCC/graficos_ob/temp/{estacao}', bbox_inches='tight', pad_inches=0, dpi=200)

    plt.show()
    
     tdzinha dos cria
 
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(len(temp_TKE_float)), td_TKE_float, 'b--', label=f'WRF TKE')
    ax1.plot(range(len(temp_TKE_float)), td_SH_float, 'm-.', label=f'WRF SH')
    ax1.plot(range(len(temp_TKE_float)), td_MYNN_float, 'r:', label=f'WRF MYNN')
    ax1.plot(range(len(temp_TKE_float)), td_est_float, c='k', label=f'Observado em {estacao}')
    ax1.plot
    plt.legend(loc='upper left')

    plt.xlabel("Horario")
    plt.ylabel("Temperatura do ponto de orvalho (Celsius)")
    plt.tick_params(axis ='x', rotation = 45)
    plt.xticks(np.arange(0, 73, 6),["18/01 00h","06h","12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])

    plt.savefig(f'/home/allan/plotsTCC/graficos_ob/td/{estacao}', bbox_inches='tight', pad_inches=0, dpi=200)

    plt.show()
    
    #agora, os proximos graficos: vento vel media horaria
    
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(len(temp_TKE_float)), wnd_TKE_float, 'b--', label=f'WRF TKE')
    ax1.plot(range(len(temp_TKE_float)), wnd_SH_float, 'm-.', label=f'WRF SH')
    ax1.plot(range(len(temp_TKE_float)), wnd_MYNN_float, 'r:', label=f'WRF MYNN')
    ax1.plot(range(len(temp_TKE_float)), wnd_est_float, c='k', label=f'Observado em {estacao}')
    ax1.plot
    plt.legend(loc='upper left')

    plt.xlabel("Horario")
    plt.ylabel("Velocidade do vento (m/s)")
    plt.tick_params(axis ='x', rotation = 45)
    plt.xticks(np.arange(0, 73, 6),["18/01 00h","06h","12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])
    #ax2=ax1.secondary_xaxis('bottom')
    #plt.xtick()

    plt.savefig(f'/home/allan/plotsTCC/graficos_ob/wnd_spd/{estacao}', bbox_inches='tight', pad_inches=0, dpi=200)
    
    #a direcao do vento
    
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(len(temp_TKE_float)), wnddir_TKE_float, 'b--', label=f'WRF TKE')
    ax1.plot(range(len(temp_TKE_float)), wnddir_SH_float, 'm-.', label=f'WRF SH')
    ax1.plot(range(len(temp_TKE_float)), wnddir_MYNN_float, 'r:', label=f'WRF MYNN')
    ax1.plot(range(len(temp_TKE_float)), wnddir_est_float, c='k', label=f'Observado em {estacao}')
    ax1.plot
    plt.legend(loc='upper left')

    plt.xlabel("Horario")
    plt.ylabel("Direção do vento (graus)")
    plt.tick_params(axis ='x', rotation = 45)
    plt.xticks(np.arange(0, 73, 6),["18/01 00h","06h","12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])
    
    #print(coluna1.split(";")[0])

    plt.savefig(f'/home/allan/plotsTCC/graficos_ob/wnd_dir/{estacao}', bbox_inches='tight', pad_inches=0, dpi=200)
    
    
    #a direcao do vento CALCULADA
    
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(len(temp_TKE_float)), wnddir_calc_TKE, 'b--', label=f'WRF TKE')
    ax1.plot(range(len(temp_TKE_float)), wnddir_calc_SH, 'm-.', label=f'WRF SH')
    ax1.plot(range(len(temp_TKE_float)), wnddir_calc_MYNN, 'r:', label=f'WRF MYNN')
    ax1.plot(range(len(temp_TKE_float)), wnddir_est_float, c='k', label=f'Observado em {estacao}')
    ax1.plot
    plt.legend(loc='upper left')

    plt.xlabel("Horario")
    plt.ylabel("Direção do vento (graus)")
    plt.tick_params(axis ='x', rotation = 45)
    #plt.xticks(np.arange(0, 73, 6),["18/01 00h","06h","12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])
    plt.xticks(np.arange(0, 61, 6),["12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])
    #print(coluna1.split(";")[0])

    plt.savefig(f'/home/allan/plotsTCC/graficos_ob/wnd_dir_calc/{estacao}', bbox_inches='tight', pad_inches=0, dpi=200)
    
    
    #a componente v do vento
    
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(61), vwind_TKE_float, 'b--', label=f'WRF TKE')
    ax1.plot(range(61), vwind_SH_float, 'm-.', label=f'WRF SH')
    ax1.plot(range(61), vwind_MYNN_float, 'r:', label=f'WRF MYNN')
    ax1.plot(range(61), vwind_est, c='k', label=f'Observado em {estacao}')
    ax1.plot
    plt.legend(loc='upper left')

    plt.xlabel("Horario")
    plt.ylabel("Velocidade da componente v do vento (m/s)")
    plt.tick_params(axis ='x', rotation = 45)
    #plt.xticks(np.arange(0, 73, 6),["18/01 00h","06h","12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])
    plt.xticks(np.arange(0, 61, 6),["12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])
    #ax2=ax1.secondary_xaxis('bottom')
    #plt.xtick()

    plt.savefig(f'/home/allan/plotsTCC/graficos_ob/vwind/{estacao}', bbox_inches='tight', pad_inches=0, dpi=200)

    #por fim, a componente u do vento :)
    
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(111)
    
    zeros=np.zeros(61)

    ax1.plot(range(61), uwind_TKE_float, 'b--', label=f'WRF TKE')
    ax1.plot(range(61), uwind_SH_float, 'm-.', label=f'WRF SH')
    ax1.plot(range(61), uwind_MYNN_float, 'r:', label=f'WRF MYNN')
    ax1.plot(range(61), uwind_est, c='k', label=f'Observado em {estacao}')
    #ax1.plot(range(len(temp_TKE_float)), zeros, c='k')
    #ax1.plot(range(len(temp_TKE_float)), zeros, c='k')
    ax1.plot
    plt.legend(loc='upper left')

    plt.xlabel("Horario")
    plt.ylabel("Velocidade da componente u do vento (m/s)")
    plt.tick_params(axis ='x', rotation = 45)
    plt.xticks(np.arange(0, 61, 6),["12h","18h","19/01 00h","06h","12h","18h","20/01 00h","06h","12h","18h","21/01 00h"])
    plt.axhline(y=0, color='r')
    vetor1=[]
    vetor1=np.append(uwind_TKE_float,uwind_SH_float)
    vetor1=np.append(vetor1,uwind_MYNN_float)
    vetor1=np.append(vetor1,uwind_est)
    maxabs=ceil_without_nan(np.max(np.abs(vetor1)))
    print(maxabs)
    plt.ylim(-(maxabs),maxabs)
    #print(coluna1.split(";")[0])

    plt.savefig(f'/home/allan/plotsTCC/graficos_ob/uwind/{estacao}', bbox_inches='tight', dpi=200)
            
            
            

            
            
            
            
            
            
    
    