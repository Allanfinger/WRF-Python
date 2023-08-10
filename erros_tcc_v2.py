
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

import math

import pandas as pd

#funcao pra passar pra float com null values
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan

#funcao componente u
def find_u(speed,degree):
    try:
        return wind_components(speed * units('m/s'), degree * units.deg)
    except ValueError:
        return np.nan
    



def E_m(y_actual, y_predicted):
    # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))

    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]
    
    #Calcular o erro medio 
    bervo = np.subtract(y_predicted,y_actual)
    erro_medio = bervo.mean()
    
    return erro_medio


def E_a(y_actual, y_predicted):
    # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))
    
    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]
    
    #calcular erro absoluto
    bervo = abs(np.subtract(y_predicted,y_actual))
    
    erro_abs = bervo.mean()
    
    return erro_abs


      
def RMSE(y_actual, y_predicted):
    # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))

    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]

    # Calcular o RMSE
    squared_error = np.square(np.subtract(y_predicted,y_actual))
    mean_squared_error = squared_error.mean()
    rmse = math.sqrt(mean_squared_error)

    return rmse


def E_mn(y_actual, y_predicted):
    # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))

    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]
    
    #Calcular o erro medio 
    bervo = np.subtract(y_predicted,y_actual)
    erro_medio = bervo.mean()
    
    #calcular o erro medio normalizado
    soma_das_m= y_predicted.mean() + y_actual.mean()
    erro_medio_normalizado=erro_medio/(soma_das_m*0.5)
    
    return erro_medio_normalizado

def E_man(y_actual, y_predicted):
    # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))
    
    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]
    
    #calcular erro absoluto
    bervo = abs(np.subtract(y_predicted,y_actual))
    
    erro_abs = bervo.mean()
    
    #calcular o erro medio absoluto normalizado
    soma_das_m= y_actual.mean() + y_predicted.mean()
    erro_medio_absoluto_normalizado = erro_abs/soma_das_m
    
    return erro_medio_absoluto_normalizado 


def E_rqmc(y_actual, y_predicted):
    # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))
    
    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]
    
    #previsto menos media do previsto
    prev_med_prev=y_predicted-y_predicted.mean()
    obs_med_obs=y_actual-y_actual.mean()
    #um menos o outro ao quadrado
    prev_obs_quadrado=(prev_med_prev-obs_med_obs)**2
    #agr o querido
    raiz_erro_medio_quadratico_centrado=(prev_obs_quadrado.mean())**1/2
    
    return raiz_erro_medio_quadratico_centrado

def E_rqmcn(y_actual, y_predicted):
        # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))
    
    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]
    
    #previsto menos media do previsto
    prev_med_prev=y_predicted-y_predicted.mean()
    obs_med_obs=y_actual-y_actual.mean()
    #um menos o outro ao quadrado
    prev_obs_quadrado=(prev_med_prev-obs_med_obs)**2
    #ele
    raiz_erro_medio_quadratico_centrado=(prev_obs_quadrado.mean())**1/2
    #calculo do amado
    erro_rqmcn=raiz_erro_medio_quadratico_centrado/(np.std(y_predicted)*np.std(y_actual))**1/2
    
    return erro_rqmcn

def E_mm(y_actual, y_predicted):
        # Converter para arrays numpy
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Obter os índices dos valores não nulos
    indices = np.where(~np.isnan(y_actual))

    # Aplicar os índices aos arrays
    y_actual = y_actual[indices]
    y_predicted = y_predicted[indices]
    
    #Calcular o erro medio 
    bervo = np.subtract(y_predicted,y_actual)
    erro_medio = bervo.mean()
    
    #calcular o erro medio normalizado
    soma_das_m=  y_predicted.mean() + y_actual.mean()
    erro_medio_normalizado=erro_medio/(soma_das_m*0.5)
    
    #calcular erro absoluto
    bervo = abs(np.subtract(y_predicted,y_actual))
    
    erro_abs = bervo.mean()
    
    #calcular o erro medio absoluto normalizado
    soma_das_m= y_actual.mean() + y_predicted.mean()
    erro_medio_absoluto_normalizado = erro_abs/soma_das_m
    #calcular o e_rqmcn
    #previsto menos media do previsto
    prev_med_prev=y_predicted-y_predicted.mean()
    obs_med_obs=y_actual-y_actual.mean()
    #um menos o outro ao quadrado
    prev_obs_quadrado=(prev_med_prev-obs_med_obs)**2
    #ele
    raiz_erro_medio_quadratico_centrado=(prev_obs_quadrado.mean())**1/2
    #calculo do amado
    erro_rqmcn=raiz_erro_medio_quadratico_centrado/(np.std(y_predicted)*np.std(y_actual))**1/2
    #agora, o cobiçado
    erro_media_metrica=(erro_medio_normalizado+erro_medio_absoluto_normalizado+erro_rqmcn)/3
    
    return erro_media_metrica




def erros(erross,caminho_csvs,EST,MYNN,SHH,TKE):
    
    for er in erross:
        header1 = ['estacao',f'{er}_temp_tke',f'{er}_temp_sh',f'{er}_temp_mynn',f'{er}_td_tke',f'{er}_td_sh',f'{er}_td_mynn',
                   f'{er}_wnd_tke',f'{er}_wnd_sh',f'{er}_wnd_mynn',f'{er}_wnddir_tke',f'{er}_wnddir_sh',f'{er}_wnddir_mynn',
                   f'{er}_uwind_tke',f'{er}_uwind_sh',f'{er}_uwind_mynn',f'{er}_vwind_tke',f'{er}_vwind_sh',f'{er}_vwind_mynn']

        with open(f'{caminho_csvs}/{er}.csv', 'w', encoding="UTF8", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(header1)
        myfile.close()
        
    
    
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
        #dir do vento arrays vazios U
        wnddir_SH=[]
        wnddir_MYNN=[]
        wnddir_TKE=[]
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
        #columnas=rows[34:107]
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
                
        #colunitas=rows_SH[0:73]
        colunitas=rows_SH[12:73]
        for f in colunitas:
            coluna1=','.join(f)
            pressao_SH=np.append(pressao_SH,coluna1.split(",")[1])
            temp_SH=np.append(temp_SH,coluna1.split(",")[3])
            wnd_SH=np.append(wnd_SH,coluna1.split(",")[5])
            uwind_SH=np.append(uwind_SH,coluna1.split(",")[6])
            vwind_SH=np.append(vwind_SH,coluna1.split(",")[7])
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
            wnddir_TKE=np.append(wnddir_TKE,coluna1.split(",")[4])
            td_TKE=np.append(td_TKE,coluna1.split(",")[2])
            
            
                    #passando pra float
        temp_TKE_float = [convert_to_float(value) for value in temp_TKE]
        wnd_TKE_float = [convert_to_float(value) for value in wnd_TKE] 
        wnddir_TKE_float = [convert_to_float(value) for value in wnddir_TKE]
        td_TKE_float = [convert_to_float(value) for value in td_TKE]
        uwind_TKE_float = [convert_to_float(value) for value in uwind_TKE]
        vwind_TKE_float = [convert_to_float(value) for value in vwind_TKE]
        
        array_var_pr=[temp_TKE_float,temp_SH_float,temp_MYNN_float,td_TKE_float,td_SH_float,td_MYNN_float,
                      wnd_TKE_float,wnd_SH_float,wnd_MYNN_float,wnddir_TKE_float,wnddir_SH_float,wnddir_MYNN_float,
                      uwind_TKE_float,uwind_SH_float,uwind_MYNN_float,vwind_TKE_float,vwind_SH_float,vwind_MYNN_float]
        
        array_var_obs=[temp_est_float,td_est_float,wnd_est_float,wnddir_est_float,uwind_est,vwind_est]
        
        #calc pra temp
        
        erro_med_array=[]
        erro_a_array=[]
        erro_rqm_array=[]
        erro_mn_array=[]
        erro_man_array=[]
        erro_rqmc_array=[]
        erro_rqmcn_array=[]
        erro_mm_array=[]
        
        j=0
        
        for i in range(0,3):
            erro_med=str(float(E_m(array_var_obs[j],array_var_pr[i])))
            erro_abs=str(float(E_a(array_var_obs[j],array_var_pr[i])))
            erro_rqm=str(float(RMSE(array_var_obs[j],array_var_pr[i])))
            erro_mn=str(float(E_mn(array_var_obs[j],array_var_pr[i])))
            erro_man=str(float(E_man(array_var_obs[j],array_var_pr[i])))
            erro_rqmc=str(float(E_rqmc(array_var_obs[j],array_var_pr[i])))
            erro_rqmcn=str(float(E_rqmcn(array_var_obs[j],array_var_pr[i])))
            erro_mm=str(float(E_mm(array_var_obs[j],array_var_pr[i])))
            
            erro_med_array=np.append(erro_med_array,erro_med)
            erro_a_array=np.append(erro_a_array,erro_abs)
            erro_rqm_array=np.append(erro_rqm_array,erro_rqm)
            erro_mn_array=np.append(erro_mn_array,erro_mn)
            erro_man_array=np.append(erro_man_array,erro_man)
            erro_rqmc_array=np.append(erro_rqmc_array,erro_rqmc)
            erro_rqmcn_array=np.append(erro_rqmcn_array,erro_rqmcn)
            erro_mm_array=np.append(erro_mm_array,erro_mm)
            
            
            
        j=j+1
            
        for i in range(3,6):
            erro_med=str(float(E_m(array_var_obs[j],array_var_pr[i])))
            erro_abs=str(float(E_a(array_var_obs[j],array_var_pr[i])))
            erro_rqm=str(float(RMSE(array_var_obs[j],array_var_pr[i])))
            erro_mn=str(float(E_mn(array_var_obs[j],array_var_pr[i])))
            erro_man=str(float(E_man(array_var_obs[j],array_var_pr[i])))
            erro_rqmc=str(float(E_rqmc(array_var_obs[j],array_var_pr[i])))
            erro_rqmcn=str(float(E_rqmcn(array_var_obs[j],array_var_pr[i])))
            erro_mm=str(float(E_mm(array_var_obs[j],array_var_pr[i])))
            
            erro_med_array=np.append(erro_med_array,erro_med)
            erro_a_array=np.append(erro_a_array,erro_abs)
            erro_rqm_array=np.append(erro_rqm_array,erro_rqm)
            erro_mn_array=np.append(erro_mn_array,erro_mn)
            erro_man_array=np.append(erro_man_array,erro_man)
            erro_rqmc_array=np.append(erro_rqmc_array,erro_rqmc)
            erro_rqmcn_array=np.append(erro_rqmcn_array,erro_rqmcn)
            erro_mm_array=np.append(erro_mm_array,erro_mm)
            
        j=j+1
        
        for i in range(6,9):
            erro_med=str(float(E_m(array_var_obs[j],array_var_pr[i])))
            erro_abs=str(float(E_a(array_var_obs[j],array_var_pr[i])))
            erro_rqm=str(float(RMSE(array_var_obs[j],array_var_pr[i])))
            erro_mn=str(float(E_mn(array_var_obs[j],array_var_pr[i])))
            erro_man=str(float(E_man(array_var_obs[j],array_var_pr[i])))
            erro_rqmc=str(float(E_rqmc(array_var_obs[j],array_var_pr[i])))
            erro_rqmcn=str(float(E_rqmcn(array_var_obs[j],array_var_pr[i])))
            erro_mm=str(float(E_mm(array_var_obs[j],array_var_pr[i])))
            
            erro_med_array=np.append(erro_med_array,erro_med)
            erro_a_array=np.append(erro_a_array,erro_abs)
            erro_rqm_array=np.append(erro_rqm_array,erro_rqm)
            erro_mn_array=np.append(erro_mn_array,erro_mn)
            erro_man_array=np.append(erro_man_array,erro_man)
            erro_rqmc_array=np.append(erro_rqmc_array,erro_rqmc)
            erro_rqmcn_array=np.append(erro_rqmcn_array,erro_rqmcn)
            erro_mm_array=np.append(erro_mm_array,erro_mm)
            
        j=j+1
            
        for i in range(9,12):
            erro_med=str(float(E_m(array_var_obs[j],array_var_pr[i])))
            erro_abs=str(float(E_a(array_var_obs[j],array_var_pr[i])))
            erro_rqm=str(float(RMSE(array_var_obs[j],array_var_pr[i])))
            erro_mn=str(float(E_mn(array_var_obs[j],array_var_pr[i])))
            erro_man=str(float(E_man(array_var_obs[j],array_var_pr[i])))
            erro_rqmc=str(float(E_rqmc(array_var_obs[j],array_var_pr[i])))
            erro_rqmcn=str(float(E_rqmcn(array_var_obs[j],array_var_pr[i])))
            erro_mm=str(float(E_mm(array_var_obs[j],array_var_pr[i])))
            
            erro_med_array=np.append(erro_med_array,erro_med)
            erro_a_array=np.append(erro_a_array,erro_abs)
            erro_rqm_array=np.append(erro_rqm_array,erro_rqm)
            erro_mn_array=np.append(erro_mn_array,erro_mn)
            erro_man_array=np.append(erro_man_array,erro_man)
            erro_rqmc_array=np.append(erro_rqmc_array,erro_rqmc)
            erro_rqmcn_array=np.append(erro_rqmcn_array,erro_rqmcn)
            erro_mm_array=np.append(erro_mm_array,erro_mm)
            
        j=j+1
            
        for i in range(12,15):
            erro_med=str(float(E_m(array_var_obs[j],array_var_pr[i])))
            erro_abs=str(float(E_a(array_var_obs[j],array_var_pr[i])))
            erro_rqm=str(float(RMSE(array_var_obs[j],array_var_pr[i])))
            erro_mn=str(float(E_mn(array_var_obs[j],array_var_pr[i])))
            erro_man=str(float(E_man(array_var_obs[j],array_var_pr[i])))
            erro_rqmc=str(float(E_rqmc(array_var_obs[j],array_var_pr[i])))
            erro_rqmcn=str(float(E_rqmcn(array_var_obs[j],array_var_pr[i])))
            erro_mm=str(float(E_mm(array_var_obs[j],array_var_pr[i])))
            
            erro_med_array=np.append(erro_med_array,erro_med)
            erro_a_array=np.append(erro_a_array,erro_abs)
            erro_rqm_array=np.append(erro_rqm_array,erro_rqm)
            erro_mn_array=np.append(erro_mn_array,erro_mn)
            erro_man_array=np.append(erro_man_array,erro_man)
            erro_rqmc_array=np.append(erro_rqmc_array,erro_rqmc)
            erro_rqmcn_array=np.append(erro_rqmcn_array,erro_rqmcn)
            erro_mm_array=np.append(erro_mm_array,erro_mm)
            
        j=j+1
            
        for i in range(15,18):
            erro_med=str(float(E_m(array_var_obs[j],array_var_pr[i])))
            erro_abs=str(float(E_a(array_var_obs[j],array_var_pr[i])))
            erro_rqm=str(float(RMSE(array_var_obs[j],array_var_pr[i])))
            erro_mn=str(float(E_mn(array_var_obs[j],array_var_pr[i])))
            erro_man=str(float(E_man(array_var_obs[j],array_var_pr[i])))
            erro_rqmc=str(float(E_rqmc(array_var_obs[j],array_var_pr[i])))
            erro_rqmcn=str(float(E_rqmcn(array_var_obs[j],array_var_pr[i])))
            erro_mm=str(float(E_mm(array_var_obs[j],array_var_pr[i])))
            
            erro_med_array=np.append(erro_med_array,erro_med)
            erro_a_array=np.append(erro_a_array,erro_abs)
            erro_rqm_array=np.append(erro_rqm_array,erro_rqm)
            erro_mn_array=np.append(erro_mn_array,erro_mn)
            erro_man_array=np.append(erro_man_array,erro_man)
            erro_rqmc_array=np.append(erro_rqmc_array,erro_rqmc)
            erro_rqmcn_array=np.append(erro_rqmcn_array,erro_rqmcn)
            erro_mm_array=np.append(erro_mm_array,erro_mm)
            
            
        os_erros=[estacao,erro_med_array,erro_a_array,erro_rqm_array,erro_mn_array,erro_man_array,
                  erro_rqmc_array,erro_rqmcn_array,erro_mm_array]
        
        
    
        
        for er,prrr in zip(erross,range(1,9)):
            bbberdos=[os_erros[0],os_erros[prrr][0],os_erros[prrr][1],os_erros[prrr][2],os_erros[prrr][3],os_erros[prrr][4],os_erros[prrr][5],
                      os_erros[prrr][6],os_erros[prrr][7],os_erros[prrr][8],os_erros[prrr][9],os_erros[prrr][10],os_erros[prrr][11],
                      os_erros[prrr][12],os_erros[prrr][13],os_erros[prrr][14],os_erros[prrr][15],os_erros[prrr][16],os_erros[prrr][17],
                      ]
            with open(f'{caminho_csvs}/{er}.csv', 'a', encoding="UTF8", newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(bbberdos)
            myfile.close()
    print(os_erros[1][0])
    #print(erro_med_array)
    return print("FOI")
            
            
        
        
EST=np.array([["A535","A540","A612","A613","A614","A623","A506","A552","A447","A522","A532","A555","A556","A518","A607","A541","A455","A538"],
             [-19.533,-18.781,-20.271,-19.988,-19.357,-18.695,-16.686,-16.160,-16.088,-17.799,-18.830,-20.031,-21.770,-21.715,-17.706,-17.007,-18.748,-18.748],
             [-41.091,-40.987,-40.306,-40.580,-40.069,-40.391,-43.844,-42.310,-39.215,-40.250,-41.977,-44.011,-42.183,-43.364,-41.344,-42.390,-39.558,-44.454]])
pr=['SH','MYNN','TKE']

SHH = "/home/allan/SH_d01/csv/"
MYNN = "/home/allan/MYNN_d01/csv/"
TKE = "/home/allan/3DTKE_d01/csv/"


erross=['e_m','e_ma','e_rqm','e_mn','e_man','e_rqmc','e_rqmcn','e_mm']
caminho_csvs='/home/allan/plotsTCC/estatistica'

los_erros=erros(erross,caminho_csvs,EST,MYNN,SHH,TKE)

#loop estacao

#tipo=['RMSE','CORR']

#for nome in



#linhas1 = ['CORR_RMSE_temp_tke','CORR_RMSE_temp_sh','CORR_RMSE_temp_mynn','CORR_RMSE_td_tke','CORR_RMSE_td_sh','CORR_RMSE_td_mynn',
#              'CORR_RMSE_wnd_tke','CORR_RMSE_wnd_sh','CORR_RMSE_wnd_mynn','CORR_RMSE_wnddir_tke','CORR_RMSE_wnddir_sh','CORR_RMSE_wnddir_mynn',
#              'CORR_RMSE_uwind_tke','CORR_RMSE_uwind_sh','CORR_RMSE_uwind_mynn','CORR_RMSE_vwind_tke','CORR_RMSE_vwind_sh','CORR_RMSE_vwind_mynn']

#header2= ['nome','dif_altura','dif_lat_wrf-est','dif_lon_wrf-est','dif_total_latlon','dif_total_altura']


          
        
        

