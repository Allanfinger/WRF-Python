#Este script serve para salvar em .csv dados pontuais do wrf.
#Ele opera pegando as latitudes e longitudes de arquivos de estações do inmet(csv) e
#pegando as variaveis desejadas nos pontos de grade mais proximos destas lat e lon.
#Com isto, formamos um arquivo .csv com as mesmas. Isto eh util no sentido de 
#ao inves de abrirmos os .nc de cada saida do wrf toda vez que quisermos usar dados pontuais,
#montamos arquivos .csv dos quais sao muito menores, nos poupando tempo de computacao.
#
#
#Este script esta MUITO mal otimizado. Pessimo, horroroso. Entao tome cuidado: ele demora muito pra fazer seu trabalho
#e costumava travar no caminho enquanto eu rodava. 


from netCDF4 import Dataset
import numpy as np




from wrf import (getvar, ll_to_xy, g_uvmet)

from metpy.calc import wind_components
from metpy.units import units


import glob

import csv


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

from itertools import zip_longest






#Nome das estacoes para catar os arquivos csv. No caso, eu utilizei as do array. Modifique os nomes com os desejados.
EST=np.array([["A535","A540","A612","A613","A614","A623","A506","A552","A447","A522","A532","A555","A556","A518","A607","A541","A455","A538","SBVT","LIN"]])


#caminho das saidas. no caso, utilizei 3 parametrizacoes diferentes de camada limite. Entao, um caminho pra cada uma delas.
#o astericos serve como 'coringa' no .glob, entao vai pegar todos os nomes dos arquivos, mas no meu caso, apenas dos valores horarios.
#por isso '*:00:*'.
SHH = "/home/allan/SH_d01/*:00:*"
MYNN = "/home/allan/MYNN_d01/*:00:*"
TKE = "/home/allan/3DTKE_d01/*:00:*"

#loop estacao

for i in range(0,18):

    #matrix1
    llxyy=[]
    
    #temperaturas array vazios
    temperatura_SH=[]
    temperatura_MYNN=[]
    temperatura_TKE=[]
    #vento arrays vazios
    vento_SH=[]
    vento_MYNN=[]
    vento_TKE=[]
    #dir do vento arrays vazios U
    udir_SH=[]
    udir_MYNN=[]
    udir_TKE=[]
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
    #umidade
    umidade_SH=[]
    umidade_MYNN=[]
    umidade_TKE=[]
    #temp ponto de orvalho
    td_SH=[]
    td_MYNN=[]
    td_TKE=[]
    #datas
    datas=[]
    
    #zerar os arquivos csv pra poder substituir
    #zerar o csv da est especifica para SH
        
    d_SH = [datas,pressao_SH,td_SH,temperatura_SH,udir_SH,vento_SH,uwind_SH,vwind_SH]
    export_data = zip_longest(*d_SH, fillvalue = '')
    with open(f'/home/allan/SH_d01/csv/SH{EST[0,i]}.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("DATA","PRESSAO","TD","T2","DIRVENTO","MAGVENTO","MAGVENTOU","MAGVENTOV"))
      wr.writerows(export_data)
    myfile.close()
    
    #zerar o csv da est espefica pra mynn
    d_MYNN = [datas,pressao_MYNN,td_MYNN,temperatura_MYNN,udir_MYNN,vento_MYNN,uwind_MYNN,vwind_MYNN]
    export_data = zip_longest(*d_MYNN, fillvalue = '')
    with open(f'/home/allan/MYNN_d01/csv/MYNN{EST[0,i]}.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("DATA","PRESSAO","TD","T2","DIRVENTO","MAGVENTO","MAGVENTOU","MAGVENTOV"))
      wr.writerows(export_data)
    myfile.close()
    
    #zerar o csv da est especifica pra 3dtke (NAO UTILIZAMOS FUNCOES. regra numero um de programacao de allan s. finger) 
    d_TKE = [datas,pressao_TKE,td_TKE,temperatura_TKE,udir_TKE,vento_TKE,uwind_TKE,vwind_TKE]
    export_data = zip_longest(*d_TKE, fillvalue = '')
    with open(f'/home/allan/3DTKE_d01/csv/TKE{EST[0,i]}.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("DATA","PRESSAO","TD","T2","DIRVENTO","MAGVENTO","MAGVENTOU","MAGVENTOV"))
      wr.writerows(export_data)
    myfile.close()
    
    
    
    
    rows = []
    #IMPORTANTE: aqui um .glob pra extrair o caminho dos arquivos das estacoes do inmet. 
    #CUIDADO: veja como o .csv da estacao esta dividido. se eh em ',' ou ';' por ex. no meu caso, estava em ','.
    pathest=glob.glob(f'/home/allan/inmet/*{EST[0,i]}*')
    #llxy = ll_to_xy(ncfile, EST[1,i], EST[2,i])
    with open(pathest[0], 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    #coluna_est_nome=''.join(rows[0])
    coluna_est_lat=''.join(rows[1])
    coluna_est_lon=''.join(rows[2])
    #coluna_est_alt=''.join(rows[3])



    #Aqui eh onde realmente pegamos os dados. Cada loop eh para uma parametrizacao, entao sinta-se a vontade de comentar 
    #um dos loops ou acrescentar mais, de acordo com sua necessidade. Cada loop acessa uma pasta. Se vc so tiver uma simulacao
    #eh util utilizar apenas um loop.
    
    for filename in sorted(glob.glob(TKE)):

        #print(filename)
    
        ncfile = Dataset(filename)
        
        #print(filename)
        
        data=filename[30:50]
        
        #variaveis
    
        temp = getvar(ncfile, "T2")
        
        vento = g_uvmet.get_uvmet10_wspd_wdir(ncfile)
        
        pressao = getvar(ncfile,"slp")
        
        td = getvar(ncfile,"td2")
        
        #pontos de grade mais proximos pra xy
    
        llxy = ll_to_xy(ncfile, float(coluna_est_lat.split(" ")[1]), float(coluna_est_lon.split(" ")[1]))
        
    
        t10=temp[llxy[1],llxy[0]]-273.15
        
        tdpont=td[llxy[1],llxy[0]]
        
        presspont=pressao[llxy[1],llxy[0]]
        
        ventodir=vento[1,llxy[1],llxy[0]]
        
        ventospd=vento[0,llxy[1],llxy[0]]
        
        #criar lista com elas
        
        uvento=wind_components(ventospd * units('m/s'), ventodir * units.deg)
        
        d_zin = [data,str(float(presspont)),str(float(tdpont)),str(float(t10)),str(float(ventodir)),str(float(ventospd)),str(float(uvento[0])),str(float(uvento[1]))]
        #zip(d_zin)
        #bolas=','.join(d_zin)
        #export_data = zip_longest(bolas, fillvalue = '')
        with open(f'/home/allan/3DTKE_d01/csv/TKE{EST[0,i]}.csv', 'a', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerow(d_zin)
        myfile.close()
        
#    
    print(f"3DTKE FOI{i+1}")
    
    



#loops pra pegar os dados do wrp
    for filename in sorted(glob.glob(MYNN)):

        #print(filename)
    
        ncfile = Dataset(filename)
        
        data=filename[30:50]
        
        #variaveis
    
        temp = getvar(ncfile, "T2")
        
        vento = g_uvmet.get_uvmet10_wspd_wdir(ncfile)
        
        pressao = getvar(ncfile,"slp")
        
        td = getvar(ncfile,"td2")
        
        #pontos de grade mais proximos pra xy
    
        llxy = ll_to_xy(ncfile, float(coluna_est_lat.split(" ")[1]), float(coluna_est_lon.split(" ")[1]))
        
        
        #variaveis pontuais
    
        t10=temp[llxy[1],llxy[0]]-273.15
        
        tdpont=td[llxy[1],llxy[0]]
        
        presspont=pressao[llxy[1],llxy[0]]
        
        ventodir=vento[1,llxy[1],llxy[0]]
        
        ventospd=vento[0,llxy[1],llxy[0]]
        

        
        uvento=wind_components(ventospd * units('m/s'), ventodir * units.deg)
        
        d_zin = [data,str(float(presspont)),str(float(tdpont)),str(float(t10)),str(float(ventodir)),str(float(ventospd)),str(float(uvento[0])),str(float(uvento[1]))]
        #zip(d_zin)
        #bolas=','.join(d_zin)
        #export_data = zip_longest(bolas, fillvalue = '')
        with open(f'/home/allan/MYNN_d01/csv/MYNN{EST[0,i]}.csv', 'a', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerow(d_zin)
        myfile.close()
        

    
    print(f"MYNN FOI{i+1}")
        
#        
#        
   
    for filename in sorted(glob.glob(SHH)):

        #print(filename)
    
        ncfile = Dataset(filename)
        
        data=filename[30:50]
        
        #variaveis
    
        temp = getvar(ncfile, "T2")
        
        vento = g_uvmet.get_uvmet10_wspd_wdir(ncfile)
        
        pressao = getvar(ncfile,"slp")
        
        td = getvar(ncfile,"td2")
        
        #pontos de grade mais proximos pra xy
    
        llxy = ll_to_xy(ncfile, float(coluna_est_lat.split(" ")[1]), float(coluna_est_lon.split(" ")[1]))
        

        
        #variaveis pontuais
    
        t10=temp[llxy[1],llxy[0]]-273.15
        
        tdpont=td[llxy[1],llxy[0]]
        
        presspont=pressao[llxy[1],llxy[0]]
        
        ventodir=vento[1,llxy[1],llxy[0]]
        
        ventospd=vento[0,llxy[1],llxy[0]]
        

        
        uvento=wind_components(ventospd * units('m/s'), ventodir * units.deg)
        
        d_zin = [data,str(float(presspont)),str(float(tdpont)),str(float(t10)),str(float(ventodir)),str(float(ventospd)),str(float(uvento[0])),str(float(uvento[1]))]
        #zip(d_zin)
        #bolas=','.join(d_zin)
        #export_data = zip_longest(bolas, fillvalue = '')
        with open(f'/home/allan/SH_d01/csv/SH{EST[0,i]}.csv', 'a', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerow(d_zin)
        myfile.close()
        

    
    print(f"SH FOI{i+1}")
   
    



