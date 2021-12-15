# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:41:45 2021
@author: shani
"""
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset 
import numpy as np
import pandas as pd
import seaborn as sns
os.chdir(r'D:/Master/era5/scripts')
from singlePoint import var1d
from matplotlib import pyplot as plt 
import math

def era5df(path, lat_lon):
    era5 = pd.DataFrame({'levels': [],'UTC': [] , 'temp(K)': [], "humidity(kg/kg)": [], "wind_U(m/s)": [],\
                         "wind_V(m/s)": [],'pressure(hPa)': [], "RH": [],"Altitude(m)": [] })
    levels = np.arange(7,138)
    os.chdir(path)
    for fname in os.listdir():
        time = pd.to_datetime(fname[1:], format = '%Y%m%d_%H')
        date = fname[1:]
        if fname[0] == 'P':
            T = var1d(fname, lat_lon, 'T')
            Q = var1d(fname,lat_lon, 'Q')
            U = var1d(fname,lat_lon, 'U')
            V = var1d(fname,lat_lon, 'V')
            
            if 'S'+date in os.listdir():
                P = var1d('S'+date, lat_lon, 'P')
                RH = var1d('S'+date,lat_lon, 'RH')
            else: 
                print('S'+date+" doesn't exist")
                continue
             
            if 'Z'+date in os.listdir():
                altitude = var1d('Z'+date,lat_lon,'Z')
            else:
                print('Z'+date+" doesn't exist")
                continue
            
            new = pd.DataFrame({'levels': levels, 'UTC': time, 'temp(K)':T, "humidity(kg/kg)":Q, "wind_U(m/s)":U,\
                          "wind_V(m/s)": V,'pressure(hPa)':P, "RH": RH,"Altitude(m)":altitude })
            era5 = era5.append(new)
    era5.set_index("UTC", inplace=True) 
    return era5

    
#%% DO NOT RUN --  build meteorological data set 
if __name__ == '__main__':

    lat_lon = [30,35]
    path = r'D:/Master/era5/netfiles'
    era5 = era5df(path, lat_lon)
    
    era5["UTC"] = era5.index
    era5.set_index(pd.to_datetime(era5.index),inplace=True)
    era5.to_csv(r'D:/Master/era5/era5.csv')
     
    
    #%%
    import scipy.integrate as integrate
    from datetime import timedelta
    
    # import the era5 data 
    era5 = pd.read_csv(r'D:/Master/era5/era5.csv')
    era5.set_index("UTC", inplace = True)
    era5.set_index(pd.to_datetime(era5.index), inplace=True)
    
    # import cyclope data
    from singlePoint import datadf 
    df = datadf(r'D:/Master/fig_sum/Seeing_Data.txt')
    df.set_index("UTC", inplace = True)
    df = df.resample('3h').mean().dropna()
    dfday = df.resample('1D').median().dropna()
    
    # compute chi and S medians of all the data per layer:
    chi_values = []
    S_values = []
    start = pd.to_datetime('2021-03-01 00:00:00')
    while start < era5.index[-1]:
        r = era5[era5.index == start]["Altitude(m)"].diff(periods=-1)
        # potential temperature gradient
        theta = era5[era5.index == start]["temp(K)"]*(1000/era5[era5.index == start]["pressure(hPa)"])**(287/1004)
        dtheta = theta.diff(periods=-1)
        chi = list(np.array(dtheta[:-1])/np.array(r[:-1]))
        chi_values.append(chi)
        
        # wind shear 
        dVx = np.array(era5[era5.index == start]["wind_U(m/s)"].diff(periods=-1))
        dVy = np.array(era5[era5.index == start]["wind_V(m/s)"].diff(periods=-1))
        S = list(np.sqrt((dVx[:-1]/r[:-1])**2 + (dVy[:-1]/r[:-1])**2))
        S_values.append(S)

        start += timedelta(hours=3)
        
    # create arrays with median for each layer
    chi_median = np.median(np.array(chi_values))
    S_median = np.median(np.array(S_values))
        
    altitude = era5[era5.index == start]["Altitude(m)"] - era5[era5.index == start]["Altitude(m)"][-1]
    l = [] 
    rm = []
    
    #%%
    ct2_median = [1.4*1E-2]*3 + [(4.8*1E-4)*(h/100)**-2.6 for h in altitude[(altitude>=50) & (altitude<100)]]+\
                  [(3.4*10**-5)*(h/1000)**-1.1 for h in altitude[(altitude <= 1000) & (altitude >= 100)]] +\
                      [5.2*10**-5]*len(altitude[(altitude>1000) & (altitude<=2000)]) +\
                          [4.2*1E-5]*len(altitude[(altitude>2000) & (altitude<=3000)]) +\
                              [2.8*1E-5]*len(altitude[(altitude>3000) & (altitude<=4000)]) +\
                       [2.5*1E-5]*len(altitude[(altitude>4000) & (altitude<=5000)]) +\
                    [1.8*1E-5]*len(altitude[(altitude>5000) & (altitude<=6000)]) +\
                [1.5*1E-5]*len(altitude[(altitude>6000) & (altitude<=7000)]) +\
             [1.6*1E-5]*len(altitude[(altitude>7000) & (altitude<=9000)]) +\
            [1.9*1E-5]*len(altitude[(altitude>9000) & (altitude<=10000)]) +\
              [2.6*1E-5]*len(altitude[(altitude>10000) & (altitude<=11000)])+\
            [3.4*1E-5]*len(altitude[(altitude>11000) & (altitude<=12000)]) +\
                   [4.4*1E-5]*len(altitude[(altitude>12000) & (altitude<=13000)]) +\
                   [4.8*1E-5]*len(altitude[(altitude>13000) & (altitude<=14000)]) +\
                     [5.5*1E-5]*len(altitude[(altitude>14000) & (altitude<=15000)]) +\
                          [6.5*1E-5]*len(altitude[(altitude>15000) & (altitude<=16000)]) +\
                           [8.3*1E-5]*len(altitude[(altitude>16000) & (altitude<=17000)]) +\
                            [1.1*1E-4]*len(altitude[(altitude>17000) & (altitude<=19000)]) +\
                            [9.5*1E-5]*len(altitude[(altitude>19000) & (altitude<=20000)]) +\
                            [8.2*1E-5]*len(altitude[(altitude>20000) & (altitude<=21000)]) +\
                            [7.4*1E-5]*len(altitude[(altitude>21000) & (altitude<=22000)]) +\
                                [7.5*1E-5]*len(altitude[(altitude>22000) & (altitude<=23000)]) +\
                                    [8.7*1E-5]*len(altitude[(altitude>23000) & (altitude<=24000)]) +\
                                        [1.1*1E-4]*len(altitude[(altitude>24000) & (altitude<=25000)]) +\
                                        [1.3*1E-4]*len(altitude[(altitude>25000) & (altitude<=26000)]) +\
                                            [1.7*1E-4]*len(altitude[(altitude>26000) & (altitude<=27000)]) +\
                                               [2.4*1E-4]*len(altitude[(altitude>27000) & (altitude<=28000)]) +\
                                                   [3*1E-4]*len(altitude[(altitude>28000) & (altitude<=29000)]) +\
                                                     [5.1*1E-5]*len(altitude[(altitude>29000) & (altitude<=30000)])
   
    
    L =9
    ct2_median = ct2_median[-L:]
    Lambda = 5.5*1E-7  #(m)                   
    seeing = [] 
    index = []
    for time in set(era5.index):
        T = era5[era5.index == time]["temp(K)"]
        P = era5[era5.index == time]["pressure(hPa)"]
        ct2 = np.array(ct2_median)*chi[-L:]/chi_median[-L:]*S[-L:]/S_median[-L:]
        cn2 = ((80*1E-6*(P[-L:]/T[-L:]**2))**2)*ct2
        cn2.name = "Cn2"
        integral = integrate.trapezoid(x=altitude[::-1][-L:],y=cn2[::-1])    
        e0 = (5.25*Lambda**(-1/5))*(integral**(3/5))*206265 
        seeing.append(e0)
        index.append(str(time))
   
    final = pd.DataFrame({'UTC': index,'seeing_model':seeing})
    final.set_index("UTC", inplace=True)
    final.set_index(pd.to_datetime(final.index),inplace=True)
    finalday = final.resample('1D').median()
    
    merge = pd.merge(df,final,how="inner",left_index=True, right_index=True)
    mergeday = pd.merge(dfday,finalday,how="inner",left_index=True, right_index=True)
    
    """
    merge[["seeing","seeing_model"]].plot.scatter("seeing_model","seeing")
    mergeday[["seeing","seeing_model"]].plot()
    """
    
    from sklearn.metrics import mean_squared_error as rmse
    strrmse = "RMSE = "+str(round(rmse(mergeday["seeing"],mergeday["seeing_model"]),3))+", L="+str(L)
    import matplotlib.dates as dates
    
    plt.scatter(merge.index, merge["seeing"], s=5)
    plt.scatter(merge.index, merge["seeing_model"], s=5)
    plt.xticks(rotation=20)    
    plt.ylabel("seeing [arcsec]", fontsize = 13)
    plt.text(dates.date2num(merge.index[40]), 3,s=strrmse)
    plt.legend(["seeing","seeing_model"])
    plt.savefig(r"D:\Master\era5\estimated_and_observed,L=" +str(L)+".pdf")
    
    #l.append(L)
    #rm.append(round(rmse(mergeday["seeing"],mergeday["seeing_model"]),3))
    
    #%% plot the seeing - observed vs estimated, on a 1:1 line 
    plt.plot(merge["seeing"],merge["seeing"])
    plt.scatter(merge["seeing"],merge["seeing_model"],c="g")
    plt.legend(["1:1 line", "estimated seeing"])
    plt.ylabel("Seeing [arcsec]", fontsize=12)
    plt.xlabel("Seeing [arcsec]",fontsize=12)
    plt.savefig(r"D:\Master\era5\model_vs_observed,L=" +str(L)+".pdf")
    
    #%% plot the RMSE vs the number of atmospheric layers used 

    plt.scatter(l,rm, s= 5)
    plt.xlabel("Number of atmospheric layers considered", fontsize = 12)
    plt.ylabel("RMSE [arcsec]",fontsize = 12)
    plt.text(55,18, "minmum RMSE = "+str(min(rm)) + ", L = "+str(l[rm.index(min(rm))])) 
    plt.scatter(l[rm.index(min(rm))],min(rm),s=30, marker='o',facecolors='none',edgecolors='r')
    plt.savefig(r"D:\Master\era5\l_rmse.pdf")
    
    #%% # use data of 1 hour for example 
    start = pd.to_datetime('2021-03-02 00:00:00')
    end = pd.to_datetime('2021-03-02 03:00:00')
    mask = (era5.index > start) & (era5.index <= end) 
    example = era5.loc[mask]
    
    Lambda = 550*1E-9  #(m)
    TempDiff = example["temp(K)"].diff(periods=-1)
    altitude = example["Altitude(m)"] - example["Altitude(m)"][-1]
    r = example["Altitude(m)"].diff(periods=-1)
    r = r[~np.isnan(r)][-37:]
    P = example["pressure(hPa)"]
    T = example["temp(K)"]
    
    # potential temperature gradient
    theta = T*(1000/P)**(287/1004)
    dtheta = theta.diff(periods=-1)
    dtehta = dtheta[~np.isnan(dtheta)][-37:]
    chi = dtehta/r
    
    # wind shear 
    dVx = example["wind_U(m/s)"].diff(periods=-1)
    dVy = example["wind_V(m/s)"].diff(periods=-1)
    dVx = dVx[~np.isnan(dVx)]
    dVy = dVy[~np.isnan(dVy)]
    
    S = np.sqrt((dVx[-37:]/r)**2 + (dVy[-37:]/r)**2)
    S_median = np.median(S[-37:])
    
    Lambda = 5.5*1E-7  #(m)                    
    T = era5[era5.index == start]["temp(K)"]
    P = era5[era5.index == start]["pressure(hPa)"]
    ct2 = np.array(ct2_median)*chi[-36:]/chi_median[-36:]*S[-36:]/S_median[-36:]
    cn2 = ((80*1E-6*(P[-36:]/T[-36:]**2))**2)*ct2
    cn2.name = "Cn2"
    
    # plot temperature structure vs altitude 
    X = [math.log10(i) for i in ct2]
    Y = (altitude/1000)[-36:]
    plt.plot(X,Y)
    plt.xlabel('log(CT2)',fontsize=13)
    plt.ylabel("Altitude [km]",fontsize=13)
    plt.savefig(r'D:/Master/era5/CT2_altitude.pdf')
    plt.show()
    plt.close()
    
    # plot cn2 vs altitude 
    x = [math.log10(i) for i in cn2]
    y = (altitude/1000)[-36:]
    plt.plot(x,y,marker='o',markersize=3,linewidth=1,markerfacecolor="pink",markeredgecolor="black")
    plt.xlabel("log(Cn2)",fontsize=13)
    plt.ylabel("Altitude[km]",fontsize=13)
    #plt.ylim([0,350])
    plt.show()
    plt.savefig(r'D:/Master/era5/cn2_altitude.pdf')
   
    #integral = np.abs(integrate.simpson(y=cn2,x=r))
    integral = integrate.trapezoid(x=altitude[::-1][-36:],y=cn2[::-1])
    print("cn2 integral = ", integral)
    
    e0 = (5.25*Lambda**(-1/5))*(integral**(3/5))*206265 
    print("e0 = ", e0, "arcsec")
    