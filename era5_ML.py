
import os
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset 
import numpy as np
import pandas as pd
os.chdir(r'D:/Master/python_scripts')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import scipy
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pickle

def LevelToP1D(fname, lat_lon):
    """fname = file name in a format of PYYYYMMDD_HH.
    lat_lon = list of [latitude,longitude].
    returns: pressure levels (pascal)"""
    pfile = Dataset(fname,"r", format='nc')
    point_index = [np.where(pfile['lat'][:]==lat_lon[0])[0][0],np.where(pfile['lon'][:]==lat_lon[1])[0][0]]
    psrf = pfile.variables["PS"][:,0,point_index[0],point_index[1]].data * 100 # convert to Pascal
    hyam = pfile.variables["hyam"][6::]#/100000
    hybm = pfile.variables["hybm"][6::]  
    p = pd.Series((hybm*psrf)+hyam)    # pressure of each level - in pascal!!
    #p0 = 101325   
    #altitude =  
    return p

def var1d(fname, lat_lon, var):
    file = Dataset(fname,"r", format='nc')
    point_index = [np.where(file['lat'][:]==lat_lon[0])[0][0],np.where(file['lon'][:]==lat_lon[1])[0][0]]
    var = pd.Series(np.squeeze(file[var][:,:,point_index[0],point_index[1]])).astype(float)
    return var 
    
def era5df(path, lat_lon, altitude):
    era5P = pd.DataFrame({'UTC': [] , 'temp(K)': [], "humidity(kg/kg)": [], "wind_U(m/s)": [], "wind_V(m/s)": [] })
    era5S = pd.DataFrame({'UTC': [], 'pressure': [], "RH": [] })
    alt = pd.DataFrame({'UTC':[],'Altitude(m)': []})
    levels = np.arange(7,138)
    os.chdir(path)
    for fname in os.listdir():
        if fname[0] == 'P':
            T = var1d(fname, lat_lon, 'T')
            Q = var1d(fname,lat_lon, 'Q')
            U = var1d(fname,lat_lon, 'U')
            V = var1d(fname,lat_lon, 'V')
            time = pd.to_datetime(fname[1:], format = '%Y%m%d_%H')
            new = pd.DataFrame({'levels': levels, 'Altitude': altitude, 'UTC': time,'temp(K)': T, "humidity(kg/kg)": Q, "wind_U(m/s)": U, "wind_V(m/s)": V })        
            era5P = pd.concat([era5P,new])           
            
        if fname[0] == 'S':
            P = var1d(fname, lat_lon, 'P')
            RH = var1d(fname,lat_lon, 'RH')
            time = pd.to_datetime(fname[1:], format = '%Y%m%d_%H')
            new2 = pd.DataFrame({'levels': levels,'UTC': time, 'pressure': P, "RH": RH })
            era5S = pd.concat([era5S,new2])
            
        if fname[0] == 'Z':
            alti = var1d(fname,lat_lon,'Z')
            time = pd.to_datetime(fname[1:], format = '%Y%m%d_%H')
            altnew = pd.DataFrame({'UTC':time,'Altitude(m)': alti})
            alt = alt.append(altnew)
            
    era5P.set_index("UTC", inplace=True, drop=False) 
    era5S.set_index("UTC", inplace=True, drop=False)
    return era5P, era5S,alt

def datadf(datafile):
    file = open(datafile, "r")
    data = []
    for line in file:               # converts the data .txt file to a list of lists 
        file = open(datafile, "r")
        stripped_line = line. strip()
        line_list = stripped_line. split()
        data.append(line_list)
        file.close() 
    
    Date = []
    UTC = []
    Seeing = []
    r0 = []
    row = 0
    while row < len(data):             # extract values of Date, Hour (LST) and seeing from data - to new lists each.
        Date.append(data[row][0])
        UTC.append(data[row][1])
        Seeing.append(float(data[row][10]))
        r0.append(float(data[row][12]))      # r0 in mm 
        row += 1

    d = {'Date': Date, 'UTCh': UTC, 'Seeing': Seeing, 'r0': r0}
    df = pd.DataFrame(data=d)
    
    time = pd.to_datetime(df['Date'] + ' ' + df['UTCh'],format = '%d/%m/%Y %H:%M:%S')
    df = pd.DataFrame({'UTC': time, 'Hour': UTC, 'seeing': df.Seeing, 'r0': df.r0 })
    return df 

#%% # import era5 data (previously saved) 

    merged = pd.read_csv(r'D:/Master/era5/era5_df_merged_std.csv')
    #merged.set_index("UTC", inplace = True)
    
    # import era5 data 
    era5P = pd.read_csv('D:/Master/era5/era5P.csv', index_col = "UTC")
    era5P.index = pd.to_datetime(era5P.index)
    era5S = pd.read_csv('D:/Master/era5/era5S.csv', index_col = "UTC")
    era5S.index = pd.to_datetime(era5S.index)

    #import seeing data 
    df = datadf(r'D:/Master/fig_sum/Seeing_Data.txt')
    df.set_index("UTC", inplace = True)
    df = df.resample('3h').mean().dropna()
    #df.index.rename("UTC", inplace = True)
    
    #%% delete "_area" from new files names
    
    path = r'D:/Master/era5/netfiles'
    os.chdir(path)
    for file in os.listdir(path):
        os.rename(file, file[0:12])
    
#%%
##################### main (import new era5 data) ########################
if __name__ == '__main__':
    import time
    start_time = time.time()
    altitude = pd.read_html("https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels")\
        [0]["Geometric Altitude [m]"][7:].reset_index(drop=True)
    # create a table with temp, humidity and wind speed (era5P), and a table with pressure and relative humidity (era5S):
    path = r'D:/Master/era5/netfiles'
    lat_lon = [30,35]
    era5Pn, era5Sn = era5df(path, lat_lon,altitude)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    era5Pn.drop("UTC", 1, inplace=True)
    era5Sn.drop("UTC", 1, inplace=True)
    era5Pn.to_csv(r"D:\Master\era5\era5P.csv")
    era5Sn.to_csv(r"D:\Master\era5\era5S.csv")
    
    # merge with existing data 
    #era5P = pd.concat([era5P, era5Pn]).sort_index()
    #era5P.to_csv(r"D:\Master\era5\era5P.csv")
    
    #era5S = pd.concat([era5S, era5Sn]).sort_index()
    #era5S.to_csv(r"D:\Master\era5\era5S.csv")
    
     # import seeing data
    df = datadf(r'D:/Master/fig_sum/Seeing_Data.txt')
    df.set_index("UTC", inplace = True)
    df = df.resample('3h').mean().dropna()
    
    altitude = var1d('Z20210621_03', lat_lon, 'Z')
    
  #%%
    # resample for every 3 hours, by standard deviation between vertical levels (beginning with "first_level")
    # take only levels >= 110 (up to ~1600m) 
    #130 ~ 205m 

    #method = "delta"
    method = "std"
    first_level = 132
    era5Pre = era5P[era5P["levels"] >= first_level].resample('3h').std().dropna().drop("levels",1)
    era5Sre = era5S[era5S["levels"] >= first_level].resample('3h').std().dropna().drop("levels",1)
    
    # merged table with all variables: 
    era5 = era5Pre.merge(era5Sre, on="UTC")
    era5["LST"] = pd.to_datetime(era5.index + pd.Timedelta(hours=3)) # in summer !! in winter delta = 2
    
    # merged table of era5 and seeing data:
    merged = df.merge(era5, on="UTC")
    
    era5.to_csv('era5_'+method+'.csv')
    merged.to_csv('era5_df_merged_'+method+".csv")
    
    f = sns.pairplot(merged[["seeing", "wind_U(m/s)", "wind_V(m/s)", "humidity(kg/kg)", "temp(K)", "pressure", "RH"]])  
    title = "levels " + str(first_level) + "-137." + method 
    f.fig.suptitle(title, y=1.02, fontsize = 14)
    #f.savefig("D:\Master\era5\pairplot_10-30m.pdf")
    
    #r2RF = [] 
    r2lm = []
    r2dtree =[] 
#%%  linear model (with scikit learn) 
    #merged = merged[merged["seeing"] <= 2.5]
    
    #       'wind_V(m/s)','pressure', 'RH'
#for i in np.arange(1000):
    X = merged[['temp(K)', 'humidity(kg/kg)','wind_U(m/s)','wind_V(m/s)','pressure', 'RH']]
    y = merged[['seeing']]
    test_per = 0.4     # percentage of data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_per, shuffle = True)
    
    lm = LinearRegression(positive=True).fit(X_train,y_train) # perform linear regression  
    
    #pickle.dump(model, open('lm.sav', 'wb')) #save a good model:
    #lm = pickle.load(open('lm.sav', 'rb')) # load a model
    
    r2_train = lm.score(X_train,y_train)  # the R^2 between the prediction on Xtrain (which will give a vector) and Ytrain 
    r2_test = lm.score(X_test,y_test)
    r2lm.append(r2_test)
    
    lm.coef_  # list of coefficients for the slope
    lm.intercept_ # intercept point on y axis 
    pred_test = lm.predict(X_test)  # make predictions
    pred_train = lm.predict(X_train)
    
    y_pred= pd.DataFrame(pred_test)
    y_pred.set_index(y_test.index, inplace=True)
    
    res_train = pd.DataFrame(np.transpose([y_train["seeing"], np.squeeze(pred_train)]), columns = ['y_train', 'prediction'])
    res_test = pd.DataFrame(np.transpose([y_test["seeing"],  np.squeeze(pred_test)]), columns = ['y_test', 'prediction'])
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(res_test["y_test"], res_test["prediction"])
    coeff = np.round(lm.coef_.transpose()) 
    coef = '\n'.join([str(i) for i in coeff.flatten()])
         
    ## plot with the measurements vs prediction 
    sns.set_style("darkgrid")
    os.chdir(r"D:\Master\era5")
    name = 'lm.'+title+'.'+str(test_per)+'_testing.'
    fig, axs = plt.subplots(2,3, figsize = [12,10])
    sns.regplot(data = res_train, x="y_train", y="prediction", ax=axs[0,0]) 
    sns.residplot(data = res_test, x = "y_test", y= "prediction", ax = axs[0,1])
    axs[0,1].set_ylim([-1.5,1.5])
    sns.boxplot(data= pd.melt(res_test), x="variable", y="value", ax = axs[1,0])
    res_test.plot(marker='o', ls= ':', ax = axs[1,1])
    sns.scatterplot(res_test["y_test"],res_test["prediction"], ax = axs[1,2])
    axs[1,2].plot(pd.Series(res_test["y_test"]),res_test["y_test"], color= 'blue')
    axs[1,2].legend(["1:1", "prediction"], loc='upper right')
    
    textbox = '\n'.join((
    r'$R^2$ training= %.2f' %r2_train,
    r'$R^2$ test =%.2f' % r2_test,
    r'std error=%.2f' % (slope, ),
    r'equation=%.2fx+%.2f' % (slope,intercept),
    r'levels:'+ str(first_level) + '-137 ',
    r'test sample size = %i %%' % (test_per*100,),
    r'Coefficients:', coef))
    axs[0,2].text(0.05,0.95,textbox, transform=axs[0,2].transAxes, fontsize=14,\
        verticalalignment='top', wrap = True)
    fig.suptitle('Linear model. variables: '+', '.join([str(i) for i in X_train.columns]), y=0.92, fontsize = 14)
    #plt.savefig(name)

    
    #%%
    # check individual variables with seeing
    norm = pd.DataFrame(merged.drop("LST",1)/merged.drop("LST",1).max())
    
    #var = "temp(K)"
    var = "wind_U(m/s)" 
    #var = "wind_V(m/s)" 
    #var = "humidity(kg/kg)"
    #var = "wind_U(m/s)"
    #var = "pressure"
    #var = 'RH'
    res = scipy.stats.linregress(norm[var], norm["seeing"])
    print("\nR^2 of %s with seeing = %.2f" %(var, res.rvalue))
    
    plt.plot(norm["seeing"],norm["seeing"])
    plt.scatter(norm["seeing"],norm[var])
    plt.xlabel(var)
    plt.ylabel("seeing")

#%% # decision tree (scikit learn)
#for i in np.arange(1000):   
    X = merged[['temp(K)', 'humidity(kg/kg)', 'wind_U(m/s)','wind_V(m/s)', 'pressure', 'RH']]
    y = merged[['seeing']]
    test_per = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_per)
    
    # Fit the regression tree  # random_state=0
    #criterion = how to check the split quality. random_state= features randomly permuted at each split
    dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf = 5)
    dtree.fit(X_train, y_train)
    
    # Predict on training data
    tr2 = dtree.predict(X_train) 
    # Predict on testing data
    y2 = dtree.predict(X_test) 
    
    r2_train = dtree.score(X_train, y_train) 
    r2_test = dtree.score(X_test, y_test)
    r2dtree.append(r2_test)
    
    res_train = pd.DataFrame(np.transpose([y_train["seeing"], tr2]), columns = ['y_train', 'prediction'])
    res_test = pd.DataFrame(np.transpose([y_test["seeing"], y2]), columns = ['y_test', 'prediction'])

    ## plot with the measurements vs prediction 
    sns.set_style("darkgrid")
    os.chdir(r"D:\Master\era5")
    name = 'lm.'+title+'.'+str(test_per)+'_testing.'
    fig, axs = plt.subplots(2,3, figsize = [12,10])
    sns.regplot(data = res_train, x="y_train", y="prediction", ax=axs[0,0]) 
    sns.residplot(data = res_test, x = "y_test", y= "prediction", ax = axs[0,1])
    axs[0,1].set_ylim([-1.5,1.5])
    sns.boxplot(data= pd.melt(res_test), x="variable", y="value", ax = axs[1,0])
    res_test.plot(marker='o', ls= ':', ax = axs[1,1])
    sns.scatterplot(res_test["y_test"],res_test["prediction"], ax = axs[1,2])
    axs[1,2].plot(pd.Series(res_test["y_test"]),res_test["y_test"], color= 'lightseagreen')
    axs[1,2].legend(["1:1", "prediction"], loc='upper right')
    
    textbox = '\n'.join((
    r'$R^2$ training= %.2f' % r2_train,
    r'$R^2$ test = %.2f' % r2_test,
    r'RMSE training = %.2f' % (np.sqrt(mean_squared_error(y_train,tr2)),),
    r'RMSE test = %.2f' % (np.sqrt(mean_squared_error(y_test,y2)),),
    r'test sample size = %i %%' % (test_per*100,)))
    axs[0,2].text(0.05,0.95,textbox, transform=axs[0,2].transAxes, fontsize=14,\
        verticalalignment='top', wrap = True)
    fig.suptitle('Decision tree. variables: '+', '.join([str(i) for i in X_train.columns]), y=0.92, fontsize = 14)
    #plt.savefig(name)
    
    from sklearn import tree
    text_representation = tree.export_text(dtree)
    print(text_representation)
    
    fig = plt.figure(figsize=(15,10))
    n = tree.plot_tree(dtree,feature_names=X_train.columns,\
                   class_names="seeing",\
                   filled=True,fontsize=9)
    plt.savefig("Decision tree.pdf")


    #%%
    ########## random forest #############
#for i in np.arange(1000):     
    X = merged[['temp(K)', 'humidity(kg/kg)', 'wind_U(m/s)','wind_V(m/s)', 'pressure', 'RH']]
    y = merged[['seeing']]
    
    test_per = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_per, shuffle = True)
    
    model_rf = RandomForestRegressor(n_estimators=1000, oob_score=True, max_depth=8, min_samples_leaf=3)
    model_rf.fit(X_train, y_train) 
    pred_train_rf= model_rf.predict(X_train)    
    pred_test_rf = model_rf.predict(X_test)
    
    res_train = pd.DataFrame(np.transpose([y_train["seeing"], pred_train_rf]), columns = ['y_train', 'prediction'])
    res_test = pd.DataFrame(np.transpose([y_test["seeing"], pred_test_rf]), columns = ['y_test', 'prediction'])
    
    r2train = model_rf.score(X_train, y_train)
    r2test = model_rf.score(X_test, y_test)
    
    #r2RF.append(r2test)
    #pd.DataFrame(r2RF).to_csv('R2RF.csv')
 
    ## plot with the measurements vs prediction 
    sns.set_style("darkgrid")
    os.chdir(r"D:\Master\era5")
    name = 'lm.'+title+'.'+str(test_per)+'_testing.'
    fig, axs = plt.subplots(2,3, figsize = [12,10])
    sns.regplot(data = res_train, x="y_train", y="prediction", ax=axs[0,0]) 
    sns.residplot(data = res_test, x = "y_test", y= "prediction", ax = axs[0,1])
    axs[0,1].set_ylim([-1.5,1.5])
    sns.boxplot(data= pd.melt(res_test), x="variable", y="value", ax = axs[1,0])
    res_test.plot(marker='o', ls= ':', ax = axs[1,1])
    sns.scatterplot(res_test["y_test"],res_test["prediction"], ax = axs[1,2])
    axs[1,2].plot(pd.Series(res_test["y_test"]),res_test["y_test"], color= 'lightseagreen')
    axs[1,2].legend(["1:1", "prediction"], loc='upper right')
    
    textbox = '\n'.join((
    r'$R^2$ training= %.2f' %r2train,
    r'$R^2$ test = %.2f' % r2test,
    r'RMSE training = %.2f' % (np.sqrt(mean_squared_error(y_train,pred_train_rf)),),
    r'RMSE test = %.2f' % (np.sqrt(mean_squared_error(y_test,pred_test_rf)),),
    r'test sample size = %i %%' % (test_per*100,)))
    axs[0,2].text(0.05,0.95,textbox, transform=axs[0,2].transAxes, fontsize=14,\
        verticalalignment='top', wrap = True)
    fig.suptitle('Random Forest. variables: '+', '.join([str(i) for i in X_train.columns]), y=0.92, fontsize = 14)
    #plt.savefig(name)
    

    # plot the different features importance: 
    importances = model_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_rf.estimators_], axis=0)
    feature_names = X_train.columns
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots(figsize = [7,5])
    forest_importances.plot.bar(yerr=std, ax=ax, fontsize = 14)
    ax.set_title("Feature importances using MDI",fontsize = 14)
    ax.set_ylabel("Mean decrease in impurity",fontsize = 14)
    fig.tight_layout()

    #%% plot R2 results of different models:
    r2RF = pd.read_csv('D:/Master/era5/R2RF.csv').drop('Unnamed: 0',1)
    data = {'LM': r2lm, 'DTree': r2dtree, 'RF': r2RF["0"]}
    r2all = pd.DataFrame(data)
    
    #sns.boxplot(pd.melt(r2all), x="variable", y="value")
    r2all.boxplot(fontsize = 14, figsize = [12,7])
    
#%%   ###### Technical #######

#  import netCDF4 files, and learn about them: 
    fname = 'Z20210526_21'
    os.chdir(r'D:/Master/era5/netfiles')
    file = Dataset(fname,"r", format='nc')

    # extract specific variable data
    # dimensions 
    for dimobj in file.dimensions.values():
        print(dimobj)
    # variables - list of variables: 
    file.variables.keys()
    # variables - with description:
    file.variables
    # specific variavble description: 
    file["SLP"]
    
    # delete excess netCDF4 files (according to seeing measurements)
    # create a list with relevant dates (dates where there are seeing measurements): 
    days = df.resample('1D').mean().dropna().index
    days = [str(x.strftime("%Y%m%d")) for x in days]
    
    path = 'E:/Shani_era5'
    os.chdir(path)
    folder = os.listdir(path)
    for netfile in folder:
        if days.count(netfile[1:9]) == 0:
            #os.remove(netfile)
            print(netfile, "was deleted")
            
     # visualize data at a specific time
    path = r'E:/Shani_era5'
    os.chdir(path)
    
    lat_lon = [30,35]
    fname = 'P20210301_12'
    T = var1d(fname, lat_lon, 'T')
    Plevels = LevelToP1D(fname, lat_lon)
    fig = plt.figure()
    plt.plot(T,Plevels)
    plt.xlabel("Temp (K)")
    plt.ylabel("Model levels")
    plt.title(fname[0:4]+'/'+fname[4:6]+'/'+ fname[6:8]+ ' '+ fname[9:]+ ':00')       
    
    #%% import a netCDF4 file 
    dat = Dataset('C:/Users/shani/Downloads/adaptor.mars.internal-1634128109.386773-14266-16-80129c29-242b-4a63-9d9e-a87563f19957.nc', 'r')
    # time variable
    time = dat.variables['time']
    datetime = netCDF4.num2date(time[:], time.units, time.calendar,only_use_cftime_datetimes=False)
    datetime = pd.to_datetime(datetime)
    
    # pressure levels
    levels = dat["level"][:]
    
    # latitude longitude
    lat = dat["latitude"][:]
    lon = dat["longitude"][:]
    
    # define the index of the wanted latitude and longitude (30,35)
    lat_i = np.where(lat == 30)[0][0]
    lon_i = np.where(lon == 35)[0][0] 
    
    # to extract variable data from netCDF4 file: netCDFfile['var'][time,levels,lat,lon]
    
    t = pd.DataFrame(dat['t'][:,:,lat_i,lon_i], columns = levels, index = datetime)
    cc = pd.DataFrame(dat['cc'][:,:,lat_i,lon_i], columns = levels, index = datetime)
    RH =  pd.DataFrame(dat['r'][:,:,lat_i,lon_i], columns = levels, index = datetime)
    clwc = pd.DataFrame(dat['clwc'][:,:,lat_i,lon_i], columns = levels, index = datetime)
    u = pd.DataFrame(dat['u'][:,:,lat_i,lon_i], columns = levels, index = datetime)
    v =  pd.DataFrame(dat['v'][:,:,lat_i,lon_i], columns = levels, index = datetime)
    w = pd.DataFrame(dat['w'][:,:,lat_i,lon_i], columns = levels, index = datetime)
    
    together = np.transpose([t.std(axis=1),cc.sum(axis=1),RH.std(axis=1),clwc.sum(axis=1),u.std(axis=1),v.std(axis=1),w.std(axis=1)])
    
    era5 =  pd.DataFrame (together, columns = ["temp_std", "cc_sum", "RH_std", "clwc_sum", "u_std", "v_std", "w_std"])\
        .set_index(datetime)
    era5.index.rename("UTC", inplace = True)
    
    merged = df.merge(era5, on = "UTC").drop("r0", 1)
    
    
    #%%
    #sns.pairplot(merged)
    sns.pairplot(merged[["seeing", "temp_std","u_std", "v_std"]])  
    
    # check individual R2 with a specific parameter
    par = "temp_std" 
    res = scipy.stats.linregress(merged[par], merged["seeing"])
    print("\nR^2 of %s with seeing = %.2f" %(par, res.rvalue))
    
    learning = 40 # percentage of data for training 
    border = int((learning/100)*len(merged))
    Xtrain = merged[["u_std", "v_std"]].iloc[:border,:] 
    Ytrain = merged[['seeing']].iloc[:border,:]
    Xtest = merged[["u_std","v_std"]].iloc[border:,:] 
    Ytest =  merged[['seeing']].iloc[border:,:]
    lm = LinearRegression().fit(Xtrain,Ytrain) # perform linear regression  
    r2 = lm.score(Xtrain,Ytrain)  # the R^2 between the prediction on Xtrain (which will give a vector) and Ytrain 
    lm.coef_  # list of coefficients for the slope
    lm.intercept_ # intercept point on y axis 
    Ypred = lm.predict(Xtest)  # make predictions
    
    Ypred= pd.DataFrame(Ypred)
    Ypred.set_index(Ytest.index, inplace=True)
    results = pd.DataFrame({'real': Ytest["seeing"], 'predicted': Ypred[0]}).reset_index(drop=True)
    
    ## plot with the measurements vs prediction 
    os.chdir(r"D:\Master\era5")
    fig, axs = plt.subplots(2,3, figsize = [12,10])
    sns.regplot(data = results, x="real", y="predicted", ax=axs[0,0]) 
    sns.residplot(data = results, x = "real", y= "predicted", ax = axs[0,1])
    axs[0,1].set_ylim([-1.5,1.5])
    sns.boxplot(data= pd.melt(results), x="variable", y="value", ax = axs[1,0])
    results.plot(marker='o', ls= ':', ax = axs[1,1])
    axs[1,2].scatter(results["real"],results["predicted"])
    axs[1,2].plot(results["real"],results["real"])
    axs[1,2].set_ylim([0,3])
    axs[1,2].set_xlabel("real")
    axs[1,2].set_ylabel("predicted")
    #plt.savefig(name)
    
    #sns.jointplot(data = results, x= "real", y= "predicted", kind="reg")
    # The coefficients
    print('\nLinear model: \nCoefficients: ', lm.coef_ , '\nR^2:',r2)
    # equation, error, and R squared: 
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(results["real"], results["predicted"])
    print("\nRegression between predicted and observed: \nEquation: %.2fx+%.2f \nR^2: %.3f \
          \nstd error %.2f \np-value (Ho = slope is zero): %.15f" %(slope,intercept,r_value**2,std_err, p_value))
   
    #%% check correlation of specific var
    var = "temp_std"
    res = scipy.stats.linregress(merged[var], merged["seeing"])
    print("\nR^2 of %s with seeing = %.2f" %(par, res.rvalue))
    
    plt.scatter(merged[var], merged["seeing"])
    plt.plot(merged["seeing"],merged["seeing"])
