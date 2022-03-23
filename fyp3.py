# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:46:05 2018

@author: YS
"""


from __future__ import division
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import fyp4 as fl

st = ['STATION']
targets = ['HOURLYDRYBULBTEMPC']
date = ['DATE']

s_features = ['HOURLYDewPointTempC','HOURLYRelativeHumidity','HOURLYWindSpeed','HOURLYWindDirection','HOURLYStationPressure']
test = ['cos_wind_dir']
features_add = test




fields = st + targets + date + s_features

df = pd.read_csv('wdata.csv', usecols=fields, parse_dates=date)

s2 = df[ df.STATION == 'WBAN:42181' ]

del df
df = s2[targets + s_features]
df = df.apply(pd.to_numeric, errors='coerce')
df = df.assign( DATE = s2[date])
df = fl.create_new_features(df)
for i in [1]:


    if   (i == 0):
        new_features = targets
    elif (i == 1):
        new_features = targets + s_features + features_add
    else:
        raise sys.SystemExit("Error Occured") 
    
    
    train_yr_start = 2007
    train_years = 9 
    test_years = 1
    
    test_yr_start = train_yr_start + train_years 
    
   
    days_later = 1    
    
    if(days_later == 1):
        
        (s_month, s_day) = (1, 1)
    else:
        raise SystemExit("Error occured")
        
        
    
    new_target = [str(days_later)+"days_later_temp_C"]
    
    
    df1 = df[date + new_features]
    df2 = df[date + new_features]
    
    df2.loc[:,"DATE"] = df1["DATE"].apply(lambda timeobj: timeobj + relativedelta(days=-days_later))
    df2.rename(columns={str(targets[0]):str(new_target[0])}, inplace=True)
    
    df1 = df1.set_index(["DATE"])
    df2 = df2.set_index(["DATE"])
    
    
    t1, t2 = df1.align(df2)
    
    t3 = t1
    t3.loc[:, new_target] = t2[new_target]
    
    range_start = datetime(train_yr_start, 1, 1, 0, 0, 0)
    range_end   = datetime(train_yr_start, 1, 1, 0, 0, 0) + relativedelta(years=train_years)
    
    df_time_train = t3[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
    df_time_train.loc[:, new_target] = df_time_train[new_target].interpolate(method='time')
    
    
    range_start = datetime(test_yr_start, s_month, s_day, 0, 0, 0)
    range_end   = datetime(test_yr_start, s_month, s_day, 0, 0, 0) + relativedelta(days=365)   
    
    df_time_test = t3[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
    
    (row_old, col_old) = df_time_test.shape    
    df_time_test = df_time_test[ df_time_test.notnull().all(axis=1) ]
    (row, col) = df_time_test.shape    
    
    print("________________________________________________________\n")
    
    model_re = fl.run_fit(df_time_train, df_time_test, new_target, new_features, poly_d_max=1, inter_only=False, plot=True)
    
    
    