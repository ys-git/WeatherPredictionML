# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:46:05 2018

@author: YS
"""


import pandas as pd  
import matplotlib  
import matplotlib.pyplot as plt  
import numpy as np  
df = pd.read_csv(r'C:\Users\YS\Desktop\fyp\Data_2.csv').set_index('date')



df.corr()[['meantempm']].sort_values('meantempm') 

predictors = ['meantempm_1',  'meantempm_2',  'meantempm_3',  
              'mintempm_1',   'mintempm_2',   'mintempm_3',
              'meandewptm_1', 'meandewptm_2', 'meandewptm_3',
              'maxdewptm_1',  'maxdewptm_2',  'maxdewptm_3',
              'mindewptm_1',  'mindewptm_2',  'mindewptm_3',
              'maxtempm_1',   'maxtempm_2',   'maxtempm_3']
df2 = df[['meantempm'] + predictors]  




plt.rcParams['figure.figsize'] = [16, 22]


fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True)


arr = np.array(predictors).reshape(6, 3)


for row, col_arr in enumerate(arr):  
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['meantempm'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='meantempm')
        else:
            axes[row, col].set(xlabel=feature)
plt.show()  

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)