# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:51:25 2022

@author: tkdal
"""

#@@
import pandas as pd

data1 = pd.read_csv('Dataset_01.csv')
#@@
#%%
data1.isna().sum().sum()

data1.isna().any(axis=1).sum()

#----------------------------------------------------------------

x_var = ['TV', 'Radio', 'Social_Media', 'Sales'] 

q2=data1[x_var].corr()

q2.drop('Sales')['Sales'].abs().idxmax()


#-------------------------------------------------------------------

q3 = data1.dropna()

from sklearn.linear_model import LinearRegression

x_var = ['TV', 'Radio', 'Social_Media']

from statsmodels.formula import ols
from statsmodels.api import OLS, add_constant

lm = LinearRegression().fit(q3[x_var], q3.Sales)

dir(lm)
lm.coef_.max()




#%%


import pandas as pd


df = pd.read_csv("Dataset_01.csv")

corr = df.corr()

dir(corr)
corr.coef_


from sklearn.linear_model import LinearRegression

df1 = df.copy()

df1 = df1.dropna()

var = ['TV', 'Radio', 'Social_Media']

lm = LinearRegression().fit(df1[var], df1.Sales)

lm.coef_











#%%




































