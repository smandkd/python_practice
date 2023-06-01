# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:16:07 2022

@author: tkdal
"""

#%%

import pandas as pd

df = pd.read_csv("Dataset_04.csv")

data4 = df.copy()

q1 = data4[data4.LOCATION == 'KOR']

q1_tab = pd.pivot_table(q1, values='Value', index='TIME', aggfunc='sum').reset_index()
q1_tab.head(2)

q1_tab.corr()


q2=data4[data4.LOCATION.isin(['KOR','JPN'])]

q2.SUBJECT.unique()

q2_tab = pd.pivot_table(q2, values='Value', 
                        columns='LOCATION', index=['TIME', 'SUBJECT'])

q2_tab = q2_tab.dropna()
from scipy.stats import ttest_rel

ttest_rel(q2_tab['KOR'], q2_tab['JPN'])
 

q3 = data4[data4.LOCATION=='KOR']

from sklearn.linear_model import LinearRegression

sub_list = q3.SUBJECT.unique()

q3_out=[]

for i in sub_list:
    temp=q3[q3.SUBJECT==i]
    lm = LinearRegression().fit(temp[['TIME']], temp['Value'])
    pred=lm.predict(temp[['TIME']])
    r2 = lm.score(temp[['TIME']], temp['Value'])
    mape = (abs(temp['Value']-pred) / temp['Value']).sum()*100/len(temp)
    q3_out.append([i, r2, mape])
    
q3_out = pd.DataFrame(q3_out, columns=['sub', 'r2', 'mape'])
q3_out.sort_values('r2', ascending = False).head(1)



















#%%
