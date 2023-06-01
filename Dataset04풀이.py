# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 13:12:00 2022

@author: tkdal
"""

#%%

import pandas as pd

df = pd.read_csv('Dataset_04.csv')

data4 = df.copy()

q1 = data4[data4.LOCATION == 'KOR']

q1_tab = pd.pivot_table(q1, index='TIME', values='Value',
                        aggfunc = 'sum').reset_index()

q1_tab.corr().loc['TIME', 'Value']


#%%

q2 = data4[data4.LOCATION.isin(['KOR', 'JPN'])]
q2.SUBJECT.unique()
q2_tab = pd.pivot_table(q2, index=['TIME', 'SUBJECT'],
                        columns='LOCATION', values='Value')

q2_tab = q2_tab.dropna()

from scipy.stats import ttest_rel

q2_out = ttest_rel(q2_tab['KOR'], q2_tab['JPN'])
q2_out.statistic





#%%


q3 = data4[data4.LOCATION=='KOR']

from sklearn.linear_model import LinearRegression

sub_list = q3.SUBJECT.unique()

q3_out=[]

for i in sub_list:
    temp=q3[q3.SUBJECT==i]
    lm = LinearRegression().fit(temp[['TIME']], temp['Value'])
    pred=lm.predict(temp[['TIME']])
    r2 = lm.score(temp[['TIME']], temp['Value'])
    mape=(abs(temp['Value']-pred) / temp['Value']).sum()*100/len(temp)
    q3_out.append([i, r2, mape])
    
q3_out = pd.DataFrame(q3_out, columns=['sub', 'r2', 'mape'])
q3_out.sort_values('r2', ascending=False).head(1)
    





















