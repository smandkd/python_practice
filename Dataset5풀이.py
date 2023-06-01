# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:33:23 2022

@author: tkdal
"""

#%%

import pandas as pd

pos = pd.read_csv('Dataset_05_Mart_POS.csv')
list1 = pd.read_csv('Dataset_05_item_list.csv')

q1=pos['Date'].value_counts().idxmax()

q1_ans=pos[pos.Date == q1]['itemDescription'].value_counts().max()
q1_ans


#%%

q2=pos.copy()

pd.to_datetime(pos['Date']).dt.year
pd.to_datetime(pos['Date']).dt.month
pd.to_datetime(pos['Date']).dt.day
pd.to_datetime(pos['Date']).dt.day_name(local='ko_kr')

q2=pos.copy()
q2['day'] = pd.to_datetime(q2['Date']).dt.day_name(locale='ko_kr')
q2['month']=pd.to_datetime(q2['Date']).dt.month

q2_merge=pd.merge(q2, list1,
                  left_on='itemDescription', 
                  right_on='prod_nm',
                  how='left')

q2_merge['week']=0
q2_merge.loc[q2_merge.day.isin(['금요일','토요일']), 'week']=1

from scipy.stats import ttest_ind

q2_merge2 = q2_merge[q2_merge.month.isin([1, 2, 3])]

q2_tab = pd.pivot_table(q2_merge2, index='Date', 
                        columns='week',
                        values='alcohol',
                        aggfunc='sum')

q2_out=ttest_ind(q2_tab[0].dropna(), q2_tab[1].dropna(), equal_var=False)
q2_out





#%%



top10_list=pos['itemDescription'].value_counts().head(10).index

q3=pos[pos['itemDescription'].isin(top10_list)]

q3_tab=pd.pivot_table(data=q3, index='Date',
                      values='itemDescription',
                      aggfunc='count').reset_index()

q3_tab['day']=pd.to_datetime(q3_tab['Date']).dt.day_name(locale='ko_kr')

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

ols1 = ols('itemDescription~day', data=q3_tab).fit()

q3_out=anova_lm(ols1)
q3_out['PR(>F)']['day']

































#%%