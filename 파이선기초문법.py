# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 16:15:03 2022

@author: tkdal
"""

#%%
for n in range(3):
    print(n)

'abc'
"abc"

text='hello'

print(text[0])

text[0:4]

text[:-1]
#%%

aa=[1,2,3]

print([1,2,[1,2,3]])

type(aa)

cc =[1,2, ['a','b']]

print(cc[2])

import numpy as np

[3] + [8]

score=['A', 'B', 'C', 'D']

score[:-1]

range(0,3)
for n in range(0,3):
    print(n)


import numpy as np

np.array()

import pandas as pd
ser= pd.Series([2, 3, 4, 5, 4, 7])
ser.max()
ser.idxmax()
ser.idxmin()
ser.isin([6,7]).sum()

df=pd.read_csv("Dataset_01.csv")

df_na = df.head(7)

df_na.head(7)
df_na.isna()


df_na.notna().sum(axis=0)
df_na['TV'].mean()
df_na
df_na.iloc[:,:-1]


df_na['TV'].max()
df_na['TV'].min()
df_na['TV'].quantile(q=0.25)
import numpy as np
x = [1,2,4,56, 3,4,54,8,9,5,3,454,22,322,3]
quantile(x, q=0.5)

array=np.array([1,2,4,56, 3,4,54,8,9,5,3,454,22,322,3])
np.median(array)


pd.crosstab(df['Influencer'],df['Sales'])

df.rename(columns={'Influencer':'IF'})

df.iloc[:, 2:4]
df.loc[:, ['Social_Media', 'Influencer']]

df["TV"].apply(func=sum)
df.info()
df['TV'].astype('int')

df2 = pd.read_csv('bike.csv')
df2.head(2)


df2['datetime'] = pd.to_datetime(df2['datetime'])

df2['month'] = df2['datetime'].dt.month
df2.head(2)

bike_agg = df2.groupby('month')['casual'].min().reset_index()

type(bike_agg)


bike = pd.read_csv("bike.csv")

bike.head(2)


bike_sub = bike.copy()

bike_sub = bike_sub.reset_index(drop=True)
bike_sub = bike_sub.set_index('datetime')

bike_1 = bike.iloc[:3, :4]
bike_2 = bike.iloc[5:8, :4]

pd.concat([bike_1, bike_2.reset_index(drop=True)], axis=1 )


df_A = pd.read_csv("join_list.csv")

import pandas as pd
dia = pd.read_csv('diamonds.csv')

pd.crosstab(dia['cut'], dia['clarity'], normalize=0).round(2)


pd.crosstab(dia['cut'], dia['clarity'], normalize=1)

dia.groupby(['cut','clarity'])['price'].mean().reset_index()




























