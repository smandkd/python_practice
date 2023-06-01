# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 22:24:43 2022

@author: tkdal
"""

#%%

import pandas as pd

data3 = pd.read_csv("Dataset_03.csv")

q1 = data3.copy()

q1['forehead_ratio'] = q1['forehead_width_cm']/ q1['forehead_height_cm']

xbar = q1['forehead_ratio'].mean()

std= q1['forehead_ratio'].std()

LL = xbar - (3*std)
UU = xbar + (3*std)


len(q1[(q1['forehead_ratio']<LL) | (q1['forehead_ratio']>UU)])
(((q1['forehead_ratio']<LL) | (q1['forehead_ratio']>UU))).sum()

#%%

from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, bartlett
#bartlett 등분산 검정용 

# X : 범주형, y : 수치형 
# X : 범주형 그룹 수 2 개만 사용 ( 2개 이하 ) -> ttest
# X : 범주형 그룹 수 3 개 이상 -> ANOVA 
q1.gender.unique() # ['Male', 'Female']
g_m = q1[q1.gender=='Male']['forehead_ratio']
g_f = q1[q1.gender=='Female']['forehead_ratio']

bartlett(g_m, g_f)

q2_out = ttest_ind(g_m, g_f, equal_var=False)# 등분산이면 True, 이분산이면 False
print(q2_out)

round(q2_out.statistic, 4)





#%%

from sklearn.model_selection import train_test_split

train, test = \
train_test_split(data3, test_size=0.3,
                 random_state=123)


len(data3)
len(train)


train.columns
x_var=train.columns.drop('gender')

from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(C=100000, solver='newton-cg', 
                           random_state=123)

logit.fit(train[x_var], train.gender)

pred = logit.predict(test[x_var])
pred_pr = logit.predict_proba(test[x_var])

from sklearn.metrics import classification_report, precision_score

print(classification_report(test.gender, pred))
precision_score(test.gender, pred, pos_label='Male') 






#%%

df = pd.read_csv("Dataset_03.csv")

df['forehead_ratio'] = df['forehead_width_cm']/df['forehead_height_cm']

df_mean = df['forehead_ratio'].mean()
df_std = df['forehead_ratio'].std()

UU = df_mean + (3*df_std)
LL = df_mean - (3*df_std)

((UU < df['forehead_ratio'])|(LL > df['forehead_ratio'])).sum()


from scipy.stats import ttest_ind

df_M = df[df.gender=='Male']['forehead_ratio']
df_F = df[df.gender=='Female']['forehead_ratio']

p = ttest_ind(df_M, df_F, equal_var=False)
p







#%%

from sklearn.model_selection import train_test_split

train, test = \
    train_test_split(data3, random_state=123, test_size=0.3)

from sklearn.linear_model import LogisticRegression

train_var = train.columns.drop('gender')

LogisticRegression().fit(train[train_var], train.gender)

train.head(2)

pred = logit.predict(test[train_var])

pred_pr = logit.predict_proba(test.drop('gender', axis=1))

from sklearn.metrics import classification_report

classification_report(test.gender, pred,labels='Male')

#%%


import pandas as pd

df = pd.read_csv('Dataset_03.csv')

df3 = df.copy()

df3['forehead_ratio'] = df3['forehead_width_cm'] / df3['forehead_height_cm']

mean = df3['forehead_ratio'].mean()
std = df3['forehead_ratio'].std()

LL = mean - (3*std)
LL
UU = mean + (3*std)

((df3['forehead_ratio'] < LL) | (df3['forehead_ratio'] > UU)).sum()


from scipy.stats import ttest_ind

tab1 = df3[df3['gender']=='Female']['forehead_ratio']
tab2 = df3[df3.gender == 'Male']['forehead_ratio']

stat,p = ttest_ind(tab1, tab2, equal_var=False)

print(stat)

df_1 = df.copy()

from sklearn.model_selection import train_test_split

train, test =\
      train_test_split(df_1, test_size=0.3, random_state=123 )

from sklearn.linear_model import LogisticRegression

var = df_1.columns.drop('gender')

logit = LogisticRegression().fit(train[var], train.gender)

from sklearn.metrics import precision_score

pred = logit.predict(test[var])
pred_pr = logit.predict_proba(test[var])

from sklearn.metrics import classification_report, precision_score

print(classification_report(test.gender, pred))
precision_score(test.gender, pred, pos_label='Male') 


logit = LogisticRegression().fit(train[var], train.gender)

from sklearn.metrics import classification_report

print(classification_report(test.gender, pred))


#%%

import pandas as pd

df= pd.read_csv('Dataset_03.csv')

df1 = df.copy()

df1['forehead_ratio'] = df['forehead_width_cm']/df['forehead_height_cm']

from scipy.stats import ttest_ind

df1_M = df1[df1.gender=='Male']['forehead_ratio']

df1_F = df1[df1.gender=='Female']['forehead_ratio']

ttest_ind(df1_M, df1_F)





from sklearn.model_selection import train_test_split

train, test=\
    train_test_split(df1, random_state=123, test_size=0.3)

from sklearn.linear_model import LogisticRegression

df1_train = df1.columns.drop('gender')

logit = LogisticRegression().fit(df1[df1_train], df1.gender)

from sklearn.metrics import precision_score

pred = logit.predict(test[df1_train])
 
from sklearn.metrics import classification_report

print(classification_report(test['gender'], pred))























