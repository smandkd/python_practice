# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 20:15:58 2022

@author: tkdal
"""

import pandas as pd
data2 = pd.read_csv("Dataset_02.csv")
#@@

#(1)
#%%
q1 = pd.crosstab(index=[data2.Sex, data2.BP],  columns=data2.Cholesterol,  normalize=True)


#%%
import numpy as np

q2 = data2.copy()

q2['Age_gr']=np.where(q2.Age < 20, 10, 
                    np.where(q2.Age < 30, 20,
                             np.where(q2.Age < 40, 30,
                                      np.where(q2.Age < 50, 40, 
                                               np.where(q2.Age < 60, 50, 60)))))

q2['Na_k_gr']=np.where(q2.Na_to_K<=10, 'Lv1',
                       np.where(q2.Na_to_K<=20, 'Lv2',
                             np.where(q2.Na_to_K<=30, 'Lv3', 'Lv4')))

# 카이스퀘어 검정 요청

from scipy.stats import chi2_contingency

tab=pd.crosstab(index=q2['Sex'], columns=q2['Drug'])
chi2_contingency(tab)

var_list = ['Sex', 'BP', 'Cholesterol', 'Age_gr', 'Na_k_gr']

q2_out = []

for i in var_list :
    tab=pd.crosstab(index=q2[i], columns=q2['Drug'])
    pvalue=chi2_contingency(tab)[1]
    q2_out.append([i, pvalue])

q2_out = pd.DataFrame(q2_out, columns=['var', 'pvalue'])
q2_out['pvalue'] < 0.05
q2_out[q2_out['pvalue'] < 0.05]['pvalue'].max()







#%%

q3 = data2.copy()
q3.columns

q3['Sex_cd'] = np.where(q3.Sex == 'M', 0, 1)
q3['BP_cd'] = np.where(q3.BP == 'LOW', 0, 
                       np.where(q3.BP =="NORMAL", 1,2 ))
q3['Ch_cd'] = np.where(q3.Cholesterol=='NORMAL', 0, 1)

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

x_var=['Age', 'Na_to_K', 'Sex_cd','BP_cd','Ch_cd']

dt=DecisionTreeClassifier().fit(q3[x_var], q3.Drug)

export_text(dt)

plot_tree(dt, max_depth=2, feature_names=x_var, class_names=q3.Drug.unique(), precision=3)




#%%

import pandas as pd
df = pd.read_csv("Dataset_02.csv")

df_condi1 = (df['BP'] == 'HIGH')
df_condi1.sum()
df_condi2 =( df['Cholesterol'] == 'NORMAL')
df_condi3 = (df['Sex'] == 'F')

(df_condi1 & df_condi2 & df_condi3).sum()

df_1 = df.copy()

import numpy as np

df_1['Age_gr'] = np.where(df_1.Age < 20 , 10, 
                          np.where(df_1.Age < 30 , 20, 
                                   np.where(df_1.Age < 40 , 30, 
                                            np.where(df_1.Age < 50 , 40, 
                                                     np.where(df_1.Age < 60 , 50, 60))))) 


df_1['Na_K_gr'] = np.where(df_1.Na_to_K <10, 'Lv1',
                           np.where(df_1.Na_to_K <20, 'Lv2',
                                    np.where(df_1.Na_to_K <30, 'Lv3', 'Lv4')))

from scipy.stats import chi2_contingency

tab = pd.crosstab(index = df_1['Age_gr'], columns = df_1['Drug'])

chi2,p,dof,ex = chi2_contingency(tab)
p

df_1.head(2)

df_1['Sex_cd'] = np.where(df_1.Sex == 'M', 0 ,1)
df_1['BP_cd'] = np.where(df_1.BP == 'LOW', 0,
                         np.where(df_1.BP == 'NORMAL', 1, 2))
df_1['Ch_cd'] = np.where(df_1.Cholesterol =='NORMAL', 0, 1)

from sklearn.tree import DecisionTreeClassifier, export_text,plot_tree


var = ['Age', 'Na_to_K', 'Sex_cd', 'BP_cd', 'Ch_cd']


pd = DecisionTreeClassifier().fit(df_1[var], df_1.Drug)
export_text(pd)
plot_tree(pd)


#%%

import pandas as pd
import numpy as np

df2 = pd.read_csv('Dataset_02.csv')

data2 = df2.copy()

data2['Age_gr'] = np.where(data2.Age < 20, 10, 
                           np.where(data2.Age < 30, 20, 
                                    np.where(data2.Age < 40, 30, 
                                             np.where(data2.Age < 50, 40, 
                                                      np.where(data2.Age < 60, 50, 60)))))

data2['Na_K_gr'] = np.where(data2.Na_to_K <= 10, 'Lv1', 
                            np.where(data2.Na_to_K <= 20, 'Lv2', 
                                     np.where(data2.Na_to_K <= 30, 'Lv3', 'Lv4')))


from scipy.stats import chi2_contingency

x_var = ['Sex', 'BP', 'Age_gr', 'Na_to_K']

tab = pd.crosstab(index = data2['Sex'], columns=data2['Drug'])

chi2_contingency(tab)


from sklearn.tree import DecisionTreeClassifier, export_text

data_2 = df2.copy()

data_2['Sex_cd'] = np.where(data_2.Sex == 'M', 0,1)
data_2['BP_cd'] = np.where(data_2.BP == 'LOW', 0, 
                           np.where(data_2.BP == 'NORMAL', 1, 2))
data_2['Ch_cd'] = np.where(data_2.Cholesterol =='NORMAL', 0, 1)

var = ['Age', 'Na_to_K', 'Sex_cd', 'Ch_cd', 'BP_cd']

model = DecisionTreeClassifier().fit(data_2[var], data_2.Drug)

export_text(model)































