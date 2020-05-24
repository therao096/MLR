# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:45:25 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
toycar =pd.read_csv("ToyotaCorolla.csv",encoding= 'unicode_escape')
############toycar.rename(columns={'Age_08_04':'Age',"Quarterly_Tax":"QuarterlyTax"})
toycar
corolla= toycar.loc[ :,["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
type(corolla)
corolla
corolla.corr()
import seaborn as sns
sns.pairplot(corolla.iloc[:,:])
corolla.columns
import statsmodels.formula.api as smf
ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=corolla).fit()
ml1.params
ml1.summary()
####preparing model based on age
m_a= smf.ols('Price~Age_08_04', data=corolla).fit()
m_a.summary() ##0.768 r squared
m_cc=smf.ols('Price~cc',data=corolla).fit()
m_cc.summary() ##0.015
m_dor=smf.ols('Price~Doors', data=corolla).fit()
m_dor.summary()#0.034
m_com=smf.ols('Price~Doors+cc', data=corolla).fit()
m_com.summary() #0.04
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
corolla_new=corolla.drop(corolla.index[[80,601,221,960]], axis=0)
corolla_new
ml_new=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=corolla_new).fit()
ml_new.params
ml_new.summary()
prid= ml_new.predict(corolla_new)
prid
###calculate VIF values of independent variables
rsq_age=smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight  ', data=corolla_new).fit().rsquared
rsq_age
vif_age=1/(1-rsq_age)
vif_age ####2.02
rsq_km= smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data= corolla_new).fit().rsquared
rsq_km
vif_km=1/(1-rsq_km)
vif_km #####1.922
rsq_hp=smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight', data= corolla_new).fit().rsquared
vif_hp=1/(1-rsq_hp)
vif_hp ####1.61
rsq_cc= smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight', data=corolla_new).fit().rsquared
vif_cc=1/(1-rsq_cc)
vif_cc###3.0187
rsq_door= smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data=corolla_new).fit().rsquared
vif_door=1/(1-rsq_door)
vif_door ####1.21
rsq_gears= smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight', data=corolla_new).fit().rsquared
vif_gears=1/(1-rsq_gears)
vif_gears ##1.101
rsq_quart=smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight', data=corolla_new).fit().rsquared
vif_quart=1/(1-rsq_quart)
vif_quart ###3.067
rsq_wt=smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax', data=corolla_new).fit().rsquared
vif_wt=1/(1-rsq_wt)
vif_wt###3.911
## as weight vif value is 3.911, let us drop weight
final_ml=smf.ols('Price~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight', data=corolla_new).fit()
final_ml.params
final_ml.summary()##0.87 r square value
print(final_ml.conf_int(0.05))
final_pred=final_ml.predict(corolla_new)
final_pred
pred_frame=pd.DataFrame(final_pred)
pred_frame
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(final_ml)
