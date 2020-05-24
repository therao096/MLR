# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:08:36 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
startup= pd.read_csv("50_Startups.csv")
######33startup
type(startup)
#####3startup.head(10)
#######startup.tail(10)
########pd.get_dummies(startup.State)
#########startup=pd.get_dummies(startup,columns=['State'])
######startup
########startup.corr()

startup['states']= startup.State.map({'Florida':0, 'New York':1,'California':2})
startup.rename(columns={'Marketing Spend':'MarketingSpend','R&D Spend':'reasearchcost'}, inplace= True)
startup
type(startup)
startup.corr()
import seaborn as sns 
sns.pairplot(startup.iloc[:,:])

startup
import statsmodels.formula.api as smf
ml1=smf.ols('Profit ~ reasearchcost+Administration +MarketingSpend +states', data=startup ).fit()
ml1.params
ml1.summary()
