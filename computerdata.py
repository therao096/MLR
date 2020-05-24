# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:27:17 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cdata=pd.read_csv("Computer_Data.csv")
cdata
cdata.columns
cdata2= cdata.loc[:,['price', 'speed', 'hd', 'ram', 'screen', 'cd', 'multi','premium', 'ads', 'trend']]
cdata2=pd.get_dummies(cdata2,columns=['cd','multi','premium'])
cdata22=cdata2.drop(cdata2.index[[1400,19]], axis=0)
cdata22
print(cdata22.corr())
cdata22.columns
import seaborn as sns
sns.pairplot(cdata2.loc[:,:])
###there seems no corelation, build a model
import statsmodels.formula.api as smf
cdata22.columns
ml1= smf.ols('price~speed+hd+ram+screen+ads+trend+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes', data=cdata22).fit()
ml1.params
ml1.summary()###r square =0.775
mlcc=smf.ols('price~cd_no', data=cdata22).fit()
mlcc.summary()
mlno=smf.ols('price~multi_no', data=cdata22).fit()
mlno.summary()
mlc=smf.ols('price~cd_no+multi_no',data=cdata22).fit()
mlc.summary()
####pval=0.188>0.05 , therefore it is significant
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

ml2=smf.ols('price~speed+hd+ram+screen+ads+trend+cd_no+cd_yes+multi_yes+premium_no+premium_yes', data=cdata22).fit()
ml2.params
ml2.summary()
pred_log=ml2.predict(cdata2)
pred_log
pred_logg=np.exp(pred_log)
pred_log
ml3=smf.ols('price~speed+hd+ram+screen+ads+trend+multi_no+cd_yes+multi_yes+premium_no+premium_yes', data=cdata22).fit()
ml2.params
ml2.summary()
##calculating vif values
rsq_cdno=smf.ols('price~cd_no', data=cdata22).fit().rsquared
vif=1/(1-rsq_cdno)
vif ##1.04
rsq_multino=smf.ols('price~multi_no', data=cdata22).fit().rsquared
vif=1/(1-rsq_multino)
vif ###1
###drop cd_no

####transforming dependent variable
ml4=smf.ols( 'price~speed+hd+ram+screen+ads+trend+cd_yes+multi_no+multi_yes+premium_no+premium_yes',data=cdata22).fit()
ml4.summary()
ml4=smf.ols('np.log(price)~speed+hd+ram+screen+ads+trend+cd_yes+multi_no+multi_yes+premium_no+premium_yes', data=cdata22).fit()
ml4.summary()
###r squr value is ####3transformiong independent variables 0.783



#######33ml5=smf.ols('price~np.log(speed)+np.log(hd)+np.log(ram)+np.log(screen)+np.log(ads)+np.log(trend)+np.log(cd_yes)+np.log(multi_no)+np.log(multi_yes)+np.log(premium_no)+np.log(premium_yes)', data=cdata22).fit()
########3ml5.summary()
######33ml5=smf.ols('price~(np.log(speed+hd+ram+screen+ads+trend+cd_yes+multi_no+multi_yes+premium_no+premium_yes))', data=cdata22).fit()
#####3ml5.summary()
pred_final=ml4.predict(cdata22)
pred_final
pred_frame=pd.DataFrame(pred_final)
pred_frame
