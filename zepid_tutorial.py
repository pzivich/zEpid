########################################################################################
# zepid: Epidemiology Tools User Guide
#   Code: pzivich
#   Date: 12/15/2017
########################################################################################

import zepid
import zepid.calc as zec
import zepid.graphics as zeg 
import zepid.ipw as zei 
import zepid.sens_analysis as zes

#import some other packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links


#Load sample dataframe
df = zepid.datex()
df.info()

#Basic Measures of association between exposure and outcome
zepid.OddsRatio(df['exposure'],df['outcome'])
zepid.RelRisk(df['exposure'],df['outcome'])
zepid.RiskDiff(df['exposure'],df['outcome'])
zepid.NNT(df['exposure'],df['outcome'])

#let's check for collinearity between a few variables
zepid.StandMeanDiff(df,'exposure','continuous')
zepid.StandMeanDiff(df,'binary','continuous')

df.category.value_counts() #so the above function only works for binary, but we can use the zepid.calc to calculate with numbers
c0mean = np.mean(df.loc[df.category==0]['continuous']) #extracting the mean
c0std = np.std(df.loc[df.category==0]['continuous']) #extracting the standard deviation
c1mean = np.mean(df.loc[df.category==1]['continuous']) #extracting the mean
c1std = np.std(df.loc[df.category==1]['continuous']) #extracting the standard deviation
c2mean = np.mean(df.loc[df.category==2]['continuous']) #extracting the mean
c2std = np.std(df.loc[df.category==2]['continuous']) #extracting the standard deviation

zec.stand_mean_diff(1900,859,c2mean,c0mean,c2std,c0std,decimal=5)
zec.stand_mean_diff(1900,668,c2mean,c1mean,c2std,c1std,decimal=5)

zepid.OddsRatio(df['exposure'],df['binary'])

pd.crosstab(df['exposure'],df['category'])
zec.oddr(80,588,747,1153)
zec.oddr(30,829,747,1153) #so we have some evidence of collinearity between exposure and the binary variables, we will ignore the method implications of this

#let's determine our functional form for a continuous variable
df.describe()
plt.hist(df.continuous)

df['cont_sq'] = df['continuous']**2
df.loc[df.continuous<20,'ccat'] = 0
df.loc[((df.continuous<40)&(df.continuous>=20)),'ccat'] = 1
df.loc[((df.continuous<60)&(df.continuous>=40)),'ccat'] = 2
df.loc[((df.continuous<80)&(df.continuous>=60)),'ccat'] = 3
df.loc[((df.continuous<=100)&(df.continuous>=80)),'ccat'] = 4
df[['rs1','rs2','rs3','rs4']] = zepid.spline(df,'continuous',n_knots=5,term=2,restricted=True) #creating a restricted quadratic spline

zeg.func_form_plot(df,'outcome','continuous') #linear
zeg.func_form_plot(df,'outcome','continuous','continuous + cont_sq') #quadratic
zeg.func_form_plot(df,'outcome','continuous','C(ccat)') #categorical
zeg.func_form_plot(df,'outcome','continuous','continuous + rs1 + rs2 + rs3 + rs4') #restricted quadratic spline

#let's fit a GLM for our results now
f = sm.families.family.Binomial(sm.families.links.logit)
mod = smf.glm('outcome ~ exposure + C(category) + binary + continuous + cont_sq',df,family=f).fit()
mod.summary()
np.exp(mod.params)
np.exp(mod.conf_int())

#let's build an IPTW model
df['exp_prob'],df['iptw_weight'] = zei.iptw(df,'exposure','continuous + binary')
zei.diagnostic.positivity(df,'iptw_weight') #positivity doesn't look great... might want to not use this model, but anyways
zei.diagnostic.p_hist(df,'exposure','exp_prob')
zei.diagnostic.p_boxplot(df,'exposure','exp_prob') #IPTW doesn't seem to fit well for our model
zei.diagnostic.standardized_diff(df,'exposure','binary','iptw_weight') #SMD > 2, so looks like problems
zei.diagnostic.standardized_diff(df,'exposure','continuous','iptw_weight',var_type='continuous') #SMD > 2, so looks like problems

df['id'] = df.index
iptmodel = zei.ipw_fit(df,'outcome ~ exposure','id','iptw_weight')
iptmodel.summary()
np.exp(iptmodel.params)
np.exp(iptmodel.conf_int())

#let's build an IPMW model to look at our missing
m,df['ipmw_weight'] = zei.ipmw(df,'outcome','C(category)')
ipmmodel = zei.ipw_fit(df,'outcome ~ exposure + C(category) + binary + continuous + cont_sq','id','ipmw_weight')
ipmmodel.summary()
np.exp(ipmmodel.params)
np.exp(ipmmodel.conf_int())

#let's condense our results into an informative graphic
labs = ['Crude','Logit','IPTW','IPMW']
resu = [1.95,2.45,1.12,2.53]
rlcl = [1.64,1.93,0.76,'2.00']
rucl = [2.32,'3.10',1.66,3.21]

pf = zeg.effectmeasure_plotdata(labs,resu,rlcl,rucl)
#after some trial and error with the 't_adjuster', we end with the following plot
zeg.effectmeasure_plot(pf,decimal=3,title='Odds Ratio Plot of Results',em='OR',ci='95% CI',scale='log',t_adjuster=0.09,size=3)
