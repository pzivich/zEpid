import numpy as np
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from statsmodels.genmod.families import family,links
import matplotlib.pyplot as plt 

import zepid as ze 
from zepid.causal.gformula import TimeFixedGFormula
from zepid.causal.doublyrobust import SimpleDoubleRobust

df = ze.load_sample_data(timevary=False)
df[['age_rs1','age_rs2']] = ze.spline(df,'age0',term=2,restricted=True)
df[['cd4_rs1','cd4_rs2']] = ze.spline(df,'cd40',term=2,restricted=True)

#Crude Model
ze.RiskRatio(df,exposure='art',outcome='dead')
ze.RiskDiff(df,exposure='art',outcome='dead')
#Adjusted Model
model = 'art + male + age0 + cd40 + dvl0'
f = sm.families.family.Binomial(sm.families.links.identity) 
linrisk = smf.glm('dead ~ '+model,df,family=f).fit()
linrisk.summary()
f = sm.families.family.Binomial(sm.families.links.log) 
log = smf.glm('dead ~ art',df,family=f).fit()
log.summary()
#g-formula
g = TimeFixedGFormula(df,exposure='art',outcome='dead')
g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
g.fit(treatment='all')
r_all = g.marginal_outcome
g.fit(treatment='none')
r_none = g.marginal_outcome
print('RD1 = ',r_all - r_none)
print('RR1 = ',r_all / r_none)
rd_results = []
rr_results = []
for i in range(500):
    dfs = df.sample(n=df.shape[0],replace=True)
    g = TimeFixedGFormula(dfs,exposure='art',outcome='dead')
    g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',print_model_results=False)
    g.fit(treatment='all')
    r_all = g.marginal_outcome
    g.fit(treatment='none')
    r_none = g.marginal_outcome
    rd_results.append(r_all - r_none)
    rr_results.append(r_all / r_none)


print('RD 95% CI:',np.percentile(rd_results,q=[2.5,97.5]))
print('RR 95% CI:',np.percentile(rr_results,q=[2.5,97.5]))
#IPTW 
model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
df['iptw'] = ze.ipw.iptw(df,treatment='art',model_denominator=model,stabilized=True)
ind = sm.cov_struct.Independence()
f = sm.families.family.Binomial(sm.families.links.identity) 
linrisk = smf.gee('dead ~ art',df['id'],df,cov_struct=ind,family=f,weights=df['iptw']).fit()
linrisk.summary()
f = sm.families.family.Binomial(sm.families.links.log) 
log = smf.gee('dead ~ art',df['id'],df,cov_struct=ind,family=f,weights=df['iptw']).fit()
log.summary()
#Double-Robust
sdr = SimpleDoubleRobust(df,exposure='art',outcome='dead')
sdr.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
sdr.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
sdr.fit()
sdr.summary()
rd = []
rr = []
for i in range(500):
    dfs = df.sample(n=df.shape[0],replace=True)
    s = SimpleDoubleRobust(dfs,exposure='art',outcome='dead')
    s.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',print_model_results=False)
    s.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',print_model_results=False)
    s.fit()
    rd.append(s.riskdiff)
    rr.append(s.riskratio)


print('RD 95% CI: ',np.percentile(rd,q=[2.5,97.5]))
print('RR 95% CI: ',np.percentile(rr,q=[2.5,97.5]))

#Risk Difference plot
labs = ['Crude','GLM','G-formula','IPTW','Double-Robust']
measure = [-0.049,np.nan,-0.077,-0.084,-0.071]
lower = ['-0.130',np.nan,-0.139,-0.157,-0.129]
upper = [0.033,np.nan,-0.012,-0.011,-0.008]
p = ze.graphics.effectmeasure_plot(label=labs,effect_measure=measure,lcl=lower,ucl=upper)
p.labels(center=0,effectmeasure='RD')
p.plot(figsize=(8.25,4),t_adjuster=0.09,max_value=0.1,min_value=-0.2)
plt.tight_layout()
plt.savefig('C:/Users/zivic/Python Programs/Development/zepid/docs/images/zepid_effrd.png',dpi=300,format='png')
plt.show()

labs = ['Crude','GLM','G-formula','IPTW','Double-Robust']
measure = [0.72,np.nan,0.58,0.54,0.57]
lower = [0.39,np.nan,0.28,0.27,0.24]
upper = [1.33,np.nan,0.93,1.06,0.95]
p = ze.graphics.effectmeasure_plot(label=labs,effect_measure=measure,lcl=lower,ucl=upper)
p.labels(center=1,effectmeasure='RR')
p.plot(figsize=(7.25,3),t_adjuster=0.015,max_value=1.5,min_value=0.2)
plt.tight_layout()
plt.savefig('C:/Users/zivic/Python Programs/Development/zepid/docs/images/zepid_effrr.png',dpi=300,format='png')
plt.show()
