import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#####################################################################################################################
# Causal Graphs
#####################################################################################################################
print("Running causal graphs...")

from zepid.causal.causalgraph import DirectedAcyclicGraph

dag = DirectedAcyclicGraph(exposure='X', outcome="Y")
dag.add_arrows((('X', 'Y'),
                ('U1', 'X'), ('U1', 'B'),
                ('U2', 'B'), ('U2', 'Y')
               ))
pos = {"X": [0, 0], "Y": [1, 0], "B": [0.5, 0.5],
       "U1": [0, 1], "U2": [1, 1]}

dag.draw_dag(positions=pos)
plt.tight_layout()
plt.savefig("../images/zepid_dag_mbias.png", format='png', dpi=300)
plt.close()

dag.calculate_adjustment_sets()
print(dag.adjustment_sets)

dag.add_arrows((('X', 'Y'),
                ('U1', 'X'), ('U1', 'B'),
                ('U2', 'B'), ('U2', 'Y'),
                ('B', 'X'), ('B', 'Y')
                ))

dag.draw_dag(positions=pos)
plt.tight_layout()
plt.savefig("../images/zepid_dag_bbias.png", format='png', dpi=300)
plt.close()

dag.calculate_adjustment_sets()
print(dag.adjustment_sets)

#####################################################################################################################
# Time-Fixed Exposure
#####################################################################################################################
print("Running time-fixed exposures...")

from zepid.graphics import EffectMeasurePlot

labels = ["Crude", "G-formula", "IPTW", "AIPTW", "TMLE", "SC-TMLE", "G-estimation"]
riskd = [-0.045, -0.076, -0.082, -0.084, -0.083, -0.083, -0.088]
lcl = [-0.128, -0.144, -0.156, -0.153, -0.152, -0.168, -0.172]
ucl = [0.038, -0.008, -0.007, -0.015, -0.013, 0.001, -0.004]

p = EffectMeasurePlot(label=labels, effect_measure=riskd, lcl=lcl, ucl=ucl)
p.labels(center=0)
p.plot(figsize=(6.5, 3), t_adjuster=0.06, max_value=0.1, min_value=-0.25, decimal=2)
plt.tight_layout()
plt.savefig("../images/zepid_effrd.png", format='png', dpi=300)
plt.close()

labels = ["G-formula", "IPTW", "AIPTW", "TMLE", "SC-TMLE", "G-estimation"]
ate = [226.90, 188.63, 195.64, 197.67, 176.93, 227.23]
lcl = [128.80,  75.89,  89.23, 102.48, -37.66, 134.23]
ucl = [325.00, 301.38, 302.06, 292.85, 391.52, 320.23]

p = EffectMeasurePlot(label=labels, effect_measure=ate, lcl=lcl, ucl=ucl)
p.labels(center=0)
p.plot(figsize=(7, 3), t_adjuster=0.06, max_value=400, min_value=-50, decimal=1)
plt.tight_layout()
plt.savefig("../images/zepid_ate.png", format='png', dpi=300)
plt.close()

#########################################
# Causal Survival Analysis
from zepid import load_sample_data, spline
from zepid.causal.gformula import SurvivalGFormula

df = load_sample_data(False).drop(columns=['cd4_wk45'])
df['t'] = np.round(df['t']).astype(int)
df = pd.DataFrame(np.repeat(df.values, df['t'], axis=0), columns=df.columns)
df['t'] = df.groupby('id')['t'].cumcount() + 1
df.loc[((df['dead'] == 1) & (df['id'] != df['id'].shift(-1))), 'd'] = 1
df['d'] = df['d'].fillna(0)

# Spline terms
df[['t_rs1', 't_rs2', 't_rs3']] = spline(df, 't', n_knots=4, term=2, restricted=True)
df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)

sgf = SurvivalGFormula(df.drop(columns=['dead']), idvar='id', exposure='art', outcome='d', time='t')
sgf.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + '
                        'cd4_rs1 + cd4_rs2 + dvl0 + t + t_rs1 + t_rs2 + t_rs3',
                  print_results=False)

sgf.fit(treatment='all')
sgf.plot(c='b')
sgf.fit(treatment='none')
sgf.plot(c='r')
plt.ylabel('Probability of death')
plt.tight_layout()
plt.savefig("../images/survival_gf_cif.png", format='png', dpi=300)
plt.close()

#####################################################################################################################
# Time-Varying Exposure
#####################################################################################################################
print("Running time-varying exposures...")

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from zepid import load_sample_data, spline
from zepid.causal.gformula import MonteCarloGFormula
from zepid.causal.ipw import IPTW, IPCW

df = load_sample_data(timevary=True)

# Background variable preparations
df['lag_art'] = df['art'].shift(1)
df['lag_art'] = np.where(df.groupby('id').cumcount() == 0, 0, df['lag_art'])
df['lag_cd4'] = df['cd4'].shift(1)
df['lag_cd4'] = np.where(df.groupby('id').cumcount() == 0, df['cd40'], df['lag_cd4'])
df['lag_dvl'] = df['dvl'].shift(1)
df['lag_dvl'] = np.where(df.groupby('id').cumcount() == 0, df['dvl0'], df['lag_dvl'])
df[['age_rs0', 'age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=4, term=2, restricted=True)  # age spline
df['cd40_sq'] = df['cd40'] ** 2  # cd4 baseline cubic
df['cd40_cu'] = df['cd40'] ** 3
df['cd4_sq'] = df['cd4'] ** 2  # cd4 current cubic
df['cd4_cu'] = df['cd4'] ** 3
df['enter_sq'] = df['enter'] ** 2  # entry time cubic
df['enter_cu'] = df['enter'] ** 3

mcgf = MonteCarloGFormula(df,  # Data set
                          idvar='id',  # ID variable
                          exposure='art',  # Exposure
                          outcome='dead',  # Outcome
                          time_in='enter',  # Start of study period
                          time_out='out')  # End of time per study period
# Pooled Logistic Model: Treatment
exp_m = ('male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + '
         'cd4 + cd4_sq + cd4_cu + dvl + enter + enter_sq + enter_cu')
mcgf.exposure_model(exp_m,
                    print_results=False,
                    restriction="g['lag_art']==0")  # Restricts to only untreated (for ITT assumption)
# Pooled Logistic Model: Outcome
out_m = ('art + male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + '
         'cd4 + cd4_sq + cd4_cu + dvl + enter + enter_sq + enter_cu')
mcgf.outcome_model(out_m,
                   print_results=False,
                   restriction="g['drop']==0")  # Restricting to only uncensored individuals
# Pooled Logistic Model: Detectable viral load
dvl_m = ('male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + '
         'lag_cd4 + lag_dvl + lag_art + enter + enter_sq + enter_cu')
mcgf.add_covariate_model(label=1,  # Order to fit time-varying models in
                         covariate='dvl',  # Time-varying confounder
                         print_results=False,
                         model=dvl_m,
                         var_type='binary')  # Variable type
# Pooled Logistic Model: CD4 T-cell count
cd4_m = ('male + age0 + age_rs0 + age_rs1 + age_rs2 +  cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + '
         'lag_dvl + lag_art + enter + enter_sq + enter_cu')
cd4_recode_scheme = ("g['cd4'] = np.maximum(g['cd4'], 1);"
                     "g['cd4_sq'] = g['cd4']**2;"
                     "g['cd4_cu'] = g['cd4']**3")
mcgf.add_covariate_model(label=2,  # Order to fit time-varying models in
                         covariate='cd4',  # Time-varying confounder
                         model=cd4_m,
                         print_results=False,
                         recode=cd4_recode_scheme,  # Recoding process to use for each iteraction of MCMC
                         var_type='continuous')  # Variable type
# Pooled Logistic Model: Censoring
cens_m = ("male + age0 + age_rs0 + age_rs1 + age_rs2 +  cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + " +
          "lag_dvl + lag_art + enter + enter_sq + enter_cu")
mcgf.censoring_model(cens_m, print_results=False)

mcgf.fit(treatment="((g['art']==1) | (g['lag_art']==1))",  # Treatment plan
         lags={'art': 'lag_art',  # Lagged variables to create each loop
               'cd4': 'lag_cd4',
               'dvl': 'lag_dvl'},
         in_recode=("g['enter_sq'] = g['enter']**2;"  # Recode statement to execute at the start
                    "g['enter_cu'] = g['enter']**3"),
         sample=20000)  # Number of resamples from population (should be large number)

# Accessing predicted outcome values
gf = mcgf.predicted_outcomes

# Fitting Kaplan Meier to Natural Course
kmn = KaplanMeierFitter()
kmn.fit(durations=gf['out'], event_observed=gf['dead'])

# Fitting Kaplan Meier to Observed Data
kmo = KaplanMeierFitter()
kmo.fit(durations=df['out'], event_observed=df['dead'], entry=df['enter'])

# Plotting risk functions
plt.step(kmn.event_table.index, 1 - kmn.survival_function_, c='k', where='post', label='Natural')
plt.step(kmo.event_table.index, 1 - kmo.survival_function_, c='gray', where='post', label='True')
plt.legend()
plt.tight_layout()
plt.savefig("../images/zepid_tvg1.png", format='png', dpi=300)
plt.close()

# Treat-all plan
mcgf.fit(treatment="all",
         lags={'art': 'lag_art',
               'cd4': 'lag_cd4',
               'dvl': 'lag_dvl'},
         in_recode=("g['enter_sq'] = g['enter']**2;"
                    "g['enter_cu'] = g['enter']**3"),
         sample=20000)
g_all = mcgf.predicted_outcomes

# Treat-none plan
mcgf.fit(treatment="none",
         lags={'art': 'lag_art',
               'cd4': 'lag_cd4',
               'dvl': 'lag_dvl'},
         in_recode=("g['enter_sq'] = g['enter']**2;"
                    "g['enter_cu'] = g['enter']**3"),
         sample=20000)
g_none = mcgf.predicted_outcomes

# Custom treatment plan
mcgf.fit(treatment="g['cd4'] <= 200",
         lags={'art': 'lag_art',
               'cd4': 'lag_cd4',
               'dvl': 'lag_dvl'},
         in_recode=("g['enter_sq'] = g['enter']**2;"
                    "g['enter_cu'] = g['enter']**3"),
         sample=20000,
         t_max=None)
g_cd4 = mcgf.predicted_outcomes

# Risk curve under treat-all
gfs = g_all.loc[g_all['uid_g_zepid'] != g_all['uid_g_zepid'].shift(-1)].copy()
kma = KaplanMeierFitter()
kma.fit(durations=gfs['out'], event_observed=gfs['dead'])

# Risk curve under treat-all
gfs = g_none.loc[g_none['uid_g_zepid'] != g_none['uid_g_zepid'].shift(-1)].copy()
kmn = KaplanMeierFitter()
kmn.fit(durations=gfs['out'], event_observed=gfs['dead'])

# Risk curve under treat-all
gfs = g_cd4.loc[g_cd4['uid_g_zepid'] != g_cd4['uid_g_zepid'].shift(-1)].copy()
kmc = KaplanMeierFitter()
kmc.fit(durations=gfs['out'], event_observed=gfs['dead'])

# Plotting risk functions
plt.step(kma.event_table.index, 1 - kma.survival_function_, c='blue', where='post', label='All')
plt.step(kmn.event_table.index, 1 - kmn.survival_function_, c='red', where='post', label='None')
plt.step(kmc.event_table.index, 1 - kmc.survival_function_, c='green', where='post', label='CD4 < 200')
plt.legend()
plt.tight_layout()
plt.savefig("../images/zepid_tvg2.png", format='png', dpi=300)
plt.close()

#####################################################################################################################
# Graphics
#####################################################################################################################
print("Running graphics...")

######################################
# Functional form assessment
import zepid as ze
from zepid.graphics import functional_form_plot

df = ze.load_sample_data(timevary=False)
df['age0_sq'] = df['age0']**2
df[['rqs0', 'rqs1']] = ze.spline(df, var='age0', term=2, n_knots=3, knots=[30, 40, 55], restricted=True)

functional_form_plot(df, outcome='dead', var='age0', discrete=True)
plt.tight_layout()
plt.savefig("../images/zepid_fform1.png", format='png', dpi=300)
plt.close()

functional_form_plot(df, outcome='dead', var='age0', discrete=True, points=True)
plt.tight_layout()
plt.savefig("../images/zepid_fform2.png", format='png', dpi=300)
plt.close()

functional_form_plot(df, outcome='dead', var='age0', f_form='age0 + age0_sq', discrete=True)
plt.tight_layout()
plt.savefig("../images/zepid_fform3.png", format='png', dpi=300)
plt.close()

functional_form_plot(df, outcome='dead', var='age0', f_form='age0 + rqs0 + rqs1', discrete=True)
plt.vlines(30, 0, 0.85, colors='gray', linestyles='--')
plt.vlines(40, 0, 0.85, colors='gray', linestyles='--')
plt.vlines(55, 0, 0.85, colors='gray', linestyles='--')
plt.tight_layout()
plt.savefig("../images/zepid_fform4.png", format='png', dpi=300)
plt.close()

######################################
# P-value plot
from zepid.graphics import pvalue_plot

pvalue_plot(point=-0.049, sd=0.042)
plt.tight_layout()
plt.savefig("../images/zepid_pvalue1.png", format='png', dpi=300)
plt.close()

from matplotlib.lines import Line2D

pvalue_plot(point=-0.049, sd=0.042, color='b', fill=False)
pvalue_plot(point=-0.062, sd=0.0231, color='r', fill=False)
plt.legend([Line2D([0], [0], color='b', lw=2),
            Line2D([0], [0], color='r', lw=2)],
           ['Our Study', 'Review'])
plt.tight_layout()
plt.savefig("../images/zepid_pvalue3.png", format='png', dpi=300)
plt.close()

######################################
# Spaghetti Plot
from zepid.graphics import spaghetti_plot

df = ze.load_sample_data(timevary=True)

spaghetti_plot(df, idvar='id', variable='cd4', time='enter')
plt.tight_layout()
plt.savefig("../images/zepid_spaghetti.png", format='png', dpi=300)
plt.close()

######################################
# Effect Measure plot
import numpy as np
from zepid.graphics import EffectMeasurePlot

labs = ['Overall', 'Adjusted', '',
        '2012-2013', 'Adjusted', '',
        '2013-2014', 'Adjusted', '',
        '2014-2015', 'Adjusted']
measure = [np.nan, 0.94, np.nan, np.nan, 1.22, np.nan, np.nan, 0.59, np.nan, np.nan, 1.09]
lower = [np.nan, 0.77, np.nan, np.nan, '0.80', np.nan, np.nan, '0.40', np.nan, np.nan, 0.83]
upper = [np.nan, 1.15, np.nan, np.nan, 1.84, np.nan, np.nan, 0.85, np.nan, np.nan, 1.44]

p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
p.labels(scale='log')
p.plot(figsize=(6.5, 3), t_adjuster=0.02, max_value=2, min_value=0.38)
plt.tight_layout()
plt.savefig("../images/zepid_effm.png", format='png', dpi=300)
plt.close()

######################################
# ROC
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family,links
from zepid.graphics import roc

df = ze.load_sample_data(timevary=False).drop(columns=['cd4_wk45']).dropna()
f = sm.families.family.Binomial(sm.families.links.logit)
df['age0_sq'] = df['age0']**2
df['cd40sq'] = df['cd40']**2
model = 'dead ~ art + age0 + age0_sq + cd40 + cd40sq + dvl0 + male'
m = smf.glm(model, df, family=f).fit()
df['predicted'] = m.predict(df)

roc(df.dropna(), true='dead', threshold='predicted')
plt.tight_layout()
plt.title('Receiver-Operator Curve')
plt.tight_layout()
plt.savefig("../images/zepid_roc.png", format='png', dpi=300)
plt.close()

######################################
# L'Abbe
from zepid.graphics import labbe_plot

labbe_plot()
plt.tight_layout()
plt.savefig("../images/zepid_labbe1.png", format='png', dpi=300)
plt.close()

labbe_plot(r1=[0.3, 0.5], r0=[0.2, 0.7], color='red')
plt.tight_layout()
plt.savefig("../images/zepid_labbe2.png", format='png', dpi=300)
plt.close()

labbe_plot(r1=[0.25, 0.5], r0=[0.1, 0.2], color='red')
plt.tight_layout()
plt.savefig("../images/zepid_labbe3.png", format='png', dpi=300)
plt.close()

######################################
# Zipper plot
from zepid.graphics import zipper_plot
lower = np.random.RandomState(80412).uniform(-0.1, 0.1, size=100)
upper = lower + np.random.RandomState(192041).uniform(0.1, 0.2, size=100)

zipper_plot(truth=0,
            lcl=lower,
            ucl=upper,
            colors=('blue', 'green'))
plt.tight_layout()
plt.savefig("../images/zipper_example.png", format='png', dpi=300)
plt.close()

#####################################################################################################################
# Sensitivity
#####################################################################################################################
print("Running sensitivity...")

from zepid.sensitivity_analysis import trapezoidal

plt.hist(trapezoidal(mini=1, mode1=1.5, mode2=3, maxi=3.5, size=250000), bins=100)
plt.tight_layout()
plt.savefig("../images/zepid_trapezoid.png", format='png', dpi=300)
plt.close()

from zepid.sensitivity_analysis import MonteCarloRR

mcrr = MonteCarloRR(observed_RR=0.73322, sample=10000)
mcrr.confounder_RR_distribution(trapezoidal(mini=0.9, mode1=1.1, mode2=1.7, maxi=1.8, size=10000))
mcrr.prop_confounder_exposed(trapezoidal(mini=0.25, mode1=0.28, mode2=0.32, maxi=0.35, size=10000))
mcrr.prop_confounder_unexposed(trapezoidal(mini=0.55, mode1=0.58, mode2=0.62, maxi=0.65, size=10000))
mcrr.fit()

mcrr.plot()
plt.tight_layout()
plt.savefig("../images/zepid_crr.png", format='png', dpi=300)
plt.close()
