import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from zepid import load_sample_data, spline


#######################################################################################################################
# Binary Outcome
#######################################################################################################################

df = load_sample_data(timevary=False)
df = df.drop(columns=['cd4_wk45'])
df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)

#############################
# Naive Risk Difference
from zepid import RiskDifference

rd = RiskDifference()
rd.fit(df, exposure='art', outcome='dead')
rd.summary()

#############################
# G-formula
from zepid.causal.gformula import TimeFixedGFormula

g = TimeFixedGFormula(df, exposure='art', outcome='dead')
g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                print_results=False)
# Estimating marginal effect under treat-all plan
g.fit(treatment='all')
r_all = g.marginal_outcome
# Estimating marginal effect under treat-none plan
g.fit(treatment='none')
r_none = g.marginal_outcome

riskd = r_all - r_none
print('RD:', riskd)

rd_results = []
for i in range(1000):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        s = df.sample(n=df.shape[0],replace=True)
        g = TimeFixedGFormula(s,exposure='art',outcome='dead')
        g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                        print_results=False)
        g.fit(treatment='all')
        r_all = g.marginal_outcome
        g.fit(treatment='none')
        r_none = g.marginal_outcome
        rd_results.append(r_all - r_none)

se = np.std(rd_results)
print('95% LCL', riskd - 1.96*se)
print('95% UCL', riskd + 1.96*se)

#############################
# IPTW
from zepid.causal.ipw import IPTW

iptw = IPTW(df, treatment='art', outcome='dead')
iptw.treatment_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                     bound=0.01, print_results=False)
iptw.marginal_structural_model('art')
iptw.fit()
iptw.summary()

#############################
# AIPTW
from zepid.causal.doublyrobust import AIPTW

aipw = AIPTW(df, exposure='art', outcome='dead')
# Treatment model
aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                    print_results=False, bound=0.01)
# Outcome model
aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
# Calculating estimate
aipw.fit()
# Printing summary results
aipw.summary()

#############################
# TMLE
from zepid.causal.doublyrobust import TMLE

tmle = TMLE(df, exposure='art', outcome='dead')
tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                    print_results=False, bound=0.01)
tmle.missing_model('art + male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
tmle.fit()
tmle.summary()

#############################
# Cross-fitting
from sklearn.ensemble import RandomForestClassifier
from zepid.superlearner import GLMSL, StepwiseSL, SuperLearner
from zepid.causal.doublyrobust import SingleCrossfitTMLE

# SuperLearner set-up
labels = ["LogR", "Step.int", "RandFor"]
candidates = [GLMSL(sm.families.family.Binomial()),
              StepwiseSL(sm.families.family.Binomial(), selection="forward", order_interaction=0),
              RandomForestClassifier(random_state=809512)]

# Single cross-fit TMLE
sctmle = SingleCrossfitTMLE(df, exposure='art', outcome='dead')
sctmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                      SuperLearner(candidates, labels, folds=10, loss_function="nloglik"),
                      bound=0.01)
sctmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                     SuperLearner(candidates, labels, folds=10, loss_function="nloglik"))
sctmle.fit(n_partitions=3, random_state=201820)
sctmle.summary()

#############################
# G-estimation
from zepid.causal.snm import GEstimationSNM

snm = GEstimationSNM(df, exposure='art', outcome='dead')
# Specify treatment model
snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
# Specify structural nested model
snm.structural_nested_model('art')
# G-estimation
snm.fit()
snm.summary()

psi = snm.psi
print('Psi:', psi)

psi_results = []
for i in range(500):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        dfs = df.sample(n=df.shape[0], replace=True)
        snm = GEstimationSNM(dfs, exposure='art', outcome='dead')
        snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        snm.structural_nested_model('art')
        snm.fit()
        psi_results.append(snm.psi)


se = np.std(psi_results)
print('95% LCL', psi - 1.96*se)
print('95% UCL', psi + 1.96*se)

snm = GEstimationSNM(df, exposure='art', outcome='dead')
snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
snm.structural_nested_model('art + art:male')
snm.fit()
snm.summary()

#######################################################################################################################
# Continuous Outcome
#######################################################################################################################

df = load_sample_data(timevary=False)
dfs = df.drop(columns=['dead']).dropna()
df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)

#############################
# G-formula
g = TimeFixedGFormula(df, exposure='art', outcome='cd4_wk45', outcome_type='normal')
g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
g.fit(treatment='all')
r_all = g.marginal_outcome
g.fit(treatment='none')
r_none = g.marginal_outcome
ate = r_all - r_none
print('ATE:', ate)

ate_results = []
for i in range(1000):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        s = df.sample(n=df.shape[0], replace=True)
        g = TimeFixedGFormula(s,exposure='art',outcome='cd4_wk45', outcome_type='normal')
        g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                        print_results=False)
        g.fit(treatment='all')
        r_all = g.marginal_outcome
        g.fit(treatment='none')
        r_none = g.marginal_outcome
        ate_results.append(r_all - r_none)

se = np.std(ate_results)
print('95% LCL', ate - 1.96*se)
print('95% UCL', ate + 1.96*se)

#############################
# IPTW

ipw = IPTW(df, treatment='art', outcome='cd4_wk45')
ipw.treatment_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                    print_results=False, bound=0.01)
ipw.marginal_structural_model('art')
ipw.fit()
ipw.summary()

#############################
# AIPTW

aipw = AIPTW(df, exposure='art', outcome='cd4_wk45')
aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                    print_results=False, bound=0.01)
aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
aipw.fit()
aipw.summary()

#############################
# TMLE

tmle = TMLE(df, exposure='art', outcome='cd4_wk45')
tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                    print_results=False, bound=0.01)
tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
tmle.fit()
tmle.summary()

#############################
# Cross-fitting
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# SuperLearner set-up
labels = ["LogR", "Step.int", "RandFor"]
b_candidates = [GLMSL(sm.families.family.Binomial()),
                StepwiseSL(sm.families.family.Binomial(), selection="forward", order_interaction=0),
                RandomForestClassifier(random_state=809512)]
c_candidates = [GLMSL(sm.families.family.Gaussian()),
                StepwiseSL(sm.families.family.Gaussian(), selection="forward", order_interaction=0),
                RandomForestRegressor(random_state=809512)]

# Single cross-fit TMLE
sctmle = SingleCrossfitTMLE(df, exposure='art', outcome='cd4_wk45')
sctmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                      SuperLearner(b_candidates, labels, folds=10, loss_function="nloglik"),
                      bound=0.01)
sctmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                     SuperLearner(c_candidates, labels, folds=10))
sctmle.fit(n_partitions=3, random_state=201820)
sctmle.summary()

#############################
# G-estimation

snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
snm.structural_nested_model('art')
snm.fit()
snm.summary()

psi = snm.psi
print('Psi:', psi)

psi_results = []
for i in range(500):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        dfs = df.sample(n=df.shape[0], replace=True)
        snm = GEstimationSNM(dfs, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        snm.structural_nested_model('art')
        snm.fit()
        psi_results.append(snm.psi)


se = np.std(psi_results, ddof=1)
print('95% LCL', psi - 1.96*se)
print('95% UCL', psi + 1.96*se)

snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                   print_results=False)
snm.structural_nested_model('art + art:male')
snm.fit()
snm.summary()
