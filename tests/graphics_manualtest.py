# Following is ignored by Travis CI and pytest
# This is meant to run to manually check all plotting features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from zepid import (load_sample_data, RiskDifference, RiskRatio, OddsRatio, IncidenceRateDifference, IncidenceRateRatio,
                   spline)
from zepid.graphics import (EffectMeasurePlot, functional_form_plot, pvalue_plot, spaghetti_plot,
                            roc, dynamic_risk_plot, labbe_plot, zipper_plot)
from zepid.causal.ipw import IPTW
from zepid.causal.gformula import MonteCarloGFormula, SurvivalGFormula, TimeFixedGFormula
from zepid.causal.doublyrobust import AIPTW, TMLE
from zepid.sensitivity_analysis import MonteCarloRR, trapezoidal
from zepid.causal.causalgraph import DirectedAcyclicGraph


def graphics_check():
    # L'Abbe Plots
    labbe_plot()
    plt.show()
    labbe_plot(r1=[0.25, 0.5], r0=[0.1, 0.2], color='r')
    plt.show()
    labbe_plot(r1=[0.3, 0.5], r0=[0.2, 0.7], scale='additive', marker='+', linestyle='')
    plt.show()
    labbe_plot(r1=[0.3, 0.5], r0=[0.2, 0.7], scale='multiplicative', markersize=10)
    plt.show()

    # 1) Check EffectMeasurePlot
    labs = ['Overall', 'Adjusted', '', '2012-2013', 'Adjusted', '', '2013-2014', 'Adjusted', '', '2014-2015',
            'Adjusted']
    measure = [np.nan, 0.94, np.nan, np.nan, 1.22, np.nan, np.nan, 0.59, np.nan, np.nan, 1.09]
    lower = [np.nan, 0.77, np.nan, np.nan, '0.80', np.nan, np.nan, '0.40', np.nan, np.nan, 0.83]
    upper = [np.nan, 1.15, np.nan, np.nan, 1.84, np.nan, np.nan, 0.85, np.nan, np.nan, 1.44]
    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.plot(figsize=[7, 4])
    plt.show()

    # 2) Check Functional form plots
    data = load_sample_data(False)
    data['age_sq'] = data['age0']**2
    functional_form_plot(data, 'dead', var='age0', loess=False)
    plt.show()
    functional_form_plot(data, 'dead', var='age0', discrete=True, loess=False)
    plt.show()
    functional_form_plot(data, 'cd40', var='age0', outcome_type='continuous', loess=False)
    plt.show()
    functional_form_plot(data, 'dead', var='age0', points=True, loess=False)
    plt.show()
    functional_form_plot(data, 'dead', var='age0', loess=True, loess_value=0.25, discrete=True)
    plt.show()
    functional_form_plot(data, 'dead', var='age0', f_form='age0 + age_sq', loess=False)
    plt.show()

    # 3) Check P-value plots
    p = pvalue_plot(point=0.23, sd=0.1)
    plt.show()
    pvalue_plot(point=0.23, sd=0.1, fill=False)
    plt.show()
    pvalue_plot(point=0.23, sd=0.1, color='r')
    plt.show()
    pvalue_plot(point=0.23, sd=0.1, null=0.05)
    plt.show()
    pvalue_plot(point=0.23, sd=0.1, alpha=0.05)
    plt.show()

    # 4) Check Spaghetti plot
    df = load_sample_data(timevary=True)
    spaghetti_plot(df, idvar='id', variable='cd4', time='enter')
    plt.show()

    # 5) Check ROC
    df = pd.DataFrame()
    df['d'] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    df['p'] = [0.1, 0.15, 0.1, 0.7, 0.5, 0.9, 0.95, 0.5, 0.4, 0.8, 0.99, 0.99, 0.89, 0.95]
    roc(df, true='d', threshold='p', youden_index=False)
    plt.show()
    roc(df, true='d', threshold='p', youden_index=True)
    plt.show()

    # 6) Check Dynamic Risk Plots
    a = pd.DataFrame([[0, 0], [1, 0.15], [2, 0.25], [4, 0.345]], columns=['timeline', 'riske']).set_index(
                     'timeline')
    b = pd.DataFrame([[0, 0], [1, 0.2], [1.5, 0.31], [3, 0.345]], columns=['timeline', 'riske']).set_index(
                     'timeline')
    dynamic_risk_plot(a, b, loess=False)
    plt.show()
    dynamic_risk_plot(a, b, measure='RR', loess=False)
    plt.show()
    dynamic_risk_plot(a, b, measure='RR', scale='log-transform', loess=False)
    plt.show()
    dynamic_risk_plot(a, b, measure='RR', scale='log', loess=False)
    plt.show()
    dynamic_risk_plot(a, b, loess=True, loess_value=0.4)
    plt.show()
    dynamic_risk_plot(a, b, loess=False, point_color='green', line_color='green')
    plt.show()

    # 7) Check Zipper Plot
    lower = np.random.uniform(-0.1, 0.1, size=100)
    upper = lower + np.random.uniform(0.1, 0.2, size=100)
    zipper_plot(truth=0,
                lcl=lower,
                ucl=upper,
                colors=('blue', 'green'))
    plt.show()


def measures_check():
    # 7) Check measures plots
    data_set = load_sample_data(False)
    rr = RiskRatio()
    rr.fit(data_set, exposure='art', outcome='dead')
    rr.plot(fmt='*', ecolor='r', barsabove=True, markersize=25)
    plt.show()
    rd = RiskDifference()
    rd.fit(data_set, exposure='art', outcome='dead')
    rd.plot()
    plt.show()
    ord = OddsRatio()
    ord.fit(data_set, exposure='art', outcome='dead')
    ord.plot()
    plt.show()
    irr = IncidenceRateRatio()
    irr.fit(data_set, exposure='art', outcome='dead', time='t')
    irr.plot()
    plt.show()
    ird = IncidenceRateDifference()
    ird.fit(data_set, exposure='art', outcome='dead', time='t')
    ird.plot()
    plt.show()


def senstivity_check():
    # 8) Check Monte Carlo Bias Analysis
    np.random.seed(101)
    mcrr = MonteCarloRR(observed_RR=0.73322, sample=10000)
    mcrr.confounder_RR_distribution(trapezoidal(mini=0.9, mode1=1.1, mode2=1.7, maxi=1.8, size=10000))
    mcrr.prop_confounder_exposed(trapezoidal(mini=0.25, mode1=0.28, mode2=0.32, maxi=0.35, size=10000))
    mcrr.prop_confounder_unexposed(trapezoidal(mini=0.55, mode1=0.58, mode2=0.62, maxi=0.65, size=10000))
    mcrr.fit()
    mcrr.plot()
    plt.show()


def causal_check():
    data = load_sample_data(False).drop(columns=['cd4_wk45'])
    data[['cd4_rs1', 'cd4_rs2']] = spline(data, 'cd40', n_knots=3, term=2, restricted=True)
    data[['age_rs1', 'age_rs2']] = spline(data, 'age0', n_knots=3, term=2, restricted=True)

    # Check TimeFixedGFormula diagnostics
    g = TimeFixedGFormula(data, exposure='art', outcome='dead')
    g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    g.run_diagnostics(decimal=3)

    # Check IPTW plots
    ipt = IPTW(data, treatment='art', outcome='dead')
    ipt.treatment_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', stabilized=True)
    ipt.marginal_structural_model('art')
    ipt.fit()
    ipt.plot_love()
    plt.tight_layout()
    plt.show()
    ipt.plot_kde()
    plt.show()
    ipt.plot_kde(measure='logit')
    plt.show()
    ipt.plot_boxplot()
    plt.show()
    ipt.plot_boxplot(measure='logit')
    plt.show()
    ipt.run_diagnostics()

    # Check AIPTW Diagnostics
    aipw = AIPTW(data, exposure='art', outcome='dead')
    aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    aipw.fit()
    aipw.run_diagnostics()
    aipw.plot_kde(to_plot='exposure')
    plt.show()
    aipw.plot_kde(to_plot='outcome')
    plt.show()
    aipw.plot_love()
    plt.show()

    # Check TMLE diagnostics
    tmle = TMLE(data, exposure='art', outcome='dead')
    tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    tmle.fit()
    tmle.run_diagnostics()
    tmle.plot_kde(to_plot='exposure')
    plt.show()
    tmle.plot_kde(to_plot='outcome')
    plt.show()
    tmle.plot_love()
    plt.show()

    # Check SurvivalGFormula plots
    df = load_sample_data(False).drop(columns=['cd4_wk45'])
    df['t'] = np.round(df['t']).astype(int)
    df = pd.DataFrame(np.repeat(df.values, df['t'], axis=0), columns=df.columns)
    df['t'] = df.groupby('id')['t'].cumcount() + 1
    df.loc[((df['dead'] == 1) & (df['id'] != df['id'].shift(-1))), 'd'] = 1
    df['d'] = df['d'].fillna(0)
    df['t_sq'] = df['t'] ** 2
    df['t_cu'] = df['t'] ** 3
    sgf = SurvivalGFormula(df, idvar='id', exposure='art', outcome='d', time='t')
    sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu')
    sgf.fit(treatment='all')
    sgf.plot()
    plt.show()
    sgf.plot(c='r', linewidth=3, alpha=0.8)
    plt.show()


def mc_gformula_check():
    df = load_sample_data(timevary=True)
    # df['weights'] = 2.1
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
    g = MonteCarloGFormula(df, idvar='id', exposure='art', outcome='dead', time_in='enter', time_out='out',
                           # weights='weights'
                           )
    exp_m = '''male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + cd4 + cd4_sq + 
            cd4_cu + dvl + enter + enter_sq + enter_cu'''
    g.exposure_model(exp_m, restriction="g['lag_art']==0")
    out_m = '''art + male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + cd4 + 
            cd4_sq + cd4_cu + dvl + enter + enter_sq + enter_cu'''
    g.outcome_model(out_m, restriction="g['drop']==0")
    dvl_m = '''male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + 
            lag_dvl + lag_art + enter + enter_sq + enter_cu'''
    g.add_covariate_model(label=1, covariate='dvl', model=dvl_m, var_type='binary')
    cd4_m = '''male + age0 + age_rs0 + age_rs1 + age_rs2 +  cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + 
            lag_dvl + lag_art + enter + enter_sq + enter_cu'''
    cd4_recode_scheme = ("g['cd4'] = np.maximum(g['cd4'],1);"
                         "g['cd4_sq'] = g['cd4']**2;"
                         "g['cd4_cu'] = g['cd4']**3")
    g.add_covariate_model(label=2, covariate='cd4', model=cd4_m,recode=cd4_recode_scheme, var_type='continuous')
    g.fit(treatment="((g['art']==1) | (g['lag_art']==1))",
          lags={'art': 'lag_art',
                'cd4': 'lag_cd4',
                'dvl': 'lag_dvl'},
          sample=50000, t_max=None,
          in_recode=("g['enter_sq'] = g['enter']**2;"
                     "g['enter_cu'] = g['enter']**3"))
    gf = g.predicted_outcomes
    kmn = KaplanMeierFitter()
    kmn.fit(durations=gf['out'], event_observed=gf['dead'])
    kmo = KaplanMeierFitter()
    kmo.fit(durations=df['out'], event_observed=df['dead'], entry=df['enter'])
    cens_m = """male + age0 + age_rs0 + age_rs1 + age_rs2 +  cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 +
             lag_dvl + art + lag_art + enter + enter_sq + enter_cu"""
    g.censoring_model(cens_m)
    g.fit(treatment="((g['art']==1) | (g['lag_art']==1))",
          lags={'art': 'lag_art',
                'cd4': 'lag_cd4',
                'dvl': 'lag_dvl'},
          sample=10000, t_max=None,
          in_recode=("g['enter_sq'] = g['enter']**2;"
                     "g['enter_cu'] = g['enter']**3"))
    gf = g.predicted_outcomes
    kmc = KaplanMeierFitter()
    kmc.fit(durations=gf['out'], event_observed=gf['dead'])
    plt.step(kmn.event_table.index, 1 - kmn.survival_function_, c='g', where='post', label='Natural')
    plt.step(kmn.event_table.index, 1 - kmc.survival_function_, c='orange', where='post', label='Censor')
    plt.step(kmo.event_table.index, 1 - kmo.survival_function_, c='k', where='post', label='True')
    plt.legend()
    plt.show()


def causalgraph_check():
    dag = DirectedAcyclicGraph(exposure="X", outcome="Y")
    dag.add_arrow(source="X", endpoint="Y")
    dag.add_arrow(source="V", endpoint="Y")
    dag.add_arrows(pairs=(("W", "X"), ("W", "Y")))
    dag.draw_dag()
    plt.show()


# graphics_check()
# senstivity_check()
# measures_check()
# causal_check()
# causalgraph_check()
# mc_gformula_check()
