.. image:: images/zepid_logo_small.png

-------------------------------------

Time-Fixed Exposure
'''''''''''''''''''''''''''''''''''''
In this section, we will go through some methods to estimate the average causal effect of a time-fixed treatment /
exposure on a specific outcome. We will review binary outcomes, continuous outcomes, and time-to-event data. To follow
along with the tutorial, run the following code to set up the data

.. code::

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.families import family

    from zepid import load_sample_data, spline, RiskDifference
    from zepid.causal.gformula import TimeFixedGFormula
    from zepid.causal.ipw import IPTW, IPMW
    from zepid.causal.snm import GEstimationSNM
    from zepid.causal.doublyrobust import AIPTW, TMLE

    df = load_sample_data(timevary=False)
    df = df.drop(columns=['cd4_wk45'])
    df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
    df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)


Which estimator should I use?
====================================
What estimator to use is an important question. Unfortunately, my answer is that it depends. Review the following list
of estimators to help you decide. Afterwards, I would recommend the following process.

First, what are you trying to estimate? Depending on what you want to estimate (the estimand), some estimators don't
make sense to use. For example, if you wanted to estimate the marginal causal effect comparing all treated versus all
untreated, then you wouldn't want to use g-estimation of structural nested models. G-estimation, as detailed below,
targets something slightly different than the target estimand. However, if you were interested in average causal effect
within defined strata, then g-estimation would be a good choice. Your causal question can (*and should*) narrow down
the list of potential estimators

Second, does your question of interest require something not available for all methods? This can also narrow down
estimators, at least ones currently available. For example, currently only `TimeFixedGFormula` allows for stochastic
treatments. See the tutorials on `Python for Epidemiologists <https://github.com/pzivich/Python-for-Epidemiologists/>`_
for further details on what each estimator can do.

Lastly, if there are multiple estimators to use, then use them all. Each has different advantages/disadvantages that
don't necessarily make one better than the other. If all the estimators provide similar answers, that can generally be
taken as a good sign. It builds some additional confidence in your results. If there are distinctly different results
across the estimators, that means that at least one assumption is being substantively broken somewhere. In these
situations, I would recommend the doubly robust estimators because they make less restrictive parametric modeling
assumptions. However, you should note the lack of agreement between estimators.

Binary Outcome
==============================================
To begin, we are interested in the average causal effect of anti-retroviral therapy (ART) on 45-week risk of death.

.. math::

    ACE = \Pr(Y^{a=1}) - \Pr(Y^{a=1})

where :math:`Y^{a}` indicates the potential outcomes under treatment :math:`a`. Unfortunately, we cannot observe these
potential outcomes (or counterfactuals after they occur). We stuck with our observational data, so we need to make
some additional assumptions to go from

.. math::

    \Pr(Y | A=1) - \Pr(Y | A=0)

to

.. math::

    \Pr(Y^{a=1}]) - \Pr(Y^{a=1})

We will assume conditional mean exchangeability, causal consistency, and positivity. These assumptions allow us to go
from our observed data to potential outcomes. See
`Hernan and Robins <https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/>`_ for further details on these
assumptions and these methods in general. We will assume conditional exchangeability by age (continuous),
gender (male / female), baseline CD4 T-cell count (continuous), and baseline detectable viral load (yes / no)
throughout. The data set we will use is a simulated data set that comes with *zEpid*

Our set of confounders for conditional exchangeability is quite large and includes some continuous variables. Therefore,
we will use parametric models (for the most part). As a result, we assume that our models are correctly specified, in
addition to the above assumptions.

Unadjusted Risk Difference
----------------------------------------
The first option is the unadjusted risk difference. We can calculate this by

.. code::

    rd = RiskDifference()
    rd.fit(df, exposure='art', outcome='dead')
    rd.summary()

By using this measure as our average causal effect, we are assuming that there is no confounding variables. However,
this is an unreasonable assumption for our observational data. However, the `RiskDifference` gives us some useful
information. In the summary, we find `LowerBound` and `UpperBound`. These bounds are the Frechet probability bounds.
Without needing the assumption of exchangeability. This is a good check. All methods below should produce values
that are within these bounds.

Therefore, the Frechet bounds allow for partial identification of the causal effect. We narrowed the range of possible
values from two unit width (-1 to 1) to unit width (-0.87 to 0.13). However, we don't have point identification. The
following methods allow for point identification under the assumption of conditional exchangeability.

Our unadjusted estimate is -0.05 (-0.13, 0.04), which we could cautiously interpret as: ART is associated with a 4.5%
point reduction (95% CL: -0.128, 0.038) in the probability of death at 45-weeks.

Parametric g-formula
----------------------------------------
The parametric g-formula allows us to estimate the average causal effect of ART on death by specifying an outcome
model. From our outcome model, we predict individuals counterfactual outcomes under our treatment plans and marginalize
over these predicted counterfactuals. This allows us to estimate the marginal risk under our treatment plan of
interest.

To estimate the parametric g-formula, we can use the following code

.. code::

    g = TimeFixedGFormula(df, exposure='art', outcome='dead')
    g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    # Estimating marginal effect under treat-all plan
    g.fit(treatment='all')
    r_all = g.marginal_outcome

    # Estimating marginal effect under treat-none plan
    g.fit(treatment='none')
    r_none = g.marginal_outcome

    riskd = r_all - r_none
    print('RD:', riskd)

which gives us an estimated risk difference of -0.076. To calculate confidence intervals, we need to use a bootstrapping
procedure. Below is an example that uses bootstrapped confidence limits.

.. code::


    rd_results = []
    for i in range(1000):
        s = dfs.sample(n=df.shape[0],replace=True)
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

In my run (your results may differ), the estimate 95% confidence limits were -0.142, -0.010. We could interpret our
results as; the 45-week risk of death when everyone was treated with ART at enrollment was 7.6% points
(95% CL: -0.142, -0.010) lower than if no one had been treated with ART at enrollment. For further details and
examples of other usage of this estimator see this
`tutorial <https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments/1_g-formula.ipynb>`_

Inverse probability of treatment weights
----------------------------------------
For the g-formula, we specified the outcome model. Another option is to specify a treatment / exposure model.
Specifically, this model predicts the probability of treatment, sometimes called propensity scores. From these
propensity scores, we can calculate inverse probability of treatment weights.

Below is some code to calculate our stabilized inverse probability of treatment weights for ART.

.. code::

    iptw = IPTW(df, treatment='art')
    iptw.regression_models('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
    iptw.fit()

After calculating the weights, there are a variety of diagnostics available to check the calculated weights. See the
below referenced tutorial for further details and examples. After calculating the weights, we can
estimate a marginal structural model. For this analysis, our marginal structural model looks like the following

.. math::

    \Pr(Y | A) = \alpha_0 + \alpha_1 A

While this model looks like a crude regression model, we are fitting it with the weighted data. The weights make it
such that there is no confounding in our pseudo-population. We will use `statsmodels` GEE to fit our marginal structural
model. The reason we use GEE is to correctly estimate the standard error. By weighting our population, we build in some
correlation between our observations. We need to account for this. While GEE does account for this, our confidence
intervals will be somewhat overly conservative.

.. code::

    ind = sm.cov_struct.Independence()
    f = sm.families.family.Binomial(sm.families.links.identity)
    linrisk = smf.gee('dead ~ art', df['id'], df,
                      cov_struct=ind, family=f, weights=iptw.Weight).fit()

    print('RD = ', np.round(linrisk.params[1], 3))
    print('95% CL:', np.round(linrisk.conf_int().iloc[1][0], 3),
          np.round(linrisk.conf_int().iloc[1][1], 3))

My results were fairly similar to the g-formula (RD = -0.082; 95% CL: -0.156, -0.007). We would interpret this in a
similar way: the 45-week risk of death when everyone was treated with ART at enrollment was 8.2% points
(95% CL: -0.156, -0.007) lower than if no one had been treated with ART at enrollment.

Both of the above formulas drop missing data. We have some missing outcome data. To account for data that is missing
at random, inverse probability of missing weights can be stacked together with IPTW. For further details and examples
see this
`tutorial <https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments/3_IPTW_intro.ipynb>`_

Augmented inverse probability weights
----------------------------------------
As you read through the previous estimators, you may have thought "is there a way to combine these approaches?" The
answer is yes! Augmented inverse probability of treatment weights require you to specify both a treatment model
(pi-model) and an outcome model (Q-model). But why would you want to specify two models? Well, by specifying both and
merging them, AIPTW becomes doubly robust. This means that as long as one model is correct, our estimate will be
unbiased on average. Essentially, we get two attempts to get our models correct.

We can calculate the AIPTW estimator through the following code

.. code::

    aipw = AIPTW(df, exposure='art', outcome='dead')

    # Treatment model
    aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    # Outcome model
    aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    # Calculating estimate
    aipw.fit()

    # Printing summary results
    aipw.summary()

In the printed results, we have an estimated risk difference of -0.085 (95% CL: -0.155, -0.015). Confidence intervals
come from the efficient influence curve. You can also bootstrap confidence intervals. For the risk ratio, you will
need to bootstrap the confidence intervals currently. Our results can be interpreted as: the 45-week risk of death
when everyone was treated with ART at enrollment was 8.5% points (95% CL: -0.155, -0.015) lower than if no one
had been treated with ART at enrollment.

For further details and examples see this
`tutorial <https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments/5_AIPTW_intro.ipynb>`_

Targeted maximum likelihood estimation
----------------------------------------
For AIPTW, we merged IPW and the g-formula. The targeted maximum likelihood estimator (TMLE) is another variation on
this procedure. TMLE uses a targeting step to update the estimate of the average causal effect. This approach is
doubly robust but keeps some of the nice properties of plug-in estimators (like the g-formula). In general, TMLE will
likely have narrower confidence intervals than AIPTW.

Below is code to generate the average causal effect of ART on death using TMLE. We specify an additional model compared
to AIPTW. TMLE has a baked in missing outcome procedure. We will take advantage of that to instead assume that
data is missing completely at random, conditional on ART, gender, age, CD4 T-cell count, and diagnosed viral load

.. code::

    tmle = TMLE(df, exposure='art', outcome='dead')

    # Specify treatment model
    tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    # Specifying missing outcome data model
    tmle.missing_model('art + male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    # Specifying outcome model
    tmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    # TMLE estimation procedure
    tmle.fit()
    tmle.summary()

Using TMLE, we estimate a risk difference of -0.080 (95% CL: -0.153, -0.008). We can interpret this as: the 45-week
risk of death when everyone was treated with ART at enrollment was 8.0% points (95% CL: -0.153, -0.008) lower than if
no one had been treated with ART at enrollment.

TMLE can also be paired with machine learning algorithms, particularly super-learner. The use of machine learning with
TMLE means we are making less restrictive parametric assumptions than all the model described above. For further
details, using super-learner / sklearn with TMLE, and examples see this
`tutorial <https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments/7_TMLE_intro.ipynb>`_

G-estimation of SNM
----------------------------------------
The final method I will review is g-estimation of structural nested mean models (SNM). G-estimation of SNM is distinct
from all of the above estimation procedures. The g-formula, IPTW, AIPTW, and TMLE all estimated the average causal
effect of ART on mortality comparing everyone treated to everyone untreated. G-estimation of SNM estimate the average
causal effect within levels of the confounders, *not* the average causal effect in the population. Therefore, if no
product terms are included in the SNM if there is effect measure modification, then the SNM will be biased due to model
misspecification. SNM are useful for learning about effect modification.

To first demonstrate g-estimation, we will assume there is no effect measure modification. For g-estimation, we specify
two models; the treatment model and the structural nested model. The treatment model is the same format as the treatment
model for IPTW / AIPTW / TMLE. The structural nested model states the interaction effects we are interested in. Since
we are assuming no interaction, we only put the treatment variable into the model.

.. code::

    snm = GEstimationSNM(df, exposure='art', outcome='dead')

    # Specify treatment model
    snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    # Specify structural nested model
    snm.structural_nested_model('art')

    # G-estimation
    snm.fit()
    snm.summary()

    psi = snm.psi
    print('Psi:', psi)

Similarly, we need to bootstrap our confidence intervals

.. code::


    psi_results = []
    for i in range(500):
        dfs = df.sample(n=df.shape[0],replace=True)
        snm = GEstimationSNM(dfs, exposure='art', outcome='dead')
        snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        snm.structural_nested_model('art')
        snm.fit()
        psi_results.append(snm.psi)


    se = np.std(psi_results)
    print('95% LCL', psi - 1.96*se)
    print('95% UCL', psi + 1.96*se)

Overall, the SNM results are similar to the other models (RD = -0.088; 95% CL: -0.172, -0.003). Instead, we interpret
this estimate as: the 45-week risk of death when everyone was treated with ART at enrollment was 8.8% points
(95% CL: -0.172, -0.003) lower than if no one had been treated with ART at enrollment across all confounder strata.

SNM can be expanded to include additional terms. Below is code to do that. For this SNM, we will assess if there is
modification by gender

.. code::

    snm = GEstimationSNM(df, exposure='art', outcome='dead')
    snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    snm.structural_nested_model('art + art:male')
    snm.fit()
    snm.summary()

The 45-week risk of death when everyone was treated with ART at enrollment was 17.6% points lower than if no one had
been treated with ART at enrollment, among women. Among men, risk of death with ART treatment at enrollment was
6.8% points lower compared to no treatment.

Remember, g-estimation of SNM is distinct from these other methods and targets a different estimand. It is a great
method to consider when you are interested in effect measure modification.

Summary
----------------------------------------
Below is a figure summarizing the results across methods.

.. image:: images/zepid_effrd.png

As we can see, all the methods provided fairly similar answers, even the misspecified structural nested model. This
will not always be the case. Differences in model results may indicate parametric model misspecification. In those
scenarios, it may be preferable to use a doubly robust estimator.

Additionally, for simplicity we dropped all missing outcome data. We made the assumption that outcome data was missing
complete at random, a strong assumption. We could relax this assumption by pairing the above methods with
inverse-probability-of-missing-weights or using built-in methods (like `TMLE`'s `missing_model`)

Continuous Outcome
==============================================
In the previous example we focused on a binary outcome, death. In this example, we will repeat the above procedure but
focus on the 45-week CD4 T-cell count. This can be expressed as

.. math::

    E[Y^{a=1}] - E[Y^{a=0}]

For illustrative purposes, we will ignore the implications of competing risks (those dying before week 45 cannot have
a CD4 T-cell count). We will start by restricting our data to only those who are not missing a week 45 T-cell count.
In an actual analysis, you wouldn't want to do this

.. code::

    df = load_sample_data(timevary=False)
    dfs = df.drop(columns=['dead']).dropna()

With our data loaded and restricted, let's compare the estimators. Overall, the estimators are pretty much
the same as the binary case. However, we are interested in estimating the average treatment effect instead. Most of the
methods auto-detect binary or continuous data in the background. Additionally, we will assume that CD4 T-cell count
is appropriately fit by a normal-distribution. Poisson is also available

Parametric g-formula
----------------------------------------
The parametric g-formula allows us to estimate the average causal effect of ART on death by specifying an outcome
model. From our outcome model, we predict individuals counterfactual outcomes under our treatment plans and marginalize
over these predicted counterfactuals. This allows us to estimate the marginal risk under our treatment plan of
interest.

To estimate the parametric g-formula, we can use the following code

.. code::

    g = TimeFixedGFormula(df, exposure='art', outcome='cd4_wk45', outcome_type='normal')
    g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    g.fit(treatment='all')
    r_all = g.marginal_outcome

    g.fit(treatment='none')
    r_none = g.marginal_outcome
    ate = r_all - r_none

    print('ATE:', ate)

To calculate confidence intervals, we need to use a bootstrapping procedure. Below is an example that uses
bootstrapped confidence limits.

.. code::


    ate_results = []
    for i in range(1000):
        s = df.sample(n=df.shape[0],replace=True)
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

In my run (your results may differ), the estimate 95% confidence limits were 158.70, 370.54.
We can interpret this estimate as: the mean 45-week CD4 T-cell count if everyone had been given ART at enrollment
was 264.62 (95% CL: 158.70, 370.54) higher than the mean if everyone has not been given ART at baseline.

Inverse probability of treatment weights
----------------------------------------
Since inverse probability of treatment weights rely on specification of the treatment-model, there is no difference
between the weight calculation and the binary outcome. This is also because we assume the same sufficient adjustment
set. We will estimate new weights since there is a different missing data pattern. Below is code to estimate our
weights

.. code::

    ipw = IPTW(df, treatment='art')
    ipw.regression_models('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    ipw.fit()
    df['iptw'] = ipw.Weight

After we calculate the weights, we can then fit the marginal structural model

.. code::

    m = smf.gee('cd4_wk45 ~ art', df.index, df,
                cov_struct=sm.cov_struct.Independence(),
                family=sm.families.family.Gaussian(),
                weights=df['iptw']).fit()
    print(m.summary())

Our marginal structural model estimates 222.56 (95% CL: 114.67, 330.46). We can interpret this estimate as: the mean
45-week CD4 T-cell count if everyone had been given ART at enrollment was 222.56 (95% CL: 114.67, 330.46) higher than
the mean if everyone has not been given ART at baseline.

Augmented inverse probability weights
----------------------------------------
Similarly to the binary outcome case, AIPTW follows the same recipe to merge IPTW and g-formula estimates. We can
calculate the AIPTW estimator through the following code

.. code::

    aipw = AIPTW(df, exposure='art', outcome='cd4_wk45')
    aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    aipw.fit()
    aipw.summary()

AIPTW produces a similar estimate to the marginal structural model (ATE = 228.22; 95% CL: 115.33, 341.11). We can
interpret this estimate as: the mean 45-week CD4 T-cell count if everyone had been given ART at enrollment was
228.22 (95% CL: 115.33, 341.11) higher than the mean if everyone has not been given ART at baseline.

Targeted maximum likelihood estimation
----------------------------------------
TMLE also supports continuous outcomes and is similarly doubly robust. Below is code to estimate TMLE for a continuous
outcome.

.. code::

    tmle = TMLE(df, exposure='art', outcome='cd4_wk45')
    tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    tmle.fit()
    tmle.summary()

Our results are fairly similar to the other models. The mean 45-week CD4 T-cell count if everyone had been given ART
at enrollment was 228.35 (95% CL: 118.97, 337.72) higher than the mean if everyone has not been given ART at baseline.

G-estimation of SNM
----------------------------------------
Recall that g-estimation of SNM estimate the average causal effect within levels of the confounders, *not* the average
causal effect in the population. Therefore, if no product terms are included in the SNM if there is effect measure
modification, then the SNM will be biased due to model misspecification.

For illustrative purposes, I will specify a one-parameter SNM. Below is code to estimate the model

.. code::

    snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
    snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    snm.structural_nested_model('art')
    snm.fit()
    snm.summary()

Overall, the SNM results are similar to the other models (ATE = 266.56). Instead, we interpret
this estimate as: the mean 45-week CD T-cell count when everyone was treated with ART at enrollment was 266.56
higher than if no one had been treated with ART at enrollment across all confounder strata.

SNM can be expanded to include additional terms. Below is code to do that. For this SNM, we will assess if there is
modification by gender

.. code::

    snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
    snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    snm.structural_nested_model('art + art:male')
    snm.fit()
    snm.summary()

The mean 45-week CD4 T-cell count when everyone was treated with ART at enrollment was 258.73 higher than if no one had
been treated with ART at enrollment, among women. Among men, CD4 T-cell count with ART treatment at enrollment was
268.28 higher compared to no treatment.

Remember, g-estimation of SNM is distinct from these other methods and targets a different estimand. It is a great
method to consider when you are interested in effect measure modification.

Summary
----------------------------------------
Below is a figure summarizing the results across methods.

.. image:: images/zepid_ate.png

There was some difference in results between outcome models and treatment models. Specifically, the g-formula and IPTW
differ. AIPTW and TMLE are similar to IPTW. This may indicate substantive misspecification of the outcome model. This
highlights why you may consider using multiple models.

Additionally, for simplicity we dropped all missing outcome data. We made the assumption that outcome data was missing
complete at random, a strong assumption. We could relax this assumption by pairing the above methods with
inverse-probability-of-missing-weights or using built-in methods (like `TMLE`'s `missing_model`)

Causal Survival Analysis
========================
Previously, we focused on the risk of death at 45-weeks. However, we may be interested in conducting a time-to-event
analysis. For the following methods, we will focus on treatment at baseline. Specifically, we will not allow the
treatment to vary over time. For methods that allow for time-varying treatment, see the tutorial for time-varying
exposures.

For the following analysis, we are interested in the average causal effect of ART treatment at baseline compare to no
treatment. We will compare the parametric g-formula and IPTW. The parametric g-formula is further described in Hernan's
"The hazards of hazard ratio" paper. For the analysis in this section, we will get a little help from the `lifelines`
library. It is a great library with a variety of survival models and procedures. We will use the `KaplanMeierFitter`
function to estimate risk function

Parametric g-formula
----------------------------------------

Inverse probability of treatment weights
----------------------------------------


Summary
----------------------------------------
Currently, only these two options are available. I plan on adding further functionalities in future updates
