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

    from zepid import load_sample_data, spline
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

    print('RD:', r_all - r_none)

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

    print('95% LCL', np.percentile(rd_results,q=2.5))
    print('95% UCL', np.percentile(rd_results,q=97.5))

In my run (your results may differ), the estimate 95% confidence limits were -0.140, -0.002. We could interpret our
results as; the 45-week risk of death when everyone was treated with ART at enrollment was 7.5% points
(95% CL: -0.140, -0.002) lower than if no one had been treated with ART at enrollment. For further details and
examples of other usage of this estimator see this
`tutorial <https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments/1_g-formula.ipynb>`_

Inverse probability of treatment weights
----------------------------------------
For the g-formula, we specified the outcome model. Another option is to specify a treatment / exposure model.
Specifically, this model predicts the probability of treatment, sometimes called propensity scores. From these
propensity scores, we can calculate inverse probability of treatment weights.

Below is some code to calculate our stabilized inverse probability of treatment weights for ART.

.. code::

    iptw = IPTW(df, treatment='art', stabilized=True)
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

My results were fairly similar to the g-formula (RD = -0.082; 95% CL: -0.156, -0.007). Both of the above formulas drop
missing data. We have some missing outcome data. To account for data that is missing at random, inverse probability
of missing weights can be stacked together with IPTW. For further details and examples see this
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
need to bootstrap the confidence intervals currently. For further details and examples see this
`tutorial <https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments/5_AIPTW_intro.ipynb>`_

Targeted maximum likelihood estimation
----------------------------------------
For AIPTW, we merged IPW and the g-formula. The targeted maximum likelihood estimator (TMLE) is another variation on
this. TMLE uses a targeting step to update the estimate of the average causal effect. This approach is doubly robust
but keeps some of the nice properties of plug-in estimators (like the g-formula). In general, TMLE will likely have
narrower confidence intervals than AIPTW.

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

Using TMLE, we estimate a risk difference of -0.080 (95% CL: -0.153, -0.008). TMLE can also be paired with machine
learning algorithms, particularly super-learner. The use of machine learning with TMLE means we are making less
restrictive parametric assumptions than all the model described above. For further details, using super-learner /
sklearn with TMLE, and examples see this
`tutorial <https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments/7_TMLE_intro.ipynb>`_

G-estimation of SNM
----------------------------------------


Summary
----------------------------------------
Below is a figure summarizing the results across methods.

...

As we can see, all the methods provided fairly similar answers, even the misspecified structural nested model. This
will not always be the case. Differences in model results may indicate parametric model misspecification. In those
scenarios, it may be preferable to use a doubly robust estimator.

Continuous Outcome
==============================================
In the previous example we focused on a binary outcome, death. In this example, we will repeat the above procedure but
focus on the 45-week CD4 T-cell count. For illustrative purposes, we will ignore the implications of competing risks
(those dying before week 45 cannot have a CD4 T-cell count. We will start by restricting our data to only those who
are not missing a week 45 T-cell count. In an actual analysis, you wouldn't want to do this

.. code::

    df = load_sample_data(timevary=False)
    dfs = df.drop(columns=['dead']).dropna()

With our data loaded and appropriately restricted, let's compare the estimators. Overall, the estimators are pretty much
the same. However, we are interested in estimating the average treatment effect instead

.. math::

    E[Y^{a=1}] - E[Y^{a=0}]

Parametric g-formula
----------------------------------------


Inverse probability of treatment weights
----------------------------------------

Augmented inverse probability weights
----------------------------------------

Targeted maximum likelihood estimation
----------------------------------------

G-estimation of SNM
----------------------------------------

Summary
----------------------------------------


Survival Analysis
==============================================
Exposure / treatment is fixed at origin

Parametric g-formula
----------------------------------------

Inverse probability of treatment weights
----------------------------------------

Summary
----------------------------------------
