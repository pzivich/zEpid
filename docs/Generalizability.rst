.. image:: images/zepid_logo_small.png

-------------------------------------

Generalizability
'''''''''''''''''''''''''''''''''''''
This section details generalizability and transportability. Throughout this section, our data comes from a randomized
trial. However, these methods can be extended to observational studies. Additionally, we have a random sample from
our target population. Our study sample has information on treatment, outcome, and modifiers. Our target population
sample only has information on modifiers.

*zEpid* comes with a simulated data set for determining generalizability and transportability. Variables included in
this data set are `A` (treatment), `Y` (outcome), `S` (indicator of being in study sample), and `L` `W` (potential
effect measure modifiers).

.. code::

    import numpy as np
    import pandas as pd

    from zepid import load_generalize_data
    from zepid.causal.generalize import IPSW, GTransportFormula, AIPSW

    df = load_generalize_data(False)

You will notice that the data set is essentially a stacked data set of the study sample (`S=1`) and the target
population sample (`S=0`). `A` and `Y` are only observed when `S=1`

Generalizability
================
Generalizability is the concept that our study sample is not a random sample from the population we want to make
inferences about (target population). The concept of generalizability is often referred to as external validity.

For demonstration, consider our simulated trial data to assess the effect of A on Y. While our trial results are
internally valid (correct estimation for our study sample), we are concerned that they are no longer reflective of our
target population. Specifically, we are concerned that the individuals who enrolled in our trial are not a random
sample of our target population. We believe that our study sample and target population are exchangeable (or a
conditional random sample) by observed variables `L` and `W`.

In addition to our trial data, we also collected basic information on the target population (assessed via
non-enrollees in our trial). With this information and assumptions, we will now look at three approaches to estimate
the effect in our target population; inverse probability of sampling weights, g-transport formula, and augmented
inverse probability of sampling weights (doubly robust).

IPSW
--------
Inverse probability of sampling weights work by re-weighting our study sample to be reflective of our target population.
To estimate the risk difference and risk ratio in the target population, we can use the following code

.. code::

    ipsw = IPSW(df, exposure='A', outcome='Y', selection='S', generalize=True)
    ipsw.regression_models('L + W + L:W', print_results=False)
    ipsw.fit()
    ipsw.summary()

Based on the summary output, the target population estimates were RD=0.10, RR=1.38. We would interpret this as; the
probability of Y given everyone in the target population had `A=1` would have been 10% points higher than if everyone
in the target population had `A=0`. For confidence intervals, we would need to use a non-parametric bootstrapping
procedure. However, we need to modify our bootstrapping procedure. Specifically, we need to account for random
variability in our study sample and the random variability in our target population selection.

For confidence intervals, we (1) divided our stacked data set, (2) sample with replacement in each of the data sets,
(3) re-stack the data sets, and (4) recalculate IPSW and the corresponding measures. Below is example code to do that
procedure with 200 resamples

.. code::

    rd = ipsw.risk_difference
    rd_bs = []

    # Step 1: divide data
    dfss = df.loc[df['S'] == 1].copy()
    dftp = df.loc[df['S'] == 0].copy()

    for i in range(200):
        # Step 2: Resample data
        dfs = dfss.sample(n=dfss.shape[0], replace=True)
        dft = dftp.sample(n=dftp.shape[0], replace=True)

        # Step 3: restack the data
        dfb = pd.concat([dfs, dft])

        # Step 4: Estimate IPSW
        ipsw = IPSW(dfb, exposure='A', outcome='Y', selection='S', generalize=True)
        ipsw.regression_models('L + W + L:W', print_results=False)
        ipsw.fit()

        rd_bs.append(ipsw.risk_difference)

    se = np.std(rd_bs, ddof=1)

    print('95% LCL:', np.round(rd - 1.96*se, 3))
    print('95% UCL:', np.round(rd + 1.96*se, 3))

In my run of the bootstrap procedure, I ended up with an estimated 95% confidence interval of (0.01, 0.19).

To account for confounding, this approach can be paired with inverse probability of treatment weights. For confidence
intervals, we would need to add a step to estimate IPTW between steps 2 and 4.

G-transport formula
-------------------
The g-transport formula is an extension of the g-formula for generalizability and transportability. Similar to the
standard parametric g-formula, we fit a parametric regression model predicting the outcome as a function of treatment
(and baseline covariates). From our estimated parametric model, we then predict the potential outcomes under the
treatment strategies for the entire population (study sample *and* target population).

The g-transport formula differs from the g-formula, in that we need to specify all modifiers within the model (and
corresponding interaction terms). If we were only interested in internal validity, our g-formula for our trial data
would only include treatment in the regression model. For the g-transport formula, we now need to include terms in the
model for all effect measure modifiers. Below is example code for the procedure

.. code::

    gtf = GTransportFormula(df, exposure='A', outcome='Y', selection='S', generalize=True)
    gtf.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
    gtf.fit()
    gtf.summary()

Based on the summary output, the target population estimates were RD=0.07, RR=1.22. We would interpret this as; the
probability of Y given everyone in the target population had `A=1` would have been 7% points higher than if everyone
in the target population had `A=0`. For confidence intervals, we would need to use a  similar non-parametric
bootstrapping procedure to IPSW. Below is example code with 200 bootstraps

.. code::

    rd = gtf.risk_difference
    rd_bs = []

    # Step 1: divide data
    dfss = df.loc[df['S'] == 1].copy()
    dftp = df.loc[df['S'] == 0].copy()

    for i in range(200):
        # Step 2: Resample data
        dfs = dfss.sample(n=dfss.shape[0], replace=True)
        dft = dftp.sample(n=dftp.shape[0], replace=True)

        # Step 3: restack the data
        dfb = pd.concat([dfs, dft])

        # Step 4: Estimate IPSW
        gtf = GTransportFormula(dfb, exposure='A', outcome='Y', selection='S', generalize=True)
        gtf.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
        gtf.fit()

        rd_bs.append(gtf.risk_difference)

    se = np.std(rd_bs, ddof=1)
    print('95% LCL:', np.round(rd - 1.96 * se, 3))
    print('95% UCL:', np.round(rd + 1.96 * se, 3))

The 95% confidence intervals for the risk difference were; -0.03, 0.16.

For observational data, the g-transport formula more naturally extends to account for confounding. To correct for
confounding, the confounding terms are included in the parametric regression model (we don't need any outside weights
or calculations). Remember that if there is an effect of treatment, then there must be modification by the confounder
on at least scale (additive / multiplicative). This suggests you want to include as many interaction terms in the
g-transport formula as possible.

AIPSW
------
At this point, you may be wondering which approach is better. Similar to other causal inference methods, there exists
a recipe to combine IPSW and the g-transport formula into a single estimate. This approach is doubly robust, such that
if either the g-transport formula or the IPSW is correctly specified, then our estimate will be unbiased. While I am
unaware of a formal name for this approach, I refer to it as augmented-IPSW.

Similar to AIPTW, AIPSW requires that we specify the g-transport formula and the IPSW models. Below is code for this
procedure

.. code::

    aipw = AIPSW(df, exposure='A', outcome='Y', selection='S', generalize=True)
    aipw.weight_model('L + W_sq', print_results=False)
    aipw.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
    aipw.fit()
    aipw.summary()

Our results are similar to the g-transport formula (RD=0.07 RR=1.23). For confidence intervals, we repeat the same
bootstrapping procedure as before

.. code::

    rd = aipw.risk_difference
    rd_bs = []

    # Step 1: divide data
    dfss = df.loc[df['S'] == 1].copy()
    dftp = df.loc[df['S'] == 0].copy()

    for i in range(200):
        # Step 2: Resample data
        dfs = dfss.sample(n=dfss.shape[0], replace=True)
        dft = dftp.sample(n=dftp.shape[0], replace=True)

        # Step 3: restack the data
        dfb = pd.concat([dfs, dft])

        # Step 4: Estimate IPSW
        aipw = AIPSW(dfb, exposure='A', outcome='Y', selection='S', generalize=True)
        aipw.weight_model('L + W + L:W', print_results=False)
        aipw.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
        aipw.fit()

        rd_bs.append(aipw.risk_difference)

    se = np.std(rd_bs, ddof=1)
    print('95% LCL:', np.round(rd - 1.96 * se, 3))
    print('95% UCL:', np.round(rd + 1.96 * se, 3))

The 95% CL were -0.02, 0.15 for the risk difference.

To extend AIPSW to observational data, we use both the IPSW approach for observation data and the g-transport formula
approach. For observational data, we need to calculate IPTW for both IPSW and AIPSW approaches.

Transportability
================
Transportability is a related concept. Rather than our study sample not being a random sample from our target
population, our study sample is not part of our target population. As an example, our study on the effect of drug X on
death may have been conducted in the United States, but we want to estimate the effect of drug X on death in Canada.
Since our study sample is not part of the target population, some authors draw a distinction between the two problems.

What this changes for our estimators is who we are marginalizing over. For generalizability, our estimates are
marginalized over the study sample and the random sample of the target population. For transportability, we only
marginalize over the random sample of the target population. Depending on the distribution of effect measure modifiers,
the generalizability and transportability estimates may differ.

Within *zEpid*, the same functions are used, but we set `generalize=False` to use transportability instead. Below are
examples

IPSW
----
IPSW takes a slightly different form for transportability compared to generalizability. Notably, IPSW becomes inverse
*odds* of sampling weights for the transportability problem. Implementation-wise, there is no large difference between
`IPSW` for generalizability and transportability. Below is how to estimate the average causal effect in the target
population

.. code::

    ipsw = IPSW(df, exposure='A', outcome='Y', selection='S', generalize=False)
    ipsw.regression_models('L + W + L:W', print_results=False)
    ipsw.fit()
    ipsw.summary()

The estimates in our target population were RD=0.10 and RR=1.36 (remember the target population is where `S=0`). We can
calculate confidence intervals using the same non-parametric bootstrapping procedure.

G-transport formula
-------------------
The g-transport formula for transportability follows the same procedure as the generalizability approach. However,
instead of marginalizing over the study sample and the target sample, we only marginalize over the target sample.
Code-wise, we only have to change `generalize=False`. Below is example code

.. code::

    gtf = GTransportFormula(df, exposure='A', outcome='Y', selection='S', generalize=False)
    gtf.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
    gtf.fit()
    gtf.summary()

The estimated RD=0.061 and RR=1.20 for our target population (`S=0`). Similarly, we can calculate confidence intervals
via non-parametric bootstrapping.

AIPSW
------
Again, AIPSW is the doubly robust procedure to merge our IPSW and g-transport formula into a singular estimate. It
follows the same approach as IPSW and g-transport formula for the transportability problem. Below is code to estimate
AIPSW

.. code::

    aipw = AIPSW(df, exposure='A', outcome='Y', selection='S', generalize=False)
    aipw.weight_model('L + W + L:W', print_results=False)
    aipw.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
    aipw.fit()
    aipw.summary()

Our estimates for AIPSW were similar to the g-transport formula (RD=0.06, RR=1.20). Confidence intervals can be
calculated using the same non-parametric bootstrap procedure.

Summary
================
Similar to other causal inference methods, each estimator requires different assumptions. Notably, the g-transport
formula requires we specify a more complex model. AIPSW, our doubly robust method, allows us to have 'two chances' to
specify our models correctly. While framed in terms of randomized study sample data, these methods extend to
observational data.

For observational data, you may be stuck with using IPSW. G-transport formula and AIPSW both require that confounders
are measured in both the study sample and the target population. The random sample from the target population (if you
did not collect it) may *not* have information on these variables. Since this information is necessary for the
g-transport formula, neither the g-transport formula nor AIPSW can be estimated.
