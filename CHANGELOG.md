## Change logs

### v0.9.1

Adding support for Python 3.9 and 3.10

Fixing scikit-learn dependency issue

Adding font size optional argument for `EffectMeasurePlot`

Switched testing from Travis CI to GitHub workflows

### v0.9.0
The 0.9.x series drops support of Python 3.5.x. Only Python 3.6+ are now supported. Support has also been added for
Python 3.8

Cross-fit estimators have been implemented for better causal inference with machine learning. Cross-fit estimators 
include `SingleCrossfitAIPTW`, `DoubleCrossfitAIPTW`, `SingleCrossfitTMLE`, and `DoubleCrossfitTMLE`. Currently 
functionality is limited to treatment and outcome nuisance models only (i.e. no model for missing data). These 
estimators also do not accept weighted data (since most of `sklearn` does not support weights)

Super-learner functionality has been added via `SuperLearner`. Additions also include emprical mean (`EmpiricalMeanSL`),
generalized linear model (`GLMSL`), and step-wise backward/forward selection via AIC (`StepwiseSL`). These new 
estimators are wrappers that are compatible with `SuperLearner` and mimic some of the R superlearner functionality.

Directed Acyclic Graphs have been added via `DirectedAcyclicGraph`. These analyze the graph for sufficient adjustment
sets, and can be used to display the graph. These rely on an optional NetworkX dependency.

`AIPTW` now supports the `custom_model` optional argument for user-input models. This is the same as `TMLE` now.

`zipper_plot` function for creating zipper plots has been added. 

Housekeeping: `bound` has been updated to new procedure, updated how `print_results` displays to be uniform, created
function to check missingness of input data in causal estimators, added warning regarding ATT and ATU variance for 
IPTW, and added back observation IDs for `MonteCarloGFormula`

Future plans: `TimeFixedGFormula` will be deprecated in favor of two estimators with different labels. This will more 
clearly delineate ATE versus stochastic effects. The replacement estimators are to be added

### v0.8.2
`IPSW` and `AIPSW` now natively support adjusting for confounding. Both now have the `treatment_model()` function, 
which calculates the inverse probability of treatment weights. How weights are handled in `AIPSW` are updated. They 
are used in both the weight and the outcome models.

`IPSW` and `AIPSW` both add censoring...

`TimeFixedGFormula` has added support for the average treatment effect in the treated (ATT), and average treatment 
effect in the untreated (ATU). 

Improved warnings when the treatment/exposure variable is not included in models that it should be in (such as the 
outcome model or in structural nested models).

Background refactoring for IPTW. `utils.py` now contains a function to calculate inverse probability of treatment 
weights. The function `iptw_calculator` is used by `IPTW`, `AIPTW`, `IPSW`, and `AIPSW` to calculate the weights now

### v0.8.1
Added support for `pygam`'s `LogisticGAM` for TMLE with custom models (Thanks darrenreger!)

Removed warning for TMLE with custom models following updates to Issue #109 I plan on creating a smarter warning
system that flags non-Donsker class machine learning algorithms and warns the user. I still need to think through 
how to do this.

### v0.8.0
`IPTW` had a massive overhaul. It now follows a similar structure to `AIPTW` and other causal inference methods. 
One *major* change is that missing data is dropped before any calculations. Therefore, if missing data was present for
certain types of data, the weights may no longer match with previous versions. While users can still call the weights 
attribute, all the calculations of the ATE are now contained within the `IPTW` class. Future updates with be other 
instances of the IPTW calculations for other methods, like `LongitudinalIPTW` and `SurvivalIPTW`. The major advantage
of this new structure is it removes some of the burden from users on how to apply IPTW to different data structures.

Diagnostic functions have been added to `TimeFixedGFormula`, `AIPTW`, and `TMLE`. The diagnostics have been restructured
for functions contained within a different file rather than function instances within specific classes. This is 
due to diagnostics commonly being shared across functions.

How missing data is handled by`AIPTW` and `IPTW` has been updated. Rather than dropping all missing data, they only drop
missing data for non-outcome variables. This behavior mimics `TMLE`. Additionally, both have gained the `missing_model`
function. This new function calculates inverse probability of censoring weights.

`bound` argument is now available to `IPTW` and `AIPTW` to truncate the predicted probabilities of the g-model. The 
behavior is the same as `TMLE`. `bound` is also available for `missing_model()`.

`IPCW` no longer supports late-entries into the data. The pooled logistic regression model will not correctly accrue 
weights when late entries occur. This is not a problem I have seen reported in the literature, but I have seen it in 
my own simulations. While you can correctly estimate IPCW with time-fixed variables, this is difficult for me to 
detect. Instead, I have `IPCW` not allow late-entries. If users would like to allow late-entries, they would need to 
"extend backwards" observations or they would need to drop the late-entries. I have update the documentation to note
this change.

S-value calculator function has been added. `s_value` returns the correspond transformed p-value into an s-value. See
documentation for details on s-values and how to interpret them.

I have also been moving around background functions. Most notably, the IPTW diagnostics have migrated to the 
`causal/utils.py` branch since these diagnostics are to be used by other causal inference methods. These reformats 
should have no change for users. This is merely maintenance on my end.

### v0.7.2
Labeling fix for `RiskDifference` summary

Adding option to extract standard errors from `TMLE` and `AIPTW`

### v0.7.1
Warning for upcoming change for `IPTW` in v0.8.0. To better align with other causal estimators, `IPTW` will no longer 
only return a vector of weights. Behind the scenes, `IPTW` will be able to estimate the marginal structural model 
and provide the results directly in v0.8.0. `IPTW` will still allow access to the `Weight` column. Other tweaks are 
coming, such as `IPTW` estimators built for different data types. For example, `SurvivalIPTW` for survival data (like
`SurvivalGFormula`).

Stochastic treatments can be estimated with the new `StochasticIPTW` class. This class is different from `IPTW` in that
it provides the estimated mean of the outcome given the treatment plan. For comparisons, multiple versions of treatment
plans need to be specified, calculated, then compared. For confidence intervals, a bootstrap procedure should be used

### v0.7.0
G-estimation of structural nested models (for a single time point) are now available through `GEstimationSNM`. Psi 
parameters can be calculated using a closed form solution or via a `scipy` optimization procedure

Survival analysis g-formula is now implemented with `SurvivalGFormula`. This g-formula implementation is for 
time-to-event data, where the treatment/exposure is determined at baseline. This does not allow for time-varying 
exposures. For time-varying exposures, `MonteCarloGFormula` or `IterativeCondGFormula` should be used instead

`summary()` functions have been updated to provide more information regarding the model

Added a calculator function for Rubin's Rule to merge multiple imputation results. Input is a list of point estimates 
and a list of variance estimates for `rubins_rules()`. This function returns a summary point estimate and summary 
variance

Weighted models are switched from `GEE` to `GLM` when possible. GEE takes extra computation time. GLM provides the 
correct point estimates, but wrong variance. Since I don't need the variance to be correct from most models, I switched 
to GLM. This improves the speed of fitting weighted models. Especially important for bootstrapping procedures

Aligned `exposure` and `outcome` references with the causal functions. All classes now use the same labels for the 
exposure and the outcome column labels.

Updated ReadTheDocs website

### v0.6.1
``AIPTW`` now supports continuous outcomes (normal or Poisson). Format is the same as `TMLE`.

`AIPTW` and `IPTW` now include the optional argument `weights`

Fixed `TMLE` attribute for average treatment effect confidence intervals, from `average_treatment_effect_ic` to
`average_treatment_effect_ci`

Fixed issue in `IPTW` assumption calculations. Depending on when `positivity()` was called, it changed the results of
`plot_love()`. 

### v0.6.0
`MonteCarloGFormula` now includes a separate `censoring_model()` function for informative censoring.
Additionally, I added a low memory option to reduce the memory burden during the Monte-Carlo procedure

``IterativeCondGFormula`` has been refactored to accept only data in a wide format. This allows for me to handle more
complex treatment assignments and specify models correctly. Additional tests have been added comparing to R's `ltmle`

There is a new branch in `zepid.causal`. This is the `generalize` branch. It contains various tools for generalizing
or transporting estimates from a biased sample to the target population of interest. Options available are 
inverse probability of sampling weights for generalizability (`IPSW`), inverse odds of sampling weights for 
transportability (`IPSW`), the g-transport formula (`GTransportFormula`), and doubly-robust augmented inverse 
probability of sampling weights (`AIPSW`)

`RiskDifference` now calculates the Frechet probability bounds

``TMLE`` now allows for specified bounds on the Q-model predictions. Additionally, avoids error when predicted
continuous values are outside the bounded values.

``AIPTW`` now has confidence intervals for the risk difference based on influence curves

``spline`` now uses `numpy.percentile` to allow for older versions of NumPy. Additionally, new function 
`create_spline_transform` returns a general function for splines, which can be used within other functions

Lots of documentation updates for all functions. Additionally, `summary()` functions are starting to be updated. 
Currently, only stylistic changes

#### v0.5.2:
While conducting further testing, I found an error in `AIPTW`. I have since corrected it and added additional tests
to `tests/test_doublyrobust.py`. Please rerun any analyses ran that used `AIPTW`

#### v0.5.1:
Added a fix to ``TMLE`` for machine learning libraries and missing outcome data

### v0.5.0:
Support for Python 3.7 has been added

``AIPW`` has been removed. It has been replaced with ``AIPTW``

``TMLE`` now supports continuous outcomes (normal or Poisson) and allows for missing outcome data to be missing at 
random. This matches more closely to the functionality to R's `tmle`

``IPMW`` allows for monotone missing data.

``MonteCarloRR`` for probabilistic bias analysis allows for random error to be incorporated

#### v0.4.3:

``TimeVaryGFormula`` is separated into ``MonteCarloGFormula`` and ``IterativeCondGFormula``. This change is for 
maintenance of the estimators and to avoid confusion since they are sufficiently distinct. Originally, I was unaware of
the iterative conditional estimator, which is why the original name was based on time-varying g-formula. While they are 
related, it is more confusing to wrap them both in the same class. ``TimeVaryGFormula`` will stick around to v0.6.0. 
Going forward it will be cut. It will not be updated going forward

L'Abbe plots are now supported. These plots are useful for visualizing additive and multiplicative interactions for 
risk estimates. These are valid for either associations or causal effects.

``IPTW.plot_love`` now displays the legend. I have previously not included this in the function (I should have)

``TMLE`` refactored to estimate machine learners via an outside function. Also converts all pd.Series to np.array to 
avoid some unhappiness with sklearn / supylearner models

#### v0.4.2:

**MAJOR CHANGES**:

``TMLE`` defaults to calculate all possible measures (risk difference, risk ratio, odds ratio) rather than individual 
ones

``TimeFixedGFormula`` allows stochastic interventions for binary exposures. For a stochastic intervention, *p* percent 
of the population is randomly treated. This process is repeated *n* times and mean is the marginal outcome. Stochastic
interventions may better align with real-world interventions (often you intervention will **not** be able to treat 
*everyone*). Additionally, conditional probabilities are implemented for stochastic interventions. For example, those 
with *C=1* might be treated randomly at *p*, while those with *C=0* are treated at *q*.

``IPTW.standard_mean_difference`` and ``IPTW.plot_love`` both support categorical variables. Categorical variables must 
be modeled with ``patsy``'s ``C(.)`` keyword. Otherwise, the dummy variables will be treated as binary variables

**MINOR CHANGES**:

Added case-control example data set. ``load_case_control_data()``

Changed rounding in Table 1 generator

#### v0.4.1:
**MAJOR CHANGES**:

``TimeFixedGFormula`` supports Poisson and normal distributed continuous outcomes

``IPTW``'s ``plot_kde`` and ``plot_boxplot`` can plot either the probabilities of treatment or the log-odds

``IPTW`` allows for sklearn or supylearner to generate predicted probabilities. Similar to ``TMLE``

``IPTW`` now allows for Love plot to be generated. These plots are valuable for assessing covariate balance via absolute
standardized mean differences. See Austin & Stuart 2015 for an example. In its current state ``IPTW.plot_love`` is 
"dumb", in the sense that it plots all variables in the model. If you have a quadratic term in the model for a 
continuous variable, it plots both the linear and quadratic terms. However, it is my understanding that you only need 
to look at the linear term. These plots are not quite for publication, rather they are useful for quick diagnostics

``IPTW.standardized_mean_differences`` now calculates for all variables automatically. This is used in the background 
for the ``plot_love``. For making publication-quality Love plots, I would recommend using the returned DataFrame from 
this function and creating a plot manually. *Note* it only returns standardized differeneces, not absolute standardized 
differences. Love plots use the standardized differences.
*WARNING:* standardized differences **only** supports binary or continuous variables. Categorical variables are NOT 
supported. This will be fixed in v0.4.2 update

**MINOR CHANGES**:

Website updated to reflect above changes and correcting errors I had missed on last check

### v0.4.0:
**MAJOR CHANGES**:

``TMLE`` has been modified to estimate the custom user models now, rather than take the input. This better corresponds 
to R's tmle (however, R does the entire process in the background. You must specify for this implementation). The reason
for this major change is that ``LTMLE`` requires an iterative process. The iterative process requires required fitting 
based on predicted values. Therefore, for ``LTMLE`` an unfitted model must be input and repeatedly fit. ``TMLE`` matches
this process.

``TimeVaryGFormula`` supports both Monte Carlo estimation and Sequential Regression (interative conditionals) this 
added approach reduces some concern over model misspecification. It is also the process used by LTMLE to estimate 
effects of interventions. Online documentation has been updated to show how the sequential regression is estimated and
demonstrates how to calculated cumulative probabilities for multiple time points

All calculator functions now return named tuples. The returned tuples can be index via ``returned[0]`` or 
``returned.point_estimate``

Documentation has been overhauled for all functions and at ReadTheDocs

Tests have been added for all currently available functions. 

Travis CI has been integrated for continuous testing

**MINOR CHANGES**:

``AIPW`` drops missing data. Similar to ``TMLE``

``IPTW`` calculation of standardized differences is now the ``stabilized_difference`` function instead of the previously
used ``StandardDifference``. This change is to follow PEP guidelines

The ``psi`` argument has been replaced with ``measure`` in ``TMLE``. The print out still refers to psi. This update is 
to help new users better understand what the argument is for

Better errors for ``IPTW`` and ``IPMW``, when a unstabilized weight is requested but a numerator for the model is 
specified

#### v0.3.2
**MAJOR CHANGES**:

``TMLE`` now allows estimation of risk ratios and odds ratios. Estimation procedure is based on ``tmle.R``

``TMLE`` variance formula has been modified to match ``tmle.R`` rather than other resources. This is beneficial for 
future implementation of missing data adjustment. Also would allow for mediation analysis with TMLE (not a priority 
for me at this time). 

``TMLE`` now includes an option to place bounds on predicted probabilities using the ``bound`` option. Default is to use
all predicted probabilities. Either symmetrical or asymmetrical truncation can be specified.

``TimeFixedGFormula`` now allows weighted data as an input. For example, IPMW can be integrated into the time-fixed 
g-formula estimation. Estimation for weighted data uses statsmodels GEE. As a result of the difference between GLM
and GEE, the check of the number of dropped data was removed.

``TimeVaryGFormula`` now allows weighted data as an input. For example, Sampling weights can be integrated into the 
time-fixed g-formula estimation. Estimation for weighted data uses statsmodels GEE.

**MINOR CHANGES**:

Added Sciatica Trial data set. Mertens, BJA, Jacobs, WCH, Brand, R, and Peul, WC. Assessment of patient-specific 
surgery effect based on weighted estimation and propensity scoring in the re-analysis of the Sciatica Trial. PLOS 
One 2014. Future plan is to replicate this analysis if possible.

Added data from Freireich EJ et al., "The Effect of 6-Mercaptopurine on the Duration of Steriod-induced
Remissions in Acute Leukemia: A Model for Evaluation of Other Potentially Useful Therapy" *Blood* 1963

``TMLE`` now allows general sklearn algorithms. Fixed issue where ``predict_proba()`` is used to generate probabilities 
within ``sklearn`` rather than ``predict``. Looking at this, I am probably going to clean up the logic behind this and 
the rest of ``custom_model`` functionality in the future

``AIPW`` object now contains ``risk_difference`` and ``risk_ratio`` to match ``RiskRatio`` and ``RiskDifference`` 
classes

#### v0.3.1
**MINOR CHANGES**:

TMLE now allows user-specified prediction models (like machine learning models). This is done by setting the option 
argument `custom_model` to a fitted model with the `predict()` function. For a full tutorial (with SuPyLearner), see 
the website.

Updated API for printing model results to the console. All branches have been updated to 
use ```print_results``` now. (Thanks Cameron Davidson-Pilon)

Semi-Bayesian function now calculates a check on the compatibility between the prior and data. It generates a warning 
if a small p-value is detected (p < 0.05). The full information on this check can be read in *Modern Epidemiology* 3rd 
edition pg340.

### v0.3.0
**BIG CHANGES**:

To conform with PEP and for clarity, all association/effect measures on a pandas dataframe are now class statements. 
This makes them distinct from the summary data calculators. Additionally, it allows users to access any part of the 
results now, unlike the previous implementation. The SD can be pulled from the corresponds results dataframe. Please 
see the updated webiste for how to use the class statements.

Name changes within the calculator branch. With the shift of the dataframe calculations to classes, now these 
functions are given more descriptive names. Additionally, all functions now return a list of the point estimate, SD, 
lower CL, upper CL. Please see the website for all the new function names

Addition of Targeted Maximum Likelihood Estimator as zepid.causal.doublyrobust.TMLE

**MINOR CHANGES**:
Added datasets from;

Glaubiger DL, Makuch R, Schwarz J, Levine AS, Johnson RE. Determination
of prognostic factors and their influence on therapeutic results in patients with Ewing's sarcoma. Cancer.
1980;45(8):2213-9 

Keil AP, Edwards JK, Richardson DB, Naimi AI, Cole SR. The parametric g-formula for time-to-event 
data: intuition and a worked example. Epidemiology. 2014;25(6):889-97

Fixed spelling error for dynamic_risk_plot that I somehow missed (previously named dyanmic_risk plot...)

Renamed func_form_plot to functional_form_plot (my abbreviations are bad, and version 0.3.0 should fix all this)

#### 0.2.1

TimeVaryGFormula speed-up: some background optimization to speed up TimeVaryGFormula. Changes include: pd.concat() 
rather than pd.append() each loop . Shuffled around some statements to execute only once rather than multiple times. In 
some testing, I went from 22 seconds to run to 3.4 seconds

#### 0.2.0
**BIG CHANGES**:

IPW all moved to zepid.causal.ipw. zepid.ipw is no longer supported

IPTW, IPCW, IPMW are now their own classes rather than functions. This was done since diagnostics are easier for IPTW 
and the user can access items directly from the models this way.

Addition of TimeVaryGFormula to fit the g-formula for time-varying exposures/confounders

effect_measure_plot() is now EffectMeasurePlot() to conform to PEP

ROC_curve() is now roc(). Also 'probability' was changed to 'threshold', since it now allows any continuous variable for 
threshold determinations

**MINOR CHANGES**:

Added sensitivity analysis as proposed by Fox et al. 2005 (MonteCarloRR)

Updated Sensitivity and Specificity functionality. Added Diagnostics, which calculates
both sensitivity and specificity. 

Updated dynamic risk plots to avoid merging warning. Input timeline is converted to a integer (x100000), merged, then 
back converted

Updated spline to use np.where rather than list comprehension

Summary data calculators are now within zepid.calc.utils

**FUTURE CHANGES**:

All pandas effect/association measure calculations will be migrating from functions to classes in a future version. 
This will better meet PEP syntax guidelines and allow users to extract elements/print results. Still deciding on the 
setup for this... No changes are coming to summary measure calculators (aside from possibly name changes). Intended as 
part of v0.3.0

Addition of Targeted Maximum Likelihood Estimation (TMLE). No current timeline developed

Addition of IPW for Interference settings. No current timeline but hopefully before 2018 ends

Further conforming to PEP guidelines (my bad)

#### 0.1.6
Removed histogram option from IPTW in favor of kernel density. Since histograms are easy to generate with matplotlib, 
just dropped the entire option.

Created causal branch. IPW functions moved inside this branch

Added depreciation warning to the IPW branch, since this will be removed in 0.2 in favor of the causal branch for 
organization of future implemented methods

Added time-fixed g-formula

Added simple double-robust estimator (based on Funk et al 2011)

#### 0.1.5
Fix to 0.1.4 and since PyPI does not allow reuse of library versions, I had to create new one. Fixes issue with 
ipcw_prep() that was a pandas error (tried to drop NoneType from columns)

#### 0.1.4
Updates: Added dynamic risk plot

Fixes: Added user option to allow late entries for ipcw_prep()

#### 0.1.3
Updates: added ROC curve generator to graphics, allows user-specification of censoring indicator to ipcw,

#### 0.1.2
Original release. Previous versions (0.1.0, 0.1.1) had errors I found when trying to install via PyPI. I forgot to 
include the `package` statement in `setup`