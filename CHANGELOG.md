### Change logs

#### v0.4.1:
** MAJOR CHANGES**:

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

#### v0.4.0:
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

``TMLE`` variance formula has been modified to match ``tmle.R`` rather than other resources. This is beneficial for future 
implementation of missing data adjustment. Also would allow for mediation analysis with TMLE (not a priority for me at
this time). 

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

#### v0.3.0
**BIG CHANGES**:

To conform with PEP and for clarity, all association/effect measures on a pandas dataframe are now class statements. This makes them distinct from the summary data calculators. Additionally, it allows users to access any part of the results now, unlike the previous implementation. The SD can be pulled from the corresponds results dataframe. Please see the updated webiste for how to use the class statements.

Name changes within the calculator branch. With the shift of the dataframe calculations to classes, now these functions are given more descriptive names. Additionally, all functions now return a list of the point estimate, SD, lower CL, upper CL. Please see the website for all the new function names

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
Removed histogram option from IPTW in favor of kernel density. Since histograms are easy to generate with matplotlib, just dropped the entire option.

Created causal branch. IPW functions moved inside this branch

Added depreciation warning to the IPW branch, since this will be removed in 0.2 in favor of the causal branch for organization of future implemented methods

Added time-fixed g-formula

Added simple double-robust estimator (based on Funk et al 2011)

#### 0.1.5
Fix to 0.1.4 and since PyPI does not allow reuse of library versions, I had to create new one. Fixes issue with ipcw_prep() that was a pandas error (tried to drop NoneType from columns)

#### 0.1.4
Updates: Added dynamic risk plot

Fixes: Added user option to allow late entries for ipcw_prep()

#### 0.1.3
Updates: added ROC curve generator to graphics, allows user-specification of censoring indicator to ipcw,

#### 0.1.2
Original release. Previous versions (0.1.0, 0.1.1) had errors I found when trying to install via PyPI. I forgot to include the `package` statement in `setup`