### Change logs

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