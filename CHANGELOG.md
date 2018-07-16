### Change logs

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