"""
_____________________________________________________________________________________________
|                                                                                           |
|       zEpid package: making epidemiology with Python e-z                                  |
|                                                                                           |
| This package is to provide easy to use tools to aid in epidemiological analyses. A full   |
| applied guide to the entire package is available at:                                      |
|       https://github.com/pzivich/Python-for-Epidemiologists                               |
|                                                                                           |
| Current contents include:                                                                 |
|       -Basics tools for pandas dataframes and summary data                                |
|       -Useful graphics                                                                    |
|       -Causal estimation methods                                                          |
|       -Sensitivity analysis tools                                                         |
|___________________________________________________________________________________________|

CONTENTS

Basic calculations (both for summary data and pandas.DataFrame):
    -Risk Ratio, Risk Difference, Number Needed to Treat, Odds Ratio, Incidence Rate Ratio,
     Incidence Rate Difference, Attributable Community Risk, Population Attributable Fraction,
     Interaction Contrast, Interaction Contrast Ratio, Sensitivity, Specificity, Splines

Causal methods:
    -Inverse Probability of Treatment Weights, Inverse Probability of Censoring Weights,
     Inverse Probability of Missing Weights, Time-Fixed G-Formula, Time-Fixed Double Robust
     Estimator

Graphics:
    -Forest Plot, Receiver Operator Curve, Functional Form Plot, P-value Distribution Plot,
     Spaghetti Plot, Dynamic Risk Plot

Sensitivity Analyses:
    -Monte Carlo correction for Risk Ratio, Trapezoidal distribution

Example)
>>>import zepid as ze
>>>ze.load_sample_data(timevary=False)
>>>ze.RiskDiff(df, exposure='art', outcome='dead') #Example of Risk Ratio Calculation

See http://zepid.readthedocs.io/en/latest/ for a full guide through all the package features
"""
from .base import (RiskRatio, RiskDiff, NNT, OddsRatio, IncRateRatio, IncRateDiff, ACR, PAF, IC, ICR,
                   Sensitivity, Specificity, Diagnostics, spline, StandMeanDiff, Table1)
from .datasets import load_sample_data

import zepid.calc
import zepid.graphics
import zepid.causal.gformula
import zepid.causal.ipw
import zepid.causal.doublyrobust
import zepid.sensitivity_analysis

from .version import __version__
