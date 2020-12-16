"""
_____________________________________________________________________________________________
|                                                                                           |
|       zEpid package: making epidemiology with Python e-z                                  |
|                                                                                           |
| This package is to provide easy to use tools to aid in epidemiological analyses. A full   |
| applied guide to the entire package is available at:                                      |
|       https://github.com/pzivich/Python-for-Epidemiologists                               |
|___________________________________________________________________________________________|

CONTENTS

Basic calculations (both for summary data and pandas.DataFrame):
    -Risk Ratio, Risk Difference, Number Needed to Treat, Odds Ratio, Incidence Rate Ratio, Incidence Rate Difference,
     Interaction Contrast, Interaction Contrast Ratio, Sensitivity, Specificity, Splines

Causal methods:
    -Inverse Probability of Treatment Weights, Inverse Probability of Censoring Weights, Inverse Probability of
     Missing Weights, Time-Fixed G-Formula, Monte-Carlo g-formula, Iterative Conditional g-formula,
     Augmented Inverse Probability Weights, Targeted Maximum Likelihood Estimation, Inverse Probability/Odds of
     Sampling Weights, g-transport formula, Augmented Inverse Probability/Odds of Sampling Weights

Graphics:
    -Forest Plot, Receiver Operator Curve, Functional Form Plot, P-value Distribution Plot, Spaghetti Plot, Dynamic
     Risk Plot

Sensitivity Analyses:
    -Monte Carlo correction for Risk Ratio, Trapezoidal distribution

See http://zepid.readthedocs.io/en/latest/ for a full guide through all the package features
"""
from .version import __version__

from .base import (RiskRatio, RiskDifference, NNT, OddsRatio, IncidenceRateRatio, IncidenceRateDifference, Sensitivity,
                   Specificity, Diagnostics, interaction_contrast, interaction_contrast_ratio, spline, table1_generator,
                   create_spline_transform)
from .datasets import (load_sample_data, load_ewing_sarcoma_data, load_gvhd_data, load_sciatica_data,
                       load_leukemia_data, load_longitudinal_data, load_binge_drinking_data, load_case_control_data,
                       load_monotone_missing_data, load_generalize_data, load_zivich_breskin_data)

import zepid.calc

import zepid.graphics

import zepid.sensitivity_analysis

import zepid.superlearner

import zepid.causal.gformula
import zepid.causal.ipw
import zepid.causal.doublyrobust
import zepid.causal.generalize

dags_available = True
try:
    import zepid.causal.causalgraph
except ImportError:
    dags_available = False
