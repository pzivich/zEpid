.. image:: images/zepid_logo.png

--------------------------------

zEpid
=====================================

*zEpid* is a Python 3.5+ epidemiology analysis toolkit. The purpose of this library is to make epidemiology e-z to do
in Python. A variety of calculations, estimators, and plots can be implemented. Current features include:

-  Basic epidemiology calculations on pandas Dataframes
  -  Risk ratio, risk difference, number needed to treat, incidence rate ratio, etc.
  -  Interaction contrasts and interaction contrast ratios
  -  Semi-bayes
-  Summary measure calculations from summary data
  -  Risk ratio, risk difference, number needed to treat, incidence rate ratio, etc.
  -  Interaction contrasts and interaction contrast ratios
  -  Semi-bayes
-  Graphics
  -  Functional form plots
  -  Forest plots (effect measure plots)
  -  P-value plots
-  Causal inference
  -  Parametric g-formula
  -  Inverse probability of treatment weights
  -  Augmented inverse probability of treatment weights
  -  Targeted maximum likelihood estimator
  -  Monte-Carlo g-formula
  -  Iterative conditional g-formula
-  Generalizability / Transportability
  -  Inverse probability of sampling weights
  -  G-transport formula
  -  Doubly-robust transport formula
-  Sensitivity analysis tools
  -  Monte Carlo bias analysis

The website contains pages with example analyses to help demonstrate the usage of this library. Additionally, examples
of graphics are displayed. The Reference page contains the full reference documentation for each function currently
implemented. For further guided tutorials of the full range of features available in *zEpid*, check out the following
`Python for Epidemiologists <https://github.com/pzivich/Python-for-Epidemiologists/>`_ tutorials. Additionally, if you
are starting to learn Python, I recommend looking at those tutorials for the basics and some other useful resources.

Contents:
-------------------------------------

.. toctree::
  :maxdepth: 3

  Causal Graphs
  Time-Fixed Exposure
  Time-Varying Exposure
  Generalizability
  Missing Data <Missing data.rst>
  Graphics
  Sensitivity Analyses
  Reference/index
  Chat on Gitter <https://gitter.im/zEpid/community>
  Create a GitHub Issue <https://github.com/pzivich/zEpid/issues>

Installation:
-------------

Dependencies are from the typical Python data-stack: Numpy, Pandas, Scipy, Statsmodels, and Matplotlib. Addtionally,
it requires Tabulate, so nice looking tables can be easily generated. Install using:

``pip install zepid``

Source code and Issue Tracker
-----------------------------

Available on Github `pzivich/zepid <https://github.com/pzivich/zepid/>`_
Please report bugs, issues, and feature extensions there.

Also feel free to contact us via `Gitter <https://gitter.im/zEpid/community>`_ email (gmail: zepidpy)
or on Twitter (@zepidpy)
