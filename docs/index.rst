.. image:: images/zepid_logo.png
-------------------------------------

zEpid
=====================================

*zEpid* is an epidemiology analysis toolkit, providing easy to use tools for epidemiologists coding in python3. The
purpose of this package is to provide a toolset to make epidemiology e-z. A variety of calculations and plots can be
generated through various functions. Current features include:

-  Basic epidemiology calculations on pandas dataframes (RR, RD, IRD, etc.)
-  Summary measure calculations 
-  Graphics (functional form plots, effect measure plots, ROC curves, etc.)
-  G-computation algorithm
-  Inverse-probability weights
-  Doubly Robust estimators
-  Sensitivity analysis tools

A narrative description of current implemented functionality are described in the corresponding sections. For specific
discussions of listed within

Contents:
-------------------------------------

.. toctree::
  :maxdepth: 3

  Measures
  Calculator
  Graphics
  Causal
  Sensitivity Analyses
  Reference/index

Installation:
------------------------------

Dependencies are from the typical Python data-stack: Numpy, Pandas, Scipy, Statsmodels, and Matplotlib. Addtionally,
it requires Tabulate, so nice looking tables can be easily generated. Install using:

``pip install zepid``

Source code and Issue Tracker
------------------------------

Available on Github `pzivich/zepid <https://github.com/pzivich/zepid/>`_
Please report bugs, issues, and feature extensions there.

For a simplified guide to Python 3.x tailored to epidemiologists, check out my
online guide on GitHub as well:
`pzivich/Python-for-Epidemiologists <https://github.com/pzivich/Python-for-Epidemiologists/>`_

Also feel free to contact us via email (gmail: zepidpy) or Twitter (@zepidpy)
