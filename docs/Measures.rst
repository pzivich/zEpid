.. image:: images/zepid_logo.png

-------------------------------------


Measures
'''''''''''''''''''''''''''''''''

*zEpid* can be used to directly calculate association/effect measures directly from a pandas dataframe object. For the following examples, we will load the following dataset (kindly provided by Jess Edwards) that comes with *zEpid*. This data set can be loaded using the following command


.. code:: python

   import zepid as ze
   df = ze.load_sample_data(timevary=False)


Measures of Effect/Association
------------------------------

There are several association measures currently implemented. We can calculate the Risk Ratio through the following code after our data set is loaded

.. code:: python

   ze.RiskRatio(df,exposure='art',outcome='dead')


Which will produce the following output

.. code::

   ======================================================================
   Comparing art=1 to art=0
   ======================================================================
   +-----+-------+-------+
   |     |   D=1 |   D=0 |
   +=====+=======+=======+
   | E=1 |    10 |    69 |
   +-----+-------+-------+
   | E=0 |    82 |   386 |
   +-----+-------+-------+
   ----------------------------------------------------------------------
   Exposed
   Risk: 0.127,  95.0% CI: (0.069, 0.22)
   Unexposed
   Risk: 0.175,  95.0% CI: (0.143, 0.212)
   ----------------------------------------------------------------------
   Risk Ratio: 0.722
   95.0% two-sided CI: (0.392 , 1.332)
   Confidence Limit Ratio:  3.399
   Standard Deviation:  0.312
   ----------------------------------------------------------------------


Other measures currently implemented included risk difference, number needed to treat, and odds ratio, implemented respectively as:

.. code:: python

   ze.RiskDiff(df,exposure='art',outcome='dead')
   ze.NNT(df,exposure='art',outcome='dead')
   ze.OddsRatio(df,exposure='art',outcome='dead')


Additionally, incidence rate measures (ratio and difference) are available as well if the data includes time contributed

.. code:: python

   ze.IncRateRatio(df,exposure='art',outcome='dead',time='t')
   ze.IncRateDiff(df,exposure='art',outcome='dead',time='t')

The following output is produced for the ``IncRateRatio``

.. code::

   ======================================================================
   Comparing art=1 to art=0
   ======================================================================
   +-----+-------+---------------+
   |     |   D=1 |   Person-time |
   +=====+=======+===============+
   | E=1 |    10 |       4077.67 |
   +-----+-------+---------------+
   | E=0 |    82 |      23236.5  |
   +-----+-------+---------------+
   ----------------------------------------------------------------------
   Exposed
   Incidence rate: 0.002, 95.0% CI: (0.001, 0.004)
   Unexposed
   Incidence rate: 0.004, 95.0% CI: (0.003, 0.004)
   ----------------------------------------------------------------------
   Incidence Rate Ratio: 0.695
   95.0% two-sided CI: ( 0.36 ,  1.34 )
   Confidence Limit Ratio:  3.717
   Standard Deviation:  0.335
   ----------------------------------------------------------------------


All of the above examples compared a binary exposure variable. If a discrete variable (for example three exposure levels 0,1,2) is instead specified as the exposure, then two comparisons will be made (1 vs 0, 2 vs 0). The reference category can be specified through the ``reference`` option. Calculations are additionally available for sensitivity and specificity implemented by:

.. code:: python

   ze.Sensitivity(df,test,disease)

   ze.Specificity(df,test,disease)


*Note* : currently, we do not have an example for these functions. The variable names are placeholders only

Other measures included are: population attributable fraction,

.. code:: python

   ze.PAF(df,exposure='art',outcome='dead')


attributable community risk,

.. code:: python

   ze.ACR(df,exposure='art',outcome='dead')


and standard mean difference

.. code:: python

   ze.StandMeanDiff(df,binary='art',continuous='cd40')



Other basic functionalities
------------------------------

Splines
^^^^^^^^^^^^

*zEpid* is able to directly calculate splines for inclusion in spline models. For a continuous variable, the are implemented through ``zepid.spline``. To implement a basic linear spline with three (automatically) determine knots, the following code is used

.. code:: python

   df[['age_lsp0','age_lsp1','age_lsp2']] = ze.spline(df,var='age0')


Instead we can generate a quadratic spline by

.. code:: python

   df[['age_qsp0','age_qsp1','age_qsp2']] = ze.spline(df,var='age0',term=2)


Any higher order spline can be requested by changing the term argument (ex. ``term=3`` produces cubic splines). The number of knots in the spline can be adjusted by specifying the optional  argument ``n_knots``, like the following

.. code:: python

   df[['age_csp0','age_csp1']] = ze.spline(df,var='age0',term=3,n_knots=2)


Furthermore, the user can specify the placement of the knots rather than having them determined
by the function. This is done by specifying the ``knots`` argument. The ``n_knots`` number must be equal to the number of knots specified in ``knots``

.. code:: python

   df[['age_sp30','age_sp45']] = ze.spline(df,var='age0',n_knots=2,knots=[30,45])


All of the previous examples are unrestricted splines. If the tails/ends of the spline deviate quite drastically, then a restricted spline can be specified. *Note* that a restricted spline returns one less column than the number of knots

.. code:: python

   df[['age_rsp0','age_rsp1']] = ze.spline(df,var='age0',n_knots=3,restricted=True)


We will return to the ``spline`` function for graphics guide. Splines are a flexible functional form and we can assess the functional form through ``statsmodels`` results and a ``matplotlib`` graph obtained from ``ze.graphics.func_form_plot``

Table 1
^^^^^^^^^^^^

Are you tired of copying your Table 1 results from raw output to an Excel document? This is something that constantly annoys me and seems like a time waster. In the hopes of making mine (and others') lives easier, I implemented a function that generates a (un)stratified descriptive table with specified summary statistics. The returned ``pandas`` dataframe can be output as a CSV, opened in Excel (or similar software), and final publication edits can be made (relabel columns/rows, set column widths, add lines, etc.). The following command generates a descriptive table

.. code:: python

   columns = ['art','dead','age0','cd40'] #list of columns of interest
   vars_type = ['category','category','continuous','continuous'] #list of variable types
   table = ze.Table1(df,columns,vars_type)
   table.to_csv('table1.csv') #outputting dataframe as a CSV


The default summary statistics for continuous variables is the median/interquartile range. Mean/standard deviation can be specified like the following

.. code:: python

   table = ze.Table1(df,columns,vars_type,continuous_measure='mean')



The two previous examples were unstratified tables. A stratified table can be stratified by categorical variable, specified like the following

.. code:: python

   columns = ['art','age0','cd40']

   vars_type = ['category','continuous','continuous']

   table = ze.Table1(df,columns,vars_type,strat_by='dead')

   table.to_csv('table1_strat.csv')



I *DO NOT* recommend attempting any operations on these generated ``pandas`` dataframes. They are purely generated for copying your results to an Excel document. Unfortunately, you will still need to do all formating and relabelling in Excel (or other software) to get your table 1 publication ready, but this should make life a little bit easier

Interaction Contrasts
^^^^^^^^^^^^^^^^^^^^^^

Lastly, the interaction contract (IC) and interaction contrast ratio (ICR) can be calculated. Both IC and ICR use ``statsmodels`` ``GLM``. The interaction contrast is calculated from a linear risk (binomial - identity GLM) implemented by 

.. code:: python

   ze.IC(df,exposure='art',outcome='dead',modifier='male')

Which produces the following ``statsmodels`` output and the following

.. code:: python

   ==============================================================================
   Dep. Variable:                   dead   No. Observations:                  547
   Model:                            GLM   Df Residuals:                      543
   Model Family:                Binomial   Df Model:                            3
   Link Function:               identity   Scale:                          1.0000
   Method:                          IRLS   Log-Likelihood:                -246.66
   Date:                Mon, 25 Jun 2018   Deviance:                       493.33
   Time:                        20:13:34   Pearson chi2:                     547.
   No. Iterations:                     2   Covariance Type:             nonrobust
   ==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
   ------------------------------------------------------------------------------
   Intercept      0.1977      0.043      4.603      0.000       0.114       0.282
   art           -0.1310      0.077     -1.692      0.091      -0.283       0.021
   male          -0.0275      0.047     -0.585      0.559      -0.120       0.065
   E1M1           0.1015      0.091      1.117      0.264      -0.077       0.280
   ==============================================================================
   ----------------------------------------------------------------------
   Interaction Contrast
   ----------------------------------------------------------------------
   IC:		0.101
   95% CI:		(-0.077, 0.28)
   ----------------------------------------------------------------------


It should be noted that ``statsmodels`` generally produces the following warning. Despite the warning, results are consistent with SAS 9.4

.. code:: python

   DomainWarning: The identity link function does not respect the domain of the Binomial family.


Unlike the IC, the ICR is slightly more complicated to calculate. To obtain the confidence intervals, the delta method or bootstrapping can be used. The default method is the delta method. If bootstrap confidence intervals are requested, be patient. For 

.. code:: python

   ze.ICR(df,exposure='art',outcome='dead',modifier='male')

Resulting in the following output

.. code:: python

   ==============================================================================
   Dep. Variable:                   dead   No. Observations:                  547
   Model:                            GLM   Df Residuals:                      543
   Model Family:                Binomial   Df Model:                            3
   Link Function:                    log   Scale:                          1.0000
   Method:                          IRLS   Log-Likelihood:                -246.66
   Date:                Mon, 25 Jun 2018   Deviance:                       493.33
   Time:                        20:22:53   Pearson chi2:                     547.
   No. Iterations:                     6   Covariance Type:             nonrobust
   ==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
   ------------------------------------------------------------------------------
   Intercept     -1.6211      0.217     -7.462      0.000      -2.047      -1.195
   E1M0          -1.0869      0.990     -1.098      0.272      -3.028       0.854
   E0M1          -0.1499      0.245     -0.612      0.540      -0.630       0.330
   E1M1          -0.3405      0.378     -0.901      0.367      -1.081       0.400
   ==============================================================================
   ----------------------------------------------------------------------
   ICR based on Risk Ratio		Alpha = 0.05
   ICR:		0.51335
   CI:		(-0.30684, 1.33353)
   ----------------------------------------------------------------------


Bootstrapped confidence intervals can be requested by the following

.. code:: python

   ze.ICR(df,exposure='art',outcome='dead',modifier='male',ci='delta',b_sample=500)


The bootstrapped confidence intervals took several seconds to run. This behavior would be expected since 501 GLM models are it in the procedure. Similar confidence intervals are obtained.

If the rare disease assumption is met, a logit model can instead be requested by specifying ``regression='logit'``. If the odds ratio does *NOT* approximate the risk ratio (i.e. the rare disease assumption is violated), then the logit model is invalid. If the logit model is specified, ``statsmodels`` won't produce a ``DomainWarning`` and logit models generally have better convergence.

If you have additional items you believe would make a good addition to the calculator functions, or *zEpid* in general, please reach out to us on GitHub