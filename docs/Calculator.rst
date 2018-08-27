.. image:: images/zepid_logo.png

-------------------------------------

Calculator
=====================================

*zEpid* includes some basic calculations which can be called directly from the ``calc`` branch. Functions within this
branch are generally used in other functions. There were substantial changes to function names in version 0.3.0 and
onward. All below documentation reflects version 0.3.0 and onward. For previous versions, please check previous
documentation

Measure of Association/Effect
'''''''''''''''''''''''''''''''''

Basic summary measures can be calculated directly when using *zEpid*. To obtain the following measures of association/effect, you can  directly call the following calculations

Risk ratio

.. code:: python

   import zepid as ze
   ze.calc.risk_ratio(12,25,193,253)

Which will return the following

.. code::

   +-----+-------+-------+
   |     |   D=1 |   D=0 |
   +=====+=======+=======+
   | E=1 |    12 |    25 |
   +-----+-------+-------+
   | E=0 |   193 |   253 |
   +-----+-------+-------+
   ----------------------------------------------------------------------
   Exposed
   Risk: 0.324,  95.0% CI: (0.171, 0.477)
   Unexposed
   Risk: 0.433,  95.0% CI: (0.387, 0.479)
   ----------------------------------------------------------------------
   Relative Risk: 0.749
   95.0% two-sided CI: (0.465 , 1.208)
   Confidence Limit Ratio:  2.596
   Standard Error:  0.243
   ----------------------------------------------------------------------

Similarly the following estimates can be generated as follows:

Risk Difference

.. code:: python

   ze.calc.risk_difference(12,25,193,253)

Number needed to Treat

.. code:: python

   ze.calc.number_needed_to_treat(12,25,193,253)

Odds Ratio

.. code:: python

   ze.calc.odds_ratio(12,25,193,253)

Incidence Rate Difference

.. code:: python

   ze.calc.incidence_rate_difference(12,25,1093,2310)

.. code::

   +-----+-------+---------------+
   |     |   D=1 |   Person-time |
   +=====+=======+===============+
   | E=1 |    12 |          1093 |
   +-----+-------+---------------+
   | E=0 |    25 |          2310 |
   +-----+-------+---------------+
   ----------------------------------------------------------------------
   Exposed
   Incidence rate: 0.011, 95.0% CI: (0.005, 0.017)
   Unexposed
   Incidence rate: 0.011, 95.0% CI: (0.007, 0.015)
   ----------------------------------------------------------------------
   Incidence Rate Difference: 0.0
   95.0% two-sided CI: ( -0.007 ,  0.008 )
   Confidence Limit Difference:  0.015
   Standard Error:  0.004
   ----------------------------------------------------------------------

Incidence Rate Ratio

.. code:: python

   ze.calc.incidence_rate_ratio(12,25,1093,2310)

Attributable Community Risk

.. code:: python

   ze.calc.attributable_community_risk(12,25,193,253)


Population Attributable Fraction

.. code:: python

   ze.calc.population_attributable_fraction(12,25,193,253)


Test Calculations
'''''''''''''''''''''''''''''''''

Aside from measures of association, *zEpid* also supports some calculations regarding sensitivity and specificity. Using set sensitivity / specificity / prevalence, either the positive predictive value or the negative predictive value can be generated as follows

.. code:: python

   ze.calc.ppv_conv(sensitivity=0.7,specificity=0.9,prevalence=0.1)

   ze.calc.npv_conv(sensitivity=0.7,specificity=0.9,prevalence=0.1)


Additionally, there is a function which allows comparisons of the relative costs of a screening program. The screening program compares two extremes (everyone is considered as test positive, everyone is considered as test negative) and compares them to the set sensitivity / specificity of the screening criteria

.. code:: python

   ze.calc.screening_cost_analyzer(cost_miss_case=2,cost_false_pos=1,prevalence=0.1,sensitivity=0.7,specificity=0.9)


Which returns the following results


.. code::

   ----------------------------------------------------------------------
   NOTE: When calculating costs, be sure to consult experts in health
   policy or related fields. Costs should encompass more than only 
   monetary costs, like relative costs (regret, disappointment, stigma, 
   disutility, etc.)
   ----------------------------------------------------------------------
   Treat everyone as Test-Negative
   Total relative cost:		 2000.0
   Per Capita relative cost:	 0.2
   ----------------------------------------------------------------------
   Treat everyone as Test-Positive
   Total relative cost:		 9000.0
   Per Capita relative cost:	 0.9
   ----------------------------------------------------------------------
   Treating by Screening Test
   Total relative cost:		 1500.0
   Per Capita relative cost:	 0.15
   ----------------------------------------------------------------------
   ----------------------------------------------------------------------

From these results, we would conclude that our test is a cost-effective strategy.


Other calculations
'''''''''''''''''''''''''''''''''

Some of the other available calculations include

Counternull p-values

.. code:: python

   ze.calc.counternull_pvalue(estimate=0.1,lcl=-0.01,ucl=0.2)

Converting odds to proportions

.. code:: python

   ze.calc.odds_to_probability(1.1)


Converting proportions to odds

.. code:: python
 

   ze.calc.probability_to_odds(0.2)

If you have additional items you believe would make a good addition to the calculator functions, or *zEpid* in general, please reach out to us on GitHub