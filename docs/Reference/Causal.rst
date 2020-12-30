Causal
======
Documentation for each of the causal inference methods implemented in zEpid

Causal Diagrams
---------------------------

.. currentmodule:: zepid.causal.causalgraph.dag

.. autosummary::
   :toctree: generated/

   DirectedAcyclicGraph


Inverse Probability Weights
---------------------------

.. currentmodule:: zepid.causal.ipw.IPTW

.. autosummary::
   :toctree: generated/

   IPTW
   StochasticIPTW

.. currentmodule:: zepid.causal.ipw.IPMW

.. autosummary::
   :toctree: generated/

   IPMW

.. currentmodule:: zepid.causal.ipw.IPCW

.. autosummary::
   :toctree: generated/

   IPCW


Time-Fixed Treatment G-Formula
------------------------------

.. currentmodule:: zepid.causal.gformula.TimeFixed

.. autosummary::
   :toctree: generated/

   TimeFixedGFormula
   SurvivalGFormula

Time-Varying Treatment G-Formula
--------------------------------

.. currentmodule:: zepid.causal.gformula.TimeVary

.. autosummary::
   :toctree: generated/

   MonteCarloGFormula
   IterativeCondGFormula

Augmented Inverse Probability Weights
-------------------------------------

.. currentmodule:: zepid.causal.doublyrobust.AIPW

.. autosummary::
   :toctree: generated/

   AIPTW

.. currentmodule:: zepid.causal.doublyrobust.crossfit

.. autosummary::
   :toctree: generated/

   SingleCrossfitAIPTW
   DoubleCrossfitAIPTW

Targeted Maximum Likelihood Estimator
-------------------------------------

.. currentmodule:: zepid.causal.doublyrobust.TMLE

.. autosummary::
   :toctree: generated/

   TMLE
   StochasticTMLE

.. currentmodule:: zepid.causal.doublyrobust.crossfit

.. autosummary::
   :toctree: generated/

   SingleCrossfitTMLE
   DoubleCrossfitTMLE

G-estimation of SNM
-------------------

.. currentmodule:: zepid.causal.snm.g_estimation

.. autosummary::
   :toctree: generated/

   GEstimationSNM

Generalizability / Transportability
-----------------------------------

.. currentmodule:: zepid.causal.generalize.estimators

.. autosummary::
   :toctree: generated/

   IPSW
   GTransportFormula
   AIPSW
