'''Tools for sensitivity analyses. I still need to read Lash & Fox to integrate more tools
for multiple bias analysis. This branch is still very much a work in progress. The goal is 
to simplify sensitivity analyses, in the hopes they become more common in publications

-MonteCarloRR(): generates a corrected RR distribution based on binary confounder
-trapezoidal(): generates a trapezoidal distribution of values
'''


from .Simple import MonteCarloRR
from .distributions import trapezoidal
