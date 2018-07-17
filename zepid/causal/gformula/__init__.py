'''Implementations of the parametric g-computation algorithm model, also known as the g-formula. 
Currently, only parametric g-formula is available

TimeFixedGFormula:
    -time-fixed exposure implementation of the g-formula. See Snowden et al. 2011 for an introduction
TimeVaryGFormula:
    -under production...
'''

from .TimeFixed import TimeFixedGFormula
from .TimeVary import TimeVaryGFormula


