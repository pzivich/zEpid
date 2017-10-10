#####################################################################
# MEASURES OF ASSOCIATION (Summary)
#####################################################################

import pandas as pd
import numpy as np
import math

#####################################################################
#Relative Risk (Summary data)
def rr(a,b,c,d,decimal=3):
    '''Calculates the Risk Ratio from count data.
    WARNING: a,b,c,d must be positive numbers.
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    r1=a/(a+b)
    print('Risk exposed:',round(r1,decimal))
    r0=c/(c+d)
    print('Risk unexposed:',round(r0,decimal))
    relrisk = r1/r0
    print('Relative Risk:',round(relrisk,decimal))
    SE=math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)));lnrr=math.log(relrisk);lcl=lnrr-(1.96*SE);ucl=lnrr+(1.96*SE)
    print('95% CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')


#####################################################################
#Risk Difference (Summary data)
def rd(a,b,c,d,decimal=3):
    '''Calculates the Risk Difference from count data.
    WARNING: a,b,c,d must be positive numbers.
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    r1=a/(a+b)
    print('Risk exposed:',round(r1,decimal))
    r0=c/(c+d)
    print('Risk unexposed:',round(r0,decimal))
    riskdiff = r1-r0
    print('Risk Difference:',round(riskdiff,decimal))	
    SE=math.sqrt(((a*b)/((((a+b)**2)*(a+b-1))))+((c*d)/(((c+d)**2)*(c+d-1))));lcl=riskdiff-(1.96*SE);ucl=riskdiff+(1.96*SE)
    print('95% CI: (',round(lcl,decimal),', ',round(ucl,decimal),')')
 
 
#####################################################################
#Interaction Contrast

    #IC = R11 - R10 - R01 - R00
    #Superadditivity: R11 - R00 > IC 

#####################################################################
#Interaction Contrast Ratio

    #ICR = RR11 - RR10 - RR01 + 1


#####################################################################
#Number Needed to Treat (Summary data)
def nnt(a,b,c,d,decimal=3):
    '''Calculates the Number Needed to Treat from count data.
    WARNING: a,b,c,d must be positive numbers
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    ratiod1=a/(a+b)
    ratiod2=c/(c+d)
    riskdiff = ratiod1-ratiod2
    print('Risk Difference: ',round(riskdiff,decimal))
    nnt = 1/(abs(riskdiff))
    print('Number Needed to Treat:',round(nnt,decimal))
    

#####################################################################
#Odds Ratio (Summary data)
def oddr(a,b,c,d,decimal=3):
    '''Calculates the Odds Ratio from count data.
    WARNING: a,b,c,d must be positive numbers
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    or1=a/b
    print('Odds exposed:',round(or1,decimal))
    or0=c/d
    print('Odds unexposed:',round(or0,decimal))
    oddsr=or1/or0
    print('Odds Ratio:',round(oddsr,decimal))
    SE=math.sqrt((1/a)+(1/b)+(1/c)+(1/d));lnor=math.log(oddsr);lcl=lnor-(1.96*SE);ucl=lnor+(1.96*SE)
    print('95% CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')
 

#####################################################################
#Incidence Rate Ratio (Summary data)
def irr(a,b,T1,T2,decimal=3):
    '''Calculates the Incidence Rate Ratio from count data.
    WARNING: a,b,T1,T2 must be positive numbers.
    
    a:
        -count of exposed with outcome
    b:
        -count of unexposed with outcome
    T1:
        -person-time contributed by those who were exposed
    T2:
        -person-time contributed by those who were unexposed
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    irate1=a/T1
    print('Incidence Rate exposed:',round(irate1,decimal))
    irate2=b/T2
    print('Incidence Rate unexposed:',round(irate2,decimal))
    irr = irate1/irate2
    print('Incidence Rate Ratio:',round(irr,decimal))
    SE=math.sqrt((1/a)+(1/b));lnirr=math.log(irr);lcl=lnirr-(1.96*SE);ucl=lnirr+(1.96*SE)
    print('95% CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')
  

#####################################################################
#Incidence Rate Difference (Summary data)
def ird(a,b,T1,T2,decimal=3):
    '''Calculates the Incidence Rate Difference from count data.
    WARNING: a,b,T1,T2 must be positive numbers.
    
    a:
        -count of exposed with outcome
    b:
        -count of unexposed with outcome
    T1:
        -person-time contributed by those who were exposed
    T2:
        -person-time contributed by those who were unexposed
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    rated1=a/T1
    print('Incidence Rate exposed:',round(rated1,decimal))
    rated2=b/T2
    print('Incidence Rate unexposed:',round(rated2,decimal))
    ird = rated1-rated2
    print('Incidence Rate Difference:',round(ird,decimal))	
    SE=math.sqrt((a/((T1)**2))+(b/((T2)**2)));lcl=ird-(1.96*SE);ucl=ird+(1.96*SE)
    print('95% CI: (',round(lcl,decimal),', ',round(ucl,decimal),')')


#####################################################################
#Population Attributable Fraction (Summary data)
def paf(a,b,c,d,decimal=3):
    '''Calculates the Population Attributable Fraction from count data.
    WARNING: a,b,c,d must be positive numbers.
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    '''
    rt=(a+c)/(a+b+c+d)
    r0=c/(c+d)
    paf=(rt-r0)/rt
    print('PAF: ',round(paf,decimal))
