#####################################################################
# MEASURES OF ASSOCIATION
#####################################################################

#Functions to calculate various epidemiologic measures of associations directly from a pandas dataframe object

#Importing other necessary packages
import pandas as pd
import numpy as np
import math


#####################################################################
#Relative Risk formula
def RelRisk(exposure,disease,decimal=3):
    '''Estimate of Relative Risk with a 95% Confidence interval. Current structure is based on Pandas crosstab.  
    WARNING: Exposure & Disease must be coded as (1: yes, 0:no). If the table has one column missing, no 
    values will be produced. Only works for binary exposures and outcomes
    
    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    b = table[0][1]
    c = table[1][0]
    d = table[0][0]
    r1 = (a/(a+b))
    r2 = (c/(c+d))
    relrisk = r1/r2
    print('Risk in exposed: ',round(r1,decimal));print('Risk in unexposed: ',round(r2,decimal));print('Relative Risk: ',round(relrisk,decimal))
    SE=math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)));lnrr=math.log(relrisk);lcl=lnrr-(1.96*SE);ucl=lnrr+(1.96*SE)
    print('95% CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')
    print('Interval width (relative precision): ',round(((math.exp(ucl))/(math.exp(lcl))),decimal))
    return relrisk


#####################################################################
#Risk Difference formula
def RiskDiff(exposure,disease,decimal=3):
    '''Estimate of Risk Difference with a 95% Confidence interval. Current structure is based on Pandas crosstab. 
    WARNING: Exposure & Disease must be coded as 1 and 0 for this to work properly (1: yes, 0:no). If the table
    has one column missing, no values will be produced.
    
    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    b = table[0][1]
    c = table[1][0]
    d = table[0][0]
    r1 = (a/(a+b))
    r2 = (c/(c+d))
    riskdiff = r1-r2
    print('Risk in exposed: ',round(r1,decimal));print('Risk in unexposed: ',round(r2,decimal));print('Risk Difference: ',round(riskdiff,decimal))
    SE=math.sqrt(((a*b)/((((a+b)**2)*(a+b-1))))+((c*d)/(((c+d)**2)*(c+d-1))));lcl=riskdiff-(1.96*SE);ucl=riskdiff+(1.96*SE)
    print('95% CI: (',round(lcl,decimal),', ',round(ucl,decimal),')');print('Interval width (relative precision): ',round((ucl-lcl),decimal))
    return riskdiff


#####################################################################
#Number Needed to Treat
def NNT(exposure,disease,decimal=3):
    '''Estimates of Number Needed to Treat. Current structure is based on Pandas crosstab. NNT confidence 
    interval presentation is based on Altman, DG (BMJ 1998).   
    WARNING: Exposure & Disease must be coded as 1 and 0 for this to work properly (1: yes, 0:no). If the 
    table has one column missing, no values will be produced.

    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    b = table[0][1]
    c = table[1][0]
    d = table[0][0]
    r1 = (a/(a+b))
    r2 = (c/(c+d))
    riskdiff = r1-r2
    print('Risk Difference: ',round(riskdiff,decimal))
    NNT = 1/riskdiff
    if riskdiff == 0:
        print('Number Needed to Treat = infinite')
    else:
        if riskdiff > 0:
            print('Number Needed to Harm: ',round(abs(NNT),decimal),'\n')
        if riskdiff < 0:
            print('Number Needed to Treat: ',round(abs(NNT),decimal),'\n')
    SE=math.sqrt(((a*b)/((((a+b)**2)*(a+b-1))))+((c*d)/(((c+d)**2)*(c+d-1))));lcl_rd=(riskdiff-(1.96*SE));ucl_rd=(riskdiff+(1.96*SE));ucl = 1/lcl_rd;lcl = 1/ucl_rd
    if lcl_rd < 0 < ucl_rd:
        print('NNH ',round(abs(lcl),decimal),'to infinity to NNT ',round(abs(ucl),decimal))
    elif 0 < lcl_rd:
        print('NNT ',round(abs(lcl),decimal),' to ',round(abs(ucl),decimal))
    else:
        print('NNH ',round(abs(lcl),decimal),' to ',round(abs(ucl),decimal))
    return NNT
    

#####################################################################
#Odds Ratio formula
def OddsRatio(exposure,disease,decimal=3):
    '''Estimates of Odds Ratio with a 95% Confidence interval. Current structure is based on Pandas crosstab.  
    WARNING: Exposure & Disease must be coded as 1 and 0 for this to work properly (1: yes, 0:no). If the 
    table has one column missing, no values will be produced.
    
    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    b = table[0][1]
    c = table[1][0]
    d = table[0][0]
    o1 = (a/b)
    o2 = (c/d)
    oddsratio = o1/o2
    print('Odds in exposed: ',round(o1,decimal));print('Odds in unexposed: ',round(o2,decimal));print('Odds Ratio: ',round(oddsratio,decimal))
    SE=math.sqrt((1/a)+(1/b)+(1/c)+(1/d));lnor=math.log(oddsratio);lcl=lnor-(1.96*SE);ucl=lnor+(1.96*SE)
    print('95% CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')');print('Interval width (relative precision): ',round((ucl-lcl),decimal))
    return oddsratio


#####################################################################
#Incidence Rate Ratio
def IncRateRatio(exposure,disease,time,decimal=3):
    '''Produces the estimate of the Incidence Rate Ratio with a 95% Confidence Interval. Current 
    structure is based on Pandas crosstab.
    WARNING: Disease must be coded as 1 and 0 (1: yes, 0: no).  If the table has one
    column missing, no values with be produced
    
    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    time:
        -total person-time contributed for each person. Stored as a variable for each row in a pandas dataframe,
         df['time'].
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    c = table[1][0]
    time_a = time.loc[exposure==1].sum()
    time_c = time.loc[exposure==0].sum()
    ir_e = (a/time_a)
    ir_u = (c/time_c)
    irr = ir_e/ir_u
    print('Incidence Rate in exposed: ',round(ir_e,decimal));print('Incidence Rate in unexposed: ',round(ir_u,decimal));print('Incidence Rate Ratio: ',round(irr,decimal))
    SE=math.sqrt((1/a)+(1/c));lnirr=math.log(irr);lcl=lnirr-(1.96*SE);ucl=lnirr+(1.96*SE)
    print('95% CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')
    return irr
    

#####################################################################
#Incidence Rate Difference
def IncRateDiff(exposure, disease, time,decimal=3):
    '''Produces the estimate of the Incidence Rate Difference with a 95% Confidence Interval. Current 
    structure is based on Pandas crosstab.
    WARNING: Disease must be coded as 1 and 0 (1: yes, 0: no).  If the table has one
    column missing, no values with be produced
    
    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    time:
        -total person-time contributed for each person. Stored as a variable for each row in a pandas dataframe,
         df['time'].
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    c = table[1][0]
    time_a = time.loc[exposure==1].sum()
    time_c = time.loc[exposure==0].sum()
    ir_e = (a/time_a)
    ir_u = (c/time_c)
    ird = ir_e-ir_u
    print('Incidence Rate in exposed: ',round(ir_e,decimal));print('Incidence Rate in unexposed: ',round(ir_u,decimal));print('Incidence Rate Difference: ',round(ird,decimal))
    SE=math.sqrt((a/((time_a)**2))+(c/((time_c)**2)));lcl=ird-(1.96*SE);ucl=ird+(1.96*SE)
    print('95% CI: (',round(lcl,decimal),', ',round(ucl,decimal),')')
    return ird
 

#####################################################################
#Attributable Community Risk / Population Attributable Risk
def ACR(exposure,disease,decimal=3):
    '''Produces the estimated Attributable Community Risk (ACR). ACR is also known as Population Attributable 
    Risk. Since this is commonly confused with the population attributable fraction, the name ACR is used to 
    clarify differences in the formulas. Current structure is based on Pandas crosstab. 
    WARNING: Exposure & Disease must be coded as 1 and 0 for this to work properly (1: yes, 0:no). If the table 
    has one column missing, no values will be produced.
    
    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    b = table[0][1]
    c = table[1][0]
    d = table[0][0]
    rt=(a+c)/(a+b+c+d)
    r0=c/(c+d)
    acr=(rt-r0)
    print('ACR: ',round(acr,decimal))


#####################################################################
#Population Attributable Fraction
def PAF(exposure, disease,decimal=3):
    '''Produces the estimated Population Attributable Fraction. Current structure is based on Pandas crosstab. 
    WARNING: Exposure & Disease must be coded as 1 and 0 for this to work properly (1: yes, 0:no). If the table
    has one column missing, no values will be produced.
    
    exposure:
        -exposure variable (column) in pandas dataframe, df['exposure']. Must be coded as binary (0,1) where 1
         is exposed. Variations in coding are not guaranteed to function as expected
    disease:
        -outcome variable (column) in pandas dataframe, df['outcome']. Must be coded as binary (0,1) where 1 is
         the outcome of interest. Variation in coding are not guaranteed to function as expected
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(exposure,disease)
    print(table,'\n')
    a = table[1][1]
    b = table[0][1]
    c = table[1][0]
    d = table[0][0]
    rt=(a+c)/(a+b+c+d)
    r0=c/(c+d)
    paf=(rt-r0)/rt
    print('PAF: ',round(paf,decimal))


