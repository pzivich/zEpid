#####################################################################
# SCREENING
#####################################################################

import pandas as pd
import numpy as np
import math

#Calculations for screening programs

#####################################################################
#Sensitivity
def sensitivity(test,disease,decimal=3):
    '''Generates the calculated sensitivity. Current structure is based on Pandas crosstab.  
    WARNING: Disease & Test must be coded as (1: yes, 0:no). If the table has one column missing, 
    no values will be produced.
    
    test:
        -column of pandas dataframe that indicates the results of the test to detect the outcome. df['test']
         Needs to be coded as binary (0,1), where 1 indicates a positive test for the individual
    disease:
        -column of pandas dataframe that indicates the true outcomes status df['outcome']
         Needs to be coded as binary (0,1), where 1 indicates the individual has the outcome
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(test,disease)
    print(table,'\n')
    a = table[1][1]
    c = table[1][0]
    sens = a/(a+c)
    print('Sensitivity: ',(round(sens,decimal)*100),'%','\n')
    return sens


#####################################################################
#Specificity
def specificity(test,disease,decimal=3):
    '''Generates the calculated specificity. Current structure is based on Pandas crosstab.  
    WARNING: Disease & Test must be coded as (1: yes, 0:no). 
    If the table has one column missing, no values will be produced.
    
    test:
        -column of pandas dataframe that indicates the results of the test to detect the outcome. df['test']
         Needs to be coded as binary (0,1), where 1 indicates a positive test for the individual
    disease:
        -column of pandas dataframe that indicates the true outcomes status df['outcome']
         Needs to be coded as binary (0,1), where 1 indicates the individual has the outcome
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    table = pd.crosstab(test,disease)
    print(table,'\n')
    b = table[0][1]
    d = table[0][0]
    spec = d/(d+b)
    print('Specificity: ',(round(spec,decimal)*100),'%','\n')
    return spec


#####################################################################
#PPV converter
def ppv_conv(sensitivity,specificity,prevalence):
    '''Generates the Positive Predictive Value from designated
    Sensitivity, Specificity, and Prevalence.  
    WARNING: sensitivity/specificity/prevalence cannot be greater than 1
    
    sensitivity:
        -sensitivity of the criteria
    specificity:
        -specificity of the criteria
    prevalence:
        -prevalence of the outcome in the population
    '''
    if ((sensitivity > 1)|(specificity > 1)|(prevalence > 1)):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    else:
        sens_prev = sensitivity*prevalence
        nspec_nprev = (1-specificity)*(1-prevalence)
        ppv = sens_prev / (sens_prev + nspec_nprev)
        return ppv


#####################################################################
#NPV converter
def npv_conv(sensitivity,specificity,prevalence):
    '''Generates the Negative Predictive Value from designated Sensitivity, Specificity, and Prevalence.  
    WARNING: sensitivity/specificity/prevalence cannot be greater than 1
    
    sensitivity:
        -sensitivity of the criteria
    specificity:
        -specificity of the criteria
    prevalence:
        -prevalence of the outcome in the population
    '''
    if ((sensitivity > 1)|(specificity > 1)|(prevalence > 1)):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    else:
        spec_nprev = specificity*(1-prevalence)
        nsens_prev = (1-sensitivity)*prevalence
        npv = spec_nprev / (spec_nprev + nsens_prev)
        return npv


#####################################################################
#Accounting for Misclassifications
def mis_count(countA,totalN,sensitivity,specificity):
    '''The formula is based on  Modern Epidemiology 3rd Edition (Rotham et al.) pg.XXX
    mis_count() totals the amount of those with outcome and not outcome
    removing the potential bias introduced by missclassification.  The
    formula returns A (count of Disease=1) and B (count of Disease=0)
    
    countA:
        -observed count of 
    totalN:
        -observed total amount of subjects
    sensitivity:
        -sensitivity of the criteria used for classification
    specificity:
        -specificity of the criteria used for classification
    '''
    if ((sensitivity > 1)|(specificity > 1)):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    elif ((countA<0)|(totalN<0)):
        raise ValueError('observed count of A and total sample size must be greater than 0')
    elif (countA>totalN):
        raise ValueError('observed count of A cannot be greater than total sample size')
    else:
        A_top = countA - (1-specificity)*totalN
        A_bottom = sensitivity + specificity - 1
        A = A_top / A_bottom
        B = totalN - A
        return A,B



#####################################################################
#Cost of a screening program
def screening_cost_analyzer(cost_miss_case,cost_false_pos,prevalence,sensitivity,specificity,population=10000,decimal=3):
    '''Compares the cost of sensivitiy/specificity of screening criteria to treating the entire population 
    as test-negative and test-positive. The lowest per capita cost is considered the ideal choice.
        
    WARNING: When calculating costs, be sure to consult experts in health policy or related fields.  Costs 
    should encompass more than just monetary costs, like relative costs (regret, disappointment, stigma, 
    disutility, etc.). Careful consideration of relative costs between false positive and false negatives
    needs to be considered.
    
    cost_miss_case:
        -The cost of missing a case relative to
    cost_false_pos:
        -The cost of a false positive case relative to
    prevalence:
        -The prevalence of the disease in the population
    sensitivity:
        -The sensitivity level of the screening test
    specificity:
        -The specificity level of the screening test
    population:
        -The population size to set. Choose a larger value since this is only necessary for total calculations. Default is 10,000
    decimal:
        -amount of decimal points to display. Default value is 3
    '''
    print('WARNING: When calculating costs, be sure to consult experts in health\npolicy or related fields.  Costs should encompass more than only monetary\ncosts, like relative costs (regret, disappointment, stigma, disutility, etc.)\n')
    if (sensitivity>1) | (specificity>1):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    else:
        disease = population*prevalence
        disease_free = population - disease
        #TEST: none 
            #total cost of not testing
        nt_cost = disease * cost_miss_case
        print('Total cost of testing no one: ',round(nt_cost,decimal))
            #per capita cost 
        pc_nt_cost = nt_cost/population
        print('Per Capita cost of testing no one: ',round(pc_nt_cost,decimal),'\n')
        #TEST: all
            #total cost of testing
        t_cost = disease_free * cost_false_pos
        print('Total cost of testing everyone: ',round(t_cost,decimal))
            #per capita cost 
        pc_t_cost = t_cost/population
        print('Per Capita cost of testing everyone: ',round(pc_t_cost,decimal),'\n')
        #TEST: criteria
            #total cost of testing
        cost_b = disease - (disease*sensitivity)
        cost_c = disease_free - (disease_free*specificity)
        ct_cost = (cost_miss_case*cost_b) + (cost_false_pos*cost_c)
        print('Total cost of testing criteria: ',round(ct_cost,decimal))
            #per capita cost 
        pc_ct_cost = ct_cost/population
        print('Per Capita cost of testing criteria: ',round(pc_ct_cost,decimal),'\n')
        if (ct_cost>nt_cost):
            print('Screening program is more costly than treating everyone as a test-negative')
        if (pc_nt_cost>pc_ct_cost>pc_t_cost):
            print('Screening program is cost efficient')
        if ((pc_t_cost<pc_ct_cost)&(pc_t_cost<pc_nt_cost)):
            print('Treating everyone is least costly')
