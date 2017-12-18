# zepid walkthrough
This is a walkthrough of the zepid package and its content functions

## First we will import some other packages and zepid
```python
#Import other packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
#Import zepid
import zepid as ze
```
Now, let's import the example dataset for the analysis
```python
#Load sample dataframe
df = ze.datex()
df.info()
```
## Basic Measures of Association
Let's calculate the Odds Ratio and its corresponding 95% Confidence Interval
```python
ze.OddsRatio(df['exposure'],df['outcome'])
```
This function will produce the following output
```
outcome    0.0  1.0
exposure           
0.0       1822  533
1.0        513  293 

----------------------------------------------------------------------
Odds in exposed:  0.571
Odds in unexposed:  0.293
----------------------------------------------------------------------
Odds Ratio:  1.952
95.0% two-sided CI: ( 1.642 ,  2.321 )
Confidence Limit Ratio:  1.413
----------------------------------------------------------------------
```
We can also calculate other measures of association by using the following functions:
```python
ze.RelRisk(df['exposure'],df['outcome']) #calculates the Risk Ratio / Relative Risk
ze.RiskDiff(df['exposure'],df['outcome']) #calculates the Risk Difference
ze.NNT(df['exposure'],df['outcome']) #calculates the Number Needed to Treat/Harm
```
## Assess Collinearity
### Binary - Continuous
To assess collinearity between a continuous variable and binary variables, we can use Standardized Mean Differences. To calculate Standard Mean Differences, we can use the following function:
```python
ze.StandMeanDiff(df,'exposure','continuous')
```
Which produces the following output
```
----------------------------------------------------------------------
Standardized Mean Difference: 0.068
----------------------------------------------------------------------
```
### Categorical - Continuous
For variables with more than two categories, piecewise Standard Mean Differences need to be calculate. To do this, we will obtain the estimates and store them as variables. We will then use zepid.calc to calculate the Standard Mean Differences from the summary information
```python
df.category.value_counts() #getting n for each category
c0mean = np.mean(df.loc[df.category==0]['continuous']) #extracting the mean
c0std = np.std(df.loc[df.category==0]['continuous']) #extracting the standard deviation
c1mean = np.mean(df.loc[df.category==1]['continuous']) #extracting the mean
c1std = np.std(df.loc[df.category==1]['continuous']) #extracting the standard deviation
c2mean = np.mean(df.loc[df.category==2]['continuous']) #extracting the mean
c2std = np.std(df.loc[df.category==2]['continuous']) #extracting the standard deviation

ze.calc.stand_mean_diff(1900,859,c2mean,c0mean,c2std,c0std,decimal=5)
ze.calc.stand_mean_diff(1900,668,c2mean,c1mean,c2std,c1std,decimal=5)
```
Which will produce the following output
```
----------------------------------------------------------------------
Standardized Mean Difference: 0.0104
----------------------------------------------------------------------
----------------------------------------------------------------------
Standardized Mean Difference: 0.01776
----------------------------------------------------------------------
```
### Binary - Binary
To see whether there is collinearity between two binary variables, we use the odds ratio
```python
ze.OddsRatio(df['exposure'],df['binary'])
```
Which produces the following output
```
binary     0.0   1.0
exposure            
0.0       1036  1534
1.0        837    20 

----------------------------------------------------------------------
Odds in exposed:  0.024
Odds in unexposed:  1.481
----------------------------------------------------------------------
Odds Ratio:  0.016
95.0% two-sided CI: ( 0.01 ,  0.025 )
Confidence Limit Ratio:  2.462
----------------------------------------------------------------------
```
Note that there is evidence of collinearity, but we will ignore this issue through the rest of our example

### Categorical - Binary
For a categorical variables, we must again use the summary calculations contained in zepid.calc
```python
pd.crosstab(df['exposure'],df['category'])
ze.calc.oddr(a=80,b=588,c=747,d=1153)
ze.calc.oddr(30,829,747,1153)
```
Which produces the following output
```
----------------------------------------------------------------------
Odds exposed: 0.136
Odds unexposed: 0.648
----------------------------------------------------------------------
Odds Ratio: 0.21
95.0% two-sided CI: ( 0.163 ,  0.27 )
----------------------------------------------------------------------
----------------------------------------------------------------------
Odds exposed: 0.036
Odds unexposed: 0.648
----------------------------------------------------------------------
Odds Ratio: 0.056
95.0% two-sided CI: ( 0.038 ,  0.081 )
----------------------------------------------------------------------
```
Again, we have evidence of collinearity but we will ignore the methodological ramifications of this in our example
## Functional form of continuous variable
To do a functional form assessment, we first have to decide on some possible functional forms and code new variables. We will assess; linear, quadratic, categorical, and restricted quadratic spline. First, creating a quadratic and categorical variable
```python
df['cont_sq'] = df['continuous']**2
df.loc[df.continuous<20,'ccat'] = 0
df.loc[((df.continuous<40)&(df.continuous>=20)),'ccat'] = 1
df.loc[((df.continuous<60)&(df.continuous>=40)),'ccat'] = 2
df.loc[((df.continuous<80)&(df.continuous>=60)),'ccat'] = 3
df.loc[((df.continuous<=100)&(df.continuous>=80)),'ccat'] = 4
```
### Spline 
To generate a spline, we can use the zepid function spline. spline is used to generate any order term spline model with either user-specified spline knots or automatically generated spline knot locations. Additionally, whether a restricted spline model is used can be chosen. For our example, we will create a restricted (restricted=True) quadratic spline (term=2) spline with five knots (n_knots=5). We will let the function determine where the knots will be located.
```python
df[['rs1','rs2','rs3','rs4']] = ze.spline(df,'continuous',n_knots=5,term=2,restricted=True)
```
### Functional form plots
Now that our potential forms are coded, we can create some function form plots. 
#### Linear
First we will look at our linear functional form
```python
ze.graphics.func_form_plot(df,'outcome','continuous',ylims=[0.15,0.5]) #linear
```
Which produces the following output
```
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                outcome   No. Observations:                 3161
Model:                            GLM   Df Residuals:                     3159
Model Family:                Binomial   Df Model:                            1
Link Function:                  logit   Scale:                             1.0
Method:                          IRLS   Log-Likelihood:                -1798.8
Date:                Mon, 18 Dec 2017   Deviance:                       3597.6
Time:                        08:02:01   Pearson chi2:                 3.17e+03
No. Iterations:                     4                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -1.4498      0.084    -17.305      0.000      -1.614      -1.286
continuous     0.0081      0.001      5.788      0.000       0.005       0.011
==============================================================================
```
and image

![alt text](https://github.com/pzivich/zepid/blob/master/images/linear_funcform.png "Linear Functional Form")

For our other functional form assessments, we obtain the following plots
#### Quadratic

![alt text](https://github.com/pzivich/zepid/blob/master/images/quad_funcform.png "Quadratic Functional Form")
#### Categorical

![alt text](https://github.com/pzivich/zepid/blob/master/images/cat_funcform.png "Categorical Functional Form")
#### Restricted Quadratic Spline

![alt text](https://github.com/pzivich/zepid/blob/master/images/rqs_funcform.png "Restricted Quadratic Spline Functional Form")
Based on these results, we will use a quadratic functional form for our continuous variable
