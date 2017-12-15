import warnings
import math 
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def effectmeasure_plotdata(label,effect_measure,lcl,ucl):
    '''Creates dataframe compatible with effectmeasure_plot() function from four lists. If you want trailing
    zeroes to displayed, enter those items as a character type. This function is adapted to take list of
    numeric mixed with character variables as inputs to ensure trailing zeroes are displayed as 
    in the table. This function is recommended to be used to prepare data for display by the effectmeasure_plot() 
    function. 
    
    Example of input data: effect_measure = [0.88,1.02,'0.90',1.11]
    
    label:
        -list of labels of lines
    effect_measure:
        -list of point estimates of measure
    lcl:
        -list of level of the lower confidence limit
    ucl:
        -list of level of the upper confidence limit
    '''
    df = pd.DataFrame();df['study'] = label;df['OR'] = effect_measure;df['LCL'] = lcl;df['UCL'] = ucl
    df['OR2'] = df['OR'].astype(str).astype(float)
    if ((all(isinstance(item,float) for item in lcl))&(all(isinstance(item,float) for item in effect_measure))):
        df['LCL_dif'] = df['OR'] - df['LCL']
    else:
        df['LCL_dif'] = (pd.to_numeric(df['OR'])) - (pd.to_numeric(df['LCL']))
    if ((all(isinstance(item,float) for item in ucl))&(all(isinstance(item,float) for item in effect_measure))):
        df['UCL_dif'] = df['UCL'] - df['OR']
    else:
        df['UCL_dif'] = (pd.to_numeric(df['UCL'])) - (pd.to_numeric(df['OR']))
    return df
        
def effectmeasure_plot(df,decimal=3,title='',em='OR',ci='95% CI',scale='log',center=0,errc='dimgrey',shape='d',pc='k',linec='gray',t_adjuster=0.01,size = 3):
    '''Creates Effect Measure plot: Necessary to format data appropriately for this function. effectmeasure_plotdata() 
    function will return an appropriately created dataframe from a series of lists. Data points must be integers or float. 
    Generates a forestplot by using features from Matplotlib.
    
    NOTE: Highly recommend using effectmeasure_plotdata() to create df compatible with function

    df: 
        -dataframe that contains labels, effect measure, lower CI, upper CI. Dataframe must have appropriate
         columns labels to function properly. Recommended that effectmeasure_plotdata() is used to generate the 
         dataframe input into this function to prevent issues.
    decimal: 
        -amount of decimals to display. Default is 3
    title: 
        -set a title for the generated plot. Default is blank title
    em: 
        -label for the effect measure. Default is 'OR', abbreviation for odds ratio
    ci: 
        -label for confidence intervals. Default is '95% CI' 
    scale: 
        -scale to set the x-axis. Default is log scale. Should remain log-scale if plotting ratio measures.
         If plotting difference measures, use 'linear' to generate a linear scale.
    errc: 
        -color of the error bars. Can be a single color or a list of valid matplotlib colors
    shape: 
        -shape of the markers. Can only be a single shape
    pc: 
        -color of the markers. Can be a single color or a list of valid matplotlib colors
    linec:
        -color of the reference line. Can only be a single color
    t_adjuster: 
        -table adjustment factor. Used to change alignment of table with plot. Depending on rows, may need to 
         change this value. As more rows are added to the table, the plot and table will begin to misalign. Only 
         small changes to this value are needed. Default is set to 0.01, which functions well for 6-8 rows
    size:
        -change the plot size. May need to change to fit all labels inside saved object. Default is 3
    '''
    tval = [] #sets values to display in side table
    ytick = [] #determines y tick marks to use
    for i in range(len(df)):
        if (np.isnan(df['OR2'][i])==False):
            if ((isinstance(df['OR'][i],float))&(isinstance(df['LCL'][i],float))&(isinstance(df['UCL'][i],float))):
                tval.append([round(df['OR2'][i],decimal),('('+str(round(df['LCL'][i],decimal))+', '+str(round(df['UCL'][i],decimal))+')')])
            else:
                tval.append([df['OR'][i],('('+str(df['LCL'][i])+', '+str(df['UCL'][i])+')')])
            ytick.append(i)
        else:
            tval.append([' ',' '])
            ytick.append(i)
    if (pd.to_numeric(df['UCL']).max() < 1):
        maxi = round(((pd.to_numeric(df['UCL'])).max() + 0.05),2) #setting x-axis maximum for UCL less than 1
    if ((pd.to_numeric(df['UCL']).max() < 9) & (pd.to_numeric(df['UCL']).max() >= 1)):
        maxi = round(((pd.to_numeric(df['UCL'])).max() + 1),0) #setting x-axis maximum for UCL less than 10
    if (pd.to_numeric(df['UCL']).max() > 9):
        maxi = round(((pd.to_numeric(df['UCL'])).max() + 10),0) #setting x-axis maximum for UCL less than 100
    if (pd.to_numeric(df['LCL']).min() > 0):
        mini = round(((pd.to_numeric(df['LCL'])).min() - 0.1),1) #setting x-axis minimum
    if (pd.to_numeric(df['LCL']).min() < 0):
        mini = round(((pd.to_numeric(df['LCL'])).min() - 0.05),2) #setting x-axis minimum
    plt.figure(figsize=(size*2,size*1)) #blank figure
    plt.suptitle(title) #sets user-defined title
    gspec = gridspec.GridSpec(1, 6) #sets up grid
    plot = plt.subplot(gspec[0, 0:4]) #plot of data
    tabl = plt.subplot(gspec[0, 4:]) # table of OR & CI 
    plot.set_ylim(-1,(len(df))) #spacing out y-axis properly
    plot.errorbar(df.OR2,df.index,xerr=[df.LCL_dif,df.UCL_dif],ecolor=errc,fmt=shape,color=pc,elinewidth=(size/size),markersize=(size*2)) #draw EM & CL
    if (scale=='log'):
        plot.set_xscale('log') #log scale for OR/RR
        center = 1
    plot.axvline(center,color=linec) #set reference line at x = 1
    plot.xaxis.set_ticks_position('bottom');plot.yaxis.set_ticks_position('left')
    plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plot.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    plot.set_yticks(ytick);plot.set_xlim([mini,maxi]);plot.set_xticks([mini,center,maxi]);plot.set_xticklabels([mini,center,maxi])
    plot.set_yticklabels(df['study'])
    plot.yaxis.set_ticks_position('none')
    plot.invert_yaxis() #invert y-axis to align values properly with table
    tb = tabl.table(cellText=tval,cellLoc='center',loc='right',colLabels=[em,ci],bbox=[0,t_adjuster,1,1])
    tabl.axis('off');tb.auto_set_font_size(False);tb.set_fontsize(12)
    for key,cell in tb.get_celld().items():
        cell.set_linewidth(0)
    plt.show()


def func_form_plot(df,outcome,var,f_form=None,outcome_type='binary',link_dist=None,ylims=None,loess_value=0.5,legend=True):
    '''Creates a LOESS plot to aid in functional form assessment for continuous variables.
    Plots can be created for binary and continuous outcomes. Default options are set to create
    a functional form plot for a binary outcome. To convert to a continuous outcome, 
    outcome_type needs to be changed, in addition to the link_dist
    
    Returns a matplotlib graph with a LOESS line (dashed red-line), regression line (sold blue-line),
    and confidence interval (shaded blue)
    
    df:
        -dataframe that contains the variables of interest
    outcome:
        -Column name of the outcome variable of interest
    var:
        -Column name of the variable of interest for the functional form assessment
    f_form:
        -Regression equation of the functional form to assess. Default is None, which will produce
         a linear functional form. Input the regression equation as the variables of interest, separated
         by +. Example) 'var + var_sq'
    outcome_type:
        -Variable type of the outcome variable. Currently, only binary and continuous variables are
         supported. Default is binary
    link_dist:
        -Link and distribution for the GLM regression equation. Change this to any valid link and 
        distributions supported by statsmodels. Default is None, which conducts logistic regression
    ylims:
        -List object of length 2 that holds the upper and lower limits of the y-axis. Y-axis limits should be 
        specified when comparing multiple graphs. These need to be user-specified since the results between
        models and datasets can be so variable. Default is None, which returns the matplotlib y-axis of best fit.
    loess_value:
        -Fraction of observations to use to fit the LOESS curve. This will need to be changed iteratively
         to determine which percent works best for the data. Default is 0.5
    legend:
        -Turn the legend on or off. Default is True, displaying the legend in the graph 
    '''
    rf = df.copy()
    rf = rf[[outcome,var]].dropna().reset_index()
    if f_form == None:
        f_form = var 
    else:
        pass
    if link_dist == None:
        link_dist = sm.families.family.Binomial(sm.families.links.logit)
    else:
        pass 
    if outcome_type=='binary':
        djm = smf.glm(outcome+'~ C('+var+')',df,family=link_dist).fit()
        djf = djm.get_prediction(df.dropna()).summary_frame()        
        dj = pd.concat([rf,djf],axis=1)
        dj.sort_values(var,inplace=True)
        yl = lowess(list(dj['mean']),list(dj[var]),frac=loess_value)
        lowess_x = list(zip(*yl))[0]
        lowess_y = list(zip(*yl))[1]
    elif outcome_type=='continuous':
        yl = lowess(list(rf[outcome]),list(rf[var]),frac=loess_value)
        lowess_x = list(zip(*yl))[0]
        lowess_y = list(zip(*yl))[1]
    else:
        raise ValueError('Functional form assessment only supports binary or continuous outcomes currently')
    ffm = smf.glm(outcome+'~ '+f_form,df,family=link_dist).fit()
    fff = ffm.get_prediction(df.dropna()).summary_frame()
    ff = pd.concat([rf,fff],axis=1)
    ff.sort_values(var,inplace=True)
    plt.fill_between(ff[var],ff['mean_ci_upper'],ff['mean_ci_lower'],alpha=0.1,color='blue',label='95% CI')
    plt.plot(ff[var],ff['mean'],'-',color='blue',label='Regression')
    plt.plot(lowess_x,lowess_y,'--',color='red',linewidth=1,label='LOESS')  
    plt.xlabel(var)
    if outcome_type=='binary':
        plt.ylabel('Probability')
    else:
        plt.ylabel(outcome)
    if legend == True:
        plt.legend()
    plt.ylim(ylims)
    plt.show()

