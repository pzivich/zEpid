#####################################################################
# Graphics
#####################################################################

#Creating graphics useful for epidemiology analyses

import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#####################################################################
#Forest Plot Generator
def forest_plot_data(study,effect_measure,lcl,ucl):
    '''Creates dataframe compatible with forest_plot() function from four lists. If you want trailing
    zeroes to displayed, enter those items as a character type. This function is adapted to take list of
    numeric mixed with character variables as inputes to ensure trailing zeroes are displayed as 
    requested. This function is recommended to be used to prepare data for display by the forest_plot 
    function. 
    
    study:
        -list of labels of lines
    effect_measure:
        -list of point estimates of measure
    lcl:
        -list of level of the lower confidence limit
    ucl:
        -list of level of the upper confidence limit
    '''
    df = pd.DataFrame();df['study'] = study;df['OR'] = effect_measure;df['LCL'] = lcl;df['UCL'] = ucl
    df['OR2'] = df['OR'].astype(str).astype(float) #Need to fix this so missing are ignored
    if ((all(isinstance(item,float) for item in lcl))&(all(isinstance(item,float) for item in effect_measure))):
        df['LCL_dif'] = df['OR'] - df['LCL']
    else:
        df['LCL_dif'] = (pd.to_numeric(df['OR'])) - (pd.to_numeric(df['LCL']))
    if ((all(isinstance(item,float) for item in ucl))&(all(isinstance(item,float) for item in effect_measure))):
        df['UCL_dif'] = df['UCL'] - df['OR']
    else:
        df['UCL_dif'] = (pd.to_numeric(df['UCL'])) - (pd.to_numeric(df['OR']))
    return df
    

def forest_plot(df,decimal=3,title='',em='OR',ci='95% CI',scale='log',errc='dimgrey',shape='d',pc='k',linec='gray',t_adjuster=0.01,size = 3):
    '''Creates Forest plot: Necessary to format data appropriately. forest_plot_data() function will return an 
    appropriately created dataframe from a series of lists. Data points must be integers or float. Generates a 
    forestplot by using features from Matplotlib.
    
    df: 
        -dataframe that contains labels, effect measure, lower CI, upper CI. Dataframe must have appropriate
         columns labels to function properly. Recommended that forest_plot_data() is used to generate the 
         dataframe input into this function to prevent issues.
    decimal: 
        -amount of decimals to display. Default is 3
    title: 
        -set a title for the generated plot. Default is no title
    em: 
        -label for the effect measure. Default is OR, abbreviation for odds ratio
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
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
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
    print(tval)
    if (pd.to_numeric(df['UCL']).max() < 9):
        maxi = round(((pd.to_numeric(df['UCL'])).max() + 1),0) #setting x-axis maximum for UCL less than 10
    if (pd.to_numeric(df['UCL']).max() > 9):
        maxi = round(((pd.to_numeric(df['UCL'])).max() + 10),0) #setting x-axis maximum for UCL less than 100
    mini = round(((pd.to_numeric(df['LCL'])).min() - 0.1),1) #setting x-axis minimum
    plt.figure(figsize=(size*2,size*1)) #blank figure
    plt.suptitle(title) #sets user-defined title
    gspec = gridspec.GridSpec(1, 6) #sets up grid
    plot = plt.subplot(gspec[0, 0:4]) #plot of data
    tabl = plt.subplot(gspec[0, 4:]) # table of OR & CI 
    plot.set_ylim(-1,(len(df))) #spacing out y-axis properly
    plot.axvline(1,color=linec) #set reference line at x = 1
    plot.errorbar(df.OR2,df.index,xerr=[df.LCL_dif,df.UCL_dif],ecolor=errc,fmt=shape,color=pc,elinewidth=(size/size),markersize=(size*2)) #draw EM & CL
    if (scale=='log'):
        plot.set_xscale('log') #log scale since OR/RR
    plot.set_yticks(ytick);plot.set_xlim([mini,maxi]);plot.set_xticks([mini,1,maxi])
    plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plot.set_yticklabels(df['study'])
    plot.invert_yaxis() #invert y-axis to align values properly with table
    plot.xaxis.set_ticks_position('bottom');plot.yaxis.set_ticks_position('left')
    tb = tabl.table(cellText=tval,cellLoc='center',loc='right',colLabels=[em,ci],bbox=[0,t_adjuster,1,1])
    tabl.axis('off');tb.auto_set_font_size(False);tb.set_fontsize(12)
    for key,cell in tb.get_celld().items():
        cell.set_linewidth(0)
    plt.show()