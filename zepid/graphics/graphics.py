"""
Contains useful graphic generators. Currently, effect measure plots and functional form assessment plots
are implemented. Uses matplotlib to generate graphics. Future inclusions include forest plots

"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker


class EffectMeasurePlot:
    """Used to generate effect measure plots. effectmeasure plot accepts four list type objects.
    effectmeasure_plot is initialized with the associated names for each line, the point estimate,
    the lower confidence limit, and the upper confidence limit.

    Notes
    ----------
    .. code::

            _____________________________________________      Measure     % CI
            |                                           |
        1   |        --------o-------                   |       x        n, 2n
            |                                           |
        2   |                   ----o----               |       w        m, 2m
            |                                           |
            |___________________________________________|
                #           #           #           #

    Examples
    -------------
    Setting up the data to plot

    >>> from zepid.graphics import EffectMeasurePlot
    >>> lab = ['One','Two']
    >>> emm = [1.01,1.31]
    >>> lcl = ['0.90',1.01]
    >>> ucl = [1.11,1.53]

    Setting up the plot, measure labels, and point colors

    >>> x = EffectMeasurePlot(lab, emm, lcl, ucl)
    >>> x.labels(effectmeasure='RR')  # Changing label of measure
    >>> x.colors(pointcolor='r')  # Changing color of the points

    Generating matplotlib axes object of forest plot

    >>> x.plot(t_adjuster=0.13)
    """
    def __init__(self, label, effect_measure, lcl, ucl):
        """Initializes effectmeasure_plot with desired data to plot. All lists should be the same
        length. If a blank space is desired in the plot, add an empty character object (' ') to
        each list at the desired point.

        Parameters
        --------------
        label : list
            List of labels to use for y-axis
        effect_measure : list
            List of numbers for point estimates to plot. If point estimate has trailing zeroes,
            input as a character object rather than a float
        lcl : list
            List of numbers for upper confidence limits to plot. If point estimate has trailing
            zeroes, input as a character object rather than a float
        ucl : list
            List of numbers for upper confidence limits to plot. If point estimate has
            trailing zeroes, input as a character object rather than a float
        """
        self.df = pd.DataFrame()
        self.df['study'] = label
        self.df['OR'] = effect_measure
        self.df['LCL'] = lcl
        self.df['UCL'] = ucl
        self.df['OR2'] = self.df['OR'].astype(str).astype(float)
        if (all(isinstance(item, float) for item in lcl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['LCL_dif'] = self.df['OR'] - self.df['LCL']
        else:
            self.df['LCL_dif'] = (pd.to_numeric(self.df['OR'])) - (pd.to_numeric(self.df['LCL']))
        if (all(isinstance(item, float) for item in ucl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['UCL_dif'] = self.df['UCL'] - self.df['OR']
        else:
            self.df['UCL_dif'] = (pd.to_numeric(self.df['UCL'])) - (pd.to_numeric(self.df['OR']))
        self.em = 'OR'
        self.ci = '95% CI'
        self.scale = 'linear'
        self.center = 1
        self.errc = 'dimgrey'
        self.shape = 'd'
        self.pc = 'k'
        self.linec = 'gray'

    def labels(self, **kwargs):
        """Function to change the labels of the outputted table. Additionally, the scale and reference
        value can be changed.

        Parameters
        -------------
        effectmeasure : string, optional
            Changes the effect measure label
        conf_int : string, optional
            Changes the confidence interval label
        scale : string, optional
            Changes the scale to either log or linear
        center : float, integer, optional
            Changes the reference line for the center
        """
        if 'effectmeasure' in kwargs:
            self.em = kwargs['effectmeasure']
        if 'conf_int' in kwargs:
            self.ci = kwargs['conf_int']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        if 'center' in kwargs:
            self.center = kwargs['center']

    def colors(self, **kwargs):
        """Function to change colors and shapes.

        Parameters
        ---------------
        errorbarcolor : string, optional
            Changes the error bar colors
        linecolor : string, optional
            Changes the color of the reference line
        pointcolor : string, optional
            Changes the color of the points
        pointshape : string, optional
            Changes the shape of points
        """
        if 'errorbarcolor' in kwargs:
            self.errc = kwargs['errorbarcolor']
        if 'pointshape' in kwargs:
            self.shape = kwargs['pointshape']
        if 'linecolor' in kwargs:
            self.linec = kwargs['linecolor']
        if 'pointcolor' in kwargs:
            self.pc = kwargs['pointcolor']

    def plot(self, figsize=(3, 3), t_adjuster=0.01, decimal=3, size=3, max_value=None, min_value=None):
        """Generates the matplotlib effect measure plot with the default or specified attributes.
        The following variables can be used to further fine-tune the effect measure plot

        Parameters
        -----------------
        figsize : tuple, optional
            Adjust the size of the figure. Syntax is same as matplotlib `figsize`
        t_adjuster : float, optional
            Used to refine alignment of the table with the line graphs. When generate plots, trial and error for this
            value are usually necessary. I haven't come up with an algorithm to determine this yet...
        decimal : integer, optional
            Number of decimal places to display in the table
        size : integer,
            Option to adjust the size of the lines and points in the plot
        max_value : float, optional
            Maximum value of x-axis scale. Default is None, which automatically determines max value
        min_value : float, optional
            Minimum value of x-axis scale. Default is None, which automatically determines min value

        Returns
        ---------
        matplotlib axes
        """
        tval = []
        ytick = []
        for i in range(len(self.df)):
            if not np.isnan(self.df['OR2'][i]):
                if ((isinstance(self.df['OR'][i], float)) & (isinstance(self.df['LCL'][i], float)) &
                        (isinstance(self.df['UCL'][i], float))):
                    tval.append([round(self.df['OR2'][i], decimal), (
                                '(' + str(round(self.df['LCL'][i], decimal)) + ', ' +
                                str(round(self.df['UCL'][i], decimal)) + ')')])
                else:
                    tval.append(
                        [self.df['OR'][i], ('(' + str(self.df['LCL'][i]) + ', ' + str(self.df['UCL'][i]) + ')')])
                ytick.append(i)
            else:
                tval.append([' ', ' '])
                ytick.append(i)
        if max_value is None:
            if pd.to_numeric(self.df['UCL']).max() < 1:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 0.05),
                             2)  # setting x-axis maximum for UCL less than 1
            if (pd.to_numeric(self.df['UCL']).max() < 9) and (pd.to_numeric(self.df['UCL']).max() >= 1):
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 1),
                             0)  # setting x-axis maximum for UCL less than 10
            if pd.to_numeric(self.df['UCL']).max() > 9:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 10),
                             0)  # setting x-axis maximum for UCL less than 100
        else:
            maxi = max_value
        if min_value is None:
            if pd.to_numeric(self.df['LCL']).min() > 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.1), 1)  # setting x-axis minimum
            if pd.to_numeric(self.df['LCL']).min() < 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.05), 2)  # setting x-axis minimum
        else:
            mini = min_value
        plt.figure(figsize=figsize)  # blank figure
        gspec = gridspec.GridSpec(1, 6)  # sets up grid
        plot = plt.subplot(gspec[0, 0:4])  # plot of data
        tabl = plt.subplot(gspec[0, 4:])  # table of OR & CI
        plot.set_ylim(-1, (len(self.df)))  # spacing out y-axis properly
        if self.scale == 'log':
            try:
                plot.set_xscale('log')
            except:
                raise ValueError('For the log scale, all values must be positive')
        plot.axvline(self.center, color=self.linec, zorder=1)
        plot.errorbar(self.df.OR2, self.df.index, xerr=[self.df.LCL_dif, self.df.UCL_dif], marker='None', zorder=2,
                      ecolor=self.errc, elinewidth=(size / size), linewidth=0)
        plot.scatter(self.df.OR2, self.df.index, c=self.pc, s=(size * 25), marker=self.shape, zorder=3,
                     edgecolors='None')
        plot.xaxis.set_ticks_position('bottom')
        plot.yaxis.set_ticks_position('left')
        plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plot.set_yticks(ytick)
        plot.set_xlim([mini, maxi])
        plot.set_xticks([mini, self.center, maxi])
        plot.set_xticklabels([mini, self.center, maxi])
        plot.set_yticklabels(self.df.study)
        plot.yaxis.set_ticks_position('none')
        plot.invert_yaxis()  # invert y-axis to align values properly with table
        tb = tabl.table(cellText=tval, cellLoc='center', loc='right', colLabels=[self.em, self.ci],
                        bbox=[0, t_adjuster, 1, 1])
        tabl.axis('off')
        tb.auto_set_font_size(False)
        tb.set_fontsize(12)
        for key, cell in tb.get_celld().items():
            cell.set_linewidth(0)
        return plot


def functional_form_plot(df, outcome, var, f_form=None, outcome_type='binary', discrete=False, link_dist=None,
                         loess=True, loess_value=0.4, legend=True, model_results=True, points=False):
    """Creates a functional form plot to aid in functional form assessment for continuous/discrete variables. Plots can
    be created for binary and continuous outcomes. Default options are set to create a functional form plot for a
    binary outcome. To convert to a continuous outcome, outcome_type needs to be changed, in addition to the link_dist

    Parameters
    ------------
    df : DataFrame
        Pandas dataframe that contains the variables of interest
    outcome : string
        Column name of the outcome variable of interest
    var : string
        Column name of the variable of interest for the functional form assessment
    f_form : string, optional
        Regression equation of the functional form to assess. Default is None, which will produce a linear functional
        form. Input the regression equation as the variables of interest, separated by +. For example, 'var + var_sq'
    outcome_type : string, optional
        Variable type of the outcome variable. Currently, only binary and continuous variables are
        supported. Default is 'binary' but 'continuous' is also supported
    link_dist : optional
        Link and distribution for the GLM regression equation. Change this to any valid link and distributions
        supported by statsmodels. Default is None, which conducts logistic regression
    loess_value : float, optional
        Fraction of observations to use to fit the LOESS curve. This will need to be changed iteratively to determine
        which percent works best for the data. Default is 0.4
    legend : bool, optional
        Turn the legend on or off. Default is True, displaying the legend in the graph
    model_results : bool, optional
        Whether to produce the model results. Default is True, which provides model results
    loess : bool, optional
        Whether to plot the LOESS curve along with the functional form. Default is True
    points : bool, optional
        Whether to plot the data points, where size is relative to the number of observations. Default is False
    discrete : bool, optional
        If your data is truly continuous, leave setting to bin the dat. Will automatically bin observations into
        categories for generation of the LOESS curve. If you data is discrete, you can set this to True to use your
        actual values. If you get a perfect SeparationError from statsmodels, it means you might have to reshift your
        categories.

    Returns
    -----------
    matplotlib axes
        Returns a matplotlib graph with a LOESS line (dashed red-line), regression line (sold blue-line), and
        confidence interval (shaded blue)

    Examples
    ------------
    Setting up the environment

    >>> from zepid import load_sample_data
    >>> from zepid.graphics import functional_form_plot
    >>> import matplotlib.pyplot as plt
    >>> df = load_sample_data(timevary=False)
    >>> df['cd4_sq'] = df['cd4']**2

    Creating a functional form plot for a linear functional form

    >>> functional_form_plot(df, outcome='dead', var='cd4')
    >>> plt.show()

    Functional form assessment for a quadractic functional form

    >>> functional_form_plot(df, outcome='dead', var='cd4', f_form='cd4 + cd4_sq')
    >>> plt.show()

    Varying the LOESS value (increased LOESS value to smooth LOESS curve further)

    >>> functional_form_plot(df, outcome='dead', var='cd4', loess_value=0.5)
    >>> plt.show()

    Removing the LOESS curve and the legend from the plot

    >>> functional_form_plot(df, outcome='dead', var='cd4', loess=False, legend=False)
    >>> plt.show()

    Adding summary points to the plot. Points are grouped together and their size reflects their relative n
    >>>functional_form_plot(df, outcome='dead', var='cd4', loess=False, legend=False, points=True)
    >>>plt.show()

    Functional form assessment for a discrete variable (age)
    >>>functional_form_plot(df, outcome='dead', var='age0', discrete=True)
    >>>plt.show()
    """
    # Copying out the dataframe to a new object we will manipulate a bit
    rf = df.copy()
    rf = rf.dropna(subset=[var, outcome]).sort_values(by=[var, outcome]).reset_index()
    warnings.warn('Warning: missing observations of model variables are dropped. ' +
                  str(int(df.shape[0] - rf.shape[0])) +
                  ' observations were dropped from the functional form assessment')

    # Functional form for the model
    if f_form is None:
        f_form = var
    else:
        pass

    # Generating Models
    if outcome_type == 'binary':
        if link_dist is None:
            link_dist = sm.families.family.Binomial()
        else:
            pass
    elif outcome_type == 'continuous':
        if link_dist is None:
            link_dist = sm.families.family.Gaussian(sm.families.links.identity)
        else:
            pass
    else:
        raise ValueError('Only binary or continuous outcomes are currently supported')

    # Generating LOESS or points if requested
    ax = plt.gca()
    if loess or points:
        if outcome_type == 'binary':
            if discrete is False:
                # Binning continuous variable into categories to get "General" functional form
                categories = int((np.max(rf[var]) - np.min(rf[var])) / 5)
                if model_results:
                    print('''A total of ''' + str(categories) + ''' categories were created. If you would like to
                            influence  the number of categories the spline is fit to, do the following\n\tIncrease:
                            multiply by constant >1\n\tDecrease: multiply by contast <1 and >0''')
                rf['vbin'] = pd.qcut(rf[var], q=categories, duplicates='drop').cat.codes
                djm = smf.glm(outcome + '~ C(vbin)', rf, family=link_dist).fit()
            else:
                djm = smf.glm(outcome + '~ C(' + var + ')', rf, family=link_dist).fit()
            djf = djm.get_prediction(rf).summary_frame()
            dj = pd.concat([rf, djf], axis=1)
            dj.sort_values(var, inplace=True)
            if points:
                pf = dj.groupby(by=[var, 'mean']).count().reset_index()
                ax.scatter(pf[var], pf['mean'], s=[100 * (n / np.max(pf[var])) for n in pf[var]],
                           color='gray', label='Data point')
            if loess:
                yl = lowess(list(dj['mean']), list(dj[var]), frac=loess_value)
                lowess_x = list(zip(*yl))[0]
                lowess_y = list(zip(*yl))[1]
                ax.plot(lowess_x, lowess_y, '--', color='red', linewidth=1, label='LOESS')
        if outcome_type == 'continuous':
            if points:
                pf = rf.groupby(by=[var, outcome]).count().reset_index()
                ax.scatter(pf[var], pf[outcome], color='gray', label='Data point')
            if loess:
                yl = lowess(list(rf[outcome]), list(rf[var]), frac=loess_value)
                lowess_x = list(zip(*yl))[0]
                lowess_y = list(zip(*yl))[1]
                ax.plot(lowess_x, lowess_y, '--', color='red', linewidth=1, label='LOESS')

    # Functional form model fitting
    ffm = smf.glm(outcome + ' ~ ' + f_form, rf, family=link_dist).fit()
    if model_results is True:
        print(ffm.summary())
        print('AIC: ', ffm.aic)
        print('BIC: ', ffm.bic)
    fff = ffm.get_prediction(rf).summary_frame()
    ff = pd.concat([rf, fff], axis=1)
    ff.sort_values(var, inplace=True)

    # Generating plot for functional form
    ax.fill_between(ff[var], ff['mean_ci_upper'], ff['mean_ci_lower'], alpha=0.1, color='blue', label='95% CI')
    ax.plot(ff[var], ff['mean'], '-', color='blue', label='Regression')
    ax.set_xlabel(var)
    ax.set_ylabel('Outcome')
    if legend is True:
        ax.legend()
    return ax


def pvalue_plot(point, sd, color='b', fill=True, null=0, alpha=None):
    """Creates a plot of the p-value distribution based on a point estimate and standard deviation.
    I find this plot to be useful to explain p-values and how much evidence weight you have in a
    specific value. I think it is useful to explain what exactly a p-value tells you. Note that this
    plot only works for measures on a linear scale (i.e. it will plot exp(log(RR)) incorrectly). It also
    helps to understand what exactly confidence intervals are telling you. These  plots are based on
    Rothman Epidemiology 2nd Edition pg 152-153 and explained more fully within.

    Parameters
    -------------
    point : float
        Point estimate. Must be on a linear scale (RD / log(RR))
    sd : float
        Standard error of the estimate. Must for linear scale (SE(RD) / SE(log(RR)))
    color : str, optional
        Change color of p-value plot
    fill : bool, optional
        Hhether to fill the curve under the p-value distribution. Setting to False prevents fill
    null : float, integer, optional
        The main value to compare to. The default is zero
    alpha : float, optional
        Whether to draw a line designating significance level area. Default is None, which does not draw this line.
        Generally, would be set to 0.05 to correspond to the widely used alpha of 0.05

    Returns
    -----------
    matplotlib axes

    Examples
    -----------
    Setting up the environment
    >>>from zepid.graphics import pvalue_plot
    >>>import matplotlib.pyplot as plt

    Basic P-value plot
    >>>pvalue_plot(point=-0.1, sd=0.061, color='r')
    >>>plt.show()

    P-value plot with significance line drawn at 'alpha'
    >>>pvalue_plot(point=-0.1, sd=0.061, color='r', alpha=0.025)
    >>>plt.show()

    P-value plot with different comparison value
    >>>pvalue_plot(point=-0.1, sd=0.061, color='r', null=0.1)
    >>>plt.show()
    """
    if point <= null:
        lower = (point - 3 * sd)
        if (point + 3 * sd) < 0:
            upper = point + 3 * sd
        else:
            upper = null + 3 * sd
    if point > null:
        upper = (point + 3 * sd)
        if (point - 3 * sd) > 0:
            lower = null - 3 * sd
        else:
            lower = point - 3 * sd
    ax = plt.gca()
    x1 = np.linspace(lower, point, 100)
    x2 = np.linspace(point, upper, 100)
    ax.plot(x2, 2 * (1 - norm.cdf(x2, loc=point, scale=sd)), c=color)
    ax.plot(x1, 2 * norm.cdf(x1, loc=point, scale=sd), c=color)
    if fill == True:
        ax.fill_between(x2, 2 * (1 - norm.cdf(x2, loc=point, scale=sd)), color=color, alpha=0.2)
        ax.fill_between(x1, 2 * norm.cdf(x1, loc=point, scale=sd), color=color, alpha=0.2)
    ax.vlines(null, 0, 1, colors='k')
    ax.set_xlim([lower, upper])
    ax.set_ylim([0, 1])
    ax.set_ylabel('P-value')
    if alpha is not None:
        ax.hlines(alpha, lower, upper)
    return ax


def spaghetti_plot(df, idvar, variable, time):
    """Create a spaghetti plot by an ID variable. A spaghetti plot can be useful for visualizing
    trends or looking at longitudinal data patterns for individuals all at once.

    Parameters
    ------------
    df : DataFrame
        Pandas dataframe containing variables of interest
    idvar : str
        ID variable for observations. This should indicate the group or individual followed over the time variable
    variable : str
        Variable of interest to see how it varies over time
    time : str
        Time or other variable in which the variable variation occurs

    Returns
    -----------
    matplotlib axes

    Examples
    -----------
    Setting up the environment
    >>>from zepid import load_sample_data
    >>>from zepid.graphics import spaghetti_plot
    >>>df = load_sample_data(timevary=True)

    Generating spaghetti plot for changing CD4 count
    >>>spaghetti_plot(df, idvar='id', variable='cd4', time='enter')
    >>>plt.show()
    """
    ax = plt.gca()
    for i in df[idvar].unique():
        s = df.loc[df[idvar] == i].copy()
        s.sort_values(time, ascending=False)
        ax.plot(s[time], s[variable])
    ax.set_xlabel(time)
    ax.set_ylabel(variable)
    return ax


def roc(df, true, threshold, youden_index=True):
    """Generate a Receiver Operator Curve from true values and predicted probabilities. Youden's Index can also be
    calculated. Youden's index is calculated as

    .. math::

        P_{Yi} = max(Se_i + Sp_i - 1)

    Parameters
    ------------
    df : DataFrame
        Pandas dataframe containing variables of interest
    true : str
        True designation of the outcome (1, 0)
    threshold : str
        Predicted probabilities for the outcome
    youden_index : bool, optional
        Whether to calculate Youden's index. Youden's index maximizes both sensitivity and specificity. The formula
        finds the maximum of (sensitivity + specificity - 1)

    Returns
    -----------
    matplotlib axes
    """
    sens = []
    fpr = []
    thresh = []
    tf = df[[threshold, true]].copy()
    if tf.isnull().values.sum() != 0:
        raise ValueError('ROC curve cannot handle missing data for probability or true values')

    # Getting all possible cutpoints
    values = (list(np.unique(tf[threshold])))
    values = [float(np.min(tf[threshold]) - 0.001)] + values + [float(np.max(tf[threshold]) + 0.001)]
    # Going through all the cutpoints and calculating Sensitivity and 1-Specificity
    for v in list(reversed(values)):
        thresh.append(v)
        prediction = np.where(tf[threshold] >= v, 1, 0)
        se = prediction[tf[true] == 1].mean()
        sens.append(se)
        sp = prediction[tf[true] == 0].mean()
        fpr.append(sp)

    # If requested, calculate Youden's Index
    if youden_index is True:
        spec = [1 - i for i in fpr]
        youdens = []
        for i, j in zip(sens, spec):
            youdens.append(i + j - 1)
        ind = np.argmax(youdens)
        print('----------------------------------------------------------------------')
        print("Youden's Index: ", thresh[ind])
        print("Predictive values at Youden's Index")
        print("\tSensitivity: ", sens[ind])
        print("\tSpecificity: ", spec[ind])
        print('----------------------------------------------------------------------')

    # Creating ROC plot
    ax = plt.gca()
    ax.plot(fpr, sens, color='blue')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    if youden_index is True:
        ax.text(0.65, 0.35, "Youden's Index:\n      " + str(round(thresh[ind], 5)))
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 -Specificity')
    return ax


def dynamic_risk_plot(risk_exposed, risk_unexposed, measure='RD', loess=True, loess_value=0.25, point_color='darkblue',
                      line_color='b', scale='linear'):
    """Creates a plot of risk measures over time. See Cole et al. "Estimation of standardized risk difference and ratio
    in a competing risks framework: application to injection drug use and progression to AIDS after initiation of
    antiretroviral therapy." Am J Epidemiol. 2015 for an example of this plot

    Parameters
    --------------
    risk_exposed : Series
        Pandas Series with the probability of the outcome among the exposed group. Index by 'timeline' where 'timeline'
        is the time. If you directly output the ``1 - survival_function_`` from lifelines.KaplanMeierFitter(), this should
        create a valid input
    risk_unexposed : Series
        Pandas Series with the probability of the outcome among the exposed group. Index by 'timeline' where 'timeline'
        is the time
    measure : str, optional
        Whether to generate the risk difference (RD) or risk ratio (RR). Default is 'RD'
    loess : bool, optional
        Whether to generate LOESS curve fit to the calculated points. Default is True
    loess_value : float, optional
        Fraction of values to fit LOESS curve to. Default is 0.25
    point_color : str, optional
        Color of the points
    line_color : str, optional
        Color of the LOESS line generated and plotted
    scale : str, optional
        Change the y-axis scale. Options are 'linear' (default), 'log', 'log-transform'. 'log' and 'log-transform' is
        only a valid option for Risk Ratio plots

    Returns
    -----------
    matplotlib axes

    Examples
    --------
    See graphics documentation or causal documentation
    """
    re = risk_exposed.drop_duplicates(keep='first').iloc[:, 0].rename('exposed').reset_index()
    ru = risk_unexposed.drop_duplicates(keep='first').iloc[:, 0].rename('unexposed').reset_index()
    re.timeline = np.round(re.timeline * 100000).astype(int) # This avoids a merge issue on floats
    ru.timeline = np.round(ru.timeline * 100000).astype(int)
    r = pd.merge(re, ru, how='outer', left_on='timeline', right_on='timeline').sort_values(by='timeline')
    r.timeline /= 100000
    r.ffill(inplace=True)
    if measure == 'RD':
        r['m'] = r['exposed'] - r['unexposed']
    elif measure == 'RR':
        r['m'] = r['exposed'] / r['unexposed']
        if scale == 'log-transform':
            r['m'] = np.log(r['m'])
    else:
        raise ValueError('Only "RD" and "RR" are currently supported')

    # Generating the plot
    ax = plt.gca()
    ax.plot(r['timeline'], r['m'], 'o', c=point_color)
    if loess is True:
        l = lowess(list(r['m']), list(r['timeline']), frac=loess_value)
        lowess_x = list(zip(*l))[0]
        lowess_y = list(zip(*l))[1]
        ax.plot(lowess_x, lowess_y, '-', c=line_color, linewidth=4)
    if measure == 'RD':
        ax.hlines(0, 0, np.max(r['timeline'] + 0.5), linewidth=1.5)
        ax.set_ylabel('Risk Difference')
    if measure == 'RR':
        if scale == 'log-transform':
            ax.hlines(0, 0, np.max(r['timeline'] + 0.5), linewidth=1.5)
            ax.set_ylabel('ln(Risk Ratio)')
        elif scale == 'log':
            ax.set_ylabel('Risk Ratio')
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.hlines(1, 0, np.max(r['timeline'] + 0.5), linewidth=1.5)
        else:
            ax.hlines(1, 0, np.max(r['timeline'] + 0.5), linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_xlim([0, np.max(r['timeline']) + 0.5])
    return ax


def labbe_plot(r1=None, r0=None, scale='both', additive_tuner=12, multiplicative_tuner=12, figsize=(7, 4), **plot_kwargs):
    """L'Abbe plots are useful for summarizing measure modification on the difference or ratio scale. Primarily
    invented for meta-analysis usage, these plots display risk differences (or ratios) by their individual risks
    by an exposure. I find them most useful for a visualization of why if there is an association and there is no
    modfication on one scale (additive or multiplicative), there must be modification on the other scale.

    Parameters
    ----------
    r1 : float, list, optional
        Single probability or a list of probabilities when exposure is 1. Default is None, which does not display points
    r0 : float, list, optional
        Single probability or a list of probabilities when exposure is 0. Default is None, which does not display points
    scale : str, optional
        Which scale to plot. The default is 'both', which generates side-by-side plots of additive scale and
        multiplicative scale. Other options are; 'additive' to display the additive plot, and 'multiplicative' to
        display the multiplicative plot
    additive_tuner : int, optional
        Optional parameter to change the number of lines displayed in the additive L'Abbe plot. Higher integer
        produces more reference lines
    multiplicative_tuner : int, optional
        Optional parameter to change the number of lines displayed in the multiplicative L'Abbe plot. Higher integer
        produces more reference lines
    figsize : set, optional
        Optional parameter to change the L'Abbe plot size. Only changes the plot size when scale='both'
    **plot_kwargs : optional
        Optional keyword arguments for matplotlib. kwargs will pass matplotlib.pyploy.plot kwargs are accepted. See
        matplotlib 'plot()' function documentation for further details

    Returns
    -------
    matplotlib axes

    Examples
    --------
    See graphics documentation
    """
    if r1 is not None or r0 is not None:
        if len(list(r1)) != len(list(r0)):
            raise ValueError('The length of `r1` must be the same as `r0`')

    ya0 = np.linspace(0.0001, 0.9999, 12)

    if scale == 'both':
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        for i in np.linspace(-1, 1, additive_tuner):
            ax[0].plot(ya0, ya0 + i, color='gray', linewidth=1)
        if r1 is not None and r0 is not None:
            if 'marker' in plot_kwargs:
                ax[0].plot(r0, r1, **plot_kwargs)
            else:  # When markers are unspecified, point estimates aren't displayed. This avoids
                ax[0].plot(r0, r1, 'o', **plot_kwargs)
        ax[0].set_xlim([0, 1])
        ax[0].set_ylim([0, 1])
        ax[0].set_yticks([0, 1])
        ax[0].set_xticks([0, 1])
        ax[0].set_xlabel("$P(Y|A=0)$")
        ax[0].set_ylabel("$P(Y|A=1)$")
        ax[0].set_title("Additive")

        ax[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        for i in np.linspace(-3.5, 3.5, multiplicative_tuner):
            ax[1].plot(ya0, np.exp(np.log(ya0) + i), color='gray', linewidth=1)
        if r1 is not None and r0 is not None:
            if 'marker' in plot_kwargs:
                ax[1].plot(r0, r1, **plot_kwargs)
            else:  # When markers are unspecified, point estimates aren't displayed. This avoids
                ax[1].plot(r0, r1, 'o', **plot_kwargs)
        ax[1].set_xlim([0, 1])
        ax[1].set_ylim([0, 1])
        ax[1].set_yticks([])
        ax[1].set_xticks([0, 1])
        ax[1].set_xlabel("$P(Y|A=0)$")
        ax[1].set_title("Multiplicative")

    elif scale == 'additive':
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        for i in np.linspace(-1, 1, additive_tuner):
            ax.plot(ya0, ya0 + i, color='gray', linewidth=1)
        if r1 is not None and r0 is not None:
            if 'marker' in plot_kwargs:
                ax.plot(r0, r1, **plot_kwargs)
            else:  # When markers are unspecified, point estimates aren't displayed. This avoids
                ax.plot(r0, r1, 'o', **plot_kwargs)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticks([0, 1])
        ax.set_xlabel("$P(Y|A=0)$")
        ax.set_ylabel("$P(Y|A=1)$")
        ax.set_title("Additive")

    elif scale == 'multiplicative':
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        for i in np.linspace(-3.2, 3.2, multiplicative_tuner):
            plt.plot(ya0, np.exp(np.log(ya0) + i), color='gray', linewidth=1)
        if r1 is not None and r0 is not None:
            if 'marker' in plot_kwargs:
                ax.plot(r0, r1, **plot_kwargs)
            else:  # When markers are unspecified, point estimates aren't displayed. This avoids
                ax.plot(r0, r1, 'o', **plot_kwargs)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticks([0, 1])
        ax.set_xlabel("$P(Y|A=0)$")
        ax.set_ylabel("$P(Y|A=1)$")
        ax.set_title("Multiplicative")

    else:
        raise ValueError("`scale` must be either 'additive', 'multplicative', or 'both'")

    return ax
