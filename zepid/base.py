import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from tabulate import tabulate

from zepid.calc.utils import (risk_ci, incidence_rate_ci, risk_ratio, risk_difference, number_needed_to_treat,
                              odds_ratio, incidence_rate_difference, incidence_rate_ratio, sensitivity, specificity)


#########################################################################################################
# Measures of effect / association
#########################################################################################################
class RiskRatio:
    """Estimate of Risk Ratio with a (1-alpha)*100% Confidence interval from a pandas dataframe. Missing data is
    ignored.

    WARNING: Outcome must be coded as (1: yes, 0:no). Only works for binary outcomes
    """

    def __init__(self, reference=0, alpha=0.05):
        """
        reference:
            -reference category for comparisons
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.reference = reference
        self.alpha = alpha
        self.risks = []
        self.risk_ratio = []
        self.results = None
        self._a_list = []
        self._b_list = []
        self._c = None
        self._d = None
        self._labels = []
        self._fit = False
        self._missing_e = None
        self._missing_d = None
        self._missing_ed = None

    def fit(self, df, exposure, outcome):
        """
        Calculates the Risk Ratio

        df:
            -pandas dataframe containing variables of interest
        exposure:
            -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
        outcome:
            -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
        """
        # Setting up holders for results
        risk_lcl = []
        risk_ucl = []
        risk_sd = []
        rr_lcl = []
        rr_ucl = []
        rr_sd = []

        # Getting unique values and dropping reference
        vals = set(df[exposure].dropna().unique())
        vals.remove(self.reference)
        self._c = df.loc[(df[exposure] == self.reference) & (df[outcome] == 1)].shape[0]
        self._d = df.loc[(df[exposure] == self.reference) & (df[outcome] == 0)].shape[0]
        self._labels.append('Ref:'+str(self.reference))
        ri, lr, ur, sd, *_ = risk_ci(events=self._c, total=(self._c + self._d), alpha=self.alpha)
        self.risks.append(ri)
        risk_lcl.append(lr)
        risk_ucl.append(ur)
        risk_sd.append(sd)
        self.risk_ratio.append(1)
        rr_lcl.append(None)
        rr_ucl.append(None)
        rr_sd.append(None)

        # Going through all the values
        for i in vals:
            self._labels.append(str(i))
            a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
            self._a_list.append(a)
            b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
            self._b_list.append(b)
            ri, lr, ur, sd, *_ = risk_ci(events=a, total=(a+b), alpha=self.alpha)
            self.risks.append(ri)
            risk_lcl.append(lr)
            risk_ucl.append(ur)
            risk_sd.append(sd)
            em, lcl, ucl, sd, *_ = risk_ratio(a=a, b=b, c=self._c, d=self._d, alpha=self.alpha)
            self.risk_ratio.append(em)
            rr_lcl.append(lcl)
            rr_ucl.append(ucl)
            rr_sd.append(sd)

        # Getting the extent of missing data
        self._missing_ed = df.loc[(df[exposure].isnull()) & (df[outcome].isnull())].shape[0]
        self._missing_e = df.loc[df[exposure].isnull()].shape[0] - self._missing_ed
        self._missing_d = df.loc[df[outcome].isnull()].shape[0] - self._missing_ed

        # Setting up results
        rf = pd.DataFrame(index=self._labels)
        rf['Risk'] = self.risks
        rf['SD(Risk)'] = risk_sd
        rf['Risk_LCL'] = risk_lcl
        rf['Risk_UCL'] = risk_ucl
        rf['RiskRatio'] = self.risk_ratio
        rf['SD(RR)'] = rr_sd
        rf['RR_LCL'] = rr_lcl
        rf['RR_UCL'] = rr_ucl
        rf['CLR'] = rf['RR_UCL'] / rf['RR_LCL']
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        for a, b, l in zip(self._a_list, self._b_list, self._labels):
            print('Comparison:'+str(self.reference)+' to '+self._labels[self._labels.index(l)+1])
            print(tabulate([['E=1', a, b], ['E=0', self._c, self._d]], headers=['', 'D=1', 'D=0'],
                           tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.results[['Risk', 'SD(Risk)', 'Risk_LCL', 'Risk_UCL']].round(decimals=decimal))
        print('======================================================================')
        print(self.results[['RiskRatio', 'SD(RR)', 'RR_LCL', 'RR_UCL']].round(decimals=decimal))
        print('======================================================================')
        print('Missing E:   ', self._missing_e)
        print('Missing D:   ', self._missing_d)
        print('Missing E&D: ', self._missing_ed)
        print('======================================================================')

    def plot(self, measure='risk_ratio', scale='linear', color='k', center=1):
        """Plot the risk ratios or the risks along with their corresponding confidence intervals. This option is an
        alternative to summary(), which displays results in a table format.

        Parameters
        ----------
        measure : str, optional
            Whether to display risk ratios or risks. Default is to display the risk ratio. Options are;
            * 'risk_ratio'  : display risk ratios
            * 'risk'        : display risks
        scale : str, optional
            Scale for the x-axis. Default is a linear scale. A log-scale can be requested by setting scale='log'
        color : str, optional
            Color to display points and confidence limits. Allows any valid matplotlib colors
        center : str, optional
            Sets a reference line. For the risk ratio, the reference line defaults to 1. For risks, no reference line is
            displayed.

        Returns
        -------
        matplotlib axes
        """
        if measure == 'risk_ratio':
            ax = _plotter(estimate=self.results['RiskRatio'], lcl=self.results['RR_LCL'], ucl=self.results['RR_UCL'],
                          labels=self.results.index,
                          center=center, color=color)
            if scale == 'log':
                try:
                    ax.set_xscale('log')
                except:
                    raise ValueError('For the log scale, all values must be positive')
            ax.set_title('Risk Ratio')
        elif measure == 'risk':
            ax = _plotter(estimate=self.results['Risk'], lcl=self.results['Risk_LCL'], ucl=self.results['Risk_UCL'],
                          labels=self.results.index,
                          center=np.nan, color=color)
            ax.set_title('Risk')
            ax.set_xlim([0, 1])
        else:
            raise ValueError('Must specify either "risk_ratio" or "risk" for plots')
        return ax


class RiskDifference:
    """Estimate of Risk Difference with a (1-alpha)*100% Confidence interval from a pandas dataframe. Missing data is
    ignored.

    WARNING: Outcome must be coded as (1: yes, 0:no). Only works for binary outcomes
    """
    def __init__(self, reference=0, alpha=0.05):
        """
        reference:
            -reference category for comparisons. Default is zero
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.reference = reference
        self.alpha = alpha
        self.risks = []
        self.risk_difference = []
        self.results = None
        self._a_list = []
        self._b_list = []
        self._c = None
        self._d = None
        self._labels = []
        self._fit = False
        self._missing_e = None
        self._missing_d = None
        self._missing_ed = None

    def fit(self, df, exposure, outcome):
        """
        Calculates the Risk Difference

        df:
            -pandas dataframe containing variables of interest
        exposure:
            -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
        outcome:
            -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
        """
        # Setting up holders for results
        risk_lcl = []
        risk_ucl = []
        risk_sd = []
        rd_lcl = []
        rd_ucl = []
        rd_sd = []

        # Getting unique values and dropping reference
        vals = set(df[exposure].dropna().unique())
        vals.remove(self.reference)
        self._c = df.loc[(df[exposure] == self.reference) & (df[outcome] == 1)].shape[0]
        self._d = df.loc[(df[exposure] == self.reference) & (df[outcome] == 0)].shape[0]
        self._labels.append('Ref:' + str(self.reference))
        ri, lr, ur, sd, *_ = risk_ci(events=self._c, total=(self._c + self._d), alpha=self.alpha)
        self.risks.append(ri)
        risk_lcl.append(lr)
        risk_ucl.append(ur)
        risk_sd.append(sd)
        self.risk_difference.append(0)
        rd_lcl.append(None)
        rd_ucl.append(None)
        rd_sd.append(None)

        # Going through all the values
        for i in vals:
            self._labels.append(str(i))
            a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
            self._a_list.append(a)
            b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
            self._b_list.append(b)
            ri, lr, ur, sd, *_ = risk_ci(events=a, total=(a + b), alpha=self.alpha)
            self.risks.append(ri)
            risk_lcl.append(lr)
            risk_ucl.append(ur)
            risk_sd.append(sd)
            em, lcl, ucl, sd, *_ = risk_difference(a=a, b=b, c=self._c, d=self._d, alpha=self.alpha)
            self.risk_difference.append(em)
            rd_lcl.append(lcl)
            rd_ucl.append(ucl)
            rd_sd.append(sd)

        # Getting the extent of missing data
        self._missing_ed = df.loc[(df[exposure].isnull()) & (df[outcome].isnull())].shape[0]
        self._missing_e = df.loc[df[exposure].isnull()].shape[0] - self._missing_ed
        self._missing_d = df.loc[df[outcome].isnull()].shape[0] - self._missing_ed

        # Setting up results
        rf = pd.DataFrame(index=self._labels)
        rf['Risk'] = self.risks
        rf['SD(Risk)'] = risk_sd
        rf['Risk_LCL'] = risk_lcl
        rf['Risk_UCL'] = risk_ucl
        rf['RiskDifference'] = self.risk_difference
        rf['SD(RD)'] = rd_sd
        rf['RD_LCL'] = rd_lcl
        rf['RD_UCL'] = rd_ucl
        rf['CLD'] = rf['RD_UCL'] - rf['RD_LCL']
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        Prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        for a, b, l in zip(self._a_list, self._b_list, self._labels):
            print('Comparison:'+str(self.reference)+' to '+self._labels[self._labels.index(l)+1])
            print(tabulate([['E=1', a, b], ['E=0', self._c, self._d]], headers=['', 'D=1', 'D=0'],
                           tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.results[['Risk', 'SD(Risk)', 'Risk_LCL', 'Risk_UCL']].round(decimals=decimal))
        print('======================================================================')
        print(self.results[['RiskDifference', 'SD(RD)', 'RD_LCL', 'RD_UCL']].round(decimals=decimal))
        print('======================================================================')
        print('Missing E:   ', self._missing_e)
        print('Missing D:   ', self._missing_d)
        print('Missing E&D: ', self._missing_ed)
        print('======================================================================')

    def plot(self, measure='risk_difference', color='k', center=0):
        """Plot the risk differences or the risks along with their corresponding confidence intervals. This option is an
        alternative to summary(), which displays results in a table format.

        Parameters
        ----------
        measure : str, optional
            Whether to display risk differences or risks. Default is to display the risk difference. Options are;
            * 'risk_difference' : display risk differences
            * 'risk'            : display risks
        color : str, optional
            Color to display points and confidence limits. Allows any valid matplotlib colors
        center : str, optional
            Sets a reference line. For the risk difference, the reference line defaults to 0. For risks, no reference
            line is displayed.

        Returns
        -------
        matplotlib axes
        """
        if measure == 'risk_difference':
            ax = _plotter(estimate=self.results['RiskDifference'], lcl=self.results['RD_LCL'],
                          ucl=self.results['RD_UCL'], labels=self.results.index,
                          center=center, color=color)
            ax.set_title('Risk Difference')
        elif measure == 'risk':
            ax = _plotter(estimate=self.results['Risk'], lcl=self.results['Risk_LCL'], ucl=self.results['Risk_UCL'],
                          labels=self.results.index,
                          center=np.nan, color=color)
            ax.set_title('Risk')
            ax.set_xlim([0, 1])
        else:
            raise ValueError('Must specify either "risk_difference" or "risk" for plots')
        return ax


class NNT:
    """Estimates of Number Needed to Treat. NNT (1-alpha)*100% confidence interval presentation is based on
    Altman, DG (BMJ 1998). Missing data is ignored.

    WARNING: Outcome must be coded as (1: yes, 0:no). Only works for binary outcomes
    """
    def __init__(self, reference=0, alpha=0.05):
        """
        reference:
            -reference category for comparisons
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.reference = reference
        self.alpha = alpha
        self.number_needed_to_treat = []
        self.results = None
        self._a_list = []
        self._b_list = []
        self._c = None
        self._d = None
        self._labels = []
        self._fit = False
        self._missing_e = None
        self._missing_d = None
        self._missing_ed = None

    def fit(self, df, exposure, outcome):
        """
        Calculates the NNT

        df:
            -pandas dataframe containing variables of interest
        exposure:
            -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
        outcome:
            -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
        """
        # Setting up holders for results
        nnt_lcl = []
        nnt_ucl = []
        nnt_sd = []

        # Getting unique values and dropping reference
        vals = set(df[exposure].dropna().unique())
        vals.remove(self.reference)
        self._c = df.loc[(df[exposure] == self.reference) & (df[outcome] == 1)].shape[0]
        self._d = df.loc[(df[exposure] == self.reference) & (df[outcome] == 0)].shape[0]
        self._labels.append('Ref:' + str(self.reference))
        self.number_needed_to_treat.append(math.inf)
        nnt_lcl.append(None)
        nnt_ucl.append(None)
        nnt_sd.append(None)

        # Going through all the values
        for i in vals:
            self._labels.append(str(i))
            a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
            self._a_list.append(a)
            b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
            self._b_list.append(b)
            em, lcl, ucl, sd, *_ = number_needed_to_treat(a=a, b=b, c=self._c, d=self._d, alpha=self.alpha)
            self.number_needed_to_treat.append(em)
            nnt_lcl.append(lcl)
            nnt_ucl.append(ucl)
            nnt_sd.append(sd)

        # Getting the extent of missing data
        self._missing_ed = df.loc[(df[exposure].isnull()) & (df[outcome].isnull())].shape[0]
        self._missing_e = df.loc[df[exposure].isnull()].shape[0] - self._missing_ed
        self._missing_d = df.loc[df[outcome].isnull()].shape[0] - self._missing_ed

        # Setting up results
        rf = pd.DataFrame(index=self._labels)
        rf['NNT'] = self.number_needed_to_treat
        rf['SD(RD)'] = nnt_sd
        rf['NNT_LCL'] = nnt_lcl
        rf['NNT_UCL'] = nnt_ucl
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        for i, r in self.results.iterrows():
            if i == self._labels[0]:
                pass
            else:
                print('======================================================================')
                if r['NNT'] == math.inf:
                    print('Number Needed to Treat = infinite')
                else:
                    if r['NNT'] > 0:
                        print('Number Needed to Harm: ', round(abs(r['NNT']), decimal))
                    if r['NNT'] < 0:
                        print('Number Needed to Treat: ', round(abs(r['NNT']), decimal))
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: ')
                if r['NNT_LCL'] < 0 < r['NNT_UCL']:
                    print('NNT ', round(abs(r['NNT_LCL']), decimal), 'to infinity to NNH ',
                          round(abs(r['NNT_UCL']), decimal))
                elif 0 < r['NNT_LCL']:
                    print('NNT ', round(abs(r['NNT_LCL']), decimal), ' to ', round(abs(r['NNT_UCL']), decimal))
                else:
                    print('NNH ', round(abs(r['NNT_LCL']), decimal), ' to ', round(abs(r['NNT_UCL']), decimal))
                print('======================================================================')
                print('======================================================================')
                print('Missing E:   ', self._missing_e)
                print('Missing D:   ', self._missing_d)
                print('Missing E&D: ', self._missing_ed)
                print('======================================================================')


class OddsRatio:
    """Estimates of Odds Ratio with a (1-alpha)*100% Confidence interval. Missing data is ignored.

    WARNING: Outcome must be coded as (1: yes, 0:no). Only works for binary outcomes
    """

    def __init__(self, reference=0, alpha=0.05):
        """
        reference:
            -reference category for comparisons
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.reference = reference
        self.alpha = alpha
        self.odds_ratio = []
        self.results = None
        self._a_list = []
        self._b_list = []
        self._c = None
        self._d = None
        self._labels = []
        self._fit = False
        self._missing_e = None
        self._missing_d = None
        self._missing_ed = None

    def fit(self, df, exposure, outcome):
        """
        Calculates the Odds Ratio

        df:
            -pandas dataframe containing variables of interest
        exposure:
            -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
        outcome:
            -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
        """
        # Setting up holders for results
        odr_lcl = []
        odr_ucl = []
        odr_sd = []

        # Getting unique values and dropping reference
        vals = set(df[exposure].dropna().unique())
        vals.remove(self.reference)
        self._c = df.loc[(df[exposure] == self.reference) & (df[outcome] == 1)].shape[0]
        self._d = df.loc[(df[exposure] == self.reference) & (df[outcome] == 0)].shape[0]
        self._labels.append('Ref:'+str(self.reference))
        self.odds_ratio.append(1)
        odr_lcl.append(None)
        odr_ucl.append(None)
        odr_sd.append(None)

        # Going through all the values
        for i in vals:
            self._labels.append(str(i))
            a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
            self._a_list.append(a)
            b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
            self._b_list.append(b)
            em, lcl, ucl, sd, *_ = odds_ratio(a=a, b=b, c=self._c, d=self._d, alpha=self.alpha)
            self.odds_ratio.append(em)
            odr_lcl.append(lcl)
            odr_ucl.append(ucl)
            odr_sd.append(sd)

        # Getting the extent of missing data
        self._missing_ed = df.loc[(df[exposure].isnull()) & (df[outcome].isnull())].shape[0]
        self._missing_e = df.loc[df[exposure].isnull()].shape[0] - self._missing_ed
        self._missing_d = df.loc[df[outcome].isnull()].shape[0] - self._missing_ed

        # Setting up results
        rf = pd.DataFrame(index=self._labels)
        rf['OddsRatio'] = self.odds_ratio
        rf['SD(OR)'] = odr_sd
        rf['OR_LCL'] = odr_lcl
        rf['OR_UCL'] = odr_ucl
        rf['CLR'] = rf['OR_UCL'] / rf['OR_LCL']
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        for a, b, l in zip(self._a_list, self._b_list, self._labels):
            print('Comparison:'+str(self.reference)+' to '+self._labels[self._labels.index(l)+1])
            print(tabulate([['E=1', a, b], ['E=0', self._c, self._d]], headers=['', 'D=1', 'D=0'],
                           tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.results[['OddsRatio', 'SD(OR)', 'OR_LCL', 'OR_UCL']].round(decimals=decimal))
        print('======================================================================')
        print('Missing E:   ', self._missing_e)
        print('Missing D:   ', self._missing_d)
        print('Missing E&D: ', self._missing_ed)
        print('======================================================================')

    def plot(self, scale='linear', color='k', center=1):
        """Plot the odds ratios along with their corresponding confidence intervals. This option is an
        alternative to summary(), which displays results in a table format.

        Parameters
        ----------
        scale : str, optional
            Scale for the x-axis. Default is a linear scale. A log-scale can be requested by setting scale='log'
        color : str, optional
            Color to display points and confidence limits. Allows any valid matplotlib colors
        center : str, optional
            Sets a reference line. The reference line defaults to 1.

        Returns
        -------
        matplotlib axes
        """
        ax = _plotter(estimate=self.results['OddsRatio'], lcl=self.results['OR_LCL'], ucl=self.results['OR_UCL'],
                      labels=self.results.index,
                      center=center, color=color)
        if scale == 'log':
            try:
                ax.set_xscale('log')
            except:
                raise ValueError('For the log scale, all values must be positive')
        ax.set_title('Odds Ratio')
        return ax


class IncidenceRateRatio:
    """Estimates of Incidence Rate Ratio with a (1-alpha)*100% Confidence interval. Missing data is ignored.

    WARNING: Outcome must be coded as (1: yes, 0:no). Only works for binary outcomes
    """
    def __init__(self, reference=0, alpha=0.05):
        """
        reference:
            -reference category for comparisons
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.reference = reference
        self.alpha = alpha
        self.incidence_rate = []
        self.incidence_rate_ratio = []
        self.results = None
        self._a_list = []
        self._a_time_list = []
        self._c = None
        self._c_time = None
        self._labels = []
        self._fit = False
        self._missing_e = None
        self._missing_d = None
        self._missing_ed = None
        self._missing_t = None

    def fit(self, df, exposure, outcome, time):
        """
        Calculate the Incidence Rate Ratio

        df:
            -pandas dataframe containing variables of interest
        exposure:
            -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
        outcome:
            -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
        """
        # Setting up holders for results
        ir_lcl = []
        ir_ucl = []
        ir_sd = []
        irr_lcl = []
        irr_ucl = []
        irr_sd = []

        # Getting unique values and dropping reference
        vals = set(df[exposure].dropna().unique())
        vals.remove(self.reference)
        self._c = df.loc[(df[exposure] == self.reference) & (df[outcome] == 1)].shape[0]
        self._c_time = df.loc[df[exposure] == self.reference][time].sum()
        self._labels.append('Ref:'+str(self.reference))
        ri, lr, ur, sd, *_ = incidence_rate_ci(events=self._c, time=self._c_time, alpha=self.alpha)
        self.incidence_rate.append(ri)
        ir_lcl.append(lr)
        ir_ucl.append(ur)
        ir_sd.append(sd)
        self.incidence_rate_ratio.append(1)
        irr_lcl.append(None)
        irr_ucl.append(None)
        irr_sd.append(None)

        # Going through all the values
        for i in vals:
            self._labels.append(str(i))
            a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
            self._a_list.append(a)
            a_t = df.loc[df[exposure] == i][time].sum()
            self._a_time_list.append(a_t)
            ri, lr, ur, sd, *_ = incidence_rate_ci(events=a, time=a_t, alpha=self.alpha)
            self.incidence_rate.append(ri)
            ir_lcl.append(lr)
            ir_ucl.append(ur)
            ir_sd.append(sd)
            em, lcl, ucl, sd, *_ = incidence_rate_ratio(a=a, t1=a_t, c=self._c, t2=self._c_time, alpha=self.alpha)
            self.incidence_rate_ratio.append(em)
            irr_lcl.append(lcl)
            irr_ucl.append(ucl)
            irr_sd.append(sd)

        # Getting the extent of missing data
        self._missing_ed = df.loc[(df[exposure].isnull()) & (df[outcome].isnull())].shape[0]
        self._missing_e = df.loc[df[exposure].isnull()].shape[0] - self._missing_ed
        self._missing_d = df.loc[df[outcome].isnull()].shape[0] - self._missing_ed
        self._missing_t = df.loc[df[time].isnull()].shape[0]

        # Setting up results
        rf = pd.DataFrame(index=self._labels)
        rf['IncRate'] = self.incidence_rate
        rf['SD(IncRate)'] = ir_sd
        rf['IncRate_LCL'] = ir_lcl
        rf['IncRate_UCL'] = ir_ucl
        rf['IncRateRatio'] = self.incidence_rate_ratio
        rf['SD(IRR)'] = irr_sd
        rf['IRR_LCL'] = irr_lcl
        rf['IRR_UCL'] = irr_ucl
        rf['CLR'] = rf['IRR_UCL'] / rf['IRR_LCL']
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        for a, a_t, l in zip(self._a_list, self._a_time_list, self._labels):
            print('Comparison:'+str(self.reference)+' to '+self._labels[self._labels.index(l)+1])
            print(tabulate([['E=1', a, a_t], ['E=0', self._c, self._c_time]], headers=['', 'D=1', 'Person-time'],
                           tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.results[['IncRate', 'SD(IncRate)', 'IncRate_LCL', 'IncRate_UCL']].round(decimals=decimal))
        print('======================================================================')
        print(self.results[['IncRateRatio', 'SD(IRR)', 'IRR_LCL', 'IRR_UCL']].round(decimals=decimal))
        print('======================================================================')
        print('Missing E:   ', self._missing_e)
        print('Missing D:   ', self._missing_d)
        print('Missing E&D: ', self._missing_ed)
        print('Missing T:   ', self._missing_t)
        print('======================================================================')

    def plot(self, measure='incidence_rate_ratio', scale='linear', color='k', center=1):
        """Plot the risk ratios or the risks along with their corresponding confidence intervals. This option is an
        alternative to summary(), which displays results in a table format.

        Parameters
        ----------
        measure : str, optional
            Whether to display incidence rate ratios or incidence rates. Default is to display the incidence rate ratio.
            Options are;
            * 'incidence_rate_ratio'  : display incidence rate ratios
            * 'incidence_rate'        : display incidence rates
        scale : str, optional
            Scale for the x-axis. Default is a linear scale. A log-scale can be requested by setting scale='log'
        color : str, optional
            Color to display points and confidence limits. Allows any valid matplotlib colors
        center : str, optional
            Sets a reference line. For the incidence rate ratio, the reference line defaults to 1. For incidence rates,
            no reference line is displayed.

        Returns
        -------
        matplotlib axes
        """
        if measure == 'incidence_rate_ratio':
            ax = _plotter(estimate=self.results['IncRateRatio'], lcl=self.results['IRR_LCL'],
                          ucl=self.results['IRR_UCL'], labels=self.results.index,
                          center=center, color=color)
            if scale == 'log':
                try:
                    ax.set_xscale('log')
                except:
                    raise ValueError('For the log scale, all values must be positive')
            ax.set_title('Incidence Rate Ratio')
        elif measure == 'incidence_rate':
            ax = _plotter(estimate=self.results['IncRate'], lcl=self.results['IncRate_LCL'],
                          ucl=self.results['IncRate_UCL'], labels=self.results.index,
                          center=np.nan, color=color)
            ax.set_title('Incidence Rate')
            ax.set_xlim([0, 1])
        else:
            raise ValueError('Must specify either "incidence_rate_ratio" or "incidence_rate" for plots')
        return ax


class IncidenceRateDifference:
    """Estimates of Incidence Rate Difference with a (1-alpha)*100% Confidence interval. Missing data is ignored.

    WARNING: Outcome must be coded as (1: yes, 0:no). Only works for binary outcomes
    """
    def __init__(self, reference=0, alpha=0.05):
        """
        reference:
            -reference category for comparisons
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.reference = reference
        self.alpha = alpha
        self.incidence_rate = []
        self.incidence_rate_difference = []
        self.results = None
        self._a_list = []
        self._a_time_list = []
        self._c = None
        self._c_time = None
        self._labels = []
        self._fit = False
        self._missing_e = None
        self._missing_d = None
        self._missing_ed = None
        self._missing_t = None

    def fit(self, df, exposure, outcome, time):
        """
        Calculates the Incidence Rate Difference

        df:
            -pandas dataframe containing variables of interest
        exposure:
            -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
        outcome:
            -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
        """
        # Setting up holders for results
        ir_lcl = []
        ir_ucl = []
        ir_sd = []
        ird_lcl = []
        ird_ucl = []
        ird_sd = []

        # Getting unique values and dropping reference
        vals = set(df[exposure].dropna().unique())
        vals.remove(self.reference)
        self._c = df.loc[(df[exposure] == self.reference) & (df[outcome] == 1)].shape[0]
        self._c_time = df.loc[df[exposure] == self.reference][time].sum()
        self._labels.append('Ref:'+str(self.reference))
        ri, lr, ur, sd, *_ = incidence_rate_ci(events=self._c, time=self._c_time, alpha=self.alpha)
        self.incidence_rate.append(ri)
        ir_lcl.append(lr)
        ir_ucl.append(ur)
        ir_sd.append(sd)
        self.incidence_rate_difference.append(0)
        ird_lcl.append(None)
        ird_ucl.append(None)
        ird_sd.append(None)

        # Going through all the values
        for i in vals:
            self._labels.append(str(i))
            a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
            self._a_list.append(a)
            a_t = df.loc[df[exposure] == i][time].sum()
            self._a_time_list.append(a_t)
            ri, lr, ur, sd, *_ = incidence_rate_ci(events=a, time=a_t, alpha=self.alpha)
            self.incidence_rate.append(ri)
            ir_lcl.append(lr)
            ir_ucl.append(ur)
            ir_sd.append(sd)
            em, lcl, ucl, sd, *_ = incidence_rate_difference(a=a, t1=a_t, c=self._c, t2=self._c_time, alpha=self.alpha)
            self.incidence_rate_difference.append(em)
            ird_lcl.append(lcl)
            ird_ucl.append(ucl)
            ird_sd.append(sd)

        # Getting the extent of missing data
        self._missing_ed = df.loc[(df[exposure].isnull()) & (df[outcome].isnull())].shape[0]
        self._missing_e = df.loc[df[exposure].isnull()].shape[0] - self._missing_ed
        self._missing_d = df.loc[df[outcome].isnull()].shape[0] - self._missing_ed
        self._missing_t = df.loc[df[time].isnull()].shape[0]

        # Setting up results
        rf = pd.DataFrame(index=self._labels)
        rf['IncRate'] = self.incidence_rate
        rf['SD(IncRate)'] = ir_sd
        rf['IncRate_LCL'] = ir_lcl
        rf['IncRate_UCL'] = ir_ucl
        rf['IncRateDiff'] = self.incidence_rate_difference
        rf['SD(IRD)'] = ird_sd
        rf['IRD_LCL'] = ird_lcl
        rf['IRD_UCL'] = ird_ucl
        rf['CLD'] = rf['IRD_UCL'] - rf['IRD_LCL']
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        for a, a_t, l in zip(self._a_list, self._a_time_list, self._labels):
            print('Comparison:'+str(self.reference)+' to '+self._labels[self._labels.index(l)+1])
            print(tabulate([['E=1', a, a_t], ['E=0', self._c, self._c_time]], headers=['', 'D=1', 'Person-time'],
                           tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.results[['IncRate', 'SD(IncRate)', 'IncRate_LCL', 'IncRate_UCL']].round(decimals=decimal))
        print('======================================================================')
        print(self.results[['IncRateDiff', 'SD(IRD)', 'IRD_LCL', 'IRD_UCL']].round(decimals=decimal))
        print('======================================================================')
        print('Missing E:   ', self._missing_e)
        print('Missing D:   ', self._missing_d)
        print('Missing E&D: ', self._missing_ed)
        print('Missing T:   ', self._missing_t)
        print('======================================================================')

    def plot(self, measure='incidence_rate_difference', color='k', center=0):
        """Plot the incidence rate differences or the incidence rates along with their corresponding confidence
        intervals. This option is an alternative to summary(), which displays results in a table format.

        Parameters
        ----------
        measure : str, optional
            Whether to display incidence rate ratios or incidence rates. Default is to display the incidence rate
            differences. Options are;
            * 'incidence_rate_difference'  : display incidence rate differences
            * 'incidence_rate'             : display incidence rates
        color : str, optional
            Color to display points and confidence limits. Allows any valid matplotlib colors
        center : str, optional
            Sets a reference line. For the incidence rate difference, the reference line defaults to 0. For incidence
            rates, no reference line is displayed.

        Returns
        -------
        matplotlib axes
        """
        if measure == 'incidence_rate_difference':
            ax = _plotter(estimate=self.results['IncRateDiff'], lcl=self.results['IRD_LCL'],
                          ucl=self.results['IRR_UCL'], labels=self.results.index,
                          center=center, color=color)
            ax.set_title('Incidence Rate Difference')
        elif measure == 'incidence_rate':
            ax = _plotter(estimate=self.results['IncRate'], lcl=self.results['IncRate_LCL'],
                          ucl=self.results['IncRate_UCL'], labels=self.results.index,
                          center=np.nan, color=color)
            ax.set_title('Incidence Rate')
            ax.set_xlim([0, 1])
        else:
            raise ValueError('Must specify either "incidence_rate_difference" or "incidence_rate" for plots')
        return ax


def _plotter(estimate, lcl, ucl, labels, center=0, color='k'):
    """Plot functionality to be used by all the measure classes. Hidden functional for all the other plotting
    functionalities
    """
    ypoints = [i for i in range(len(labels))]

    ax = plt.gca()
    # ax.errorbar(estimate, ypoints, xerr=[lcl, ucl], marker='None', ecolor=color, elinewidth=1, linewidth=0)
    ax.hlines(ypoints, lcl, ucl, colors=color, zorder=3)
    ax.scatter(estimate, ypoints, c=color, s=100, marker='o', edgecolors='None', zorder=2)
    if np.isnan(center):
        pass
    else:
        ax.axvline(center, color='gray', zorder=1)
    ax.set_yticklabels(labels)
    ax.set_yticks(ypoints)
    return ax


#########################################################################################################
# Testing measures
#########################################################################################################
class Sensitivity:
    """Generates the sensitivity and (1-alpha)% confidence interval, comparing test results to disease status
    from pandas dataframe

    WARNING: Disease & Test must be coded as (1: yes, 0:no)
    """
    def __init__(self, alpha=0.05):
        """
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.alpha = alpha
        self.sensitivity = None
        self.results = None
        self._fit = False
        self._a = None
        self._b = None

    def fit(self, df, test, disease):
        """
        Calculates the Sensitivity

        df:
            -pandas dataframe containing variables of interest
        test:
            -column name of test results to detect the outcome. Needs to be coded as binary (0,1), where 1 indicates a
            positive test for the individual
        disease:
            -column name of true outcomes status. Needs to be coded as binary (0,1), where 1 indicates the individual
             has the outcome
        """
        self._a = df.loc[(df[test] == 1) & (df[disease] == 1)].shape[0]
        self._b = df.loc[(df[test] == 1) & (df[disease] == 0)].shape[0]
        se, ls, us, sd = sensitivity(detected=self._a, cases=(self._a + self._b), alpha=self.alpha)
        self.sensitivity = se

        # Setting up results
        rf = pd.DataFrame()
        rf['Sensitivity'] = [se]
        rf['SD(Se)'] = [sd]
        rf['Se_LCL'] = [ls]
        rf['Se_UCL'] = [us]
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        Prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        print(tabulate([['T+', self._a, self._b]], headers=['', 'D+', 'D-'], tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.results[['Sensitivity', 'SD(Se)', 'Se_LCL', 'Se_UCL']].round(decimals=decimal))
        print('======================================================================')


class Specificity:
    """Generates the sensitivity and (1-alpha)% confidence interval, comparing test results to disease status
    from pandas dataframe

    WARNING: Disease & Test must be coded as (1: yes, 0:no)
    """
    def __init__(self, alpha=0.05):
        """
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.alpha = alpha
        self.specificity = None
        self.results = None
        self._fit = False
        self._c = None
        self._d = None

    def fit(self, df, test, disease):
        """
        Calculates the Specificity

        df:
            -pandas dataframe containing variables of interest
        test:
            -column name of test results to detect the outcome. Needs to be coded as binary (0,1), where 1 indicates a
            positive test for the individual
        disease:
            -column name of true outcomes status. Needs to be coded as binary (0,1), where 1 indicates the individual
             has the outcome
        """
        self._c = df.loc[(df[test] == 0) & (df[disease] == 1)].shape[0]
        self._d = df.loc[(df[test] == 0) & (df[disease] == 0)].shape[0]
        sp, ls, us, sd = specificity(detected=self._c, noncases=(self._c + self._d), alpha=self.alpha)
        self.specificity = sp

        # Setting up results
        rf = pd.DataFrame()
        rf['Specificity'] = [sp]
        rf['SD(Sp)'] = [sd]
        rf['Sp_LCL'] = [ls]
        rf['Sp_UCL'] = [us]
        self.results = rf
        self._fit = True

    def summary(self, decimal=3):
        """
        Prints the summary results

        decimal:
            -amount of decimal points to display. Default is 3
        """
        if self._fit is False:
            raise ValueError('fit() function must be completed before results can be obtained')

        print(tabulate([['T-', self._c, self._d]], headers=['', 'D+', 'D-'], tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.results[['Specificity', 'SD(Sp)', 'Sp_LCL', 'Sp_UCL']].round(decimals=decimal))
        print('======================================================================')


class Diagnostics:
    """Generates the Sensitivity, Specificity, and the corresponding (1-alpha)% confidence intervals, comparing test
    results to disease status from pandas dataframe

    WARNING: Disease & Test must be coded as (1: yes, 0:no)
    """
    def __init__(self, alpha=0.05):
        """
        alpha:
            -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
        """
        self.alpha = alpha
        self.sensitivity = None
        self.specificity = None
        self.results = None
        self._fit = False
        self._a = None
        self._b = None
        self._c = None
        self._d = None

    def fit(self, df, test, disease):
        """
        Calculates the Sensitivity and Specificity

        df:
            -pandas dataframe containing variables of interest
        test:
            -column name of test results to detect the outcome. Needs to be coded as binary (0,1), where 1 indicates a
            positive test for the individual
        disease:
            -column name of true outcomes status. Needs to be coded as binary (0,1), where 1 indicates the individual
             has the outcome
        """
        self.sensitivity = Sensitivity(alpha=self.alpha)
        self.sensitivity.fit(df=df, test=test, disease=disease)
        self.specificity = Specificity(alpha=self.alpha)
        self.specificity.fit(df=df, test=test, disease=disease)

    def summary(self, decimal=3):
        """
        Prints the results

        decimal:
            -number of decimal places to display. Default is 3
        """
        print(tabulate([['T+', self.sensitivity._a, self.sensitivity._b],
                        ['T-', self.specificity._c, self.specificity._d]],
                       headers=['', 'D+', 'D-'], tablefmt='grid'), '\n')
        print('======================================================================')
        print(self.sensitivity.results[['Sensitivity', 'SD(Se)', 'Se_LCL', 'Se_UCL']].round(decimals=decimal))
        print(self.specificity.results[['Specificity', 'SD(Sp)', 'Sp_LCL', 'Sp_UCL']].round(decimals=decimal))
        print('======================================================================')


#########################################################################################################
# Interaction contrasts
#########################################################################################################
def interaction_contrast(df, exposure, outcome, modifier, adjust=None, decimal=3, print_results=True):
    """Calculate the Interaction Contrast (IC) using a pandas dataframe and statsmodels to fit a linear
    binomial regression. Can ONLY be used for a 0,1 coded exposure and modifier (exposure = {0,1}, modifier = {0,1},
    outcome = {0,1}). Can handle adjustment for other confounders in the regression model. Prints the fit
    of the linear binomial regression, the IC, and the corresponding IC 95% confidence interval.

    NOTE: statsmodels may produce a domain error in some versions.

    df:
        -pandas dataframe containing variables of interest
    exposure:
        -column name of exposure variable. Must be coded as (0,1) where 1 is exposure
    outcome:
        -column name of outcome variable. Must be coded as (0,1) where 1 is outcome of interest
    modifier:
        -column name of modifier variable. Must be coded as (0,1) where 1 is modifier
    adjust:
        -string of other variables to adjust for, in correct statsmodels format. Default is None
        NOTE: variables can NOT be named {E1M0,E0M1,E1M1} since this function creates variables with those names.
              Answers will be incorrect
         Ex) '+ C1 + C2 + C3 + Z'
    decimal:
        -Number of decimals to display in result. Default is 3


    Example of Output)
                     Generalized Linear Model Regression Results
    ==============================================================================
    Dep. Variable:                      D   No. Observations:                  210
    Model:                            GLM   Df Residuals:                      204
    Model Family:                Binomial   Df Model:                            5
    Link Function:               identity   Scale:                             1.0
    Method:                          IRLS   Log-Likelihood:                -97.450
    Date:                Thu, 03 May 2018   Deviance:                       194.90
    Time:                        18:46:13   Pearson chi2:                     198.
    No. Iterations:                    79
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.6101      0.542      1.125      0.260      -0.453       1.673
    X              0.2049      0.056      3.665      0.000       0.095       0.314
    Z              0.1580      0.049      3.207      0.001       0.061       0.255
    E1M1          -0.2105      0.086     -2.447      0.014      -0.379      -0.042
    var1        7.544e-05    6.7e-05      1.125      0.260    -5.6e-05       0.000
    var2          -0.0248      0.022     -1.125      0.260      -0.068       0.018
    ==============================================================================

    ----------------------------------------------------------------------
    Interaction Contrast
    ----------------------------------------------------------------------

    IC:		-0.21047
    95% CI:		(-0.37908, -0.04186)
    ----------------------------------------------------------------------
    """
    df.loc[((df[exposure] == 1) & (df[modifier] == 1)), 'E1M1'] = 1
    df.loc[((df[exposure] != 1) | (df[modifier] != 1)), 'E1M1'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E1M1'] = np.nan
    if adjust is None:
        eq = outcome + ' ~ ' + exposure + ' + ' + modifier + ' + E1M1'
    else:
        eq = outcome + ' ~ ' + exposure + ' + ' + modifier + ' + E1M1 + ' + adjust
    f = sm.families.family.Binomial(sm.families.links.identity)
    model = smf.glm(eq, df, family=f).fit()
    ic = model.params['E1M1']
    lcl = model.conf_int().loc['E1M1'][0]
    ucl = model.conf_int().loc['E1M1'][1]
    if print_results:
        print(model.summary())
        print('\n----------------------------------------------------------------------')
        print('Interaction Contrast')
        print('----------------------------------------------------------------------')
        print('\nIC:\t\t' + str(round(ic, decimal)))
        print('95% CI:\t\t(' + str(round(lcl, decimal)) + ', ' + str(round(ucl, decimal)) + ')')
        print('----------------------------------------------------------------------')
    return ic, lcl, ucl


def interaction_contrast_ratio(df, exposure, outcome, modifier, adjust=None, regression='logit', ci='delta',
                               b_sample=200, alpha=0.05, decimal=5, print_results=True):
    """Calculate the Interaction Contrast Ratio (ICR) using a pandas dataframe, and conducts either log binomial
    or logistic regression through statsmodels. Can ONLY be used for a 0,1 coded exposure and modifier (exposure = {0,1},
    modifier = {0,1}, outcome = {0,1}). Can handle missing data and adjustment for other confounders in the regression
    model. Prints the fit of the binomial regression, the ICR, and the corresponding ICR confidence interval. Confidence
    intervals can be generated using the delta method or bootstrap method

    NOTE: statsmodels may produce a domain error for log binomial models in some versions

    df:
        -pandas dataframe containing variables of interest
    exposure:
        -column name of exposure variable. Must be coded as (0,1) where 1 is exposure
    outcome:
        -column name of outcome variable. Must be coded as (0,1) where 1 is outcome of interest
    modifier:
        -column name of modifier variable. Must be coded as (0,1) where 1 is modifier
    adjust:
        -string of other variables to adjust for, in correct statsmodels format. Default is none
        NOTE: variables can NOT be named {E1M0,E0M1,E1M1} since this function creates variables with those names.
              Answers will be incorrect
         Ex) '+ C1 + C2 + C3 + Z'
    regression:
        -Type of regression model to fit. Default is log binomial.
         Options include:
            'log':      Log-binomial model. Estimates the Relative Risk (RR)
            'logit':    Logistic (logit) model. Estimates the Odds Ratio (OR). Note, this is only valid when the
                        OR approximates the RR
    ci:
        -Type of confidence interval to return. Default is the delta method. Options include:
            'delta':      Delta method as described by Hosmer and Lemeshow (1992)
            'bootstrap':  bootstrap method (Assmann et al. 1996). The delta method is more time efficient than bootstrap
    b_sample:
        -Number of times to resample to generate bootstrap confidence intervals. Only important if bootstrap confidence
         intervals are requested. Default is 1000
    alpha:
        -Alpha level for confidence interval. Default is 0.05
    decimal:
        -Number of decimal places to display in result. Default is 3
    """
    df.loc[((df[exposure] == 1) & (df[modifier] == 0)), 'E1M0'] = 1
    df.loc[((df[exposure] != 1) | (df[modifier] != 0)), 'E1M0'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E1M0'] = 0
    df.loc[((df[exposure] == 0) & (df[modifier] == 1)), 'E0M1'] = 1
    df.loc[((df[exposure] != 0) | (df[modifier] != 1)), 'E0M1'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E0M1'] = 0
    df.loc[((df[exposure] == 1) & (df[modifier] == 1)), 'E1M1'] = 1
    df.loc[((df[exposure] != 1) | (df[modifier] != 1)), 'E1M1'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E1M1'] = np.nan
    if regression == 'logit':
        f = sm.families.family.Binomial(sm.families.links.logit)
        print('Note: Using the Odds Ratio to calculate the ICR is only valid when\nthe OR approximates the RR')
        # TODO replace this with a warning looking at prevalence. Now should default to logit
    elif regression == 'log':
        f = sm.families.family.Binomial(sm.families.links.log)
    if adjust is None:
        eq = outcome + ' ~ E1M0 + E0M1 + E1M1'
    else:
        eq = outcome + ' ~ E1M0 + E0M1 + E1M1 + ' + adjust
    model = smf.glm(eq, df, family=f).fit()
    em10 = math.exp(model.params['E1M0'])
    em01 = math.exp(model.params['E0M1'])
    em11 = math.exp(model.params['E1M1'])
    em_expect = em10 + em01 - 1
    icr = em11 - em_expect
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    if ci == 'delta':
        cov_matrix = model.cov_params()
        vb10 = cov_matrix.loc['E1M0']['E1M0']
        vb01 = cov_matrix.loc['E0M1']['E0M1']
        vb11 = cov_matrix.loc['E1M1']['E1M1']
        cvb10_01 = cov_matrix.loc['E1M0']['E0M1']
        cvb10_11 = cov_matrix.loc['E1M0']['E1M1']
        cvb01_11 = cov_matrix.loc['E0M1']['E1M1']
        varICR = (((em10 ** 2) * vb10) + ((em01 ** 2) * vb01) + ((em11 ** 2) * vb11) + (
        (em10 * em01 * 2 * cvb10_01)) + (-1 * em10 * em11 * 2 * cvb10_11) + (-1 * em01 * em11 * 2 * cvb01_11))
        icr_lcl = icr - zalpha * math.sqrt(varICR)
        icr_ucl = icr + zalpha * math.sqrt(varICR)
    elif ci == 'bootstrap':
        print('Running bootstrap... please wait...')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bse_icr = []
            ul = 1 - alpha / 2
            ll = 0 + alpha / 2
            for i in range(b_sample):
                dfs = df.sample(n=df.shape[0], replace=True)
                try:
                    bmodel = smf.glm(eq, dfs, family=f).fit()
                    em_bexpect = math.exp(bmodel.params['E1M0']) + math.exp(bmodel.params['E0M1']) - 1
                    bicr = math.exp(bmodel.params['E1M1']) - em_bexpect
                    sigma = bicr - icr
                    bse_icr.append(sigma)
                except:
                    bse_icr.append(np.nan)
            bsdf = pd.DataFrame()
            bsdf['sigma'] = bse_icr
            lsig, usig = bsdf['sigma'].dropna().quantile(q=[ll, ul])
            icr_lcl = lsig + icr
            icr_ucl = usig + icr
    else:
        raise ValueError('Please specify a supported confidence interval type')
    if print_results:
        print(model.summary())
        print('\n----------------------------------------------------------------------')
        if regression == 'logit':
            print('ICR based on Odds Ratio\t\tAlpha = ' + str(alpha))
            print('Note: Using the Odds Ratio to calculate the ICR is only valid when\nthe OR approximates the RR')
        elif regression == 'log':
            print('ICR based on Risk Ratio\t\tAlpha = ' + str(alpha))
        print('\nICR:\t\t' + str(round(icr, decimal)))
        print('CI:\t\t(' + str(round(icr_lcl, decimal)) + ', ' + str(round(icr_ucl, decimal)) + ')')
        print('----------------------------------------------------------------------')
    return icr, icr_lcl, icr_ucl


#########################################################################################################
# Other
#########################################################################################################
def spline(df, var, n_knots=3, knots=None, term=1, restricted=False):
    """Creates spline dummy variables based on either user specified knot locations or automatically
    determines knot locations based on percentiles. Options are available to set the number of knots,
    location of knots (value), term (linear, quadratic, etc.), and restricted/unrestricted.

    Returns a pandas dataframe containing the spline variables (labeled 0 to n_knots)

    df:
        -pandas dataframe containing the variables of interest
    var:
        -continuous variable to generate spline for
    n_knots:
        -number of knots requested. Options for knots include any positive integer if the location of
         knots are specified. If knot locations are not specified, n_knots must be an integer between
         1 to 7, including both. Default is set to 3
    knots:
        -Location of specified knots in a list. To specify the location of knots, put desired numbers for
         knots into a list. Be sure that the length of the list is the same as the specified number of knots.
         Default is None, so that the function will automatically determine knot locations without user specification
    term:
        -High order term for the spline terms. To calculate a quadratic spline change to 2, cubic spline
         change to 3, etc. Default is 1, so a linear spline is returned
    restricted:
        -Whether to return a restricted spline. Note that the restricted spline returns one less column
         than the number of knots. An unrestricted spline returns the same number of columns as the number of knots.
         Default is False, providing an unrestricted spline


    Example of Output)
           rspline0     rspline1   rspline2
    0   9839.409066  1234.154601   2.785600
    1    446.391437     0.000000   0.000000
    2   7107.550298   409.780251   0.000000
    3   4465.272901     7.614501   0.000000
    4  10972.041543  1655.208555  52.167821
    ..          ...          ...        ...
    """
    if knots is None:
        if n_knots == 1:
            knots = [0.5]
        elif n_knots == 2:
            knots = [1 / 3, 2 / 3]
        elif n_knots == 3:
            knots = [0.05, 0.5, 0.95]
        elif n_knots == 4:
            knots = [0.05, 0.35, 0.65, 0.95]
        elif n_knots == 5:
            knots = [0.05, 0.275, 0.50, 0.725, 0.95]
        elif n_knots == 6:
            knots = [0.05, 0.23, 0.41, 0.59, 0.77, 0.95]
        elif n_knots == 7:
            knots = [0.025, 11 / 60, 26 / 75, 0.50, 79 / 120, 49 / 60, 0.975]
        else:
            raise ValueError(
                'When the knot locations are not pre-specified, the number of specified knots must be'
                ' an integer between 1 and 7')
        pts = list(df[var].quantile(q=knots))
    else:
        if n_knots != len(knots):
            raise ValueError('The number of knots and the number of specified knots must match')
        else:
            pass
        pts = knots
    if sorted(pts) != pts:
        raise ValueError('Knots must be in ascending order')
    colnames = []
    sf = df.copy()
    for i in range(len(pts)):
        colnames.append('spline' + str(i))
        sf['spline' + str(i)] = np.where(sf[var] > pts[i], (sf[var] - pts[i]) ** term, 0)
        sf['spline' + str(i)] = np.where(sf[var].isnull(), np.nan, sf['spline' + str(i)])
    if restricted is False:
        return sf[colnames]
    elif restricted is True:
        rsf = sf.copy()
        colnames = []
        for i in range(len(pts) - 1):
            colnames.append('rspline' + str(i))
            rsf['rspline' + str(i)] = np.where(rsf[var] > pts[i],
                                               rsf['spline' + str(i)] - rsf['spline' + str(len(pts) - 1)], 0)
            rsf['rspline' + str(i)] = np.where(rsf[var].isnull(), np.nan, rsf['rspline' + str(i)])
        return rsf[colnames]
    else:
        raise ValueError('restricted must be set to either True or False')


def table1_generator(df, cols, variable_type, continuous_measure='median', strat_by=None, decimal=3):
    """Code to automatically generate a descriptive table of your study population (often referred to as a
    Table 1). Personally, I hate copying SAS/R/Python output from the interpreter to an Excel or other
    spreadsheet software. This code will generate a pandas dataframe object. This object will be a formatted
    table which can be exported as a CSV, opened in Excel, then final formatting changes/renaming can be done.
    Variables with np.nan values are counted as missing

    Categorical variables will be divided into the unique numbers and have a percent calculated. Additionally,
    missing data will be counted (but is not included in the percent). Additionally, a single categorical variable
    can be used to present the results

    Continuous variables either have median/IQR or mean/SE calculated depending on what is requested. Missing are
    counted as a separate category

    Returns a pandas dataframe object containing a formatted Table 1. It is not recommended that this table is used
    in any part of later analysis, since is id difficult to parse through the table. This function is only meant to
    reduce the amount of copying from output needed.

    df:
        -pandas dataframe object containing all variables of interest
    cols:
        -list of columns of variable names to include in the table. Ex) ['X',var1','var2']
    variable_types:
        -list of strings indicating the variable types. Ex) ['category','continuous','continuous']
         Options
            'category'      :   variable with categories only
            'continuous'    :   continuous variable
    continuous_measure:
        -Whether to use the medians or the means. Default is median
         Options
            'median'    :   returns medians and IQR for continuous variables
            'mean'      :   returns means and SE for continuous variables
    strat_by:
        -What categorical variable to stratify by. Default is None (no stratification)
    decimal:
        -Number of decimals to display in the table. Default is 3


    Example of Output)
    _                                D=0                             D=1
    __                           % / IQR           n             % / IQR          n

    Variable
    TOTAL                                 310.000000                      74.000000
    X        1.0                0.608187  104.000000            0.692308  27.000000
             0.0                0.391813   67.000000            0.307692  12.000000
             Missing                      139.000000                      35.000000
    Z        1.0                0.722581  224.000000            0.635135  47.000000
             0.0                0.277419   86.000000            0.364865  27.000000
             Missing                        0.000000                       0.000000
    var1              [468.231, 525.312]  497.262978  [481.959, 538.964] 507.286133
             Missing                        0.000000                       0.000000
    var2                [24.454, 25.731]   25.058982      [24.1, 25.607]  24.816898
             Missing                        0.000000                       0.000000
    var3                [24.446, 25.685]   25.037731    [24.388, 25.563]  24.920583
             Missing                        0.000000
    """
    if len(cols) != len(variable_type):
        raise ValueError('List of columns must be the same length as the list of variable types')
    if continuous_measure != 'median' and continuous_measure != 'mean':
        raise ValueError("'median' or 'mean' must be requested as the continuous_measure")

    # Unstratified Table 1
    if strat_by is None:
        rlist = []
        for i in cols:
            vn = cols.index(i)
            if continuous_measure == 'median':
                if variable_type[vn] == 'continuous':
                    rf = pd.DataFrame({'n / Median': [np.median(df[i].dropna()), df[i].isnull().sum()],
                                       '% / IQR': [np.percentile(df[i].dropna(), [25, 75]).round(decimals=decimal),
                                                   '']}, index=['', 'Missing'])
                if variable_type[vn] == 'category':
                    x = df[i].value_counts()
                    m = df[i].isnull().sum()
                    rf = pd.DataFrame({'n / Median': x, '% / IQR': x / x.sum()})
                    rf = rf.append(pd.DataFrame({'n / Median': m, '% / IQR': ''}, index=['Missing']))
            if continuous_measure == 'mean':
                if variable_type[vn] == 'continuous':
                    rf = pd.DataFrame({'n / Mean': [np.mean(df[i].dropna()), df[i].isnull().sum()],
                                       '% / SE': [np.std(df[i].dropna()).round(decimals=decimal), '']},
                                      index=['', 'Missing'])
                if variable_type[vn] == 'category':
                    x = df[i].value_counts(dropna=False)
                    y = df[i].value_counts(dropna=False)
                    rf = pd.DataFrame({'n / Mean': x, '% / SE': y / y.sum()})
            rlist.append(rf)
        srf = pd.concat(rlist, keys=cols, names=['Variable'])
        if continuous_measure == 'median':
            return srf[['n / Median', '% / IQR']]
        if continuous_measure == 'mean':
            return srf[['n / Mean', '% / SE']]

    # Stratified Table 1
    if strat_by is not None:
        v = df[strat_by].dropna().unique()
        slist = []
        nlist = []
        for j in v:
            sf = df.loc[df[strat_by] == j].copy()
            rlist = []
            for i in cols:
                vn = cols.index(i)
                if variable_type[vn] == 'category':
                    x = sf[i].value_counts()
                    m = sf[i].isnull().sum()
                    if continuous_measure == 'median':
                        rf = pd.DataFrame({'n / Median': x, '% / IQR': x / x.sum()})
                        rf = rf.append(pd.DataFrame({'n / Median': m, '% / IQR': ''}, index=['Missing']))
                    if continuous_measure == 'mean':
                        rf = pd.DataFrame({'n / Mean': x, '% / SD': x / x.sum()})
                        rf = rf.append(pd.DataFrame({'n / Mean': m, '% / SD': ''}, index=['Missing']))
                if variable_type[vn] == 'continuous':
                    if continuous_measure == 'median':
                        rf = pd.DataFrame({'n / Median': [np.median(sf[i].dropna()), sf[i].isnull().sum()],
                                           '% / IQR': [np.percentile(sf[i].dropna(), [25, 75]).round(decimals=decimal),
                                                       '']}, index=['', 'Missing'])
                    if continuous_measure == 'mean':
                        rf = pd.DataFrame({'n / Mean': [np.mean(sf[i].dropna()), sf[i].isnull().sum()],
                                          '% / SD': [np.std(sf[i].dropna()).round(decimals=decimal), '']},
                                          index=['', 'Missing'])
                rlist.append(rf)
            if continuous_measure == 'median':
                c = pd.DataFrame({'n / Median': len(sf), '% / IQR': ''}, index=[''])
            if continuous_measure == 'mean':
                c = pd.DataFrame({'n / Mean': len(sf), '% / SD': ''}, index=[''])
            try:  # avoids pandas error
                rff = pd.concat([c] + rlist, keys=['TOTAL'] + cols, names=['Variable'], axis=0, sort=False)
            except TypeError:  # gets around pandas <0.22 error
                rff = pd.concat([c] + rlist, keys=['TOTAL'] + cols, names=['Variable'], axis=0)
            slist.append(rff)
            if continuous_measure == 'median':
                nlist.append((strat_by + '=' + str(j), '% / IQR'))
            if continuous_measure == 'mean':
                nlist.append((strat_by + '=' + str(j), '% / SD'))
            nlist.append((strat_by + '=' + str(j), 'n'))
        index = pd.MultiIndex.from_tuples(nlist, names=['_', '__'])
        try:  # avoids pandas error
            srf = pd.concat(slist, keys=cols, names=['Variable'], axis=1, sort=False)
        except TypeError:
            srf = pd.concat(slist, keys=cols, names=['Variable'], axis=1)
        print(srf.columns)
        print(index)
        srf.columns = index
        return srf
