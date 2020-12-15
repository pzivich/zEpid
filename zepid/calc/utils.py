import warnings
from collections import namedtuple

import numpy as np
from scipy.stats import norm

Results = namedtuple('Results',
                     ['point_estimate', 'lower_bound', 'upper_bound', 'standard_error', 'alpha', 'measure']
                     )


def normal_ppf(z):
    return norm.ppf(z, loc=0, scale=1)


def check_positivity_or_throw(*args):
    for arg in args:
        if arg <= 0:
            raise ValueError('Value must be positive, however %f is not positive' % arg)


def check_nonnegativity_or_throw(*args):
    for arg in args:
        if arg < 0:
            raise ValueError('Value must be non-negative, however %f is negative' % arg)


def warn_if_normal_approximation_invalid(*args):
    for arg in args:
        if arg <= 5:
            warnings.warn('At least one cell count is less than 5, therefore confidence '
                          'interval approximation is invalid', UserWarning)
            # just print once
            break


def risk_ci(events, total, alpha=0.05, confint='wald'):
    r"""Calculate two-sided risk confidence intervals

    Risk is calculated from

    .. math::

        R = \frac{a}{a+b}

    Wald standard error is

    .. math::

        SE_{Wald} = \left(\frac{1}{a} - \frac{1}{b}\right)^{\frac{1}{2}}

    Hypergeometric standard error is

    .. math::

        SE_{HypGeo} = \left(\frac{a b}{(a+b)^2  (a+b-1)}\right)^{\frac{1}{2}}

    Parameters
    ----------
    events : integer, float
        Number of events/outcomes that occurred
    total : integer, float
        Total number of subjects that could have experienced the event
    alpha : float, optional
        Alpha level. Default is 0.05
    confint : string, optional
        Type of confidence interval to generate. Current options include Wald or Hypergeometric confidence intervals

    Returns
    ---------
    tuple
        Tuple containing risk, lower CL, upper CL, SE

    Note
    ----
    Confidence intervals rely on the central limit theorem, so there must be at least 5 events and 5 nonevents

    Examples
    --------
    Estimate the risk, standard error, and confidence intervals

    >>> from zepid.calc import risk_ci
    >>> r = risk_ci(45, 100)

    Extracting the estimated risk

    >>> r.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> r.lower_bound
    >>> r.upper_bound

    Extracting the standard error

    >>> r.standard_error
    """
    risk = events / total
    c = 1 - alpha / 2
    zalpha = normal_ppf(c)
    if confint == 'wald':
        sd = np.sqrt((risk * (1 - risk)) / total)
        # follows SAS9.4: http://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/viewer.htm#procstat_
        # freq_details37.htm#procstat.freq.freqbincl
        lower = risk - zalpha * sd
        upper = risk + zalpha * sd
    elif confint == 'hypergeometric':
        sd = np.sqrt(events * (total - events) / (total ** 2 * (total - 1)))
        lower = risk - zalpha * sd
        upper = risk + zalpha * sd
    else:
        raise ValueError('Only wald and hypergeometric confidence intervals are currently supported')
    return Results(risk, lower, upper, sd, alpha, 'risk')


def incidence_rate_ci(events, time, alpha=0.05):
    r"""Calculate two-sided incidence rate confidence intervals. Only Wald-type confidence intervals are currently
    implemented.

    Incidence rate is calculated from

    .. math::

        I = \frac{a}{t}

    Incidence rate standard error is

    .. math::

        SE = \left(\frac{a}{t^2}\right)^\frac{1}{2})

    Parameters
    -------------
    events : integer, float
        Number of events/outcomes that occurred
    time : integer, float
        Total person-time contributed in this group
    alpha : float, optional
        Alpha level. Default is 0.05


    Returns
    ------------
    tuple
        Tuple containing incidence rate, lower CL, upper CL, SE

    Examples
    --------
    Estimate the incidence rate, standard error, and confidence intervals

    >>> from zepid.calc import incidence_rate_ci
    >>> i = incidence_rate_ci(56, 503)

    Extracting the estimated incidence rate

    >>> i.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> i.lower_bound
    >>> i.upper_bound

    Extracting the standard error

    >>> i.standard_error
    """
    c = 1 - alpha / 2
    ir = events / time
    zalpha = normal_ppf(c)
    sd = np.sqrt(events / (time ** 2))
    # https://www.researchgate.net/post/How_to_estimate_standard_error_from_incidence_rate_and_population
    # Incidence rate confidence intervals are a mess, with not sources agreeing...
    # http://www.openepi.com/PersonTime1/PersonTime1.htm
    # zEpid uses a normal approximation. There are too many options for CI's...
    lower = ir - zalpha * sd
    upper = ir + zalpha * sd
    return Results(ir, lower, upper, sd, alpha, 'incidence rate')


def risk_ratio(a, b, c, d, alpha=0.05):
    r"""Calculates the risk ratio and confidence intervals from count data.

    Risk ratio is calculated from

    .. math::

        RR = \frac{a}{a + b} / \frac{c}{c + d}

    Risk ratio standard error is

    .. math::

        SE = \left(\frac{1}{a} - \frac{1}{a + b} + \frac{1}{c} - \frac{1}{c + d}\right)^{\frac{1}{2}}

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    b : integer, float
        Count of unexposed individuals with outcome
    c : integer, float
        Count of exposed individuals without outcome
    d : integer, float
        Count of unexposed individuals without outcome
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval

    Returns
    --------------
    tuple
        Tuple of risk ratio, lower CL, upper CL, SE

    Examples
    --------
    Estimate the risk ratio, standard error, and confidence intervals

    >>> from zepid.calc import risk_ratio
    >>> rr = risk_ratio(45, 55, 21, 79)

    Extracting the estimated risk ratio

    >>> rr.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> rr.lower_bound
    >>> rr.upper_bound

    Extracting the standard error

    >>> rr.standard_error
    """
    check_positivity_or_throw(a, b, c, d)
    warn_if_normal_approximation_invalid(a, b, c, d)

    zalpha = normal_ppf(1 - alpha / 2)
    r1 = a / (a + b)
    r0 = c / (c + d)
    relrisk = r1 / r0
    sd = np.sqrt((1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + d)))
    lnrr = np.log(relrisk)
    lcl = np.exp(lnrr - (zalpha * sd))
    ucl = np.exp(lnrr + (zalpha * sd))
    return Results(relrisk, lcl, ucl, sd, alpha, 'risk ratio')


def risk_difference(a, b, c, d, alpha=0.05):
    r"""Calculates the risk difference and confidence intervals from count data.

    Risk difference is calculated as

    .. math::

        RD = \frac{a}{a + b} - \frac{c}{c + d}

    Risk difference standard error is calculated as

    .. math::

        SE = \left(\frac{R_1 \times (1 - R_1)}{a+b} + \frac{R_0 \times (1-R_0)}{c+d}\right)^{\frac{1}{2}}

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    b : integer, float
        Count of unexposed individuals with outcome
    c : integer, float
        Count of exposed individuals without outcome
    d : integer, float
        Count of unexposed individuals without outcome
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval

    Returns
    --------------
    tuple
        Tuple of risk difference, lower CL, upper CL, SE

    Examples
    --------
    Estimate the risk difference, standard error, and confidence intervals

    >>> from zepid.calc import risk_difference
    >>> rd = risk_difference(45, 55, 21, 79)

    Extracting the estimated risk difference

    >>> rd.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> rd.lower_bound
    >>> rd.upper_bound

    Extracting the standard error

    >>> rd.standard_error
    """
    check_positivity_or_throw(a, b, c, d)
    warn_if_normal_approximation_invalid(a, b, c, d)
    zalpha = normal_ppf(1 - alpha / 2)
    r1 = a / (a + b)
    r0 = c / (c + d)
    riskdiff = r1 - r0
    sd = np.sqrt((r1 * (1 - r1)) / (a + b) + (r0 * (1 - r0)) / (c + d))
    # TODO hypergeometric CL for later implementation
    # sd = np.sqrt(((a * b) / ((a + b) ** 2 * (a + b - 1))) + ((c * d) / (((c + d) ** 2) * (c + d - 1))))
    lcl = riskdiff - (zalpha * sd)
    ucl = riskdiff + (zalpha * sd)
    return Results(riskdiff, lcl, ucl, sd, alpha, 'risk difference')


def number_needed_to_treat(a, b, c, d, alpha=0.05):
    r"""Calculates the number needed to treat and confidence intervals from count data.

    Number needed to treat is calculated as

    .. math::

        NNT = \frac{1}{RD} = \frac{1}{\frac{a}{a + b} - \frac{c}{c + d}}

    Confidence intervals are calculated by taking the inverse of the lower and upper confidence limits of the risk
    difference. The formula for the risk difference standard error is

    .. math::

        SE = \left(\frac{R_1 \times (1 - R_1)}{a+b} + \frac{R_0 \times (1-R_0)}{c+d}\right)^{\frac{1}{2}}

    Note
    ----
    If the risk difference confidence limits cover the null (RD=0), that means the interpretation will switch from
    NNT to NNH (number needed to harm). See Altman 1998 for further details on interpretation is this scenario

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    b : integer, float
        Count of unexposed individuals with outcome
    c : integer, float
        Count of exposed individuals without outcome
    d : integer, float
        Count of unexposed individuals without outcome
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval

    Returns
    --------------
    tuple
        Tuple of NNT, lower CL, upper CL, SE

    Examples
    --------
    Estimate the number needed to treat, standard error, and confidence intervals

    >>> from zepid.calc import number_needed_to_treat
    >>> nnt = number_needed_to_treat(45, 55, 21, 79)

    Extracting the estimated number needed to treat

    >>> nnt.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> nnt.lower_bound
    >>> nnt.upper_bound

    Extracting the standard error

    >>> nnt.standard_error
    """
    check_positivity_or_throw(a, b, c, d)
    warn_if_normal_approximation_invalid(a, b, c, d)

    zalpha = normal_ppf(1 - alpha / 2)
    r1 = a / (a + b)
    r0 = c / (c + d)
    riskdiff = r1 - r0
    sd = np.sqrt((r1 * (1 - r1)) / (a + b) + (r0 * (1 - r0)) / (c + d))
    # TODO hypergeometric CL for later implementation
    # sd = np.sqrt(((a * b) / ((a + b) ** 2 * (a + b - 1))) + ((c * d) / (((c + d) ** 2) * (c + d - 1))))
    lcl_rd = riskdiff - (zalpha * sd)
    ucl_rd = riskdiff + (zalpha * sd)
    if riskdiff != 0:
        numbnt = 1 / riskdiff
    else:
        numbnt = np.inf
    if lcl_rd != 0:
        lcl = 1 / lcl_rd
    else:
        lcl = np.inf
    if ucl_rd != 0:
        ucl = 1 / ucl_rd
    else:
        ucl = np.inf
    return Results(numbnt, lcl, ucl, sd, alpha, 'number needed to treat')


def odds_ratio(a, b, c, d, alpha=0.05):
    r"""Calculates the odds ratio and confidence interval from count data

    Odds ratio is calculated from

    .. math::

        OR = \frac{a}{b} / \frac{c}{d}

    Odds ratio standard error is

    .. math::

        SE = \left(\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}\right)^{\frac{1}{2}}

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    b : integer, float
        Count of unexposed individuals with outcome
    c : integer, float
        Count of exposed individuals without outcome
    d : integer, float
        Count of unexposed individuals without outcome
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval

    Returns
    --------------
    tuple
        Tuple of OR, lower CL, upper CL, SE

    Examples
    --------
    Estimate the odds ratio, standard error, and confidence intervals

    >>> from zepid.calc import odds_ratio
    >>> odr = odds_ratio(45, 55, 21, 79)

    Extracting the estimated odds ratio

    >>> odr.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> odr.lower_bound
    >>> odr.upper_bound

    Extracting the standard error

    >>> odr.standard_error
    """
    check_positivity_or_throw(a, b, c, d)
    warn_if_normal_approximation_invalid(a, b, c, d)

    zalpha = normal_ppf(1 - alpha / 2)
    or1 = a / b
    or0 = c / d
    oddsr = or1 / or0
    sd = np.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
    lnor = np.log(oddsr)
    lcl = np.exp(lnor - (zalpha * sd))
    ucl = np.exp(lnor + (zalpha * sd))
    return Results(oddsr, lcl, ucl, sd, alpha, 'odds ratio')


def incidence_rate_ratio(a, c, t1, t2, alpha=0.05):
    r"""Calculates the incidence rate ratio and confidence intervals from count data

    Incidence rate ratio is calculated from

    .. math::

        IR = \frac{a}{t_1} / \frac{c}{t_2}

    Incidence rate ratio standard error is

    .. math::

        SE = \left(\frac{1}{a} + \frac{1}{c}\right)^{\frac{1}{2}}

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    c : integer, float
        Count of unexposed individuals with outcome
    t1 : integer, float
        Person-time contributed by those who were exposed
    t2 : integer, float
        Person-time contributed by those who were unexposed
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval

    Returns
    --------------
    tuple
        Tuple of IR, lower CL, upper CL, SE

    Examples
    --------
    Estimate the incidence rate ratio, standard error, and confidence intervals

    >>> from zepid.calc import incidence_rate_ratio
    >>> ir = incidence_rate_ratio(45, 21, 109, 158)

    Extracting the estimated incidence rate ratio

    >>> ir.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> ir.lower_bound
    >>> ir.upper_bound

    Extracting the standard error

    >>> ir.standard_error
    """
    check_positivity_or_throw(a, c)
    check_nonnegativity_or_throw(t2, t1)
    warn_if_normal_approximation_invalid(a, c)

    zalpha = normal_ppf(1 - alpha / 2)
    irate1 = a / t1
    irate2 = c / t2
    irater = irate1 / irate2
    sd = np.sqrt((1 / a) + (1 / c))
    lnirr = np.log(irater)
    lcl = np.exp(lnirr - (zalpha * sd))
    ucl = np.exp(lnirr + (zalpha * sd))
    return Results(irater, lcl, ucl, sd, alpha, 'incidence rate ratio')


def incidence_rate_difference(a, c, t1, t2, alpha=0.05):
    r"""Calculates the incidence rate difference and confidence intervals from count data

    Incidence rate difference is calculated from

    .. math::

        ID = \frac{a}{t_1} - \frac{c}{t_2}

    Incidence rate difference standard error is

    .. math::

        SE = \left(\frac{a}{t_1^2} + \frac{c}{t_2^2}\right)^{\frac{1}{2}}

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    c : integer, float
        Count of unexposed individuals with outcome
    t1 : integer, float
        Person-time contributed by those who were exposed
    t2 : integer, float
        Person-time contributed by those who were unexposed
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval

    Returns
    --------------
    tuple
        Tuple of ID, lower CL, upper CL, SE)

    Examples
    --------
    Estimate the incidence rate ratio, standard error, and confidence intervals

    >>> from zepid.calc import incidence_rate_difference
    >>> ird = incidence_rate_difference(45, 21, 109, 158)

    Extracting the estimated incidence rate ratio

    >>> ird.point_estimate

    Extracting the lower and upper confidence intervals, respectively

    >>> ird.lower_bound
    >>> ird.upper_bound

    Extracting the standard error

    >>> ird.standard_error
    """
    check_positivity_or_throw(a, c)
    check_nonnegativity_or_throw(t2, t1)
    warn_if_normal_approximation_invalid(a, c)

    zalpha = normal_ppf(1 - alpha / 2)
    rated1 = a / t1
    rated2 = c / t2
    irated = rated1 - rated2
    sd = np.sqrt((a / (t1 ** 2)) + (c / (t2 ** 2)))
    lcl = irated - (zalpha * sd)
    ucl = irated + (zalpha * sd)
    return Results(irated, lcl, ucl, sd, alpha, 'incidence rate difference')


def attributable_community_risk(a, b, c, d):
    r"""Calculates the estimated attributable community risk (ACR) from count data. ACR is also known as Population
    Attributable Risk. Since this is commonly confused with the population attributable fraction, the name ACR is used
    to clarify differences in the formulas

    Attributable community risk is calculated as

    .. math::

        ACR = \frac{a + c}{a + b + c + d} - \frac{c}{c + d} = R - R_0

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    b : integer, float
        Count of unexposed individuals with outcome
    c : integer, float
        Count of exposed individuals without outcome
    d : integer, float
        Count of unexposed individuals without outcome

    Returns
    --------------
    float
        Attributable community risk

    Examples
    --------
    Return the attributable community risk

    >>> from zepid.calc import attributable_community_risk
    >>> attributable_community_risk(45, 55, 21, 79)
    """
    check_positivity_or_throw(a, b, c, d)

    rt = (a + c) / (a + b + c + d)
    r0 = c / (c + d)
    return rt - r0


def population_attributable_fraction(a, b, c, d):
    r"""Calculates the population attributable fraction (PAF) from count data

    Population attributable fraction is calculated as

    .. math::

        PAF = \left(\frac{a + c}{a + b + c + d} - \frac{c}{c + d}\right) / \frac{a + c}{a + b + c + d} = (R - R_0) / R

    Parameters
    ------------
    a : integer, float
        Count of exposed individuals with outcome
    b : integer, float
        Count of unexposed individuals with outcome
    c : integer, float
        Count of exposed individuals without outcome
    d : integer, float
        Count of unexposed individuals without outcome

    Returns
    --------------
    float
        Population attributable fraction

    Examples
    --------
    Return the population attributable fraction

    >>> from zepid.calc import population_attributable_fraction
    >>> population_attributable_fraction(45, 55, 21, 79)
    """
    check_positivity_or_throw(a, b, c, d)

    rt = (a + c) / (a + b + c + d)
    r0 = c / (c + d)
    return (rt - r0) / rt


def probability_to_odds(prob):
    r"""Converts given probability (proportion) to odds

    Probability is converted to odds using

    .. math::

        O = \frac{\Pr}{1 - \Pr}

    Parameters
    ---------------
    prob : float, NumPy array
        Probability or array of probabilities to transform into odds

    Returns
    ----------
    odds
        Float or array of odds

    Examples
    --------
    Convert a single probability to an odds

    >>> from zepid.calc import probability_to_odds
    >>> probability_to_odds(0.3)

    Convert an array of probabilities to odds

    >>> import numpy as np
    >>> probs = np.array([0.3, 0.1, 0.2, 0.5, 0.01])
    >>> probability_to_odds(probs)
    """
    return prob / (1 - prob)


def odds_to_probability(odds):
    r"""Converts given odds to probability (proportion)

    Probability is converted to odds using

    .. math::

        \Pr = \frac{O}{1 + O}

    Parameters
    ---------------
    odds : float, NumPy array
        Odds or array of odds to transform into probabilities

    Returns
    ----------
    prob
        Float or array of probabilities

    Examples
    --------
    Convert a single odds to a probability

    >>> from zepid.calc import odds_to_probability
    >>> odds_to_probability(0.45)

    Convert an array of odds to probabilities

    >>> import numpy as np
    >>> odds = np.array([0.3, 0.5, 1, 3.1, 1.1])
    >>> odds_to_probability(odds)
    """
    return odds / (1 + odds)


def logit(prob):
    """Logit transformation of probabilities. Input can be a single probability of array of probabilities

    Parameters
    ----------
    prob : float, array
        A single probability or an array of probabilities

    Returns
    -------
    logit-transformed probabilities
    """
    return np.log(prob / (1 - prob))


def inverse_logit(logodds):
    """Inverse logit transformation. Returns probabilities

    Parameters
    ----------
    logodds : float, array
        A single log-odd or an array of log-odds

    Returns
    -------
    inverse-logit transformed results (i.e. probabilities for log-odds)
    """
    return 1 / (1 + np.exp(-logodds))


def counternull_pvalue(estimate, lcl, ucl, sided='two', alpha=0.05, decimal=3):
    r"""Calculates the counternull p-value. It is useful to prevent over-interpretation of results

    Parameters
    -------------
    estimate : float
        Point estimate for result
    lcl : float
        Lower confidence limit
    ucl : float
        Upper confidence limit
    sided : string, optional
        Whether to compute the upper one-sided, lower one-sided, or two-sided counternull
        p-value. Default is the two-sided
            * 'upper'     Upper one-sided p-value
            * 'lower'     Lower one-sided p-value
            * 'two'       Two-sided p-value
    alpha : float, optional
        Alpha level for p-value. Default is 0.05. Verify that this is the same alpha used to generate confidence
        intervals
    decimal : integer, optional
        Number of decimal places to display. Default is three

    Returns
    ----------
    None
        Function does not return an object. It prints results to the console

    Notes
    --------
    Make sure that the confidence interval points put into the equation match the alpha level calculation

    Examples
    --------
    Calculate the counternull p-value for a single estimate and confidence interval

    >>> from zepid.calc import counternull_pvalue
    >>> counternull_pvalue(-0.1, -0.3, 0.1)

    References
    ----------
    Rosenthal R, Rubin DB. (1994). The counternull value of an effect size: A new statistic. Psychological Science,
    5(6), 329-334.
    """
    zalpha = normal_ppf(1 - alpha / 2)
    se = (ucl - lcl) / (zalpha * 2)
    cnull = 2 * estimate
    up_cn = norm.cdf(x=cnull, loc=estimate, scale=se)
    lp_cn = 1 - up_cn
    lowerp = norm.cdf(x=estimate, loc=cnull, scale=se)
    upperp = 1 - lowerp
    twosip = 2 * min([up_cn, lp_cn])
    print('----------------------------------------------------------------------')
    print('Alpha = ', alpha)
    print('----------------------------------------------------------------------')
    print('Counternull estimate = ', cnull)
    if sided == 'upper':
        print('Upper one-sided counternull p-value: ', round(upperp, decimal))
    elif sided == 'lower':
        print('Lower one-sided counternull p-value: ', round(lowerp, decimal))
    else:
        print('Two-sided counternull p-value: ', round(twosip, decimal))
    print('----------------------------------------------------------------------\n')


def semibayes(prior_mean, prior_lcl, prior_ucl, mean, lcl, ucl, ln_transform=False, alpha=0.05,
              decimal=3, print_results=True):
    r"""A simple Bayesian Analysis. Note that this analysis assumes a normal distribution for the
    continuous measure. See chapter 18 of Modern Epidemiology 3rd Edition (specifically pages 334, 340 for this
    procedure)

    The posterior estimate is calculated as

    .. math::

        E_{posterior} = \frac{\left(E_{prior} \times \frac{1}{Var_{prior}}\right) +
        (E \times \frac{1}{Var})}{E_{prior} \times \frac{1}{Var_{prior}}}

    and the posterior variance is

    .. math::

        Var_{posterior} = \frac{1}{\frac{1}{Var_{prior}} + \frac{1}{Var}}

    Parameters
    ------------
    prior_mean : float
        Prior designated point estimate
    prior_lcl : float
        Prior designated lower confidence limit
    prior_ucl : float
        Prior designated upper confidence limit
    mean : float
        Point estimate result obtained from analysis
    lcl : float
        Lower confidence limit estimate obtained from analysis
    ucl : float
        Upper confidence limit estimate obtained from analysis
    ln_transform : bool, optional
        Whether to natural log transform results before conducting analysis. Should be used for
        RR, OR, or or other Ratio measure. Default is False (use for RD and other absolute measures)
    alpha : float, optional
        Alpha level for confidence intervals. Default is 0.05
    decimal : float, optional
        Number of decimal places to display. Default is three
    print_results : bool, optional
        Whether to print the results of the semi-Bayesian calculations. Default is True

    Returns
    ---------
    tuple
        Tuple of posterior mean, posterior lower CL, posterior upper CL

    Note
    ----
    Make sure that the alpha used to generate the confidence intervals matches the alpha used in this calculation.
    Additionally, this calculation can only handle normally distributed priors and observed

    Examples
    --------
    Posterior Risk Difference

    >>> from zepid.calc import semibayes
    >>> semibayes(prior_mean=-0.15, prior_lcl=-0.5, prior_ucl=0.2, mean=-0.1, lcl=-0.3, ucl=0.1, print_results=False)

    Posterior Risk Ratio

    >>> semibayes(prior_mean=0.9, prior_lcl=0.75, prior_ucl=1.2, mean=0.85, lcl=0.77, ucl=0.91, ln_transform=True)

    References
    ----------
    Rothman KJ, Greenland S, Lash TL. (2008). Modern epidemiology (Vol. 3). Philadelphia: Wolters Kluwer
    Health/Lippincott Williams & Wilkins.
    """
    # Transforming to log scale if ratio measure
    if ln_transform:
        prior_mean = np.log(prior_mean)
        prior_lcl = np.log(prior_lcl)
        prior_ucl = np.log(prior_ucl)
        mean = np.log(mean)
        lcl = np.log(lcl)
        ucl = np.log(ucl)
    zalpha = normal_ppf(1 - alpha / 2)

    # Extracting prior SD
    prior_sd = (prior_ucl - prior_lcl) / (2 * zalpha)
    prior_var = prior_sd ** 2
    prior_w = 1 / prior_var

    # Extracting observed SD
    sd = (ucl - lcl) / (2 * zalpha)
    var = sd ** 2
    w = 1 / var

    # Checking Prior
    check = (mean - prior_mean) / ((var + prior_var) ** (1 / 2))  # ME3 pg340
    approx_check = norm.sf(abs(check)) * 2
    if approx_check < 0.05:
        warnings.warn('The approximate homogeneity check between the prior and the data indicates the prior may be  '
                      'incompatible with the data. Therefore, it may be misleading to summarize the results via a '
                      'semi-Bayesian approach. Either the prior, data, or both are misleading.')

    # Calculating posterior
    post_mean = ((prior_mean * prior_w) + (mean * w)) / (prior_w + w)
    post_var = 1 / (prior_w + w)
    sd = np.sqrt(post_var)
    post_lcl = post_mean - zalpha * sd
    post_ucl = post_mean + zalpha * sd

    # Transforming back if ratio measure
    if ln_transform:
        post_mean = np.exp(post_mean)
        post_lcl = np.exp(post_lcl)
        post_ucl = np.exp(post_ucl)
        prior_mean = np.exp(prior_mean)
        prior_lcl = np.exp(prior_lcl)
        prior_ucl = np.exp(prior_ucl)
        mean = np.exp(mean)
        lcl = np.exp(lcl)
        ucl = np.exp(ucl)

    # Presenting Results
    if print_results:
        print('----------------------------------------------------------------------')
        print('Prior-Data Compatibility Check: p =', round(approx_check, 5))
        print('For compatibility check, small p-values indicate misleading results')
        print('----------------------------------------------------------------------')
        print('Prior Estimate: ', round(prior_mean, decimal))
        print(str(round((1 - alpha) * 100, 1)) + '% Prior Confidence Interval: (', round(prior_lcl, decimal), ', ',
              round(prior_ucl, decimal), ')')
        print('----------------------------------------------------------------------')
        print('Data Estimate: ', round(mean, decimal))
        print(str(round((1 - alpha) * 100, 1)) + '% Confidence Interval: (',
              round(lcl, decimal), ', ', round(ucl, decimal), ')')
        print('----------------------------------------------------------------------')
        print('Posterior Estimate: ', round(post_mean, decimal))
        print(str(round((1 - alpha) * 100, 1)) + '% Posterior Probability Interval: (', round(post_lcl, decimal), ', ',
              round(post_ucl, decimal), ')')
        print('----------------------------------------------------------------------\n')
    return post_mean, post_lcl, post_ucl


def sensitivity(detected, cases, alpha=0.05, confint='wald'):
    r"""Calculate the sensitivity from number of detected cases and the number of total true cases.

    Parameters
    ---------------
    detected : integer, float
        Number of true cases detected via testing criteria
    cases : integer, float
        Total number of true/actual cases
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
    confint : string, optional
        Type of confidence interval to generate. Current options include Wald or Hypergeometric confidence intervals


    Returns
    --------
    tuple
        Tuple of sensitivity, lower CL, upper CL, SE

    Examples
    --------
    Calculating sensitivity

    >>> from zepid.calc import sensitivity
    >>> se = sensitivity(90, 100)

    Extract sensitivity

    >>> se[0]

    Extract confidence intervals for sensitivity

    >>> se[1:3]

    Extract standard error

    >>> se[3]
    """
    check_positivity_or_throw(detected, cases)
    warn_if_normal_approximation_invalid(cases)

    if detected > cases:
        raise ValueError('Detected true cases must be less than or equal to the total number of cases')

    sens = detected / cases
    zalpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    if confint == 'wald':
        sd = np.sqrt((sens * (1 - sens)) / cases)
        # follows SAS9.4: http://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/viewer.htm#procstat_
        # freq_details37.htm#procstat.freq.freqbincl
        lower = sens - zalpha * sd
        upper = sens + zalpha * sd
    elif confint == 'hypergeometric':
        sd = np.sqrt(detected * (cases - detected) / (cases ** 2 * (cases - 1)))
        lower = sens - zalpha * sd
        upper = sens + zalpha * sd
    else:
        raise ValueError('Please specify a valid confidence interval')
    return sens, lower, upper, sd


def specificity(detected, noncases, alpha=0.05, confint='wald'):
    """Calculate the specificity from number of false detections and the number of total non-cases.

    Parameters
    ---------------
    detected : integer, float
        Number of false cases detected via testing criteria
    noncases : integer, float
        Total number of non-cases
    alpha : float, optional
        Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
    confint : string, optional
        Type of confidence interval to generate. Current options include Wald or Hypergeometric confidence intervals

    Returns
    --------
    tuple
        Tuple of specificity, lower CL, upper CL, SE

    Examples
    --------
    Calculating specificity

    >>> from zepid.calc import specificity
    >>> sp = specificity(88, 100)

    Extract specificity

    >>> sp[0]

    Extract confidence intervals for specificity

    >>> sp[1:3]

    Extract standard error

    >>> sp[3]
    """
    check_positivity_or_throw(detected, noncases)
    warn_if_normal_approximation_invalid(noncases)

    if detected > noncases:
        raise ValueError('Detected true cases must be less than or equal to the total number of cases')
    spec = 1 - (detected / noncases)
    zalpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    if confint == 'wald':
        sd = np.sqrt((spec * (1 - spec)) / noncases)
        # follows SAS9.4: http://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/viewer.htm#procstat_
        # freq_details37.htm#procstat.freq.freqbincl
        lower = spec - zalpha * sd
        upper = spec + zalpha * sd
    elif confint == 'hypergeometric':
        sd = np.sqrt(detected * (noncases - detected) / (noncases ** 2 * (noncases - 1)))
        lower = spec - zalpha * sd
        upper = spec + zalpha * sd
    else:
        raise ValueError('Please specify a valid confidence interval')
    return spec, lower, upper, sd


def ppv_converter(sensitivity, specificity, prevalence):
    r"""Generates the positive predictive value from designated sensitivity, specificity, and prevalence.

    Positive predictive value is calculated using

    .. math::

        PPV = \frac{Se \times P}{Se \times P + (1-Sp) (1-P)}

    Parameters
    -------------
    sensitivity : float
        Sensitivity of the testing criteria
    specificity : float
        Specificity of the testing criteria
    prevalence : float
        Prevalence of the outcome in the population

    Returns
    ------------
    float
        Positive predictive value

    Examples
    --------
    Calculate the positivity predictive value

    >>> from zepid.calc import ppv_converter
    >>> ppv_converter(0.9, 0.88, 0.15)
    """
    if (sensitivity > 1) or (specificity > 1) or (prevalence > 1):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    if (sensitivity < 0) or (specificity < 0) or (prevalence < 0):
        raise ValueError('sensitivity/specificity/prevalence cannot be less than 0')
    sens_prev = sensitivity * prevalence
    nspec_nprev = (1 - specificity) * (1 - prevalence)
    ppv = sens_prev / (sens_prev + nspec_nprev)
    return ppv


def npv_converter(sensitivity, specificity, prevalence):
    r"""Generates the negative predictive value from designated sensitivity, specificity, and prevalence.

    .. math::

        NPV = \frac{Sp \times (1-P)}{(1-Se) \times P + Sp \times (1-P)}

    Parameters
    -------------
    sensitivity : float
        Sensitivity of the testing criteria
    specificity : float
        Specificity of the testing criteria
    prevalence : float
        Prevalence of the outcome in the population

    Returns
    ------------
    float
        Negative predictive value

    Examples
    --------
    Calculate the positivity predictive value

    >>> from zepid.calc import npv_converter
    >>> npv_converter(0.9, 0.88, 0.15)
    """
    if (sensitivity > 1) or (specificity > 1) or (prevalence > 1):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    if (sensitivity < 0) or (specificity < 0) or (prevalence < 0):
        raise ValueError('sensitivity/specificity/prevalence cannot be less than 0')
    spec_nprev = specificity * (1 - prevalence)
    nsens_prev = (1 - sensitivity) * prevalence
    npv = spec_nprev / (spec_nprev + nsens_prev)
    return npv


def screening_cost_analyzer(cost_miss_case, cost_false_pos, prevalence, sensitivity,
                            specificity, population=10000, decimal=3):
    """Compares the cost of sensivitiy/specificity of screening criteria to treating the entire population
    as test-negative and test-positive. The lowest per capita cost is considered the ideal choice. Note that
    this function only provides relative costs

    Parameters
    ----------------
    cost_miss_case : float
        The relative cost of missing a case, compared to false positives. In general, set this to 1 then change the
        value under 'cost_false_pos' to reflect the relative cost
    cost_false_pos : float
        The relative cost of a false positive case, compared to a missed case
    prevalence : float
        The prevalence of the disease in the population. Must be a float
    sensitivity : float
        The sensitivity level of the screening test. Must be a float
    specificity : float
        The specificity level of the screening test. Must be a float
    population : float
        The population size to set. Choose a larger value since this is only necessary for total calculations. Default
        is 10,000
    decimal : integer
        Amount of decimal points to display. Default value is 3

    Returns
    ---------
    None
        Prints results to console

    Note
    ----
    When calculating costs, be sure to consult experts in health policy or related fields.  Costs should encompass more
    than just monetary costs, like relative costs (regret, disappointment, stigma, disutility, etc.). Careful
    consideration of relative costs between false positive and false negatives needs to be considered.

    Examples
    --------
    Calculate the (relative) cost for the proposed screening strategy

    >>> from zepid.calc import screening_cost_analyzer
    >>> screening_cost_analyzer(cost_miss_case=1, cost_false_pos=3, prevalence=0.15, sensitivity=0.9, specificity=0.88)
    """
    warnings.warn("NOTE: When calculating costs, be sure to consult experts in health policy or related fields."
                  " Costs should encompass more than only monetary costs, like relative costs (regret, "
                  "disappointment, stigma, disutility, etc.)", UserWarning)

    if (sensitivity > 1) | (specificity > 1):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    disease = population * prevalence
    disease_free = population - disease

    # TEST: no positives
    nt_cost = disease * cost_miss_case
    pc_nt_cost = nt_cost / population

    # TEST: all postives
    t_cost = disease_free * cost_false_pos
    pc_t_cost = t_cost / population

    # TEST: criteria
    cost_b = disease - (disease * sensitivity)
    cost_c = disease_free - (disease_free * specificity)
    ct_cost = (cost_miss_case * cost_b) + (cost_false_pos * cost_c)
    pc_ct_cost = ct_cost / population

    # Present results
    print('----------------------------------------------------------------------')
    print('Treat everyone as Test-Negative')
    print('Total relative cost:\t\t', round(nt_cost, decimal))
    print('Per Capita relative cost:\t', round(pc_nt_cost, decimal))
    print('----------------------------------------------------------------------')
    print('Treat everyone as Test-Positive')
    print('Total relative cost:\t\t', round(t_cost, decimal))
    print('Per Capita relative cost:\t', round(pc_t_cost, decimal))
    print('----------------------------------------------------------------------')
    print('Treating by Screening Test')
    print('Total relative cost:\t\t', round(ct_cost, decimal))
    print('Per Capita relative cost:\t', round(pc_ct_cost, decimal))
    print('----------------------------------------------------------------------')
    if pc_ct_cost > pc_nt_cost:
        print('Screening program is more costly than treating everyone as a test-negative')
    if pc_nt_cost > pc_ct_cost > pc_t_cost:
        print('Screening program is cost efficient')
    if (pc_t_cost < pc_ct_cost) and (pc_t_cost < pc_nt_cost):
        print('Treating everyone as test-positive is least costly')
    print('----------------------------------------------------------------------\n')


def rubins_rules(point_estimates, std_error):
    r"""Function to merge multiple imputed data sets into a single summary estimate and variance. Results are based on
    Rubin's Rules for merging estimates. The summary point estimate is calculated via

    .. math::

        \bar{\beta} = m^{-1} \sum_{k=1}^m \hat{\beta_k}

    where m is the number of imputed data sets. The variance is calculated via

    .. math::

        Var(\hat{\beta}) = m_{-1} \sum_{k=1}^m Var(\hat{\beta}) + (1 + m^{-1})(m-1)^{-1} \sum_{k=1}^m
        (\hat{\beta_k} - \bar{\beta})^2

    The variance is constructed from the within-sample variance and the between sample variance

    Notes
    -----
    If your point estimates correspond to ratios, be sure to provide the natural-log transformed point estimates and
    the variance of the natural-log estimate

    Parameters
    ----------
    point_estimates : list
        Container object of the point estimates
    std_error : list
        Container object of the estimate standard errors

    Returns
    -------
    tuple
        Tuple of summary beta, summary standard error

    Examples
    --------
    >>> from zepid.calc import rubins_rules
    >>> rr_est = []
    >>> rr_std = []

    Calculating summary estimate

    >>> b = rubins_rules(rr_est, rr_std)

    Printing the summary risk ratio

    >>> print("RR = ", np.exp(b[0]))
    >>> print("95% LCL:", np.exp(b[0] - 1.96*b[1]))
    >>> print("95% UCL:", np.exp(b[0] + 1.96*b[1]))

    References
    ----------
    Rubin DB. (2004). Multiple imputation for nonresponse in surveys (Vol. 81). John Wiley & Sons.
    """
    if len(point_estimates) != len(std_error):
        raise ValueError("The number of point estimates and variances must be the same")

    # Calculate summary point estimate
    beta = np.mean(point_estimates)

    # Calculate summary variance estimate
    variance = np.array(std_error)**2
    var_between = np.sum((np.array(point_estimates) - beta)**2)
    var_within = np.mean(variance)
    var = var_within + (1 + len(variance)**-1)*((len(variance) - 1)**-1)*var_between

    return beta, np.sqrt(var)


def s_value(pvalue):
    r"""Function to calculate the S-value. The 'S' stands for Shannon information or surprisal values. The name comes
    from Claude Shannon for this work to information theory. S-values are calculated from p-values using the
    following transformation

    .. math::

        s = -\log(p)

    The S-value transformation allows a more intuitive explanation of what p-values tell us about the null hypothesis
    and alternative hypothesis compatibility. The S-value tells us how many 'bits' of information exist against the null
    hypothesis. For an example, a S-value of 5.1 is no more surprising than seeing heads for 5 fair coin tosses. The
    S-value should be rounded down in the interpretation

    Note
    ----
    S-values do NOT have a significant cut-point. Rather this transformation is to help build intuition what information
    a p-values is providing and the corresponding 'surprisal' of a result

    Parameters
    ----------
    pvalue : float, container
        P-value (or array of p-values) to convert into a S-value(s)

    Returns
    -------
    array
        NumPy array of calculated S-values

    Examples
    --------
    >>> from zepid.calc import s_value
    >>> s_value(pvalue=0.05)

    References
    ----------
    Greenland S. (2019). Valid P-values behave exactly as they should: Some misleading criticisms of P-values and their
    resolution with S-values. The American Statistician, 73(sup1), 106-114.

    Amrhein V, Trafimow D, & Greenland S. (2018). Inferential Statistics as Descriptive Statistics: There is No
    Replication Crisis if We Don’t Expect Replication. The American Statistician.
    """
    return -1 * np.log2(np.array(pvalue))


def probability_bounds(v, bounds):
    """Function to generate bounded values for probabilities. Specifically this function is used in multiple
    estimators to generate bounded probabilities. This is available for both symmetric and asymmetric bounds.

    Parameters
    ----------
    v : numpy.array
        Array of values to bound
    bounds : float, list, numpy.array
        Bounds to apply to v. If only a single value is provided, then symmetric bounds are used.

    Returns
    -------
    numpy.array of bounded values
    """
    v = np.asarray(v)
    if type(bounds) is float:  # Symmetric Bounding
        if bounds < 0 or bounds > 1:
            raise ValueError('Bound value must be between (0, 1)')
        v[v < bounds] = bounds
        v[v > 1 - bounds] = 1 - bounds

    elif type(bounds) is str:  # Catching string inputs
        raise ValueError('Bounds must either be a float between (0, 1), or a collection of floats between (0, 1)')

    elif type(bounds) is int:  # Catching string inputs
        raise ValueError('Bounds must either be a float between (0, 1), or a collection of floats between (0, 1)')

    else:  # Asymmetric bounds
        if bounds[0] > bounds[1]:
            raise ValueError('Bound thresholds must be listed in ascending order')
        if len(bounds) > 2:
            warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                          'specified bounds are used by the bound statement. So only ' +
                          str(bounds[0:2]) + ' will be used', UserWarning)
        if type(bounds[0]) is str or type(bounds[1]) is str:
            raise ValueError('Bounds must be floats between (0, 1)')
        if (bounds[0] < 0 or bounds[1] > 1) or (bounds[0] < 0 or bounds[1] > 1):
            raise ValueError('Both bound values must be between (0, 1)')
        v[v < bounds[0]] = bounds[0]
        v[v > bounds[1]] = bounds[1]

    return v
