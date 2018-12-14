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
    """Calculate two-sided risk confidence intervals

    Risk is calculated from

    .. math::

        R = \frac{a}{a+b}

    Wald standard error is

    .. math::

        SE_{Wald} = (\frac{1}{a} - \frac{1}{b})^{\frac{1}{2}}

    Hypergeometric standard error is

    .. math::

        SE_{HypGeo} = (\frac{a*b}{(a+b)^2 * (a+b-1)})^{\frac{1}{2}}

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
    Relies on the Central Limit Theorem, so there must be at least 5 events
    and 5 nonevents
    """
    risk = events / total
    c = 1 - alpha / 2
    zalpha = normal_ppf(c)
    if confint == 'wald':
        sd = np.sqrt((risk * (1-risk)) / total)
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
    """Calculate two-sided incidence rate confidence intervals. Only Wald-type confidence intervals are currently
    implemented.

    Incidence rate is calculated from

    .. math::

        I = \frac{a}{T}

    Incidence rate standard error is

    .. math::

        SE = (\frac{a}{T^2})^\frac{1}{2})

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
    """Calculates the risk ratio and confidence intervals from count data.

    Risk ratio is calculated from

    .. math::

        RR = \frac{a}{a + b} / \frac{c}{c + d}

    Risk ratio standard error is

    .. math::

        SE = (\frac{1}{a} - \frac{1}{a + b} + \frac{1}{c} - \frac{1}{c + d})^{\frac{1}{2}}

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
    """Calculates the risk difference and confidence intervals from count data.

    Risk difference is calculated as

    .. math::

        RD = \frac{a}{a + b} - \frac{c}{c + d}

    Risk difference standard error is calculated as

    .. math::

        SE = (\frac{a*b}{(a+b)^2 * (a+b-1)} + \frac{c*d}{(c*d)^2 * (c+d-1)})^{\frac{1}{2}}

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
    """
    check_positivity_or_throw(a, b, c, d)
    warn_if_normal_approximation_invalid(a, b, c, d)
    zalpha = normal_ppf(1 - alpha / 2)
    r1 = a / (a + b)
    r0 = c / (c + d)
    riskdiff = r1 - r0
    sd = np.sqrt((r1*(1-r1))/(a+b) + (r0*(1-r0))/(c+d))
    # TODO hypergeometric CL for later implementation
    # sd = np.sqrt(((a * b) / ((a + b) ** 2 * (a + b - 1))) + ((c * d) / (((c + d) ** 2) * (c + d - 1))))
    lcl = riskdiff - (zalpha * sd)
    ucl = riskdiff + (zalpha * sd)
    return Results(riskdiff, lcl, ucl, sd, alpha, 'risk difference')


def number_needed_to_treat(a, b, c, d, alpha=0.05):
    """Calculates the number needed to treat and confidence intervals from count data.

    Number needed to treat is calculated as

    .. math::

        NNT = \frac{1}{RD}

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
    """
    check_positivity_or_throw(a, b, c, d)
    warn_if_normal_approximation_invalid(a, b, c, d)

    zalpha = normal_ppf(1 - alpha / 2)
    r1 = a / (a + b)
    r0 = c / (c + d)
    riskdiff = r1 - r0
    sd = np.sqrt((r1*(1-r1))/(a+b) + (r0*(1-r0))/(c+d))
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
    """Calculates the odds ratio and confidence interval from count data

    Odds ratio is calculated from

    .. math::

        OR = \frac{a}{b} / \frac{c}{d}

    Odds ratio standard error is

    .. math::

        SE = (\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d})^{\frac{1}{2}}

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
    """Calculates the incidence rate ratio and confidence intervals from count data

    Incidence rate ratio is calculated from

    .. math::

        IR = \frac{a}{t1} / \frac{c}{t2}

    Incidence rate ratio standard error is

    .. math::

        SE = (\frac{1}{a} + \frac{1}{c})^{\frac{1}{2}}

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
    """Calculates the incidence rate difference and confidence intervals from count data

    Incidence rate difference is calculated from

    .. math::

        ID = \frac{a}{t1} - \frac{c}{t2}

    Incidence rate difference standard error is

    .. math::

        SE = (\frac{a}{t1^2} + \frac{c}{t2^2})^{\frac{1}{2}}

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
    """
    check_positivity_or_throw(a, c)
    check_nonnegativity_or_throw(t2, t1)
    warn_if_normal_approximation_invalid(a, c)

    zalpha = normal_ppf(1 - alpha / 2)
    rated1 = a / t1
    rated2 = c / t2
    irated = rated1 - rated2
    sd = np.sqrt((a / (t1**2)) + (c / (t2**2)))
    lcl = irated - (zalpha * sd)
    ucl = irated + (zalpha * sd)
    return Results(irated, lcl, ucl, sd, alpha, 'incidence rate difference')


def attributable_community_risk(a, b, c, d):
    """Calculates the estimated attributable community risk (ACR) from count data. ACR is also known as Population
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
    """
    check_positivity_or_throw(a, b, c, d)

    rt = (a + c) / (a + b + c + d)
    r0 = c / (c + d)
    return rt - r0


def population_attributable_fraction(a, b, c, d):
    """Calculates the population attributable fraction (PAF) from count data

    Population attributable fraction is calculated as

    .. math::

        PAF = (\frac{a + c}{a + b + c + d} - \frac{c}{c + d}) / \frac{a + c}{a + b + c + d} = (R - R_0) / R

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
    """
    check_positivity_or_throw(a, b, c, d)

    rt = (a + c) / (a + b + c + d)
    r0 = c / (c + d)
    return (rt - r0) / rt


def probability_to_odds(prob):
    """Convert proportion to odds

    Probability is converted to odds using

    .. math::

        O = \frac{P}{1 - P}

    Parameters
    ---------------
    prob : float, NumPy array
        Probability or array of probabilities to transform into odds

    Returns
    ----------
    odds
        Float or array of odds
    """
    return prob / (1 - prob)


def odds_to_probability(odds):
    """Convert odds to proportion

    Probability is converted to odds using

    .. math::

        P = \frac{O}{1 + O}

    Parameters
    ---------------
    odds : float, NumPy array
        Odds or array of odds to transform into probabilities

    Returns
    ----------
    prob
        Float or array of probabilities
    """
    return odds / (1 + odds)


def counternull_pvalue(estimate, lcl, ucl, sided='two', alpha=0.05, decimal=3):
    """Calculates the counternull based on Rosenthal R & Rubin DB (1994). It is useful to prevent over-interpretation
    of results. For a full discussion and how to interpret the estimate and p-value, see Rosenthal & Rubin.

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
    """A simple Bayesian Analysis. Note that this analysis assumes normal distribution for the
    continuous measure. See chapter 18 of Modern Epidemiology 3rd Edition (specifically pages 334, 340)

    The posterior estimate and variance are calculated as

    .. math::

        E_{posterior} = \frac{(E_{prior}*\frac{1}{Var_{prior}}) + (E*\frac{1}{Var})}{E_{prior}*\frac{1}{Var_{prior}}}

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

    Notes
    -------------
    Warning; Make sure that the alpha used to generate the confidence intervals matches the alpha
    used in this calculation. Additionally, this calculation can only handle normally distributed
    priors and observed
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
              round(lcl, decimal), ', ', round(ucl, decimal),')')
        print('----------------------------------------------------------------------')
        print('Posterior Estimate: ', round(post_mean, decimal))
        print(str(round((1 - alpha) * 100, 1)) + '% Posterior Probability Interval: (', round(post_lcl, decimal), ', ',
              round(post_ucl, decimal), ')')
        print('----------------------------------------------------------------------\n')
    return post_mean, post_lcl, post_ucl


def sensitivity(detected, cases, alpha=0.05, confint='wald'):
    """Calculate the sensitivity from number of detected cases and the number of total true cases.

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
    """
    check_positivity_or_throw(detected, cases)
    warn_if_normal_approximation_invalid(cases)

    if detected > cases:
        raise ValueError('Detected true cases must be less than or equal to the total number of cases')

    sens = detected / cases
    zalpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    if confint == 'wald':
        sd = np.sqrt((sens * (1-sens)) / cases)
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
    """
    check_positivity_or_throw(detected, noncases)
    warn_if_normal_approximation_invalid(noncases)

    if detected > noncases:
        raise ValueError('Detected true cases must be less than or equal to the total number of cases')
    spec = 1 - (detected / noncases)
    zalpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    if confint == 'wald':
        sd = np.sqrt((spec * (1-spec)) / noncases)
        # follows SAS9.4: http://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/viewer.htm#procstat_
        # freq_details37.htm#procstat.freq.freqbincl
        lower = spec - zalpha * sd
        upper = spec + zalpha * sd
    elif confint == 'hypergeometric':
        sd = np.sqrt(detected * (noncases - detected) / (noncases ** 2 * (cases - 1)))
        lower = spec - zalpha * sd
        upper = spec + zalpha * sd
    else:
        raise ValueError('Please specify a valid confidence interval')
    return spec, lower, upper, sd


def ppv_converter(sensitivity, specificity, prevalence):
    """Generates the positive predictive value from designated sensitivity, specificity, and prevalence.

    Positive predictive value is calculated using

    .. math::

        PPV = \frac{Se*P}{Se*P + (1-Sp)*(1-P)}

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
    """Generates the negative predictive value from designated sensitivity, specificity, and prevalence.

    .. math::

        NPV = \frac{Sp*(1-P)}{(1-Se)*P + Sp*(1-P)}

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
        The population size to set. Choose a larger value since this is only necessary for total calculations. Default is 10,000
    decimal : integer
        Amount of decimal points to display. Default value is 3

    Returns
    ---------
    None
        Prints results to console

    Notes
    --------
    When calculating costs, be sure to consult experts in health policy or related fields.  Costs should encompass more
    than just monetary costs, like relative costs (regret, disappointment, stigma, disutility, etc.). Careful
    consideration of relative costs between false positive and false negatives needs to be considered.
    """
    print('----------------------------------------------------------------------')
    print('''NOTE: When calculating costs, be sure to consult experts in health\npolicy or related fields.  
        Costs should encompass more than only monetary\ncosts, like relative costs (regret, disappointment, stigma, 
        disutility, etc.)''')
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
