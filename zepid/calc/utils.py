import warnings
import math
from scipy.stats import norm


def risk_ci(events, total, alpha=0.05, confint='wald'):
    """Calculate two-sided (1-alpha)% Confidence interval of Risk. Note
    relies on the Central Limit Theorem, so there must be at least 5 events
    and 5 nonevents

    Returns (risk, lower CL, upper CL, SE)

    events:
        -Number of events/outcomes that occurred
    total:
        -total number of subjects that could have experienced the event
    alpha:
        -Alpha level. Default is 0.05
    confint:
        -Type of confidence interval to generate. Current options are
            wald
            hypergeometric
    """
    risk = events / total
    c = 1 - alpha / 2
    zalpha = norm.ppf(c, loc=0, scale=1)
    if confint == 'wald':
        sd = math.sqrt((risk * (1-risk)) / total)
        # follows SAS9.4: http://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/viewer.htm#procstat_
        # freq_details37.htm#procstat.freq.freqbincl
        lower = risk - zalpha * sd
        upper = risk + zalpha * sd
    elif confint == 'hypergeometric':
        sd = math.sqrt(events * (total - events) / (total ** 2 * (total - 1)))
        lower = risk - zalpha * sd
        upper = risk + zalpha * sd
    else:
        raise ValueError('Only wald and hypergeometric confidence intervals are currently supported')
    return risk, lower, upper, sd


def incidence_rate_ci(events, time, alpha=0.05):
    """Calculate two-sided (1-alpha)% Wald Confidence interval of Incidence Rate

    Returns (incidence rate, lower CL, upper CL, SE)

    events:
        -number of events/outcomes that occurred
    time:
        -total person-time contributed in this group
    alpha:
        -alpha level. Default is 0.05
    """
    c = 1 - alpha / 2
    ir = events / time
    zalpha = norm.ppf(c, loc=0, scale=1)
    sd = math.sqrt(events / (time ** 2))
    # https://www.researchgate.net/post/How_to_estimate_standard_error_from_incidence_rate_and_population
    # Incidence rate confidence intervals are a mess, with not sources agreeing...
    # http://www.openepi.com/PersonTime1/PersonTime1.htm
    # zEpid uses a normal approximation. There are too many options for CI's...
    lower = ir - zalpha * sd
    upper = ir + zalpha * sd
    return ir, lower, upper, sd


def risk_ratio(a, b, c, d, alpha=0.05):
    """Calculates the Risk Ratio from count data.

    Returns (risk ratio, lower CL, upper CL, SE)

    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (a < 0) or (b < 0) or (c < 0) or (d < 0):
        raise ValueError('All numbers must be positive')
    if (a <= 5) or (b <= 5) or (c <= 5) or (d <= 5):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    r1 = a / (a + b)
    r0 = c / (c + d)
    relrisk = r1 / r0
    sd = math.sqrt((1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + d)))
    lnrr = math.log(relrisk)
    lcl = math.exp(lnrr - (zalpha * sd))
    ucl = math.exp(lnrr + (zalpha * sd))
    return relrisk, lcl, ucl, sd


def risk_difference(a, b, c, d, alpha=0.05):
    """Calculates the Risk Difference from count data.

    Returns (risk difference, lower CL, upper CL, SE)

    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (a < 0) or (b < 0) or (c < 0) or (d < 0):
        raise ValueError('All numbers must be positive')
    if (a <= 5) or (b <= 5) or (c <= 5) or (d <= 5):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    r1 = a / (a + b)
    r0 = c / (c + d)
    riskdiff = r1 - r0
    sd = math.sqrt((r1*(1-r1))/(a+b) + (r0*(1-r0))/(c+d))
    # TODO hypergeometric CL for later implementation
    # sd = math.sqrt(((a * b) / ((a + b) ** 2 * (a + b - 1))) + ((c * d) / (((c + d) ** 2) * (c + d - 1))))
    lcl = riskdiff - (zalpha * sd)
    ucl = riskdiff + (zalpha * sd)
    return riskdiff, lcl, ucl, sd


def number_needed_to_treat(a, b, c, d, alpha=0.05):
    """Calculates the Number Needed to Treat from count data

    Returns (NNT, lower CL, upper CL, SE)

    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (a < 0) or (b < 0) or (c < 0) or (d < 0):
        raise ValueError('All numbers must be positive')
    if (a <= 5) or (b <= 5) or (c <= 5) or (d <= 5):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    r1 = a / (a + b)
    r0 = c / (c + d)
    riskdiff = r1 - r0
    sd = math.sqrt((r1*(1-r1))/(a+b) + (r0*(1-r0))/(c+d))
    # TODO hypergeometric CL for later implementation
    # sd = math.sqrt(((a * b) / ((a + b) ** 2 * (a + b - 1))) + ((c * d) / (((c + d) ** 2) * (c + d - 1))))
    lcl_rd = riskdiff - (zalpha * sd)
    ucl_rd = riskdiff + (zalpha * sd)
    if riskdiff != 0:
        numbnt = 1 / riskdiff
    else:
        numbnt = math.inf
    if lcl_rd != 0:
        lcl = 1 / lcl_rd
    else:
        lcl = math.inf
    if ucl_rd != 0:
        ucl = 1 / ucl_rd
    else:
        ucl = math.inf
    return numbnt, lcl, ucl, sd


def odds_ratio(a, b, c, d, alpha=0.05):
    """Calculates the Odds Ratio from count data

    Returns (Odds Ratio, lower CL, upper CL, SE)

    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (a < 0) or (b < 0) or (c < 0) or (d < 0):
        raise ValueError('All numbers must be positive')
    if (a <= 5) or (b <= 5) or (c <= 5) or (d <= 5):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    or1 = a / b
    or0 = c / d
    oddsr = or1 / or0
    sd = math.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
    lnor = math.log(oddsr)
    lcl = math.exp(lnor - (zalpha * sd))
    ucl = math.exp(lnor + (zalpha * sd))
    return oddsr, lcl, ucl, sd


def incidence_rate_ratio(a, c, t1, t2, alpha=0.05):
    """Calculates the Incidence Rate Ratio from count data

    Returns (incidence rate ratio, lower CL, upper CL, SE)

    a:
        -count of exposed with outcome
    b:
        -count of unexposed with outcome
    T1:
        -person-time contributed by those who were exposed
    T2:
        -person-time contributed by those who were unexposed
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (a < 0) or (c < 0) or (t1 <= 0) | (t2 <= 0):
        raise ValueError('All numbers must be positive')
    if (a <= 5) or (c <= 5):
        warnings.warn('At least one event count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    irate1 = a / t1
    irate2 = c / t2
    irater = irate1 / irate2
    sd = math.sqrt((1 / a) + (1 / c))
    lnirr = math.log(irater)
    lcl = math.exp(lnirr - (zalpha * sd))
    ucl = math.exp(lnirr + (zalpha * sd))
    return irater, lcl, ucl, sd


def incidence_rate_difference(a, c, t1, t2, alpha=0.05):
    """Calculates the Incidence Rate Difference from count data

    Returns (incidence rate difference, lower CL, upper CL, SE)

    a:
        -count of exposed with outcome
    b:
        -count of unexposed with outcome
    T1:
        -person-time contributed by those who were exposed
    T2:
        -person-time contributed by those who were unexposed
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (a < 0) or (c < 0) or (t1 <= 0) or (t2 <= 0):
        raise ValueError('All numbers must be positive')
    if (a <= 5) or (c <= 5):
        warnings.warn('At least one event count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    rated1 = a / t1
    rated2 = c / t2
    irated = rated1 - rated2
    sd = math.sqrt((a / (t1**2)) + (c / (t2**2)))
    lcl = irated - (zalpha * sd)
    ucl = irated + (zalpha * sd)
    return irated, lcl, ucl, sd


def attributable_community_risk(a, b, c, d):
    """Calculates the estimated Attributable Community Risk (ACR) from count data. ACR is also
    known as Population Attributable Risk. Since this is commonly confused with the population
    attributable fraction, the name ACR is used to clarify differences in the formulas

    Return attributable community risk

    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    """
    if (a < 0) or (b < 0) or (c < 0) or (d < 0):
        raise ValueError('All numbers must be positive')
    rt = (a + c) / (a + b + c + d)
    r0 = c / (c + d)
    return rt - r0


def population_attributable_fraction(a, b, c, d):
    """Calculates the Population Attributable Fraction from count data

    Returns population attribuable fraction

    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    """
    if (a < 0) or (b < 0) or (c < 0) or (d < 0):
        raise ValueError('All numbers must be positive')
    rt = (a + c) / (a + b + c + d)
    r0 = c / (c + d)
    return (rt - r0) / rt


def probability_to_odds(prop):
    """Convert proportion to odds. Returns the corresponding odds

    Returns odds

    prop:
        -proportion that is desired to transform into odds
    """
    return prop / (1 - prop)


def odds_to_probability(odds):
    """Convert odds to proportion. Returns the corresponding proportion

    Return proportion/probability

    odds:
        -odds that is desired to transform into a proportion
    """
    return odds / (1 + odds)


def counternull_pvalue(estimate, lcl, ucl, sided='two', alpha=0.05, decimal=3):
    """Calculates the counternull based on Rosenthal R & Rubin DB (1994). It is useful to prevent over-interpretation
    of results. For a full discussion and how to interpret the estimate and p-value, see Rosenthal & Rubin.

    NOTE: Make sure that the confidence interval points put into the equation match the alpha level calculation

    estimate:
        -Point estimate for result
    lcl:
        -Lower confidence limit
    ucl:
        -Upper confidence limit
    sided:
        -Whether to compute the upper one-sided, lower one-sided, or two-sided counternull
         p-value. Default is the two-sided
            'upper'     Upper one-sided p-value
            'lower'     Lower one-sided p-value
            'two'       Two-sided p-value
    alpha:
        -Alpha level for p-value. Default is 0.05. Verify that this is the same alpha used to
         generate confidence intervals
    decimal:
        -Number of decimal places to display. Default is three
    """
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
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

    Warning: Make sure that the alpha used to generate the confidence intervals matches the alpha
    used in this calculation. Additionally, this calculation can only handle normally distributed
    priors and observed

    Returns (posterior mean, posterior lower CL, posterior upper CL)

    prior_mean:
        -Prior designated point estimate
    prior_lcl:
        -Prior designated lower confidence limit
    prior_ucl:
        -Prior designated upper confidence limit
    mean:
        -Point estimate result obtained from analysis
    lcl:
        -Lower confidence limit estimate obtained from analysis
    ucl:
        -Upper confidence limit estimate obtained from analysis
    ln_transform:
        -Whether to natural log transform results before conducting analysis. Should be used for
         RR, OR, or or other Ratio measure. Default is False (use for RD and other absolute measures)
    alpha:
        -Alpha level for confidence intervals. Default is 0.05
    decimal:
        -Number of decimal places to display. Default is three
    print_results:
        -Whether to print the results of the semi-Bayesian calculations. Default is True
    """
    # Transforming to log scale if ratio measure
    if ln_transform:
        prior_mean = math.log(prior_mean)
        prior_lcl = math.log(prior_lcl)
        prior_ucl = math.log(prior_ucl)
        mean = math.log(mean)
        lcl = math.log(lcl)
        ucl = math.log(ucl)
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)

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
    sd = math.sqrt(post_var)
    post_lcl = post_mean - zalpha * sd
    post_ucl = post_mean + zalpha * sd

    # Transforming back if ratio measure
    if ln_transform:
        post_mean = math.exp(post_mean)
        post_lcl = math.exp(post_lcl)
        post_ucl = math.exp(post_ucl)
        prior_mean = math.exp(prior_mean)
        prior_lcl = math.exp(prior_lcl)
        prior_ucl = math.exp(prior_ucl)
        mean = math.exp(mean)
        lcl = math.exp(lcl)
        ucl = math.exp(ucl)

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
    """
    Calculate the Sensitivity from number of detected cases and the number of total true cases.

    Returns (sensitivity, lower CL, upper CL, SE)

    detected:
        -number of true cases detected via testing criteria
    cases:
        -total number of true/actual cases
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (detected < 0) or (cases < 0):
        raise ValueError('All numbers must be positive')
    if detected > cases:
        raise ValueError('Detected true cases must be less than or equal to the total number of cases')
    if cases <= 5:
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    sens = detected / cases
    zalpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    if confint == 'wald':
        sd = math.sqrt((sens * (1-sens)) / cases)
        # follows SAS9.4: http://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/viewer.htm#procstat_
        # freq_details37.htm#procstat.freq.freqbincl
        lower = sens - zalpha * sd
        upper = sens + zalpha * sd
    elif confint == 'hypergeometric':
        sd = math.sqrt(detected * (cases - detected) / (cases ** 2 * (cases - 1)))
        lower = sens - zalpha * sd
        upper = sens + zalpha * sd
    else:
        raise ValueError('Please specify a valid confidence interval')
    return sens, lower, upper, sd


def specificity(detected, noncases, alpha=0.05, confint='wald'):
    """
    Calculate the Sensitivity from number of detected cases and the number of total true cases.

    Returns (specificity, lower CL, upper CL, SE)

    detected:
        -number of false cases detected via testing criteria
    cases:
        -total number of true/actual noncases
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    """
    if (detected < 0) or (noncases < 0):
        raise ValueError('All numbers must be positive')
    if detected > noncases:
        raise ValueError('Detected true cases must be less than or equal to the total number of cases')
    if noncases <= 5:
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    spec = 1 - (detected / noncases)
    zalpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    if confint == 'wald':
        sd = math.sqrt((spec * (1-spec)) / noncases)
        # follows SAS9.4: http://support.sas.com/documentation/cdl/en/procstat/67528/HTML/default/viewer.htm#procstat_
        # freq_details37.htm#procstat.freq.freqbincl
        lower = spec - zalpha * sd
        upper = spec + zalpha * sd
    elif confint == 'hypergeometric':
        sd = math.sqrt(detected * (noncases - detected) / (noncases ** 2 * (cases - 1)))
        lower = spec - zalpha * sd
        upper = spec + zalpha * sd
    else:
        raise ValueError('Please specify a valid confidence interval')
    return spec, lower, upper, sd


def ppv_converter(sensitivity, specificity, prevalence):
    """Generates the Positive Predictive Value from designated Sensitivity, Specificity, and Prevalence.

    Returns positive predictive value

    sensitivity:
        -sensitivity of the criteria
    specificity:
        -specificity of the criteria
    prevalence:
        -prevalence of the outcome in the population
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
    """Generates the Negative Predictive Value from designated Sensitivity, Specificity, and Prevalence.

    Returns negative predictive value

    sensitivity:
        -sensitivity of the criteria
    specificity:
        -specificity of the criteria
    prevalence:
        -prevalence of the outcome in the population
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

    WARNING: When calculating costs, be sure to consult experts in health policy or related fields.  Costs
    should encompass more than just monetary costs, like relative costs (regret, disappointment, stigma,
    disutility, etc.). Careful consideration of relative costs between false positive and false negatives
    needs to be considered.

    cost_miss_case:
        -The relative cost of missing a case, compared to false positives. In general, set this to 1 then
         change the value under 'cost_false_pos' to reflect the relative cost
    cost_false_pos:
        -The relative cost of a false positive case, compared to a missed case
    prevalence:
        -The prevalence of the disease in the population. Must be a float
    sensitivity:
        -The sensitivity level of the screening test. Must be a float
    specificity:
        -The specificity level of the screening test. Must be a float
    population:
        -The population size to set. Choose a larger value since this is only necessary for total calculations. Default is 10,000
    decimal:
        -amount of decimal points to display. Default value is 3
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
