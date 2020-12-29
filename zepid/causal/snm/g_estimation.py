import warnings
import patsy
import numpy as np
import pandas as pd
import scipy.optimize

from zepid.calc.utils import probability_bounds
from zepid.causal.utils import propensity_score, check_input_data


class GEstimationSNM:
    r"""G-estimation for structural nested mean models. G-estimation is distinct from the other g-methods (inverse
    probability weights and g-formula) in the parameter it estimates. Rather than estimating the average causal effect
    of treating everyone versus treating no one, g-estimation estimates the average causal effect within strata of L.
    It does this by specifying a structural nested model. The structural nested mean model looks like the following for
    additive effects

    .. math::

        E[Y^a |A=a, V] - E[Y^{a=0}|A=a, V] = \psi a + \psi a*V

    There are two items to note in the structural nested model; (1) there is no intercept or term for L, and (2) we
    need the potential outcomes to solve for psi. The first item means that we are estimating fewer parameters, making
    g-estimation less susceptible to model misspecification than the g-formula. The second means we cannot solve the
    above equation directly.

    Under the assumption of conditional exchangeability, we can solve for psi using another equation. Specifically, we
    can work to solve the following model

    .. math::

        logit(\Pr(A=1|Y^{a=0}, L)) = alpha + alpha Y^{a=0} + alpha Y{a=0} V + alpha L

    Under the assumption of conditional exchangeability, the alpha term for the potential outcome Y should be equal to
    zero! Therefore, we need to find the value of psi that results in that alpha term equaling zero. For the additive
    model, we can solve for psi in the first equation by

    .. math::

        H(\psi) = Y - (\psi A + \psi A L)

    meaning we solve for when alpha is approximately zero under

    .. math::

        logit(\Pr(A=1|Y^{a=0}, L)) = alpha + alpha H(\psi) + alpha H(\psi) V + alpha L

    To find the values for the psi's where the alpha for those terms is approximately zero, we have two options;
    (1) grid-search or (2) closed form. The closed form is ultimately faster since we are only required to do some
    basic matrix manipulation to solve. For the grid search, we need to search across the potential values that
    minimize the values of alphas. We use SciPy's Nelder-Mead optimization procedure for the heavy lifting.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame object containing all variables of interest
    exposure : str
        Column name of the exposure variable. Currently only binary is supported
    outcome : str
        Column name of the outcome variable. Either continuous or binary outcomes are supported
    weights :
        Column name of weights. Weights allow for items like sampling weights, missing weights, and censoring weights
        to estimate effects

    Notes
    -----
    Similar to marginal structural models, g-estimation cannot inherently account for missing at random data. To
    account for missing outcome data, inverse probability of missing weights should be used

    The grid-search approach does allow for some unique sensitivity analyses that are not incorporated into the
    closed-form. Specifically, we can imagine that there is some unobserved confounding. With unobserved confounding,
    we know that the alpha value will not exactly equal zero. We can optimize for slightly different alphas to see
    how sensitive our results are to some assumptions regarding unobserved confounding. For further details on
    translating unobserved confounding to alpha values, see Scharfstein et al. 1999 in the references

    If you continuous variable takes on large values, you may see the closed-form and grid-search start to diverge in
    results. This is because of the tolerance value. If you have large outcome values, I recommend rescaling them to
    prevent any issues with the grid-search

    Examples
    --------
    Set up the environment and the data set

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.snm import GEstimationSNM
    >>> df = load_sample_data(timevary=False).drop(columns=['dead'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

    One-parameter structural nested mean model via closed-form solution

    >>> snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
    >>> snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> snm.structural_nested_model(model='art')
    >>> snm.fit()
    >>> snm.summary()

    One-parameter structural nested mean model via grid-search

    >>> snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
    >>> snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> snm.structural_nested_model(model='art')
    >>> snm.fit(solver='search')

    One-parameter structural nested mean model via grid-search with different alphas

    >>> snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
    >>> snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> snm.structural_nested_model(model='art')
    >>> snm.fit(solver='search', alpha_value=0.03)

    Two-parameter structural nested mean model via closed-form

    >>> snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
    >>> snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> snm.structural_nested_model(model='art + art:male')
    >>> snm.fit()

    Two-parameter structural nested mean model via grid-search and starting values

    >>> snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
    >>> snm.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> snm.structural_nested_model(model='art + art:male')
    >>> snm.fit(solver='search', starting_value=[-0.05, 0.0])

    References
    ----------
    Naimi AI, Cole SR, Kennedy EH. (2017). An introduction to g methods. International journal of epidemiology,
    46(2), 756-762.

    Robins JM. (2000). Marginal structural models versus structural nested models as tools for causal inference.
    In Statistical models in epidemiology, the environment, and clinical trials (pp. 95-133). Springer, New York, NY.

    Vansteelandt S, Joffe M. (2014). Structural nested models and G-estimation: the partially realized promise.
    Statistical Science, 29(4), 707-731.

    Wallace MP, Moodie EE, Stephens DA. (2017). An R package for G-estimation of structural nested mean models.
    Epidemiology, 28(2), e18-e20.

    Scharfstein DO, Rotnitzky A, Robins JM. (1999). Adjusting for nonignorable drop-out using semiparametric
    nonresponse models. Journal of the American Statistical Association, 94(448), 1096-1120.
    """
    def __init__(self, df, exposure, outcome, weights=None):
        self.exposure = exposure
        self.outcome = outcome
        self._missing_indicator = '__missing_indicator__'
        self.df, self._miss_flag, continuous = check_input_data(data=df,
                                                                exposure=exposure,
                                                                outcome=outcome,
                                                                estimator="GEstimationSNM",
                                                                drop_censoring=False,
                                                                drop_missing=True,
                                                                binary_exposure_only=True)

        self.psi = None
        self.psi_labels = None

        self._weight_ = weights
        self.ipmw = None
        self._alphas = None
        self._treatment_model = None
        self._fit_missing_ = False
        self._predicted_A = None
        self._snm_ = None
        self._print_results = True
        self._scipy_solver_obj = None

    def exposure_model(self, model, print_results=True):
        r"""Specify the treatment model to satisfy conditional exchangeability. Behind the scenes, `GestimationSNM` will
        add the necessary H(psi) terms. The only variables that need to be specified are the set of L's to satisfy
        conditional exchangeability.

        .. math::

            logit(\Pr(A=1| L)) = alpha + alpha L

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. For example, 'var1 + var2 + var3'
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results). This only applies to the
            closed-form solution. The grid-search procedure fits lots of regression models and these results are
            not printed

        Notes
        -----
        H(psi) terms are only necessary for the grid-search solution to g-estimation. For the closed form, we directly
        generate predicted values of A from this model. As a result, the `exposure_model()` function is agnostic to
        the estimation approach. It only requires specifying the sufficient adjustment set via patsy format
        """
        self._treatment_model = model
        self._print_results = print_results

    def structural_nested_model(self, model):
        r"""Specify the structural nested mean model to fit. The structural nested model should include the treatment of
        interest, as well as any interactions with L's that are necessary. G-estimation assumes that this model is
        correctly specified and ALL interactions with confounders are included in this model. One way to ensure
        this assumption is to saturate the structural nested mean model (or allow for as much flexibility as possible).

        The structural nested mean model takes the following form

        .. math::

            E[Y^a |A=a, V] - E[Y^a|A=a, V] = \psi a + \psi a*V

        Parameters
        ----------
        model : str
            Structural nested mean model to estimate. Model should include treatment and relevant treatment-confounder
            interaction terms as needed. Interactions should be indicated via patsy magic.
            For example, 'A + A:V + A:C'
        """
        if self.exposure not in model:
            warnings.warn("It looks like '" + self.exposure + "' is not included in the structural nested model.")

        self._snm_ = model

    def missing_model(self, model_denominator, model_numerator=None, stabilized=True, bound=False, print_results=True):
        """Estimation of Pr(M=0|A=a,L), which is the missing data mechanism for the outcome. The corresponding
        observation probabilities are used to account for informative censoring by observed variables. The missing_model
        only accounts for missing outcome data.

        The inverse probability weights calculated by this function account for informative censoring (missing data on
        the outcome) by observed variables.

        Note
        ----
        The treatment variable should be included in the model

        Parameters
        ----------
        model_denominator: str
            String listing variables predicting missingness of outcomes via `patsy` syntax. For example, `
            'var1 + var2 + var3'. This is for the predicted probabilities of the denominator
        model_numerator : str, optional
            Optional string listing variables to predict the exposure, separated by +. Only used to calculate the
            numerator. Default (None) calculates the probability of censoring by treatment only. In general this is
            recommended. If assessing effect modifcation, this variable should be included in the numerator as well.
            Argument is only used when calculating stabilized weights
        stabilized : bool, optional
            Whether to use stabilized inverse probability of censoring weights
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            inference becomes limited to the restricted population. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation. A collection of
            floats can be provided for asymmetric trunctation
        print_results: bool, optional
        """
        # Error if no missing outcome data
        if not self._miss_flag:
            raise ValueError("No missing outcome data is present in the data set")

        # Warning if exposure is not included in the missingness of outcome model
        if self.exposure not in model_denominator:
            warnings.warn("For the specified missing outcome model, the exposure variable should be included in the "
                          "model", UserWarning)

        miss_model = self._missing_indicator + ' ~ ' + model_denominator
        fitmodel = propensity_score(self.df, miss_model, print_results=print_results)

        if stabilized:
            if model_numerator is None:
                mnum = self.exposure
            else:
                mnum = model_numerator
            numerator_model = propensity_score(self.df, self._missing_indicator + ' ~ ' + mnum,
                                               weights=self._weight_,
                                               print_results=print_results)
            n = numerator_model.predict(self.df)
        else:
            n = 1

        if bound:  # Bounding predicted probabilities if requested
            d = probability_bounds(fitmodel.predict(self.df), bounds=bound)
        else:
            d = fitmodel.predict(self.df)

        self.ipmw = np.where(self.df[self._missing_indicator] == 1, n / d, np.nan)
        self._fit_missing_ = True

    def fit(self, solver='closed', starting_value=None, alpha_value=0, tolerance=1e-7, verbose_solver=False,
            maxiter=500):
        """Using the treatment model and the format of the structural nested mean model, the solutions for psi are
        calculated.

        Parameters
        ----------
        solver : str, optional
            Method to use to solve for the values of psi. Default is 'closed' which uses the closed form solution,
            which is computationally less intensive.
            By specifying 'search', SciPy's optimize class is used to solve for the values of psi. One advantage of
            this approach is a sensitivity analysis approach.
        starting_value : list, array, container
            If using the 'search ' method, the initial guesses for psi can be specified. By default, no starting values
            of psi are used.
        alpha_value : float, list, array, container
            G-estimation allows for some easy and interesting sensitivity analyses. Specifically, we can imagine that
            an unmeasured confounder exists. This unmeasured confounder would mean our minimized alpha would not be
            zero anymore. We can change the alpha_value parameter to minimize alpha to different numbers. This allows
            us to see how unmeasured confounders may influence our results. This option is only available for the
            solver='search' option. See the online guide for more information on this parameter.
        verbose_solver : optional, bool
            For this grid-search procedure, setting this option to true print results for each iteration of the SciPy
            optimization procedure. Each iteration prints the psi values and the correspond alpha values from the
            fitted model
        maxiter : optional, int
            Maximum number of iterations to perform. If the maximum number of iterations is hit, the optimization
            procedure will stop and SciPy will say the convergence failed
        """
        if self._snm_ is None or self._treatment_model is None:
            raise ValueError("The exposure_model() and structural_nested_model() must be specified before the fit "
                             "procedure")

        if self._miss_flag and not self._fit_missing_:
            warnings.warn("All missing outcome data is assumed to be missing completely at random. To relax this "
                          "assumption to outcome data is missing at random please use the `missing_model()` "
                          "function", UserWarning)

        # Assigning label for weights
        df = self.df.copy()
        if self.ipmw is None:
            if self._weight_ is None:
                weight_col = None
            else:
                weight_col = self._weight_
        else:
            weight_col = '_w_'
            if self._weight_ is None:
                df['_w_'] = self.ipmw
            else:
                df['_w_'] = self.ipmw * df[self._weight_]

        # Pulling out data set up for SNM via patsy
        df = df.dropna()
        snm = patsy.dmatrix(self._snm_ + ' - 1', df, return_type='dataframe')
        self.psi_labels = snm.columns.values.tolist()  # Grabs labels for the solved psi values

        if solver == 'closed':
            # Pulling array of outcomes with the interaction terms (copy and rename column to get right interactions)
            yf = df.copy().drop(columns=[self.exposure])
            yf = yf.rename(columns={self.outcome: self.exposure})
            y_vals = patsy.dmatrix(self._snm_ + ' - 1', yf, return_type='dataframe')

            # Solving for the array of Psi values
            self.psi = self._closed_form_solver_(treat=self.exposure,
                                                 model=self.exposure + ' ~ ' + self._treatment_model,
                                                 df=df,
                                                 snm_matrix=snm, y_matrix=y_vals,
                                                 weights=weight_col, print_results=self._print_results)

        elif solver == 'search':
            # Adding other potential SNM variables to the input data
            sf = pd.concat([df, snm.drop(columns=[self.exposure])], axis=1)

            # Resolving if not initial parameters
            if starting_value is None:
                starting_value = [0.0] * len(self.psi_labels)

            # Passing to optimization procedure
            self._scipy_solver_obj = self._grid_search_(data_set=sf,
                                                        treatment=self.exposure, outcome=self.outcome,
                                                        weights=weight_col,
                                                        model=self._treatment_model, snm_terms=self.psi_labels,
                                                        start_vals=starting_value, alpha_shift=np.array(alpha_value),
                                                        tolerance=tolerance, verbose_solver=verbose_solver,
                                                        max_iter=maxiter)
            self._alphas = np.array(alpha_value)
            self.psi = self._scipy_solver_obj.x

        else:
            raise ValueError("`solver` must be specified as either 'closed' or as 'search'")

    def summary(self, decimal=3):
        """Summary of results
        """
        snm_form = ''
        is_first = False
        for l in self.psi_labels:
            if not is_first:
                is_first = True
                snm_form += 'psi*' + l
            else:
                snm_form += ' + psi*' + l

        print('======================================================================')
        print('           G-estimation of Structural Nested Mean Model               ')
        print('======================================================================')
        fmt = 'Treatment:        {:<22} No. Observations:     {:<10}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<22} No. Missing Outcome:  {:<10}'
        print(fmt.format(self.outcome, np.sum(self.df[self.outcome].isnull())))

        fmt = 'Missing model:    {:<15}'
        if self._fit_missing_:
            m = 'Logistic'
        else:
            m = 'None'

        print(fmt.format(m))

        # Printing scipy optimization if possible
        if self._scipy_solver_obj is not None:
            self._print_scipy_results(self._scipy_solver_obj, self._alphas)
        else:
            self._print_closed_results()

        print('----------------------------------------------------------------------')
        print('SNM:     ' + snm_form)
        print('----------------------------------------------------------------------')

        fmt = '{:<25} {:<30}'
        for p, pl in zip(self.psi, self.psi_labels):  # Printing all psi's and their labels
            print(fmt.format(pl, np.round(p, decimals=decimal)))
            # print(pl+':  ', np.round(p, decimals=decimal))

        print('======================================================================')

    @staticmethod
    def _grid_search_(data_set, treatment, outcome, model, snm_terms, start_vals,
                      alpha_shift, tolerance, verbose_solver, max_iter, weights):
        """Background function to perform the optimization procedure for psi
        """
        # Creating function for scipy to optimize
        def function_to_optimize(data, psi, snm_terms, y, a, pi_model, alpha_shift, weights):
            # loop through all psi values to calculate the corresponding H(psi) value based on covariate pattern
            snm = data[y] - data[snm_terms].mul(psi, axis='columns').sum(axis=1)
            data['H_psi'] = snm

            # Creating new terms to add to model
            h_terms_list = [w.replace(treatment, 'H_psi') for w in snm_terms]
            h_terms = ''
            for h in h_terms_list:
                h_terms += ' + ' + h

            # Estimating the necessary model
            fm = propensity_score(df=data, model=a + ' ~ ' + pi_model + h_terms,
                                  weights=weights, print_results=False)

            # Pulling elements from fitted model
            alpha = fm.params[h_terms_list] - alpha_shift  # Estimated alphas with the shift
            if verbose_solver:
                print('Psi:  ', np.array(psi))
                print('Alpha:', np.array(alpha))

            return np.abs(np.array(alpha)), psi

        def return_abs_alpha(psi):
            result = function_to_optimize(psi=psi, data=data_set,
                                          snm_terms=snm_terms,
                                          y=outcome, a=treatment,
                                          pi_model=model, alpha_shift=alpha_shift,
                                          weights=weights)
            return np.sum(result[0])

        return scipy.optimize.minimize(fun=return_abs_alpha, x0=start_vals,
                                       method='Nelder-Mead',
                                       tol=tolerance,
                                       options={'maxiter': max_iter,
                                                'disp': verbose_solver})

    @staticmethod
    def _closed_form_solver_(treat, model, df, snm_matrix, y_matrix, weights, print_results):
        """Background function to calculate the closed form solution for psi
        """
        # Calculate predictions
        fm = propensity_score(df=df, model=model, weights=weights, print_results=print_results)
        pred_treat = fm.predict(df)

        diff = df[treat] - pred_treat
        if weights is not None:
            diff = diff * df[weights]

        # D-dimensional psi-matrix
        lhm = np.dot(snm_matrix.mul(diff, axis=0).transpose(), snm_matrix)  # Dot product to produce D-by-D matrix

        # Array of outcomes
        y_matrix = y_matrix.mul(diff, axis=0)
        rha = y_matrix.sum()

        # Solving matrix and array for psi values
        psi_array = np.linalg.solve(lhm, rha)
        return psi_array

    @staticmethod
    def _print_scipy_results(optimized_function, alpha_values):
        """Background print function to the scipy optimized results results
        """
        fmt = 'Method:           {:<24} No. Iterations:   {:<10}'
        print(fmt.format('Nelder-Mead', optimized_function.nit))
        fmt = 'Alpha values:     {:<24} Optimized:        {:<10}'
        print(fmt.format(np.str(alpha_values), str(optimized_function.success)))

    @staticmethod
    def _print_closed_results():
        """Background print function to print the closed-form info
        """
        print('Method:           Closed-form')
