import warnings
import numpy as np
import pandas as pd

from zepid.causal.ipw.utils import propensity_score


class GEstimationSNM:
    """G-estimation for structural nested models. G-estimation is distinct from the other g-methods (inverse
    probability weights and g-formula) in the parameter it estimates. Rather than estimating the average causal effect
    of treating everyone versus treating no one, g-estimation estimates the average causal effect within strata of L.
    It does this by specifying a structural nested model. The structural nested model looks like the following for
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
    (1) grid search or (2) closed form. The closed form is ultimately faster since we are only required to do some
    basic matrix manipulation to solve. For the grid search, we need to search across the potential values that
    minimize the values of alphas. We use SciPy's optimize for the heavy lifting.

    Parameters
    ----------
    df :
        -
    exposure :
        -
    outcome :
        -
    weights :
        -

    Notes
    -----

    Examples
    --------
    """
    def __init__(self, df, exposure, outcome, weights=None):
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, GEstimationSNM will drop all missing "
                          "data. GEstimationSNM will fit " + str(df.dropna().shape[0]) + ' of ' + str(df.shape[0]) +
                          " observations", UserWarning)
        self.df = df.copy().dropna().reset_index()

        if self.df[outcome].value_counts().index.isin([0, 1]).all():
            raise ValueError("GEstimationSNM only supports continuous outcomes currently")

        if not self.df[exposure].value_counts().index.isin([0, 1]).all():
            raise ValueError("GEstimationSNM only supports binary exposures currently")

        self.treatment = exposure
        self.outcome = outcome

        self._weights = weights

        self._treatment_model = None
        self._predicted_A = None
        self._snm_ = None

    def treatment_model(self, model, print_results=True):
        """Specify the treatment model to satisfy conditional exchangeability. Behind the scenes, `GestimationSNM` will
        add the necessary H(psi) terms. The only variables that need to be specified are the set of L's to satisfy
        conditional exchangeability.

        .. math::

            logit(\Pr(A=1| L)) = alpha + alpha L

        Notes
        -----
        H(psi) terms are only necessary for the grid-search solution to g-estimation. For the closed form, we directly
        generate predicted values of A from this model. As a result, the `treatment_model()` function is agnostic to
        the estimation approach.
        """
        self._treatment_model = model
        # TODO move below to closed form solver
        # self._treatment_model = propensity_score(df=self.df, model=self.treatment + ' ~ ' + model,
        #                                         weights=self._weights, print_results=print_results)
        # self._predicted_A = self._treatment_model.predict(self.df)

    def structural_nested_model(self, model):
        """Specify the structural nested mean model to fit. The structural nested model should include the treatment of
        interest, as well as any interactions with L's that are necessary. G-estimation assumes that this model is
        correctly specified and ALL interactions are included in this model. One way to ensure this assumption is to
        saturate the structural nested model (or allow for as much flexibility as possible).

        The structural nested mean model takes the following form

        .. math::

            E[Y^a |A=a, L] - E[Y^a|A=a, L] = \psi a + \psi a*L

        Parameters
        ----------
        model :
            -

        """
        self._snm_ = model

    def fit(self, solver='closed', starting_value=None, alpha_value=0):
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
        """
        if solver == 'closed':
            self._closed_form_solver_()

        elif solver == 'search':
            self._grid_search_()

        else:
            raise ValueError("`solver` must be specified as either 'closed' or as 'search'")

    def summary(self):
        """Summary of results
        """
        # TODO add SciPy optimization results if possible. Not needed for closed form
        # TODO need unique labels for each psi

    def _grid_search_(self):
        """Background function to perform the optimization procedure for psi
        """
        # TODO put something here to designate all the psi / H(psi) needed to calculate / minimize
        # TODO will need to find all then pass to SciPy to minimize

    def _closed_form_solver_(self):
        """Background function to calculate the closed form solution for psi
        """
        # TODO needs to do some matrix manipulations to calculate
