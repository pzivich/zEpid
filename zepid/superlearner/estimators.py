import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from itertools import combinations
from sklearn.base import BaseEstimator


class EmpiricalMeanSL(BaseEstimator):
    """Empirical mean estimator in the format of SciKit learn. This estimator is for use with the SuperLearner
    functionality.

    Note
    ----
    Generally, I do not recommend its use outside of SuperLearner. Essentially the empirical mean is a
    baseline estimator with which to compare other estimators included in SuperLearner.

    Parameters
    ----------
    None

    Examples
    --------
    Setup the environment and data set

    >>> from zepid import load_sample_data
    >>> from zepid.superlearner import EmpiricalMeanSL
    >>> df = load_sample_data(False).dropna()
    >>> X = np.asarray(df[['art', 'male', 'age0']])
    >>> y = np.asarray(df['dead'])

    EmpiricalMean estimation

    >>> emp_mean = EmpiricalMeanSL()
    >>> emp_mean.fit(X=X, y=y)

    EmpiricalMean prediction

    >>> emp_mean.predict(X=X)
    """

    def __init__(self):
        self.empirical_mean = np.nan

    def fit(self, X, y):
        """Estimate the empirical mean based on X and y. While X in input, it has no effect on the estimated empirical
        mean (since the empirical mean is the average for the full y).

        Parameters
        ----------
        X : numpy.array
            Training data
        y : numpy.array
            Target values

        Returns
        -------
        None
        """
        # Error Checking
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations (rows).")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("It looks like there is missing values in X or y. EmpiricalMeanSL does not support "
                             "missing data.")

        self.empirical_mean = np.mean(y)
        return self

    def predict(self, X):
        """Predict the value of y given a set of X covariates. Because X has no effect on the empirical mean, the
        mean from the data used in the fit() step is returned for all observations.

        Parameters
        ----------
        X : numpy.array
            NumPy array of covariates

        Returns
        -------
        NumPy array of predicted empirical means of the dimension X.shape[0]
        """
        return np.array([self.empirical_mean] * X.shape[0])


class GLMSL:
    """Generalized Linear Model for use with SuperLearner. This function is a wrapper function for the statsmodels
    `GLM` class. This is because the GLM implementation in statsmodels is not natively compatible with the sklearn /
    SuperLearner class. Compatible with all options available in the statsmodels families.

    Parameters
    ----------
    family: statsmodels.families.family
        Family to use for the model. All statsmodels supported families are also supported
    verbose : bool, optional

    Examples
    --------
    Setup the environment and data set

    >>> import statsmodels.api as sm
    >>> from zepid import load_sample_data
    >>> from zepid.superlearner import GLMSL
    >>> df = load_sample_data(False).dropna()
    >>> X = np.asarray(df[['art', 'male', 'age0']])
    >>> y = np.asarray(df['dead'])

    GLMSL example (logit model)

    >>> f = sm.families.family.Binomial()
    >>> glm = GLMSL(family=f)
    >>> glm.fit(X, y)
    """
    def __init__(self, family, verbose=False):
        self._family_ = family
        self._verbose_ = verbose

        # Storage items
        self.model = None

    def fit(self, X, y):
        """Estimate the GLM

        Parameters
        ----------
        X : numpy.array
            Training data
        y : numpy.array
            Target values

        Returns
        -------
        None
        """
        # Error Checking
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations (rows).")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("It looks like there is missing values in X or y. GLMSL does not support "
                             "missing data.")

        # Final results
        self.model = sm.GLM(y, np.hstack([np.zeros([X.shape[0], 1]) + 1, X]),  # Adds intercept into model
                            family=self._family_).fit()
        if self._verbose_:
            print(self.model.summary())

        return self

    def predict(self, X):
        """Predict using the fitted GLM.

        Parameters
        ----------
        X : numpy.array
            Samples following the same pattern as the X array input into the fit() statement.

        Returns
        -------
        Returns predicted values from the GLM
        """
        # Adding intercept
        Xd = np.hstack([np.zeros([X.shape[0], 1]) + 1, X])

        # Generating predictions to return
        return self.model.predict(Xd)

    def get_params(self, deep=True):
        """For sklearn.base.clone() compatibility"""
        return {"family": self._family_,
                "verbose": self._verbose_}

    def set_params(self, **parameters):
        """For sklearn.base.clone() compatibility"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class StepwiseSL:
    """Step-wise model selection for Generalized Linear Model selection for use with SuperLearner. Briefly, each
    combination of models is compared by AIC with the best one selected. The model selection procedure continues until
    there are no improvements in the model by AIC. The optimal is the best model estimated by the step-wise selection
    procedure and the lowest AIC value.

    Parameters
    ----------
    family: statsmodels.families.family
        Family to use for the model. All statsmodels supported families are also supported
    selection : str, optional
        Method of step-wise selection to use. Options are `'forward'` and `'backward'`. Default is backward, which
        starts from the full model inclusion and removes terms one at a time.
    order_interaction : int, optional
        Order of interactions to explore. For example, `interaction_order=0` explores only the main effects.
    verbose : bool, optional

    Examples
    --------
    Setup the environment and data set

    >>> import statsmodels.api as sm
    >>> from zepid import load_sample_data
    >>> from zepid.superlearner import StepwiseSL
    >>> df = load_sample_data(False).dropna()
    >>> X = np.asarray(df[['art', 'male', 'age0']])
    >>> y = np.asarray(df['dead'])

    StepwiseSL estimation with no interactions

    >>> f = sm.families.family.Binomial()
    >>> step_sl = StepwiseSL(family=f, method="backward", order_interaction=0)
    >>> step_sl.fit(X, y)

    StepwiseSL prediction

    >>> step_sl.predict(X=X)

    StepwiseSL with all first-order interactions

    >>> step_sl = StepwiseSL(family=f, method="backward", order_interaction=1)
    >>> step_sl.fit(X, y)

    StepwiseSL with forward selection and all second-order interactions

    >>> step_sl = StepwiseSL(family=f, method="forward", order_interaction=2)
    >>> step_sl.fit(X, y)
    """
    def __init__(self, family, selection="backward", order_interaction=0, verbose=False):
        # Error Checking
        if selection.lower() not in ["forward", "backward"]:
            raise ValueError("`method` must be one of the following: backward, forward")
        if order_interaction < 0 or type(order_interaction) is not int:
            raise ValueError("interaction_order must be a non-negative integer")

        self._family_ = family
        self._verbose_ = verbose
        self._selection_ = selection
        self._order_ = order_interaction

        # Storage items
        self.model_optim = None
        self.cols_optim = None

    def fit(self, X, y):
        """Estimate the optimal GLM

        Parameters
        ----------
        X : numpy.array
            Training data
        y : numpy.array
            Target values

        Returns
        -------
        None
        """
        # Error Checking
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations (rows).")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("It looks like there is missing values in X or y. StepWiseSL does not support "
                             "missing data.")
        if X.shape[1] < self._order_:  # Checking X shape with order_interaction
            warnings.warn("order_interaction is greater than the number of columns. This is not possible to assess, "
                          "so order_interaction="+str(int(X.shape[1]))+" will be assessed instead.", UserWarning)
            self._order_ = X.shape[1]

        # Creating all x-order interaction terms for assessment
        Xu = self._all_order_interactions_(X, self._order_)

        # Determining method of selection
        if self._selection_ == "backward":
            # Estimating full model as starting point
            full_model = sm.GLM(y, np.hstack([np.zeros([X.shape[0], 1]) + 1, Xu]),  # Adds intercept into model
                                family=self._family_).fit()
            if np.isnan(full_model.aic):
                raise ValueError("Saturated model is having trouble converging. Reduce the number of covariates, the "
                                 "order_interaction, or use selection=forward instead")
            best_aic = full_model.aic
            if self._verbose_:
                print(full_model.summary())
                print("Full-Model AIC:", best_aic)

            # Determining best AIC via backwards step-wise selection
            best_cols = list(range(Xu.shape[1]))
            best_alt_aic = best_aic
            best_alt_model = full_model
            best_alt_cols = best_cols.copy()

            while best_aic >= best_alt_aic and len(best_cols) - 1 >= -1:
                best_aic = best_alt_aic
                best_model = best_alt_model
                best_cols = best_alt_cols
                if self._verbose_:
                    print("\nCurrent Optim:", best_cols)

                if len(best_cols) - 1 == -1:  # necessary break for intercept-only to be outpute correctly
                    break
                if self._verbose_:
                    print("\nValid Combinations...")
                alt_models = list(combinations(best_cols, len(best_cols) - 1))
                best_alt_model, best_alt_cols, best_alt_aic = None, None, np.inf
                for alt in alt_models:
                    alt_model = sm.GLM(y, np.hstack([np.zeros([X.shape[0], 1]) + 1, Xu[:, alt]]),  # Adds intercept
                                       family=self._family_).fit()
                    if self._verbose_:
                        print("Columns:", alt)
                        print("AIC:    ", alt_model.aic)
                    if alt_model.aic < best_alt_aic:
                        best_alt_model = alt_model
                        best_alt_aic = alt_model.aic
                        best_alt_cols = alt

        # Determining method of selection
        if self._selection_ == "forward":
            # Estimating null model as starting point
            null_model = sm.GLM(y, np.zeros([X.shape[0], 1]) + 1,  # intercept-only model
                                family=self._family_).fit()
            best_aic = null_model.aic
            if self._verbose_:
                print(null_model.summary())
                print("Full-Model AIC:", best_aic)

            # Determining best AIC via forwards step-wise selection
            best_cols = ()
            best_alt_aic = best_aic
            best_alt_model = null_model
            best_alt_cols = ()
            vars_to_select = list(range(Xu.shape[1]))
            best_alt_var = None

            while best_aic >= best_alt_aic and len(best_cols) <= Xu.shape[1]:
                best_aic = best_alt_aic
                best_model = best_alt_model
                best_cols = best_alt_cols

                # Removing the previous best variable from those to add
                vars_to_select = list(vars_to_select)
                try:
                    vars_to_select.remove(best_alt_var)
                except ValueError:
                    pass
                vars_to_select = tuple(vars_to_select)

                if self._verbose_:
                    print("\nCurrent Optim:", best_cols)

                if len(best_cols) == Xu.shape[1]:  # necessary break for saturated to be output correctly
                    break
                if self._verbose_:
                    print("\nValid Combinations...")
                best_alt_model, best_alt_cols, best_alt_aic = None, None, np.inf
                for var in vars_to_select:
                    alt = best_cols + (var, )
                    alt_model = sm.GLM(y, np.hstack([np.zeros([X.shape[0], 1]) + 1, Xu[:, alt]]),
                                       family=self._family_).fit()
                    if self._verbose_:
                        print("Columns:", alt)
                        print("AIC:    ", alt_model.aic)
                    if alt_model.aic < best_alt_aic and not np.isnan(alt_model.aic):
                        best_alt_model = alt_model
                        best_alt_aic = alt_model.aic
                        best_alt_cols = alt
                        best_alt_var = var

        # Final results
        self.model_optim = best_model
        self.cols_optim = best_cols
        if self._verbose_:
            print(self.model_optim.summary())

        return self

    def predict(self, X):
        """Predict using the optimal GLM, where optimal is defined as the lowest AIC for the step-wise selection
        procedure used.

        Parameters
        ----------
        X : numpy.array
            Samples following the same pattern as the X array input into the fit() statement. All order_interaction
            terms are created in this step for the input X (i.e. the user does not need to create any of the x-order
            interaction terms)

        Returns
        -------
        Returns predicted values from the optimal GLM
        """
        # Creating all x-order interaction terms for assessment
        Xu = self._all_order_interactions_(X, self._order_)

        # Adding intercept
        Xd = np.hstack([np.zeros([X.shape[0], 1]) + 1, Xu[:, self.cols_optim]])

        # Generating predictions to return
        return self.model_optim.predict(Xd)

    @staticmethod
    def _all_order_interactions_(X, x_order):
        """Background function

        Parameters
        ----------
        X : numpy.array
            Input data array
        x_order : int
            Order of interactions to generate. x_order=1 generates all first order interactions

        Returns
        -------
        Array containing the original X and all corresponding x-order interactions
        """
        Xu = X.copy()
        assessed_order = 1
        while x_order >= assessed_order:
            assessed_order += 1

            # Generating unique combinations of columns
            combos_order = list(combinations(range(X.shape[1]), assessed_order))

            # Creating interaction term for all unique combinations
            for co in combos_order:
                interaction_term = np.prod(X[:, co], axis=1).reshape((X.shape[0], 1))
                Xu = np.hstack([Xu, interaction_term])

        return Xu

    def get_params(self, deep=True):
        """For sklearn.base.clone() compatibility"""
        return {"family": self._family_,
                "selection": self._selection_,
                "order_interaction": self._order_,
                "verbose": self._verbose_}

    def set_params(self, **parameters):
        """For sklearn.base.clone() compatibility"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
