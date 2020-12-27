import copy
import warnings
import numpy as np
import pandas as pd
import sklearn.model_selection as ms

from scipy.optimize import nnls
from sklearn import clone

from zepid.calc import logit, inverse_logit, probability_bounds


class SuperLearnerError(Exception):
    """Class for errors in the SuperLearner procedures. Nothing special besides directing user to specific issues are
    to the SuperLearner side
    """
    pass


class SuperLearner:
    r"""`SuperLearner` is an implementation of the super learner algorithm, which is a generalized stacking algorithm.
    Super learner is an approach to combine multiple predictive functions into a singular predictive function that has
    performance at least as good as the best candidate estimator included (asymptotically). Additionally, it should be
    noted that super learner converges at the rate with which the best candidate estimator converges.

    Briefly, super learner takes an input of candidate estimators for a function. Each of the estimators is run through
    a train-test cross-validation algorithm. From the candidate estimators, either the best overall performing
    candidate (discrete super learner) or a weighted combination of the algorithms is used as the updated predictive
    function.

    Note
    ----
    `SuperLearner` does not accept missing data. All missing data decisions have to occur prior to trying to use the
    `SuperLearner` procedure.


    `SuperLearner` accepts estimators that are of the SciKit-Learn format. Specifically, the candidate estimators must
    follow the `estimator.fit(X, y)` and `estimator.predict(X)` format. Performance has currently been checked for
    `sklearn`, `pygam`, and the estimators included in `zepid.superlearner`. Please consider opening an issue on
    GitHub if you find Python libraries that are not supported (but follow the SciKit-Learn style).

    Note
    ----
    `SuperLearner(discrete=True)` returns predictions from the candidate estimator with the greatest coefficient. In the
    case of a tie, the first candidate estimator with the greatest coefficient is used (as per `numpy.argmax` behavior).


    To compare performances easily, `SuperLearner` provides both Cross-Validated Error and the Relative Efficiency. The
    Cross-Validated Error calculation depends on the chosen loss function. For L2, the loss function is

    .. math::

        \frac{1}{n} \sum_i (Y_i - \widehat{Y}_i)^2

    For the negative-log-likelihood loss function,

    .. math::

        \frac{1}{n} \sum_i Y_i \times \ln(\widehat{Y}_i) + (1-Y_i) \times \ln(1 - \widehat{Y}_i)

    Relative efficiency is the Cross-Validated Error for the candidate estimator divided by the Cross-Validated Error
    for the chosen super learner.

    Parameters
    ----------
    estimators : list, array
        Candidate estimators. Must follow sklearn style and not be fit yet
    estimator_labels : list, array
        Labels for the candidate estimators being included
    folds : int, optional
        Number of folds to use during the cross-validation procedure. It is recommended to be between 10-20. The default
        value is 10-fold cross-validation.
    loss_function : str, optional
        Loss function to use. Options include: L2, NLogLik. L2 should be used for continuous outcomes and NLogLik for
        binary outcomes
    solver : str, optional
        Optimization algorithm to use to determine the super learner weights. Currently only Non-Negative Least Squares
        is available.
    bounds : float, collection, optional
        Bounding to use for probability. The bounding prevents values of exactly 0 or 1, which will break the loss
        function evaluation. Default is 1e-6.
    discrete : bool, optional
        Whether to use only the estimator with the greatest weight (discrete super learner). Default is False, which
        uses the super learner including all estimators
    verbose : bool, optional
        Whether to print progress to the console as super learner is being fit. Default is False.

    Examples
    --------
    Setup the environment and data set

    >>> import statsmodels.api as sm
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from zepid import load_sample_data
    >>> from zepid.superlearner import EmpiricalMeanSL, StepwiseSL, SuperLearner

    >>> fb = sm.families.family.Gaussian()
    >>> fc = sm.families.family.Binomial()
    >>> df = load_sample_data(False).dropna()
    >>> X = np.asarray(df[['art', 'male', 'age0']])
    >>> y = np.asarray(df['dead'])

    SuperLearner for binary outcomes

    >>> # Setting up estimators
    >>> emp = EmpiricalMeanSL()
    >>> log = LogisticRegression()
    >>> step = StepwiseSL(family=fb, selection="backward", order_interaction=1)
    >>> sl = SuperLearner(estimators=[emp, log, step], estimator_labels=["Mean", "Log", "Step"], loss_function='nloglik')
    >>> fsl = sl.fit(X, y)
    >>> fsl.summary()  # Summary of Cross-Validated Errors
    >>> fsl.predict(X)  # Generating predicted values from super learner

    SuperLearner for continuous outcomes

    >>> emp = EmpiricalMeanSL()
    >>> lin = LinearRegression()
    >>> step = StepwiseSL(family=fc, selection="backward", order_interaction=1)
    >>> sl = SuperLearner(estimators=[emp, log, step], estimator_labels=["Mean", "Lin", "Step"], loss_function='L2')
    >>> fsl = sl.fit(X, y)
    >>> fsl.summary()  # Summary of Cross-Validated Errors
    >>> fsl.predict(X)  # Generating predicted values from super learner

    Discrete Super Learner

    >>> sl = SuperLearner([emp, log, step], ["Mean", "Lin", "Step"], loss_function='L2', discrete=True)
    >>> sl.fit(X, y)

    References
    ----------
    Van der Laan MJ, Polley EC, Hubbard AE. (2007). Super learner. Statistical Applications in Genetics and Molecular
    Biology, 6(1).

    Rose S. (2013). Mortality risk score prediction in an elderly population using machine learning. American Journal
    of Epidemiology, 177(5), 443-452.
    """
    def __init__(self, estimators, estimator_labels, folds=10, loss_function="L2", solver="nnls",
                 bounds=1e-6, discrete=False, verbose=False):
        # TODO the R SuperLearner library supports an alt-NNLS, LS, and nnloglik optimization routines. Find in Python
        # Checking for errors
        if len(estimators) != len(estimator_labels):
            raise ValueError("estimators and estimator_labels must be of the same length")
        if solver.lower() not in ["nnls"]:
            raise ValueError("The solver " + str(solver) + " is not currently available. Please select one of the "
                                                           "following: NNLS")
        if loss_function.lower() not in ["l2", "nloglik"]:
            raise ValueError("The loss function " + str(loss_function) + " is not currently available. Please select "
                                                                         "one of the following: L2, NLogLik")

        # Parameters
        self.estimators = estimators
        self.labels = estimator_labels
        self.k = folds
        self.loss_function = loss_function.lower()
        self.discrete = discrete
        self.solver = solver.lower()
        self._verbose_ = verbose
        self._bounds_ = bounds

        # Storage items for results
        self.est_performance = pd.DataFrame()
        self.est_performance['estimator'] = list(estimator_labels)
        self.coefficients = None
        self.fit_estimators = []
        self._include_estimator_ = []

    def fit(self, X, y):
        """Fit SuperLearner given the variables `X` to predict `y`. These variables are directly passed to the
        candidate estimators. If there is any pre-processing to do outside of the estimator, please do so before passing
        to fit.

        Parameters
        ----------
        X : numpy.array
            Covariates to predict the target values
        y : numpy.array
            Target values to predict

        Returns
        -------
        None
        """
        # Some checks
        X = np.asarray(X)  # this line saves a lot of headaches
        y = np.asarray(y)  # this line saves a lot of headaches
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("It looks like there is missing values in X or y. SuperLearner does not support missing "
                             "data.")
        if np.all(np.in1d(y, np.array([0, 1]))) and (self.loss_function == "l2"):
            # Allows for the algorithm to proceed (but throws warning to the user about the chosen loss function)
            warnings.warn("It looks like your `y` is binary, and the `L2` loss function should be used for "
                          "continuous outcomes", UserWarning)

        # Step 1) input data and algorithms
        n_obs = X.shape[0]  # number of observations
        n_est = len(self.estimators)  # number of candidate estimators
        cv_pred = np.full([n_obs, n_est], np.nan)  # NaN for blank results

        # Step 2) Cross-Validation Splits
        if self._verbose_:
            print("Starting cross-validation procedure...")
        current_fold = 0
        for train, test in ms.KFold(self.k, shuffle=False).split(range(n_obs)):
            current_fold += 1

            # Step 3) Test-Train for each algorithm for each fold (note that train is only a 1/folds of the hold-out)
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for est_id in range(n_est):
                if self._verbose_:
                    print("...fitting "+str(self.labels[est_id])+" fold-"+str(current_fold))
                cv_est = clone(self.estimators[est_id])
                cv_est.fit(X_train, y_train)

                # Step 4) Filling in cross-validated Y predictions
                cv_pred[test, est_id] = self._predict_(estimator=cv_est, X=X_test)

        # Step 5) Calculating Cross-Validated Error for each candidate
        self.est_performance['cv_error'] = np.nan
        for est_id in range(n_est):
            self.est_performance.loc[est_id, "cv_error"] = self._error_term_(y, cv_pred[:, est_id])

        # Step 6) Determine coefficients
        if self.solver == "nnls":
            coefs, _ = nnls(cv_pred, y)
        else:
            raise ValueError(str(self.solver) + " is not currently available")

        coefs = np.array(coefs)
        machine_limit = np.finfo(np.double).eps
        coefs[coefs < np.sqrt(machine_limit)] = 0  # dropping coefs below certain precision
        self.coefficients = coefs / np.sum(coefs)
        self.est_performance['coefs'] = self.coefficients

        # Step 7) Fit algorithms to full data
        if self._verbose_:
            print("Fitting candidate(s) to full data...")

        if self.discrete:  # Step 7.a) discrete super learner
            discrete_sl_id = np.argmax(self.coefficients)
            if self._verbose_:
                print("Discrete super learner is: " + str(self.labels[discrete_sl_id]))
            for est_id in range(n_est):
                est = clone(self.estimators[est_id])
                # Estimating only the Discrete Super Learner to save computation time
                if est_id == discrete_sl_id:
                    if self._verbose_:
                        print("...fitting " + str(self.labels[est_id]))
                    est.fit(X, y)
                    self.fit_estimators.append(est)
                    self.coefficients[est_id] = 1
                # Skipping over others but retaining the shape of input estimators
                else:
                    self.fit_estimators.append(est)
                    self.coefficients[est_id] = 0

        else:  # Step 7.b) super learner
            # No prior selection (unlike discrete super learner)
            for est_id in range(n_est):
                est = clone(self.estimators[est_id])
                # Estimating only candidate estimators with coefficient > 0
                if self.coefficients[est_id] > 0:
                    if self._verbose_:
                        print("...fitting " + str(self.labels[est_id]))
                    est.fit(X, y)
                    self.fit_estimators.append(est)
                # Skipping over candidates with coefficient == 0
                else:
                    if self._verbose_:
                        print("...skipping " + str(self.labels[est_id]))
                    self.fit_estimators.append(est)

        return self

    def predict(self, X):
        """Generate predictions using the fit SuperLearner.

        Parameters
        ----------
        X : numpy.array
            Covariates to generate predictions of y. Note that X should be in the same format as the X used during the
            fit() function

        Returns
        -------
        numpy.array of predicted values using either discrete super learner or super learner
        """
        X = np.asarray(X)  # this line saves a lot of headaches
        if self.coefficients is None:
            raise ValueError("fit() must be called before predict()")
        if np.any(np.isnan(X)):
            raise ValueError("It looks like there is missing values in X. SuperLearner does not support missing data.")

        n_obs = X.shape[0]
        n_est = len(self.estimators)
        cv_pred = np.full([n_obs, n_est], np.nan)  # NaN for blank results

        # Step 8) Predictions from SuperLearner
        for est_id in range(n_est):
            if self.coefficients[est_id] > 0:
                cv_pred[:, est_id] = self._predict_(self.fit_estimators[est_id], X)
            else:
                cv_pred[:, est_id] = 0

        # Combining predictions
        if self.loss_function == "l2":
            y_pred = np.dot(cv_pred, self.coefficients)
        if self.loss_function == 'nloglik':
            cv_pred_bound = probability_bounds(cv_pred, bounds=self._bounds_)
            logodds = logit(cv_pred_bound)
            logodds_pred = np.dot(logodds, self.coefficients)
            y_pred = inverse_logit(logodds_pred)

        return y_pred

    def summary(self):
        """Prints the summary information for the fit SuperLearner to the console.

        Returns
        -------
        None
        """
        if self.coefficients is None:
            raise ValueError("fit() must be called before summary()")
        print('======================================================================')
        print("            Super Learner Candidate Estimator Performance             ")
        print('======================================================================')
        print(self.est_performance.set_index("estimator"))
        print('======================================================================')

    def _predict_(self, estimator, X):
        """Background function to generate predictions based on the designated loss-function
        """
        if self.loss_function == 'l2':
            pred = estimator.predict(X)

        if self.loss_function == 'nloglik':
            if hasattr(estimator, "predict_proba"):
                try:  # Allows for use with PyGAM
                    pred = estimator.predict_proba(X)[:, 1]
                except IndexError:
                    pred = estimator.predict_proba(X)
                if pred.min() < 0 or pred.max() > 1:
                    raise SuperLearnerError("Probability less than zero or greater than one")
            else:
                pred = estimator.predict(X)
                if pred.min() < 0 or pred.max() > 1:
                    raise SuperLearnerError("Probability less than zero or greater than one")

        return pred

    def _error_term_(self, y_obs, y_pred):
        """Calculates the Error term based on the loss function
        """
        if self.loss_function == "l2":
            error = np.sum((y_obs - y_pred) ** 2) / y_obs.shape[0]
        if self.loss_function == 'nloglik':
            y_pred_bound = probability_bounds(v=y_pred, bounds=self._bounds_)
            error = - np.sum(y_obs*y_pred_bound + (1-y_obs)*np.log(1-y_pred_bound)) / y_obs.shape[0]
        return error
