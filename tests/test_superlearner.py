import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression, LinearRegression

from zepid.superlearner import EmpiricalMeanSL, GLMSL, StepwiseSL, SuperLearner


@pytest.fixture
def data():
    data = pd.DataFrame()
    data['C'] = [5, 10, 12, 13, -10, 0, 37]
    data['B'] = [0, 0, 0, 1, 1, 1, 1]
    data['M'] = [0, 0, 1, np.nan, 0, 1, 1]
    return data


@pytest.fixture
def data_test():
    # True Models: y ~ a + w + w*x + N(0, 1)
    # True Models: Pr(b=1) ~ logit(a + w - w*x)
    data = pd.DataFrame()
    data['X'] = [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    data['W'] = [-3, 2, -2, -1, 2, -2, 2, -2, -1, -1, 1, 2, -1, 0, -2, -1, -1, -3, -1, 1]
    data['A'] = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    data['Y'] = [-6.6, 4.2, -2.0, -0.6, 6.6, -2.2, 1.2, -4.9, -2.2, 0.8, 1.3, 3.4, 0.3, 1.4, -1.8, -2.4, -1.6,
                 -4.1, -2.5, 2.5]
    data['B'] = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    return data


class TestEmpiricalMeanSL:

    def test_error_missing_data(self, data):
        empm = EmpiricalMeanSL()
        with pytest.raises(ValueError, match="missing values in X or y"):
            empm.fit(np.asarray(data['M']), np.asarray(data['C']))

        with pytest.raises(ValueError, match="missing values in X or y"):
            empm.fit(np.asarray(data['C']), np.asarray(data['M']))

    def test_error_shapes(self, data):
        empm = EmpiricalMeanSL()
        with pytest.raises(ValueError, match="same number of observations"):
            empm.fit(np.asarray(data['C']), np.array([0, 1, 1]))

    def test_mean_correct(self, data):
        empm = EmpiricalMeanSL()

        # Continuous
        empm.fit(X=np.asarray(data['B']), y=np.asarray(data['C']))
        npt.assert_allclose(empm.empirical_mean, np.mean(data['C']))

        # Binary
        empm.fit(X=np.asarray(data['C']), y=np.asarray(data['B']))
        npt.assert_allclose(empm.empirical_mean, np.mean(data['B']))

    def test_predict(self, data):
        empm = EmpiricalMeanSL()

        # Continuous
        empm.fit(X=np.asarray(data['B']), y=np.asarray(data['C']))
        X_pred = np.array([1, 1, 1])
        pred_y = empm.predict(X=X_pred)
        assert pred_y.shape[0] == X_pred.shape[0]  # Same shape in output
        npt.assert_allclose(empm.empirical_mean,
                            [np.mean(data['C'])] * X_pred.shape[0])

        # Binary
        empm.fit(X=np.asarray(data['C']), y=np.asarray(data['B']))
        X_pred = np.array([1, 1, 1, 0])
        pred_y = empm.predict(X=X_pred)
        assert pred_y.shape[0] == X_pred.shape[0]  # Same shape in output
        npt.assert_allclose(empm.empirical_mean,
                            [np.mean(data['B'])] * X_pred.shape[0])


class TestGLMSL:

    def test_error_missing_data(self, data):
        f = sm.families.family.Binomial()
        glm = GLMSL(f)
        with pytest.raises(ValueError, match="missing values in X or y"):
            glm.fit(np.asarray(data['M']), np.asarray(data['C']))

        with pytest.raises(ValueError, match="missing values in X or y"):
            glm.fit(np.asarray(data['C']), np.asarray(data['M']))

    def test_error_shapes(self, data):
        f = sm.families.family.Binomial()
        glm = GLMSL(f)
        with pytest.raises(ValueError, match="same number of observations"):
            glm.fit(np.asarray(data['C']), np.array([0, 1, 1]))

    def test_match_statsmodels_continuous(self, data_test):
        f = sm.families.family.Gaussian()
        glm = GLMSL(f)
        glm.fit(np.asarray(data_test[['A', 'W', 'X']]), np.asarray(data_test['Y']))

        # Checking chosen covariates
        sm_glm = smf.glm("Y ~ A + W + X", data_test, family=f).fit()
        npt.assert_allclose(glm.model.params,
                            sm_glm.params)

        # Checking predictions from model
        step_preds = glm.predict(np.asarray(data_test.loc[0:5, ['A', 'W', 'X']]))
        npt.assert_allclose(step_preds,
                            sm_glm.predict(data_test.loc[0:5, ]))

    def test_match_statsmodels_binary(self, data_test):
        f = sm.families.family.Binomial()
        glm = GLMSL(f)
        glm.fit(np.asarray(data_test[['A', 'W']]), np.asarray(data_test['B']))

        # Checking chosen covariates
        sm_glm = smf.glm("B ~ A + W", data_test, family=f).fit()
        npt.assert_allclose(glm.model.params,
                            sm_glm.params)

        # Checking predictions from model
        step_preds = glm.predict(np.asarray(data_test.loc[0:5, ['A', 'W']]))
        npt.assert_allclose(step_preds,
                            sm_glm.predict(data_test.loc[0:5, ]))


class TestStepWiseSL:

    def test_error_setup(self):
        f = sm.families.family.Binomial()
        # Testing selection method error
        with pytest.raises(ValueError, match="`method` must be one"):
            StepwiseSL(f, selection="wrong")
        # Testing interaction_order < 0
        with pytest.raises(ValueError, match="interaction_order"):
            StepwiseSL(f, order_interaction=-1)
        # Testing interaction_order != int
        with pytest.raises(ValueError, match="interaction_order"):
            StepwiseSL(f, order_interaction=0.4)

    def test_error_missing_data(self, data):
        f = sm.families.family.Binomial()
        step = StepwiseSL(f)
        with pytest.raises(ValueError, match="missing values in X or y"):
            step.fit(np.asarray(data['M']), np.asarray(data['C']))

        with pytest.raises(ValueError, match="missing values in X or y"):
            step.fit(np.asarray(data['C']), np.asarray(data['M']))

    def test_error_shapes(self, data):
        f = sm.families.family.Binomial()
        step = StepwiseSL(f)
        with pytest.raises(ValueError, match="same number of observations"):
            step.fit(np.asarray(data['C']), np.array([0, 1, 1]))

    def test_warn_backward_saturated(self, data_test):
        f = sm.families.family.Binomial()
        step = StepwiseSL(f, selection="backward", order_interaction=3, verbose=False)
        with pytest.warns(UserWarning, match="order_interaction is greater"):
            step.fit(np.asarray(data_test[['A', 'W']]), np.asarray(data_test['B']))

    def test_forward_continuous(self, data_test):
        f = sm.families.family.Gaussian()
        step = StepwiseSL(f, selection="forward", order_interaction=1)
        step.fit(np.asarray(data_test[['A', 'W', 'X']]), np.asarray(data_test['Y']))

        # Checking chosen covariates
        best_x_indices = np.asarray((1, 5, 4, 3))  # This is the order the AIC's got forward
        npt.assert_array_equal(np.asarray(step.cols_optim),
                               best_x_indices)

        # Checking predictions from model
        best_x_preds = np.array([-6.79917101,  5.38279072, -1.86983794, -1.22659046,  5.38279072, -1.86983794])
        step_preds = step.predict(np.asarray(data_test.loc[0:5, ['A', 'W', 'X']]))
        npt.assert_allclose(step_preds,
                            best_x_preds)

    def test_backward_continuous(self, data_test):
        f = sm.families.family.Gaussian()
        step = StepwiseSL(f, selection="backward", order_interaction=1)
        step.fit(np.asarray(data_test[['A', 'W', 'X']]), np.asarray(data_test['Y']))

        # Checking chosen covariates
        best_x_indices = np.asarray((1, 3, 4, 5))  # This is the order the AIC's got backward
        npt.assert_array_equal(np.asarray(step.cols_optim),
                               best_x_indices)

        # Checking predictions from model
        best_x_preds = np.array([-6.79917101,  5.38279072, -1.86983794, -1.22659046,  5.38279072, -1.86983794])
        step_preds = step.predict(np.asarray(data_test.loc[0:5, ['A', 'W', 'X']]))
        npt.assert_allclose(step_preds,
                            best_x_preds)

    def test_forward_binary(self, data_test):
        f = sm.families.family.Binomial()
        step = StepwiseSL(f, selection="forward", order_interaction=1)
        step.fit(np.asarray(data_test[['A', 'W', 'X']]), np.asarray(data_test['B']))

        # Checking chosen covariates
        best_x_indices = np.asarray((1, 3))  # This is the order the AIC's got backward
        npt.assert_array_equal(np.asarray(step.cols_optim),
                               best_x_indices)

        # Checking predictions from model
        best_x_preds = np.array([0.00646765, 0.96985036, 0.7380893, 0.45616085, 0.96985036, 0.7380893])
        step_preds = step.predict(np.asarray(data_test.loc[0:5, ['A', 'W', 'X']]))
        npt.assert_allclose(step_preds,
                            best_x_preds, rtol=1e-5)

    def test_backward_binary(self, data_test):
        f = sm.families.family.Binomial()
        step = StepwiseSL(f, selection="backward", order_interaction=1)
        step.fit(np.asarray(data_test[['A', 'X']]), np.asarray(data_test['B']))

        # Checking chosen covariates
        best_x_indices = np.asarray([])  # This is the order the AIC's got backward
        npt.assert_array_equal(np.asarray(step.cols_optim),
                               best_x_indices)

        # Checking predictions from model
        best_x_preds = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        step_preds = step.predict(np.asarray(data_test.loc[0:5, ['A', 'X']]))
        npt.assert_allclose(step_preds,
                            best_x_preds, rtol=1e-5)


class TestSuperLearner:

    @pytest.fixture
    def load_estimators_continuous(self):
        emp = EmpiricalMeanSL()
        linr = LinearRegression()
        step = StepwiseSL(family=sm.families.family.Gaussian(), selection="forward", order_interaction=1)
        return [emp, linr, step]

    @pytest.fixture
    def load_estimators_binary(self):
        emp = EmpiricalMeanSL()
        logr = LogisticRegression()
        step = StepwiseSL(family=sm.families.family.Binomial(), selection="forward", order_interaction=1)
        return [emp, logr, step]

    def test_error_estimator_length(self, load_estimators_continuous):
        with pytest.raises(ValueError, match="estimators and estimator_labels"):
            SuperLearner(estimators=load_estimators_continuous, estimator_labels=["wrong", "number"])

    def test_error_solver(self, load_estimators_continuous):
        with pytest.raises(ValueError, match="The solver INVALID_SOLVER is not currently"):
            SuperLearner(estimators=load_estimators_continuous, estimator_labels=["Mean", "LineR", "Step"],
                         solver="INVALID_SOLVER")

    def test_error_lossf(self, load_estimators_continuous):
        with pytest.raises(ValueError, match="The loss function INVALID_LOSSF is not currently"):
            SuperLearner(estimators=load_estimators_continuous, estimator_labels=["Mean", "LineR", "Step"],
                         loss_function="INVALID_LOSSF")

    def test_error_shapes(self, data, load_estimators_continuous):
        sl = SuperLearner(estimators=load_estimators_continuous, estimator_labels=["Mean", "LineR", "Step"])
        with pytest.raises(ValueError, match="same number of observations"):
            sl.fit(np.asarray(data['C']), np.array([0, 1, 1]))

        with pytest.raises(ValueError, match="same number of observations"):
            sl.fit(np.array([0, 1, 1]), np.asarray(data['C']))

    def test_error_nan(self, data, load_estimators_continuous):
        sl = SuperLearner(estimators=load_estimators_continuous, estimator_labels=["Mean", "LineR", "Step"], folds=2)
        with pytest.raises(ValueError, match="missing values in X or y"):
            sl.fit(np.asarray(data['C']), np.asarray(data['M']))

        with pytest.raises(ValueError, match="missing values in X or y"):
            sl.fit(np.asarray(data['M']), np.asarray(data['C']))

        fsl = sl.fit(np.asarray(data['B']).reshape(-1, 1), np.asarray(data['C']))
        with pytest.raises(ValueError, match="missing values in X"):
            fsl.predict(np.asarray(data['M']))

    def test_error_before_fit(self, data, load_estimators_continuous):
        sl = SuperLearner(estimators=load_estimators_continuous, estimator_labels=["Mean", "LineR", "Step"])
        with pytest.raises(ValueError, match="must be called before"):
            sl.predict(np.asarray(data['C']))

        with pytest.raises(ValueError, match="must be called before"):
            sl.summary()

    def test_warn_lossf(self, data_test, load_estimators_binary):
        sl = SuperLearner(estimators=load_estimators_binary, estimator_labels=["Mean", "LineR", "Step"], folds=3)
        with pytest.warns(UserWarning, match="looks like your `y` is binary"):
            sl.fit(np.asarray(data_test[['A', 'W', 'X']]), np.asarray(data_test['B']))

    def test_continuous_superlearner(self, data_test, load_estimators_continuous):
        sl = SuperLearner(estimators=load_estimators_continuous, estimator_labels=["Mean", "LineR", "Step"], folds=5)
        fsl = sl.fit(np.asarray(data_test[['A', 'W', 'X']]), np.asarray(data_test['Y']))

        # Coefficients and CV-Error
        expected = pd.DataFrame.from_records([{"estimator": "Mean",  "cv_error": 10.2505625, "coefs": 0.097767},
                                              {"estimator": "LineR", "cv_error": 1.90231789, "coefs": 0.357968},
                                              {"estimator": "Step",  "cv_error": 1.66769069, "coefs": 0.544265}])
        pdt.assert_frame_equal(fsl.est_performance,
                               expected)

        # Predicted values
        expected = np.array([-5.65558813, 4.45487519, -1.91811241, -1.46252119, 4.45487519, -1.91811241])
        npt.assert_allclose(fsl.predict(np.asarray(data_test.loc[0:5, ["A", "W", "X"]])),
                            expected)

    def test_binary_superlearner(self, data_test, load_estimators_binary):
        sl = SuperLearner(estimators=load_estimators_binary, estimator_labels=["Mean", "LogR", "Step"],
                          loss_function='nloglik', folds=5)
        fsl = sl.fit(np.asarray(data_test[['A', 'X']]), np.asarray(data_test['B']))

        # Coefficients and CV-Error
        expected = pd.DataFrame.from_records([{"estimator": "Mean", "cv_error": -0.049431, "coefs": 0.966449},
                                              {"estimator": "LogR", "cv_error": -0.030154, "coefs": 0.033551},
                                              {"estimator": "Step", "cv_error": 1.797190,  "coefs": 0.}])
        pdt.assert_frame_equal(fsl.est_performance,
                               expected)

        # Predicted values
        expected = np.array([0.69634645, 0.70191334, 0.70322108, 0.69766808, 0.70191334, 0.70322108])
        npt.assert_allclose(fsl.predict(np.asarray(data_test.loc[0:5, ["A", "X"]])),
                            expected)
