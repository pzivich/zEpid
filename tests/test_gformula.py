import pytest
import warnings
import numpy as np
import pandas as pd
import numpy.testing as npt
from scipy.stats import logistic

from zepid.causal.gformula import TimeFixedGFormula, TimeVaryGFormula


@pytest.fixture
def sim_t_fixed_data():
    n = 10000
    np.random.seed(1011)
    df = pd.DataFrame()
    df['W1'] = np.random.normal(size=n)
    df['W2'] = np.random.binomial(1, size=n, p=logistic.cdf(df['W1']))
    df['W3'] = np.random.normal(size=n)
    df['A'] = np.random.binomial(1, size=n, p=logistic.cdf(-1 + 2 * df['W1'] ** 2))
    df['Ya1'] = np.random.binomial(1, size=n,
                                   p=logistic.cdf(-0.5 + 2 * df['W1'] ** 2 + 0.5 * df['W2'] - 0.5 * 1 +
                                                  1.1 * df['W3']))
    df['Ya0'] = np.random.binomial(1, size=n,
                                   p=logistic.cdf(-0.5 + 2 * df['W1'] ** 2 + 0.5 * df['W2'] +
                                                  1.1 * df['W3']))
    df['Y'] = np.where(df['A'] == 1, df['Ya1'], df['Ya0'])
    df['W1_sq'] = df['W1'] ** 2
    df['t'] = 1
    df['id'] = df.index
    return df


class TestTimeFixedGFormula:

    @pytest.fixture
    def data(self):
        df = pd.DataFrame()
        df['L'] = [0 for i in range(125)] + [1 for i in range(75)]
        df['A'] = [0 for i in range(75)] + [1 for i in range(50)] + [0 for i in range(50)] + [1 for i in range(25)]
        df['Y'] = ([0 for i in range(45)] + [1 for i in range(30)] + [0 for i in range(22)] + [1 for i in range(28)] +
                   [0 for i in range(28)] + [1 for i in range(22)] + [0 for i in range(10)] + [1 for i in range(15)])
        df['w'] = [5 for i in range(125)] + [1 for i in range(75)]
        return df

    @pytest.fixture
    def continuous_data(self):
        n = 10000
        np.random.seed(1011)
        df = pd.DataFrame()
        df['W1'] = np.random.normal(size=n)
        df['W2'] = np.random.binomial(1, size=n, p=logistic.cdf(df['W1']))
        df['W3'] = np.random.normal(size=n)
        df['A'] = np.random.binomial(1, size=n, p=logistic.cdf(-1 + 2 * df['W1'] ** 2))
        df['Y'] = -0.5 + 2*df['W1'] + 0.5*df['W2'] - 0.5*df['A'] + 1.1*df['W3'] + np.random.normal(size=n)
        df['t'] = 1
        df['id'] = df.index
        return df

    @pytest.fixture
    def cat_data(self):
        n = 10000
        np.random.seed(1011)
        df = pd.DataFrame()
        df['A1'] = [0 for i in range(7500)] + [1 for i in range(2500)]
        df['A2'] = [0 for i in range(5500)] + [1 for i in range(2000)] + [0 for i in range(2500)]
        df['Y'] = np.random.binomial(1, size=n, p=logistic.cdf(-0.5 + 2 * df['A1'] + 0.5 * df['A2']))
        df['id'] = df.index
        return df

    def test_error_outcome_type(self, continuous_data):
        with pytest.raises(ValueError):
            g = TimeFixedGFormula(continuous_data, exposure='A', outcome='Y', outcome_type='categorical')

    def test_error_wrong_treatment_object(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment=5)

    def test_g_formula1(self, sim_t_fixed_data):
        g = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        g.outcome_model(model='A + W1_sq + W2 + W3', print_results=False)
        g.fit(treatment='all')
        r1 = g.marginal_outcome
        g.fit(treatment='none')
        r0 = g.marginal_outcome
        npt.assert_allclose(r1 - r0, -0.075186, rtol=1e-2)

    def test_g_formula2(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)
        g.fit(treatment='all')
        r1 = g.marginal_outcome
        npt.assert_allclose(r1, 0.575)
        g.fit(treatment='none')
        r0 = g.marginal_outcome
        npt.assert_allclose(r0, 0.415)

    def test_custom_treatment(self, sim_t_fixed_data):
        g = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        g.outcome_model(model='A + W1_sq + W2 + W3', print_results=False)
        g.fit(treatment="g['W3'] > 2")
        npt.assert_allclose(g.marginal_outcome, 0.682829, rtol=1e-5)

    def test_directions_correct(self, sim_t_fixed_data):
        g = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        g.outcome_model(model='A + W1_sq + W2 + W3', print_results=False)
        g.fit(treatment='all')
        r1 = g.marginal_outcome
        g.fit(treatment='none')
        r0 = g.marginal_outcome
        g.fit(treatment="g['W3'] > 2")
        rc = g.marginal_outcome
        assert r1 < rc < r0

    def test_continuous_outcome(self, continuous_data):
        g = TimeFixedGFormula(continuous_data, exposure='A', outcome='Y', outcome_type='continuous')
        g.outcome_model(model='A + W1 + W2 + W3', print_results=False)
        g.fit(treatment='all')
        npt.assert_allclose(g.marginal_outcome, -0.730375, rtol=1e-5)

    def test_error_binary_exposure_list_treatments(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment=['True', 'False'])

    def test_categorical_treat(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], outcome='Y')
        g.outcome_model(model='A1 + A2', print_results=False)
        g.fit(treatment=["False", "False"])
        npt.assert_allclose(g.marginal_outcome, 0.373091, rtol=1e-5)
        g.fit(treatment=["True", "False"])
        npt.assert_allclose(g.marginal_outcome, 0.8128, rtol=1e-5)
        g.fit(treatment=["False", "True"])
        npt.assert_allclose(g.marginal_outcome, 0.5025, rtol=1e-5)

    def test_error_noncategorical_treatment(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], outcome='Y')
        g.outcome_model(model='A1 + A2', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment='all')

    def test_error_mismatch_categorical_treatment(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], outcome='Y')
        g.outcome_model(model='A1 + A2', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment=['True', 'False', 'False'])

    def test_warn_categorical_treatment(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], outcome='Y')
        g.outcome_model(model='A1 + A2', print_results=False)
        with pytest.warns(UserWarning):
            g.fit(treatment=['True', 'True'])

    def test_weighted_data(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y', weights='w')
        g.outcome_model(model='A + L + A:L', print_results=False)
        g.fit(treatment='all')
        r1 = g.marginal_outcome
        npt.assert_allclose(r1, 0.564286, rtol=1e-5)
        g.fit(treatment='none')
        r0 = g.marginal_outcome
        npt.assert_allclose(r0, 0.404286, rtol=1e-5)


class TestTimeVaryGFormula:

    def test_sequential_regression_for_single_t(self, sim_t_fixed_data):
        # Estimating sequential regression for single t
        gt = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y',
                             time_out='t', method='SequentialRegression')
        out_m = '''A + W1_sq + W2 + W3'''
        gt.outcome_model(out_m, print_results=False)
        gt.fit(treatment="all")

        # Estimating with TimeFixedGFormula
        gf = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        gf.outcome_model(model=out_m, print_results=False)
        gf.fit(treatment='all')

        # Expected behavior; same results between the estimation methods
        npt.assert_allclose(gf.marginal_outcome, gt.predicted_outcomes)

