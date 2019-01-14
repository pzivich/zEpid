import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
from scipy.stats import logistic
from lifelines import KaplanMeierFitter

from zepid import load_sample_data, spline
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
    df['t0'] = 0
    df['id'] = df.index
    return df


class TestTimeFixedGFormula:

    @pytest.fixture
    def data(self):
        df = pd.DataFrame()
        df['L'] = [0]*125 + [1]*75
        df['A'] = [0]*75 + [1]*50 + [0]*50 + [1]*25
        df['Y'] = [0]*45 + [1]*30 + [0]*22 + [1]*28 + [0]*28 + [1]*22 + [0]*10 + [1]*15
        df['w'] = [5]*125 + [1]*75
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
        df['A1'] = [0]*7500 + [1]*2500
        df['A2'] = [0]*5500 + [1]*2000 + [0]*2500
        df['Y'] = np.random.binomial(1, size=n, p=logistic.cdf(-0.5 + 2 * df['A1'] + 0.5 * df['A2']))
        df['id'] = df.index
        return df

    def test_error_outcome_type(self, continuous_data):
        with pytest.raises(ValueError):
            TimeFixedGFormula(continuous_data, exposure='A', outcome='Y', outcome_type='categorical')

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
        g = TimeFixedGFormula(continuous_data, exposure='A', outcome='Y', outcome_type='normal')
        g.outcome_model(model='A + W1 + W2 + W3', print_results=False)
        g.fit(treatment='all')
        npt.assert_allclose(g.marginal_outcome, -0.730375, rtol=1e-5)

    def test_error_binary_exposure_list_treatments(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment=['True', 'False'])

    def test_categorical_treat(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], exposure_type='categorical', outcome='Y')
        g.outcome_model(model='A1 + A2', print_results=False)
        g.fit(treatment=["False", "False"])
        npt.assert_allclose(g.marginal_outcome, 0.373091, rtol=1e-5)
        g.fit(treatment=["True", "False"])
        npt.assert_allclose(g.marginal_outcome, 0.8128, rtol=1e-5)
        g.fit(treatment=["False", "True"])
        npt.assert_allclose(g.marginal_outcome, 0.5025, rtol=1e-5)

    def test_error_noncategorical_treatment(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], exposure_type='categorical', outcome='Y')
        g.outcome_model(model='A1 + A2', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment='all')

    def test_error_mismatch_categorical_treatment(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], exposure_type='categorical', outcome='Y')
        g.outcome_model(model='A1 + A2', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment=['True', 'False', 'False'])

    def test_warn_categorical_treatment(self, cat_data):
        g = TimeFixedGFormula(cat_data, exposure=['A1', 'A2'], exposure_type='categorical', outcome='Y')
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

    # TODO add test for Poisson distributed outcome

    def test_stochastic_conditional_probability(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)
        with pytest.raises(ValueError):
            g.fit_stochastic(p=[0.0], conditional=["g['L']==1", "g['L']==0"])

    def test_stochastic_matches_marginal(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)

        g.fit(treatment='all')
        rm = g.marginal_outcome
        g.fit_stochastic(p=1.0)
        rs = g.marginal_outcome
        npt.assert_allclose(rm, rs, rtol=1e-7)

        g.fit(treatment='none')
        rm = g.marginal_outcome
        g.fit_stochastic(p=0.0)
        rs = g.marginal_outcome
        npt.assert_allclose(rm, rs, rtol=1e-7)

    def test_stochastic_between_marginal(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)
        g.fit(treatment='all')
        r1 = g.marginal_outcome
        g.fit(treatment='none')
        r0 = g.marginal_outcome
        g.fit_stochastic(p=0.5)
        r_star = g.marginal_outcome
        assert r0 < r_star < r1

    def test_weight_stochastic_matches_marginal(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y', weights='w')
        g.outcome_model(model='A + L + A:L', print_results=False)

        g.fit(treatment='all')
        rm = g.marginal_outcome
        g.fit_stochastic(p=1.0)
        rs = g.marginal_outcome
        npt.assert_allclose(rm, rs, rtol=1e-7)

        g.fit(treatment='none')
        rm = g.marginal_outcome
        g.fit_stochastic(p=0.0)
        rs = g.marginal_outcome
        npt.assert_allclose(rm, rs, rtol=1e-7)

    def test_weight_stochastic_between_marginal(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y', weights='w')
        g.outcome_model(model='A + L + A:L', print_results=False)
        g.fit(treatment='all')
        r1 = g.marginal_outcome
        g.fit(treatment='none')
        r0 = g.marginal_outcome
        g.fit_stochastic(p=0.5)
        r_star = g.marginal_outcome
        assert r0 < r_star < r1

    def test_conditional_stochastic_matches_marginal(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)

        g.fit(treatment='all')
        rm = g.marginal_outcome
        g.fit_stochastic(p=[1.0, 1.0], conditional=["g['L']==1", "g['L']==0"])
        rs = g.marginal_outcome
        npt.assert_allclose(rm, rs, rtol=1e-7)

        g.fit(treatment='none')
        rn = g.marginal_outcome
        g.fit_stochastic(p=[0.0, 0.0], conditional=["g['L']==1", "g['L']==0"])
        rs = g.marginal_outcome
        npt.assert_allclose(rn, rs, rtol=1e-7)

    def test_conditional_weight_stochastic_matches_marginal(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y', weights='w')
        g.outcome_model(model='A + L + A:L', print_results=False)

        g.fit(treatment='all')
        rm = g.marginal_outcome
        g.fit_stochastic(p=[1.0, 1.0], conditional=["g['L']==1", "g['L']==0"])
        rs = g.marginal_outcome
        npt.assert_allclose(rm, rs, rtol=1e-7)

        g.fit(treatment='none')
        rn = g.marginal_outcome
        g.fit_stochastic(p=[0.0, 0.0], conditional=["g['L']==1", "g['L']==0"])
        rs = g.marginal_outcome
        npt.assert_allclose(rn, rs, rtol=1e-7)

    def test_conditional_stochastic_warning(self):
        data = pd.DataFrame()
        data['A'] = [1]*50 + [0]*50
        data['L'] = [1]*25 + [0]*25 + [1]*40 + [0]*10
        data['Y'] = [1]*25 + [0]*25 + [1]*25 + [0]*25
        g = TimeFixedGFormula(data, exposure='A', outcome='Y')
        g.outcome_model(model='A + L + A:L', print_results=False)
        with pytest.warns(UserWarning):
            g.fit_stochastic(p=[1.0, 1.0], conditional=["(g['L']==1) | (g['L']==0)", "g['L']==0"])


class TestTimeVaryGFormula:

    @pytest.fixture
    def longdata(self):
        df = pd.DataFrame()
        df['id'] = [1, 1, 1]
        df['t'] = [0, 1, 2]
        df['A'] = [0, 1, 1]
        df['Y'] = [0, 0, 1]
        df['W'] = [5, 5, 5]
        df['L'] = [25, 20, 31]
        return df

    @pytest.fixture
    def data(self):
        df = pd.DataFrame()
        np.random.seed(555)
        n = 1000
        df['W'] = np.random.normal(size=n)
        df['L1'] = np.random.normal(size=n) + df['W']
        df['A1'] = np.random.binomial(1, size=n, p=logistic.cdf(df['L1']))
        df['Y1'] = np.random.binomial(1, size=n, p=logistic.cdf(-1 + 0.7 * df['L1'] - 0.3 * df['A1']))
        df['L2'] = 0.5 * df['L1'] - 0.9 * df['A1'] + np.random.normal(size=n)
        df['A2'] = np.random.binomial(1, size=n, p=logistic.cdf(1.5 * df['A1'] + 0.8 * df['L2']))
        df['A2'] = np.where(df['A1'] == 1, 1, df['A2'])
        df['Y2'] = np.random.binomial(1, size=n, p=logistic.cdf(-1 + 0.7 * df['L2'] - 0.3 * df['A2']))
        df['Y2'] = np.where(df['Y1'] == 1, np.nan, df['Y2'])
        df['L3'] = 0.5 * df['L2'] - 0.9 * df['A2'] + np.random.normal(size=n)
        df['A3'] = np.random.binomial(1, size=n, p=logistic.cdf(1.5 * df['A2'] + 0.8 * df['L3']))
        df['A2'] = np.where(df['A1'] == 1, 1, df['A2'])
        df['Y3'] = np.random.binomial(1, size=n, p=logistic.cdf(-1 + 0.7 * df['L3'] - 0.3 * df['A3']))
        df['Y3'] = np.where((df['Y2'] == 1) | (df['Y1'] == 1), np.nan, df['Y3'])
        df['id'] = df.index
        d1 = df[['id', 'Y1', 'A1', 'L1', 'W']].copy()
        d1.rename(mapper={'Y1': 'Y', 'A1': 'A', 'L1': 'L'}, axis='columns', inplace=True)
        d1['t0'] = 0
        d1['t'] = 1
        d1['t2'] = 0
        d1['w'] = 2
        d2 = df[['id', 'Y2', 'A2', 'L2', 'W']].copy()
        d2.rename(mapper={'Y2': 'Y', 'A2': 'A', 'L2': 'L'}, axis='columns', inplace=True)
        d2['t0'] = 1
        d2['t'] = 2
        d2['t2'] = 5
        d2['w'] = 2
        d3 = df[['id', 'Y3', 'A3', 'L3', 'W']].copy()
        d3.rename(mapper={'Y3': 'Y', 'A3': 'A', 'L3': 'L'}, axis='columns', inplace=True)
        d3['t0'] = 2
        d3['t'] = 3
        d3['t2'] = 10
        d3['w'] = 2
        df = pd.concat([d1, d2, d3], sort=False).dropna()
        return df

    def test_error_continuous_treatment(self, sim_t_fixed_data):
        with pytest.raises(ValueError):
            TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='W1', outcome='Y', time_out='t', time_in='t0')

    def test_error_continuous_outcome(self, sim_t_fixed_data):
        with pytest.raises(ValueError):
            TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='W1', time_out='t', time_in='t0')

    def test_error_monte_carlo1(self, sim_t_fixed_data):
        with pytest.raises(ValueError):
            TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='W1', time_out='t')

    def test_error_estimation_method(self, sim_t_fixed_data):
        with pytest.raises(ValueError):
            TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='W1', time_out='t', method='Fail')

    def test_error_covariate_label(self, sim_t_fixed_data):
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.add_covariate_model(label='first', covariate='W1', model='W2')

    def test_error_covariate_type(self, sim_t_fixed_data):
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.add_covariate_model(label=1, covariate='W1', model='W2', var_type='categorical')

    def test_error_no_outcome_model(self, sim_t_fixed_data):
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.fit(treatment='all')
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y',
                             time_out='t', method='SequentialRegression')
        with pytest.raises(ValueError):
            g.fit(treatment='all')

    def test_error_treatment_type(self, sim_t_fixed_data):
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.fit(treatment=1)
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y',
                             time_out='t', method='SequentialRegression')
        with pytest.raises(ValueError):
            g.fit(treatment=1)

    def test_error_sr_other_models(self, sim_t_fixed_data):
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y',
                             time_out='t', method='SequentialRegression')
        g.outcome_model('A + W1_sq + W2 + W3', print_results=False)
        g.exposure_model('W1_sq', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment='all')

    def test_error_sr_natural_course(self, sim_t_fixed_data):
        g = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y',
                             time_out='t', method='SequentialRegression')
        g.outcome_model('A + W1_sq + W2 + W3', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment='natural')

    def test_error_sr_recurrent_outcomes(self):
        df = pd.DataFrame()
        df['id'] = [1, 1, 1]
        df['Y'] = [0, 1, 1]
        df['A'] = [0, 1, 1]
        df['t'] = [0, 1, 2]
        g = TimeVaryGFormula(df, idvar='id', exposure='A', outcome='Y', time_out='t', method='SequentialRegression')
        g.outcome_model('A', print_results=False)
        with pytest.raises(ValueError):
            g.fit(treatment='all')

    def test_long_to_wide_conversion(self, longdata):
        g = TimeVaryGFormula(longdata, idvar='id', exposure='A', outcome='Y',
                             time_out='t', method='SequentialRegression')
        lf = g._long_to_wide(longdata, id='id', t='t')
        expected_lf = pd.DataFrame.from_records([{'A_0': 0, 'A_1': 1, 'A_2': 1,
                                                  'Y_0': 0, 'Y_1': 0, 'Y_2': 1,
                                                  'W_0': 5, 'W_1': 5, 'W_2': 5,
                                                  'L_0': 25, 'L_1': 20, 'L_2': 31,
                                                  'id': 1}]).set_index('id')
        pdt.assert_frame_equal(lf, expected_lf[['A_0', 'A_1', 'A_2', 'Y_0', 'Y_1', 'Y_2', 'W_0', 'W_1', 'W_2', 'L_0',
                                                'L_1', 'L_2']], check_names=False)

    def test_monte_carlo_for_single_t(self, sim_t_fixed_data):
        # Estimating monte carlo for single t
        gt = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y',
                              time_out='t', time_in='t0')
        gt.outcome_model('A + W1_sq + W2 + W3', print_results=False)
        gt.exposure_model('W1_sq', print_results=False)
        gt.fit(treatment="all", sample=1000000)  # Keep this a high number to reduce simulation errors

        # Estimating with TimeFixedGFormula
        gf = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        gf.outcome_model(model='A + W1_sq + W2 + W3', print_results=False)
        gf.fit(treatment='all')

        # Expected behavior; same results between the estimation methods
        npt.assert_allclose(gf.marginal_outcome, np.mean(gt.predicted_outcomes['Y']), rtol=1e-3)

    def test_sequential_regression_for_single_t(self, sim_t_fixed_data):
        # Estimating sequential regression for single t
        gt = TimeVaryGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y',
                              time_out='t', method='SequentialRegression')
        gt.outcome_model('A + W1_sq + W2 + W3', print_results=False)
        gt.fit(treatment="all")

        # Estimating with TimeFixedGFormula
        gf = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        gf.outcome_model(model='A + W1_sq + W2 + W3', print_results=False)
        gf.fit(treatment='all')

        # Expected behavior; same results between the estimation methods
        npt.assert_allclose(gf.marginal_outcome, gt.predicted_outcomes)

    def test_complete_mc_procedure_completes(self):
        df = load_sample_data(timevary=True)
        df['lag_art'] = df['art'].shift(1)
        df['lag_art'] = np.where(df.groupby('id').cumcount() == 0, 0, df['lag_art'])
        df['lag_cd4'] = df['cd4'].shift(1)
        df['lag_cd4'] = np.where(df.groupby('id').cumcount() == 0, df['cd40'], df['lag_cd4'])
        df['lag_dvl'] = df['dvl'].shift(1)
        df['lag_dvl'] = np.where(df.groupby('id').cumcount() == 0, df['dvl0'], df['lag_dvl'])
        df[['age_rs0', 'age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=4, term=2, restricted=True)  # age spline
        df['cd40_sq'] = df['cd40'] ** 2
        df['cd40_cu'] = df['cd40'] ** 3
        df['cd4_sq'] = df['cd4'] ** 2
        df['cd4_cu'] = df['cd4'] ** 3
        df['enter_sq'] = df['enter'] ** 2
        df['enter_cu'] = df['enter'] ** 3
        g = TimeVaryGFormula(df, idvar='id', exposure='art', outcome='dead', time_in='enter', time_out='out')
        exp_m = '''male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + cd4 + cd4_sq + 
                cd4_cu + dvl + enter + enter_sq + enter_cu'''
        g.exposure_model(exp_m, restriction="g['lag_art']==0")
        out_m = '''art + male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + cd4 + 
                cd4_sq + cd4_cu + dvl + enter + enter_sq + enter_cu'''
        g.outcome_model(out_m, restriction="g['drop']==0")
        dvl_m = '''male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + 
                lag_dvl + lag_art + enter + enter_sq + enter_cu'''
        g.add_covariate_model(label=1, covariate='dvl', model=dvl_m, var_type='binary')
        cd4_m = '''male + age0 + age_rs0 + age_rs1 + age_rs2 +  cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + 
                lag_dvl + lag_art + enter + enter_sq + enter_cu'''
        cd4_recode_scheme = ("g['cd4'] = np.maximum(g['cd4'],1);"
                             "g['cd4_sq'] = g['cd4']**2;"
                             "g['cd4_cu'] = g['cd4']**3")
        g.add_covariate_model(label=2, covariate='cd4', model=cd4_m, recode=cd4_recode_scheme, var_type='continuous')
        g.fit(treatment="((g['art']==1) | (g['lag_art']==1))",
              lags={'art': 'lag_art',
                    'cd4': 'lag_cd4',
                    'dvl': 'lag_dvl'},
              sample=10000, t_max=None,
              in_recode=("g['enter_sq'] = g['enter']**2;"
                         "g['enter_cu'] = g['enter']**3"))
        assert isinstance(g.predicted_outcomes, type(pd.DataFrame()))

    def test_match_r_ltmle(self, data):
        g = TimeVaryGFormula(data, idvar='id', exposure='A', outcome='Y',
                             time_out='t', method='SequentialRegression')
        out_m = 'A + L'
        g.outcome_model(out_m, print_results=False)
        g.fit(treatment="all")
        npt.assert_allclose(g.predicted_outcomes, 0.4051569, rtol=1e-5)
        g.fit(treatment="none")
        npt.assert_allclose(g.predicted_outcomes, 0.661226, rtol=1e-5)

    def test_sr_gap_time(self, data):
        g = TimeVaryGFormula(data, idvar='id', exposure='A', outcome='Y', time_out='t2', method='SequentialRegression')
        out_m = 'A + L'
        g.outcome_model(out_m, print_results=False)
        g.fit(treatment="all")
        npt.assert_allclose(g.predicted_outcomes, 0.4051569, rtol=1e-5)
        g.fit(treatment="none")
        npt.assert_allclose(g.predicted_outcomes, 0.661226, rtol=1e-5)

    def test_sr_weights1(self, data):
        g = TimeVaryGFormula(data, idvar='id', exposure='A', outcome='Y', time_out='t2', method='SequentialRegression',
                             weights='w')
        out_m = 'A + L'
        g.outcome_model(out_m, print_results=False)
        g.fit(treatment="all")
        npt.assert_allclose(g.predicted_outcomes, 0.4051569, rtol=1e-5)
        g.fit(treatment="none")
        npt.assert_allclose(g.predicted_outcomes, 0.661226, rtol=1e-5)

    def test_sr_custom_treatment(self, data):
        g = TimeVaryGFormula(data, idvar='id', exposure='A', outcome='Y', time_out='t2', method='SequentialRegression')
        out_m = 'A + L'
        g.outcome_model(out_m, print_results=False)
        g.fit(treatment="g['t'] != 2")
        npt.assert_allclose(g.predicted_outcomes, 0.48543, rtol=1e-5)

    def test_sr_custom_time_point(self, data):
        g = TimeVaryGFormula(data, idvar='id', exposure='A', outcome='Y', time_out='t', method='SequentialRegression')
        g.outcome_model('A + L', print_results=False)
        # values come from R's ltmle package
        g.fit(treatment="all", t_max=2)
        npt.assert_allclose(g.predicted_outcomes, 0.33492, rtol=1e-5)
        g.fit(treatment="none", t_max=2)
        npt.assert_allclose(g.predicted_outcomes, 0.51228, rtol=1e-5)

    def test_sr_warning_outside_time_point(self, data):
        g = TimeVaryGFormula(data, idvar='id', exposure='A', outcome='Y', time_out='t2', method='SequentialRegression')
        g.outcome_model('A + L', print_results=False)
        with pytest.warns(UserWarning):
            g.fit(treatment="all", t_max=6)
        npt.assert_allclose(g.predicted_outcomes, 0.33492, rtol=1e-5)

    # TODO still need to come up with an approach to test MC-g-formula results...
