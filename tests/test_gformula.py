import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
from scipy.stats import logistic

from zepid import load_sample_data, spline, load_gvhd_data, load_longitudinal_data
from zepid.causal.gformula import TimeFixedGFormula, SurvivalGFormula, MonteCarloGFormula, IterativeCondGFormula


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

    def test_exposed_standardized1(self, data):
        g = TimeFixedGFormula(data, exposure='A', outcome='Y', standardize='exposed')
        g.outcome_model(model='A + L + A:L', print_results=False)
        g.fit(treatment='all')
        rm = g.marginal_outcome
        g.fit(treatment='none')
        rs = g.marginal_outcome
        npt.assert_allclose(rm - rs, 0.16, rtol=1e-5)
        npt.assert_allclose(rm / rs, 1.387097, rtol=1e-5)

        g = TimeFixedGFormula(data, exposure='A', outcome='Y', standardize='unexposed')
        g.outcome_model(model='A + L + A:L', print_results=False)
        g.fit(treatment='all')
        rm = g.marginal_outcome
        g.fit(treatment='none')
        rs = g.marginal_outcome
        npt.assert_allclose(rm - rs, 0.16, rtol=1e-5)
        npt.assert_allclose(rm / rs, 1.384615, rtol=1e-5)


class TestSurvivalGFormula:

    @pytest.fixture
    def data(self):
        df = load_sample_data(False).drop(columns=['cd4_wk45'])
        df['t'] = np.round(df['t']).astype(int)
        df = pd.DataFrame(np.repeat(df.values, df['t'], axis=0), columns=df.columns)
        df['t'] = df.groupby('id')['t'].cumcount() + 1
        df.loc[((df['dead'] == 1) & (df['id'] != df['id'].shift(-1))), 'd'] = 1
        df['d'] = df['d'].fillna(0)
        df['t_sq'] = df['t'] ** 2
        df['t_cu'] = df['t'] ** 3
        return df.drop(columns=['dead'])

    def test_error_continuous_a(self, data):
        with pytest.raises(ValueError):
            SurvivalGFormula(data, idvar='id', exposure='cd40', outcome='d', time='t')

    def test_error_no_outcome_model(self, data):
        sgf = SurvivalGFormula(data, idvar='id', exposure='art', outcome='d', time='t')
        with pytest.raises(ValueError):
            sgf.fit(treatment='all')

    def test_error_no_fit(self, data):
        sgf = SurvivalGFormula(data, idvar='id', exposure='art', outcome='d', time='t')
        with pytest.raises(ValueError):
            sgf.plot(treatment='all')

    def test_treat_all(self, data):
        sgf = SurvivalGFormula(data, idvar='id', exposure='art', outcome='d', time='t')
        sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu', print_results=False)
        sgf.fit(treatment='all')

        npt.assert_allclose(sgf.marginal_outcome.iloc[0], 0.009365, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[-1], 0.088415, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[9], 0.056269, atol=1e-5)

    def test_treat_none(self, data):
        sgf = SurvivalGFormula(data, idvar='id', exposure='art', outcome='d', time='t')
        sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu', print_results=False)
        sgf.fit(treatment='none')

        npt.assert_allclose(sgf.marginal_outcome.iloc[0], 0.016300, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[-1], 0.147303, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[9], 0.095469, atol=1e-5)

    def test_treat_natural(self, data):
        sgf = SurvivalGFormula(data, idvar='id', exposure='art', outcome='d', time='t')
        sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu', print_results=False)
        sgf.fit(treatment='natural')

        npt.assert_allclose(sgf.marginal_outcome.iloc[0], 0.015030, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[-1], 0.135694, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[9], 0.088382, atol=1e-5)

    def test_treat_custom(self, data):
        sgf = SurvivalGFormula(data, idvar='id', exposure='art', outcome='d', time='t')
        sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu', print_results=False)
        sgf.fit(treatment="((g['age0']>=25) & (g['male']==0))")

        npt.assert_allclose(sgf.marginal_outcome.iloc[0], 0.015090, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[-1], 0.137336, atol=1e-5)
        npt.assert_allclose(sgf.marginal_outcome.iloc[9], 0.088886, atol=1e-5)


class TestMonteCarloGFormula:

    def test_error_continuous_treatment(self, sim_t_fixed_data):
        with pytest.raises(ValueError):
            MonteCarloGFormula(sim_t_fixed_data, idvar='id', exposure='W1', outcome='Y', time_out='t', time_in='t0')

    def test_error_continuous_outcome(self, sim_t_fixed_data):
        with pytest.raises(ValueError):
            MonteCarloGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='W1', time_out='t', time_in='t0')

    def test_error_covariate_label(self, sim_t_fixed_data):
        g = MonteCarloGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.add_covariate_model(label='first', covariate='W1', model='W2')

    def test_error_covariate_type(self, sim_t_fixed_data):
        g = MonteCarloGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.add_covariate_model(label=1, covariate='W1', model='W2', var_type='categorical')

    def test_error_no_outcome_model(self, sim_t_fixed_data):
        g = MonteCarloGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.fit(treatment='all')

    def test_error_treatment_type(self, sim_t_fixed_data):
        g = MonteCarloGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        with pytest.raises(ValueError):
            g.fit(treatment=1)

    def test_monte_carlo_for_single_t(self, sim_t_fixed_data):
        # Estimating monte carlo for single t
        gt = MonteCarloGFormula(sim_t_fixed_data, idvar='id', exposure='A', outcome='Y', time_out='t', time_in='t0')
        gt.outcome_model('A + W1_sq + W2 + W3', print_results=False)
        gt.exposure_model('W1_sq', print_results=False)
        gt.fit(treatment="all", sample=1000000)  # Keep this a high number to reduce simulation errors
        print(gt.predicted_outcomes)

        # Estimating with TimeFixedGFormula
        gf = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        gf.outcome_model(model='A + W1_sq + W2 + W3', print_results=False)
        gf.fit(treatment='all')

        # Expected behavior; same results between the estimation methods
        npt.assert_allclose(gf.marginal_outcome, np.mean(gt.predicted_outcomes['Y']), rtol=1e-3)

    def test_mc_detect_censoring(self):
        df = load_sample_data(timevary=True)

        not_censored = np.where((df['id'] != df['id'].shift(-1)) & (df['dead'] == 0), 0, 1)
        not_censored = np.where(df['out'] == np.max(df['out']), 1, not_censored)

        g = MonteCarloGFormula(df, idvar='id', exposure='art', outcome='dead', time_in='enter', time_out='out')

        npt.assert_equal(np.array(g.gf['__uncensored__']), not_censored)

    def test_mc_detect_censoring2(self):
        df = load_gvhd_data()
        g = MonteCarloGFormula(df, idvar='id', exposure='gvhd', outcome='d', time_in='day', time_out='tomorrow')

        npt.assert_equal(np.array(g.gf['__uncensored__']), 1 - df['censlost'])

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
        g = MonteCarloGFormula(df, idvar='id', exposure='art', outcome='dead', time_in='enter', time_out='out')
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
        cens_m = """male + age0 + age_rs0 + age_rs1 + age_rs2 +  cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 +
                 lag_dvl + lag_art + enter + enter_sq + enter_cu"""
        g.censoring_model(cens_m)
        g.fit(treatment="((g['art']==1) | (g['lag_art']==1))",
              lags={'art': 'lag_art',
                    'cd4': 'lag_cd4',
                    'dvl': 'lag_dvl'},
              sample=5000, t_max=None,
              in_recode=("g['enter_sq'] = g['enter']**2;"
                         "g['enter_cu'] = g['enter']**3"))
        assert isinstance(g.predicted_outcomes, type(pd.DataFrame()))

    # TODO still need to come up with an approach to test MC-g-formula results...


class TestIterativeGFormula:

    @pytest.fixture
    def simple_data(self):
        df = pd.DataFrame()
        df['Y1'] = [0, 1, 1]
        df['A1'] = [0, 1, 1]
        df['Y2'] = [0, 1, 0]
        df['A2'] = [0, 1, 1]
        return df

    def test_error_misalignment(self, simple_data):
        with pytest.raises(ValueError):
            g = IterativeCondGFormula(simple_data, exposures=['A1', 'A2'], outcomes=['Y1'])
        with pytest.raises(ValueError):
            g = IterativeCondGFormula(simple_data, exposures=['A1'], outcomes=['Y1', 'Y2'])

    def test_no_continuous_outcomes(self):
        df = pd.DataFrame()
        df['Y1'] = [0, 0.2, 1]
        df['A1'] = [0, 1, 1]
        with pytest.raises(ValueError):
            g = IterativeCondGFormula(df, exposures=['A1'], outcomes=['Y1'])

    def test_treatment_dimension_error1(self):
        df = load_longitudinal_data()
        icgf = IterativeCondGFormula(df, exposures=['A1', 'A2', 'A3'], outcomes=['Y1', 'Y2', 'Y3'])
        icgf.outcome_model(models=['A1 + L1', 'A2 + L2', 'A3 + L3'], print_results=False)
        with pytest.raises(ValueError):
            icgf.fit(treatments=[1, 1])

    def test_treatment_dimension_error2(self):
        df = load_longitudinal_data()
        icgf = IterativeCondGFormula(df, exposures=['A1', 'A2', 'A3'], outcomes=['Y1', 'Y2', 'Y3'])
        icgf.outcome_model(models=['A1 + L1', 'A2 + L2', 'A3 + L3'], print_results=False)
        with pytest.raises(ValueError):
            icgf.fit(treatments=[[1, 1, 1], [0, 0, 0]])

    def test_error_iterative_recurrent_outcomes(self, simple_data):
        with pytest.raises(ValueError):
            g = IterativeCondGFormula(simple_data, exposures=['A1', 'A2'], outcomes=['Y1', 'Y2'])

    def test_error_iterative_no_outcome(self, simple_data):
        g = IterativeCondGFormula(simple_data, exposures=['A1'], outcomes=['Y1'])
        with pytest.raises(ValueError):
            g.fit(treatments=[1])

    def test_iterative_for_single_t(self, sim_t_fixed_data):
        # Estimating sequential regression for single t
        gt = IterativeCondGFormula(sim_t_fixed_data, exposures=['A'], outcomes=['Y'])
        gt.outcome_model(['A + W1_sq + W2 + W3'], print_results=False)
        gt.fit(treatments=[1])

        # Estimating with TimeFixedGFormula
        gf = TimeFixedGFormula(sim_t_fixed_data, exposure='A', outcome='Y')
        gf.outcome_model(model='A + W1_sq + W2 + W3', print_results=False)
        gf.fit(treatment='all')

        # Expected behavior; same results between the estimation methods
        npt.assert_allclose(gf.marginal_outcome, gt.marginal_outcome)

    def test_match_r_ltmle1(self):
        df = load_longitudinal_data()
        icgf = IterativeCondGFormula(df, exposures=['A1', 'A2', 'A3'], outcomes=['Y1', 'Y2', 'Y3'])
        icgf.outcome_model(models=['A1 + L1', 'A2 + L2', 'A3 + L3'], print_results=False)
        icgf.fit(treatments=[1, 1, 1])
        npt.assert_allclose(icgf.marginal_outcome, 0.4140978, rtol=1e-5)
        icgf.fit(treatments=[0, 0, 0])
        npt.assert_allclose(icgf.marginal_outcome, 0.6464508, rtol=1e-5)

    def test_match_r_ltmle2(self):
        df = load_longitudinal_data()
        icgf = IterativeCondGFormula(df, exposures=['A1', 'A2'], outcomes=['Y1', 'Y2'])
        icgf.outcome_model(models=['A1 + L1', 'A2 + L2'], print_results=False)
        icgf.fit(treatments=[1, 1])
        npt.assert_allclose(icgf.marginal_outcome, 0.3349204, rtol=1e-5)
        icgf.fit(treatments=[0, 0])
        npt.assert_allclose(icgf.marginal_outcome, 0.5122774, rtol=1e-5)

    def test_match_r_ltmle3(self):
        df = load_longitudinal_data()
        icgf = IterativeCondGFormula(df, exposures=['A1', 'A2', 'A3'], outcomes=['Y1', 'Y2', 'Y3'])
        icgf.outcome_model(models=['A1 + L1', 'A2 + A1 + L2', 'A3 + A2 + L3'], print_results=False)
        icgf.fit(treatments=[1, 1, 1])
        npt.assert_allclose(icgf.marginal_outcome, 0.4334696, rtol=1e-5)
        icgf.fit(treatments=[0, 0, 0])
        npt.assert_allclose(icgf.marginal_outcome, 0.6282985, rtol=1e-5)

    def test_match_r_custom_treatment(self):
        df = load_longitudinal_data()
        icgf = IterativeCondGFormula(df, exposures=['A1', 'A2', 'A3'], outcomes=['Y1', 'Y2', 'Y3'])
        icgf.outcome_model(models=['A1 + L1', 'A2 + L2', 'A3 + L3'], print_results=False)
        icgf.fit(treatments=[1, 0, 1])
        npt.assert_allclose(icgf.marginal_outcome, 0.4916937, rtol=1e-5)

        icgf.fit(treatments=[0, 1, 0])
        npt.assert_allclose(icgf.marginal_outcome, 0.5634683, rtol=1e-5)
