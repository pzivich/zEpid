import pytest
import numpy.testing as npt
from sklearn.linear_model import LogisticRegression, LinearRegression

import zepid as ze
from zepid.causal.doublyrobust import TMLE, AIPTW


class TestTMLE:

    @pytest.fixture
    def df(self):
        df = ze.load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = ze.spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = ze.spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.drop(columns=['cd4_wk45']).dropna()

    @pytest.fixture
    def cf(self):
        df = ze.load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = ze.spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = ze.spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.drop(columns=['dead']).dropna()

    def test_drop_missing_data(self):
        df = ze.load_sample_data(False)
        tmle = TMLE(df, exposure='art', outcome='dead')
        assert df.dropna().shape[0] == tmle.df.shape[0]

    def test_error_when_no_models_specified1(self, df):
        tmle = TMLE(df, exposure='art', outcome='dead')
        with pytest.raises(ValueError):
            tmle.fit()

    def test_error_when_no_models_specified2(self, df):
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        with pytest.raises(ValueError):
            tmle.fit()

    def test_error_when_no_models_specified3(self, df):
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        with pytest.raises(ValueError):
            tmle.fit()

    def test_match_r_epsilons(self, df):
        r_epsilons = [-0.016214091, 0.003304079]
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle._epsilon, r_epsilons, rtol=1e-5)

    def test_match_r_tmle_riskdifference(self, df):
        r_rd = -0.08440622
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.risk_difference, r_rd)

    def test_match_r_tmle_rd_ci(self, df):
        r_ci = -0.1541104, -0.01470202
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.risk_difference_ci, r_ci, rtol=1e-5)

    def test_match_r_tmle_riskratio(self, df):
        r_rr = 0.5344266
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.risk_ratio, r_rr)

    def test_match_r_tmle_rr_ci(self, df):
        r_ci = 0.2773936, 1.0296262
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.risk_ratio_ci, r_ci, rtol=1e-5)

    def test_match_r_tmle_oddsratio(self, df):
        r_or = 0.4844782
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.odds_ratio, r_or)

    def test_match_r_tmle_or_ci(self, df):
        r_ci = 0.232966, 1.007525
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.odds_ratio_ci, r_ci, rtol=1e-5)

    def test_symmetric_bounds_on_gW(self, df):
        r_rd = -0.08203143
        r_ci = -0.1498092, -0.01425363
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=0.1, print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.risk_difference, r_rd)
        npt.assert_allclose(tmle.risk_difference_ci, r_ci, rtol=1e-5)

    def test_asymmetric_bounds_on_gW(self, df):
        r_rd = -0.08433208
        r_ci = -0.1541296, -0.01453453
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=[0.025, 0.9], print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.risk_difference, r_rd)
        npt.assert_allclose(tmle.risk_difference_ci, r_ci, rtol=1e-5)

    def test_no_risk_with_continuous(self, cf):
        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=[0.025, 0.9], print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        assert tmle.risk_difference is None
        assert tmle.risk_ratio is None
        assert tmle.odds_ratio is None
        assert tmle.risk_difference_ci is None
        assert tmle.risk_ratio_ci is None
        assert tmle.odds_ratio_ci is None

    def test_no_ate_with_binary(self, df):
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=[0.025, 0.9], print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        assert tmle.average_treatment_effect is None
        assert tmle.average_treatment_effect_ic is None

    def test_match_r_epsilons_continuous(self, cf):
        r_epsilons = [-0.0046411652, 0.0002270186]
        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle._epsilon, r_epsilons, rtol=1e-4, atol=1e-4)

    def test_match_r_continuous_outcome(self, cf):
        r_ate = 223.4022
        r_ci = 118.6037, 328.2008

        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.average_treatment_effect, r_ate, rtol=1e-3)
        npt.assert_allclose(tmle.average_treatment_effect_ic, r_ci, rtol=1e-3)

    def test_match_r_continuous_outcome_gbounds(self, cf):
        r_ate = 223.3958
        r_ci = 118.4178, 328.3737

        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=[0.025, 0.9], print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.average_treatment_effect, r_ate, rtol=1e-3)
        npt.assert_allclose(tmle.average_treatment_effect_ic, r_ci, rtol=1e-3)

    def test_match_r_continuous_poisson(self, cf):
        r_ate = 223.4648
        r_ci = 118.6276, 328.3019

        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False, continous_distribution='poisson')
        tmle.fit()
        npt.assert_allclose(tmle.average_treatment_effect, r_ate, rtol=1e-3)
        npt.assert_allclose(tmle.average_treatment_effect_ic, r_ci, rtol=1e-3)

    def test_sklearn_in_tmle(self, df):
        log = LogisticRegression(penalty='l1', C=1.0, random_state=201)
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + cd40 + dvl0', custom_model=log)
        tmle.outcome_model('art + male + age0 + cd40 + dvl0', custom_model=log)
        tmle.fit()
        # Dropping since Linux RNG does not match my OS (windows)
        # npt.assert_allclose(tmle.psi, -0.07507877527854623)
        # npt.assert_allclose(tmle.confint, [-0.15278930211034644, 0.002631751553253986], rtol=1e-5)
        # TODO Test now only checks no errors are thrown when adding sklearn

    def test_sklearn_in_tmle2(self, cf):
        log = LogisticRegression(penalty='l1', C=1.0, random_state=201)
        lin = LinearRegression()
        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + cd40 + dvl0', custom_model=log)
        tmle.outcome_model('art + male + age0 + cd40 + dvl0', custom_model=lin)
        tmle.fit()
        # Dropping since Linux RNG does not match my OS (windows)
        # npt.assert_allclose(tmle.psi, -0.07507877527854623)
        # npt.assert_allclose(tmle.confint, [-0.15278930211034644, 0.002631751553253986], rtol=1e-5)
        # TODO Test now only checks no errors are thrown when adding sklearn


class TestAIPTW:

    @pytest.fixture
    def df(self):
        df = ze.load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = ze.spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = ze.spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.drop(columns=['cd4_wk45']).dropna()

    def test_drop_missing_data(self):
        df = ze.load_sample_data(False)
        aipw = AIPTW(df, exposure='art', outcome='dead')
        assert df.dropna().shape[0] == aipw.df.shape[0]

    def test_error_when_no_models_specified1(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        with pytest.raises(ValueError):
            aipw.fit()

    def test_error_when_no_models_specified2(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        with pytest.raises(ValueError):
            aipw.fit()

    def test_error_when_no_models_specified3(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        with pytest.raises(ValueError):
            aipw.fit()

    def test_match_rd(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_difference, -0.06857139263248598)

    def test_match_rr(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_ratio, 0.5844630051393351)
