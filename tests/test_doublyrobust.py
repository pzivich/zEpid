import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
from sklearn.linear_model import LogisticRegression, LinearRegression

import zepid as ze
from zepid.causal.doublyrobust import TMLE, AIPTW, StochasticTMLE


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

    @pytest.fixture
    def mf(self):
        df = ze.load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = ze.spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = ze.spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.drop(columns=['cd4_wk45'])

    @pytest.fixture
    def mcf(self):
        df = ze.load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = ze.spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = ze.spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.drop(columns=['dead'])

    def test_drop_missing_data(self):
        df = ze.load_sample_data(False)
        tmle = TMLE(df, exposure='art', outcome='dead')
        assert df.dropna(subset=['cd4_wk45']).shape[0] == tmle.df.shape[0]

    def test_error_when_no_models_specified(self, df):
        tmle = TMLE(df, exposure='art', outcome='dead')
        with pytest.raises(ValueError):
            tmle.fit()

        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        with pytest.raises(ValueError):
            tmle.fit()

        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        with pytest.raises(ValueError):
            tmle.fit()

    def test_continuous_processing(self):
        a_list = [0, 1, 1, 0, 1, 1, 0, 0]
        y_list = [1, -1, 5, 0, 0, 0, 10, -5]
        df = pd.DataFrame()
        df['A'] = a_list
        df['Y'] = y_list

        tmle = TMLE(df=df, exposure='A', outcome='Y', continuous_bound=0.0001)

        # Checking all flagged parts are correct
        assert tmle._continuous_outcome is True
        assert tmle._continuous_min == -5
        assert tmle._continuous_max == 10
        assert tmle._cb == 0.0001

        # Checking that TMLE bounding works as intended
        y_bound = [2 / 5, 4 / 15, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 0.9999, 0.0001]
        pdt.assert_series_equal(pd.Series(y_bound),
                                tmle.df['Y'],
                                check_dtype=False, check_names=False)

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
        assert tmle.average_treatment_effect_ci is None

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
        npt.assert_allclose(tmle.average_treatment_effect_ci, r_ci, rtol=1e-3)

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
        npt.assert_allclose(tmle.average_treatment_effect_ci, r_ci, rtol=1e-3)

    def test_match_r_continuous_poisson(self, cf):
        r_ate = 223.4648
        r_ci = 118.6276, 328.3019

        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False, continuous_distribution='poisson')
        tmle.fit()
        npt.assert_allclose(tmle.average_treatment_effect, r_ate, rtol=1e-3)
        npt.assert_allclose(tmle.average_treatment_effect_ci, r_ci, rtol=1e-3)

    def test_sklearn_in_tmle(self, df):
        log = LogisticRegression(C=1.0)
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + cd40 + dvl0', custom_model=log)
        tmle.outcome_model('art + male + age0 + cd40 + dvl0', custom_model=log)
        tmle.fit()

        # Testing RD match
        npt.assert_allclose(tmle.risk_difference, -0.091372098)
        npt.assert_allclose(tmle.risk_difference_ci, [-0.1595425678, -0.0232016282], rtol=1e-5)
        # Testing RR match
        npt.assert_allclose(tmle.risk_ratio, 0.4998833415)
        npt.assert_allclose(tmle.risk_ratio_ci, [0.2561223823, 0.9756404452], rtol=1e-5)
        # Testing OR match
        npt.assert_allclose(tmle.odds_ratio, 0.4496171689)
        npt.assert_allclose(tmle.odds_ratio_ci, [0.2139277755, 0.944971255], rtol=1e-5)

    def test_sklearn_in_tmle2(self, cf):
        log = LogisticRegression(C=1.0)
        lin = LinearRegression()
        tmle = TMLE(cf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + cd40 + dvl0', custom_model=log)
        tmle.outcome_model('art + male + age0 + cd40 + dvl0', custom_model=lin)
        tmle.fit()

        npt.assert_allclose(tmle.average_treatment_effect, 236.049719, rtol=1e-5)
        npt.assert_allclose(tmle.average_treatment_effect_ci, [135.999264, 336.100175], rtol=1e-5)

    def test_missing_binary_outcome(self, mf):
        r_rd = -0.08168098
        r_rd_ci = -0.15163818, -0.01172378
        r_rr = 0.5495056
        r_rr_ci = 0.2893677, 1.0435042
        r_or = 0.4996546
        r_or_ci = 0.2435979, 1.0248642

        tmle = TMLE(mf, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()

        npt.assert_allclose(tmle.risk_difference, r_rd)
        npt.assert_allclose(tmle.risk_difference_ci, r_rd_ci, rtol=1e-5)
        npt.assert_allclose(tmle.risk_ratio, r_rr)
        npt.assert_allclose(tmle.risk_ratio_ci, r_rr_ci, rtol=1e-5)
        npt.assert_allclose(tmle.odds_ratio, r_or)
        npt.assert_allclose(tmle.odds_ratio_ci, r_or_ci, rtol=1e-5)

    def test_no_missing_data(self, df):
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        with pytest.raises(ValueError):
            tmle.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                               print_results=False)

    def test_missing_continuous_outcome(self, mcf):
        r_ate = 211.8295
        r_ci = 107.7552, 315.9038

        tmle = TMLE(mcf, exposure='art', outcome='cd4_wk45')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.average_treatment_effect, r_ate, rtol=1e-3)
        npt.assert_allclose(tmle.average_treatment_effect_ci, r_ci, rtol=1e-3)

    def test_sklearn_in_tmle_missing(self, mf):
        log = LogisticRegression(C=1.0)
        tmle = TMLE(mf, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + cd40 + dvl0', custom_model=log, print_results=False)
        tmle.missing_model('male + age0 + cd40 + dvl0', custom_model=log, print_results=False)
        tmle.outcome_model('art + male + age0 + cd40 + dvl0', custom_model=log, print_results=False)
        tmle.fit()

        # Testing RD match
        npt.assert_allclose(tmle.risk_difference, -0.090086, rtol=1e-5)
        npt.assert_allclose(tmle.risk_difference_ci, [-0.160371, -0.019801], rtol=1e-4)
        # Testing RR match
        npt.assert_allclose(tmle.risk_ratio, 0.507997, rtol=1e-5)
        npt.assert_allclose(tmle.risk_ratio_ci, [0.256495, 1.006108], rtol=1e-4)
        # Testing OR match
        npt.assert_allclose(tmle.odds_ratio, 0.457541, rtol=1e-5)
        npt.assert_allclose(tmle.odds_ratio_ci, [0.213980, 0.978331], rtol=1e-4)


class TestStochasticTMLE:

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

    @pytest.fixture
    def simple_df(self):
        expected = pd.DataFrame([[1, 1, 1, 1, 1],
                                 [0, 0, 0, -1, 2],
                                 [0, 1, 0, 5, 1],
                                 [0, 0, 1, 0, 0],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 1, 0, 0],
                                 [0, 1, 0, 10, 1],
                                 [0, 0, 0, -5, 0],
                                 [1, 1, 0, -5, 2]],
                                columns=["W", "A", "Y", "C", "S"],
                                index=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        return expected

    def test_error_continuous_exp(self, df):
        with pytest.raises(ValueError):
            StochasticTMLE(df=df, exposure='cd40', outcome='dead')

    def test_error_fit(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        with pytest.raises(ValueError):
            stmle.fit(p=0.5)

        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=0.5)

        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=0.5)

    def test_error_p_oob(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=1.1)

        with pytest.raises(ValueError):
            stmle.fit(p=-0.1)

    def test_error_p_cond_len(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.fit(p=[0.1], conditional=["df['male']==1", "df['male']==0"])

        with pytest.raises(ValueError):
            stmle.fit(p=[0.1, 0.3], conditional=["df['male']==1"])

    def test_error_summary(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        stmle.outcome_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        with pytest.raises(ValueError):
            stmle.summary()

    def test_continuous_processing(self):
        a_list = [0, 1, 1, 0, 1, 1, 0, 0]
        y_list = [1, -1, 5, 0, 0, 0, 10, -5]
        df = pd.DataFrame()
        df['A'] = a_list
        df['Y'] = y_list

        stmle = StochasticTMLE(df=df, exposure='A', outcome='Y', continuous_bound=0.0001)

        # Checking all flagged parts are correct
        assert stmle._continuous_outcome is True
        assert stmle._continuous_min == -5
        assert stmle._continuous_max == 10
        assert stmle._cb == 0.0001

        # Checking that TMLE bounding works as intended
        y_bound = [2 / 5, 4 / 15, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 0.9999, 0.0001]
        pdt.assert_series_equal(pd.Series(y_bound),
                                stmle.df['Y'],
                                check_dtype=False, check_names=False)

    def test_marginal_vector_length_stoch(self, df):
        stmle = StochasticTMLE(df=df, exposure='art', outcome='dead')
        stmle.exposure_model('male')
        stmle.outcome_model('art + male + age0')
        stmle.fit(p=0.4, samples=7)
        assert len(stmle.marginals_vector) == 7

    def test_qmodel_params(self, simple_df):
        # Comparing to SAS logit model
        sas_params = [-1.0699, -0.9525, 1.5462]
        sas_preds = [0.3831332, 0.2554221, 0.1168668, 0.2554221, 0.6168668, 0.6168668, 0.1168668, 0.2554221, 0.3831332]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='Y')
        stmle.outcome_model('A + W')
        est_params = stmle._outcome_model.params
        est_preds = stmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_qmodel_params2(self, simple_df):
        # Comparing to SAS linear model
        sas_params = [0.3876, 0.3409, -0.2030, -0.0883]
        sas_preds = [0.437265, 0.210957, 0.6402345, 0.3876202, 0.0963188, 0.1846502, 0.6402345, 0.38762016, 0.34893314]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='C')
        stmle.outcome_model('A + W + S', continuous_distribution='normal')
        est_params = stmle._outcome_model.params
        est_preds = stmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_qmodel_params3(self, simple_df):
        # Comparing to SAS Poisson model
        sas_params = [-1.0478, 0.9371, -0.5321, -0.2733]
        sas_preds = [0.4000579, 0.2030253, 0.6811115, 0.3507092, 0.1567304, 0.20599265, 0.6811115, 0.3507092, 0.3043857]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='C')
        stmle.outcome_model('A + W + S', continuous_distribution='Poisson')
        est_params = stmle._outcome_model.params
        est_preds = stmle._Qinit_

        npt.assert_allclose(sas_params, est_params, atol=1e-4)
        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    def test_gmodel_params(self, simple_df):
        # Comparing to SAS Poisson model
        sas_preds = [2.0, 1.6666666667, 2.5, 1.6666666667, 2, 2, 2.5, 1.6666666667, 2]

        stmle = StochasticTMLE(df=simple_df, exposure='A', outcome='C')
        stmle.exposure_model('W')
        est_preds = 1 / stmle._denominator_

        npt.assert_allclose(sas_preds, est_preds, atol=1e-6)

    # TODO check bounding

    # TODO compare to R in several versions


class TestAIPTW:

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

    @pytest.fixture
    def dat(self):
        df = pd.DataFrame()
        df['L'] = [1]*10000 + [0]*40000
        df['A'] = [1]*2000 + [0]*8000 + [1]*30000 + [0]*10000
        df['Y'] = [1]*500 + [0]*1500 + [1]*4000 + [0]*4000 + [1]*10000 + [0]*20000 + [1]*4000 + [0]*6000
        return df

    def test_drop_missing_data(self):
        df = ze.load_sample_data(False)
        aipw = AIPTW(df, exposure='art', outcome='dead')
        assert df.dropna(subset=['cd4_wk45']).shape[0] == aipw.df.shape[0]

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
        npt.assert_allclose(aipw.risk_difference, -0.0848510605)

    def test_match_rr(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_ratio, 0.5319812235)

    def test_double_robustness(self, dat):
        aipw = AIPTW(dat, exposure='A', outcome='Y')
        aipw.exposure_model('L', print_results=False)
        aipw.outcome_model('L + A + A:L', print_results=False)
        aipw.fit()
        both_correct_rd = aipw.risk_difference
        both_correct_rr = aipw.risk_ratio

        aipw = AIPTW(dat, exposure='A', outcome='Y')
        aipw.exposure_model('L', print_results=False)
        aipw.outcome_model('A + L', print_results=False)
        aipw.fit()
        wrong_y_rd = aipw.risk_difference
        wrong_y_rr = aipw.risk_ratio

        # Testing
        npt.assert_allclose(both_correct_rd, wrong_y_rd)
        npt.assert_allclose(both_correct_rr, wrong_y_rr)

        aipw = AIPTW(dat, exposure='A', outcome='Y')
        aipw.exposure_model('1', print_results=False)
        aipw.outcome_model('A + L + A:L', print_results=False)
        aipw.fit()
        wrong_a_rd = aipw.risk_difference
        wrong_a_rr = aipw.risk_ratio

        # Testing
        npt.assert_allclose(both_correct_rd, wrong_a_rd)
        npt.assert_allclose(both_correct_rr, wrong_a_rr)

    def test_weighted_rd(self, df):
        df['weights'] = 2
        aipw = AIPTW(df, exposure='art', outcome='dead', weights='weights')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_difference, -0.0848510605)

    def test_weighted_rr(self, df):
        df['weights'] = 2
        aipw = AIPTW(df, exposure='art', outcome='dead', weights='weights')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_ratio, 0.5319812235)

    def test_continuous_outcomes(self, cf):
        aipw = AIPTW(cf, exposure='art', outcome='cd4_wk45')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.average_treatment_effect, 225.13767, rtol=1e-3)
        npt.assert_allclose(aipw.average_treatment_effect_ci, [118.64677, 331.62858], rtol=1e-3)

    def test_poisson_outcomes(self, cf):
        aipw = AIPTW(cf, exposure='art', outcome='cd4_wk45')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           continuous_distribution='poisson', print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.average_treatment_effect, 225.13767, rtol=1e-3)
        npt.assert_allclose(aipw.average_treatment_effect_ci, [118.64677, 331.62858], rtol=1e-3)

    def test_weighted_continuous_outcomes(self, cf):
        cf['weights'] = 2
        aipw = AIPTW(cf, exposure='art', outcome='cd4_wk45', weights='weights')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.average_treatment_effect, 225.13767, rtol=1e-3)
        assert aipw.average_treatment_effect_ci is None

    def test_bounds(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=0.1, print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()

        npt.assert_allclose(aipw.risk_difference, -0.0819506956)
        npt.assert_allclose(aipw.risk_difference_ci, (-0.1498808287, -0.0140205625))

    def test_bounds2(self, df):
        aipw = AIPTW(df, exposure='art', outcome='dead')
        aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=[0.2, 0.9], print_results=False)
        aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        aipw.fit()

        npt.assert_allclose(aipw.risk_difference, -0.0700780176)
        npt.assert_allclose(aipw.risk_difference_ci, (-0.1277925885, -0.0123634468))
