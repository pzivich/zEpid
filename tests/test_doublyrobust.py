import pytest
import numpy.testing as npt

import zepid as ze
from zepid.causal.doublyrobust import TMLE


class TestTMLE:

    @pytest.fixture
    def df(self):
        df = ze.load_sample_data(False)
        df[['cd4_rs1', 'cd4_rs2']] = ze.spline(df, 'cd40', n_knots=3, term=2, restricted=True)
        df[['age_rs1', 'age_rs2']] = ze.spline(df, 'age0', n_knots=3, term=2, restricted=True)
        return df.dropna()

    def test_drop_missing_data(self, df):
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
        npt.assert_allclose(tmle.psi, r_rd)

    def test_match_r_tmle_rd_ci(self, df):
        r_ci = -0.1541104, -0.01470202
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.confint, r_ci, rtol=1e-5)

    def test_match_r_tmle_riskratio(self, df):
        r_rr = 0.5344266
        tmle = TMLE(df, exposure='art', outcome='dead', psi='risk_ratio')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.psi, r_rr)

    def test_match_r_tmle_rr_ci(self, df):
        r_ci = 0.2773936, 1.0296262
        tmle = TMLE(df, exposure='art', outcome='dead', psi='risk_ratio')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.confint, r_ci, rtol=1e-5)

    def test_match_r_tmle_oddsratio(self, df):
        r_rr = 0.4844782
        tmle = TMLE(df, exposure='art', outcome='dead', psi='odds_ratio')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.psi, r_rr)

    def test_match_r_tmle_or_ci(self, df):
        r_ci = 0.232966, 1.007525
        tmle = TMLE(df, exposure='art', outcome='dead', psi='odds_ratio')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.confint, r_ci, rtol=1e-5)

    def test_symmetric_bounds_on_gW(self, df):
        r_rd = -0.08203143
        r_ci = -0.1498092, -0.01425363
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=0.1, print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.psi, r_rd)
        npt.assert_allclose(tmle.confint, r_ci, rtol=1e-5)

    def test_asymmetric_bounds_on_gW(self, df):
        r_rd = -0.08433208
        r_ci = -0.1541296, -0.01453453
        tmle = TMLE(df, exposure='art', outcome='dead')
        tmle.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                            bound=[0.025, 0.9], print_results=False)
        tmle.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        tmle.fit()
        npt.assert_allclose(tmle.psi, r_rd)
        npt.assert_allclose(tmle.confint, r_ci, rtol=1e-5)

    # TODO add check or sklearn/superlearner
