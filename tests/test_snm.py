import pytest
import numpy as np
import numpy.testing as npt

from zepid import load_sample_data
from zepid.causal.ipw import IPMW

from zepid.causal.snm import GEstimationSNM


class TestGEstimationSNM:

    @pytest.fixture
    def data_c(self):
        df = load_sample_data(False).drop(columns=['dead']).dropna()
        df['age_sq'] = df['age0']**2
        df['age_cu'] = df['age0']**3
        df['cd4_sq'] = df['cd40']**2
        df['cd4_cu'] = df['cd40']**3
        return df

    @pytest.fixture
    def data_b(self):
        df = load_sample_data(False).drop(columns=['cd4_wk45']).dropna()
        df['age_sq'] = df['age0']**2
        df['age_cu'] = df['age0']**3
        df['cd4_sq'] = df['cd40']**2
        df['cd4_cu'] = df['cd40']**3
        return df

    def test_continuous_exp_error(self, data_c):
        with pytest.raises(ValueError):
            GEstimationSNM(data_c, exposure='cd40', outcome='cd4_wk45')

    def test_error_when_no_models_specified1(self, data_c):
        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        with pytest.raises(ValueError):
            snm.fit()

    def test_error_when_no_models_specified2(self, data_c):
        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        with pytest.raises(ValueError):
            snm.fit()

    def test_error_when_no_models_specified3(self, data_c):
        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        snm.structural_nested_model('art')
        with pytest.raises(ValueError):
            snm.fit()

    def test_invalid_solver(self, data_c):
        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art')
        with pytest.raises(ValueError):
            snm.fit(solver='grid_search')

    def test_solvermatch_1param_continuous(self, data_c):
        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art')
        snm.fit(solver='closed')
        closed = snm.psi

        snm.fit(solver='search', starting_value=200)
        search = snm.psi

        npt.assert_allclose(closed, search)

    def test_solvermatch_2param_continuous(self, data_c):
        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art + art:male')
        snm.fit(solver='closed')
        closed = snm.psi

        snm.fit(solver='search', starting_value=[200, -50])
        search = snm.psi

        npt.assert_allclose(closed, search)

    def test_solvermatch_3param_continuous(self, data_c):
        data_c['cd4_wk45'] /= 100

        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art + art:male')
        snm.fit(solver='closed')
        closed = snm.psi

        snm.fit(solver='search')
        search = snm.psi

        npt.assert_allclose(closed, search)

    def test_solvermatch_1param_binary(self, data_b):
        snm = GEstimationSNM(data_b, exposure='art', outcome='dead')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art')
        snm.fit(solver='closed')
        closed = snm.psi

        snm.fit(solver='search')
        search = snm.psi

        npt.assert_allclose(closed, search)

    def test_solvermatch_2param_binary(self, data_b):
        snm = GEstimationSNM(data_b, exposure='art', outcome='dead')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art + art:male')
        snm.fit(solver='closed')
        closed = snm.psi

        snm.fit(solver='search')
        search = snm.psi

        npt.assert_allclose(closed, search, rtol=1e-5)

    def test_solvermatch_3param_binary(self, data_b):
        snm = GEstimationSNM(data_b, exposure='art', outcome='dead')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art + art:male')
        snm.fit(solver='closed')
        closed = snm.psi

        snm.fit(solver='search')
        search = snm.psi

        npt.assert_allclose(closed, search, rtol=1e-5)

    def test_weighted_model(self):
        df = load_sample_data(False).drop(columns=['dead'])
        df['age_sq'] = df['age0'] ** 2
        df['age_cu'] = df['age0'] ** 3
        df['cd4_sq'] = df['cd40'] ** 2
        df['cd4_cu'] = df['cd40'] ** 3

        ipmw = IPMW(df, missing_variable='cd4_wk45')
        ipmw.regression_models('art + male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0',
                               print_results=False)
        ipmw.fit()
        df['ipcw'] = ipmw.Weight

        snm = GEstimationSNM(df.dropna(), exposure='art', outcome='cd4_wk45', weights='ipcw')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art')
        snm.fit()

        npt.assert_allclose(snm.psi, [244.379181], atol=1e-5)

        snm = GEstimationSNM(df.dropna(), exposure='art', outcome='cd4_wk45', weights='ipcw')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art')
        snm.fit(solver='search')

        npt.assert_allclose(snm.psi, [244.379181], atol=1e-5)

    def test_missing_model(self):
        df = load_sample_data(False).drop(columns=['dead'])
        df['age_sq'] = df['age0'] ** 2
        df['age_cu'] = df['age0'] ** 3
        df['cd4_sq'] = df['cd40'] ** 2
        df['cd4_cu'] = df['cd40'] ** 3

        snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.missing_model('art + male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0',
                          stabilized=False, print_results=False)
        snm.structural_nested_model('art')
        snm.fit()

        npt.assert_allclose(snm.psi, [244.379181], atol=1e-5)

        snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.missing_model('art + male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0',
                          stabilized=False, print_results=False)
        snm.structural_nested_model('art')
        snm.fit(solver='search')

        npt.assert_allclose(snm.psi, [244.379181], atol=1e-5)

    def test_missing_w_weights(self):
        df = load_sample_data(False).drop(columns=['dead'])
        df['age_sq'] = df['age0'] ** 2
        df['age_cu'] = df['age0'] ** 3
        df['cd4_sq'] = df['cd40'] ** 2
        df['cd4_cu'] = df['cd40'] ** 3
        df['weight'] = 2

        snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45', weights='weight')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.missing_model('art + male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0',
                          stabilized=False, print_results=False)
        snm.structural_nested_model('art')
        snm.fit()

        npt.assert_allclose(snm.psi, [244.379181], atol=1e-5)

        snm = GEstimationSNM(df, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.missing_model('art + male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0',
                          stabilized=False, print_results=False)
        snm.structural_nested_model('art')
        snm.fit(solver='search')

        npt.assert_allclose(snm.psi, [244.379181], atol=1e-5)

    def test_match_r_continuous(self, data_c):
        # Comparing to R's DTRreg
        r_1param = [251.4032]
        r_2param = [289.4268, -47.8836]
        r_3param = [436.0397, -81.0284, -0.4397]
        r_5param = [974.4983, -50.3237, -285.2829, -0.3569, -8.5426]

        # One-parameter SNM
        snm = GEstimationSNM(data_c, exposure='art', outcome='cd4_wk45')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_1param, rtol=1e-4)

        # Two-parameter SNM
        snm.structural_nested_model('art + art:male')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_2param, rtol=1e-4)

        # Three-parameter SNM
        snm.structural_nested_model('art + art:male + art:cd40')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_3param, rtol=1e-4)

        # Five-parameter SNM
        snm.structural_nested_model('art + art:male + art:dvl0 + art:cd40 + art:age0')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_5param, rtol=1e-4)

    def test_match_r_binary(self, data_b):
        # Comparing to R's DTRreg
        r_1param = [-0.0895]
        r_2param = [-0.1760, 0.1084]
        r_3param = [-0.2316, 0.1245, 0.000165]
        r_5param = [0.3022, 0.1627, -0.1374, 0.0002, -0.0120]

        # One-parameter SNM
        snm = GEstimationSNM(data_b, exposure='art', outcome='dead')
        snm.exposure_model('male + age0 + age_sq + age_cu + cd40 + cd4_sq + cd4_cu + dvl0', print_results=False)
        snm.structural_nested_model('art')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_1param, rtol=1e-3)

        # Two-parameter SNM
        snm.structural_nested_model('art + art:male')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_2param, rtol=1e-3)

        # Three-parameter SNM
        snm.structural_nested_model('art + art:male + art:cd40')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_3param, atol=1e-3)

        # Five-parameter SNM
        snm.structural_nested_model('art + art:male + art:dvl0 + art:cd40 + art:age0')
        snm.fit(solver='closed')
        closed = snm.psi
        npt.assert_allclose(closed, r_5param, atol=1e-3)
