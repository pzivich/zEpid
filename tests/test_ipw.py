import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family, links

from zepid import load_sample_data, spline
from zepid.causal.ipw import IPTW, IPMW, IPCW


@pytest.fixture
def sdata():
    df = load_sample_data(False)
    df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
    df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)
    return df


class TestIPTW:

    @pytest.fixture
    def data(self):
        df = pd.DataFrame()
        df['A'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        df['L'] = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
        return df

    def test_probability_calc(self, data):
        ipt = IPTW(data, treatment='A', stabilized=True)
        ipt.regression_models(model_denominator='L', print_results=False)
        ipt.fit()
        pd = ipt.ProbabilityDenominator
        pn = ipt.ProbabilityNumerator
        npt.assert_allclose(pn, [0.5]*10)
        npt.assert_allclose(pd, [1/3, 1/3, 0.75, 0.75, 0.75, 1/3, 1/3, 1/3, 1/3, 0.75])

    def test_unstabilized_weights(self, data):
        ipt = IPTW(data, treatment='A', stabilized=False)
        ipt.regression_models(model_denominator='L', print_results=False)
        ipt.fit()
        npt.assert_allclose(ipt.Weight, [3, 3, 4/3, 4/3, 4/3, 1.5, 1.5, 1.5, 1.5, 4])

    def test_stabilized_weights(self, data):
        ipt = IPTW(data, treatment='A', stabilized=True)
        ipt.regression_models(model_denominator='L', print_results=False)
        ipt.fit()
        npt.assert_allclose(ipt.Weight, [1.5, 1.5, 2/3, 2/3, 2/3, 3/4, 3/4, 3/4, 3/4, 2])

    def test_positivity_calculator(self, data):
        ipt = IPTW(data, treatment='A', stabilized=True)
        ipt.regression_models(model_denominator='L', print_results=False)
        ipt.fit()
        ipt.positivity()
        npt.assert_allclose(ipt._pos_avg, 1)
        npt.assert_allclose(ipt._pos_sd, 0.456435, rtol=1e-5)
        npt.assert_allclose(ipt._pos_min, 2/3)
        npt.assert_allclose(ipt._pos_max, 2)

    def test_unstabilized_positivity_warning(self, data):
        ipt = IPTW(data, treatment='A', stabilized=False)
        ipt.regression_models(model_denominator='L', print_results=False)
        ipt.fit()
        with pytest.warns(UserWarning):
            ipt.positivity()

    def test_match_sas_unstabilized(self, sdata):
        sas_w_sum = 1038.051
        sas_rd = -0.081519085
        sas_rd_ci = -0.156199938, -0.006838231
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', stabilized=False)
        ipt.regression_models(model)
        ipt.fit()
        sdata['iptw'] = ipt.Weight
        npt.assert_allclose(np.sum(sdata.dropna()['iptw']), sas_w_sum, rtol=1e-4)

        # Estimating GEE
        ind = sm.cov_struct.Independence()
        f = sm.families.family.Binomial(sm.families.links.identity)
        linrisk = smf.gee('dead ~ art', sdata['id'], sdata, cov_struct=ind, family=f, weights=sdata['iptw']).fit()
        npt.assert_allclose(linrisk.params[1], sas_rd, rtol=1e-5)
        npt.assert_allclose((linrisk.conf_int()[0][1], linrisk.conf_int()[1][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_stabilized(self, sdata):
        sas_w_sum = 515.6177
        sas_rd = -0.081519085
        sas_rd_ci = -0.156199938, -0.006838231
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', stabilized=True)
        ipt.regression_models(model)
        ipt.fit()
        sdata['iptw'] = ipt.Weight
        npt.assert_allclose(np.sum(sdata.dropna()['iptw']), sas_w_sum, rtol=1e-4)

        # Estimating GEE
        ind = sm.cov_struct.Independence()
        f = sm.families.family.Binomial(sm.families.links.identity)
        linrisk = smf.gee('dead ~ art', sdata['id'], sdata, cov_struct=ind, family=f, weights=sdata['iptw']).fit()
        npt.assert_allclose(linrisk.params[1], sas_rd, rtol=1e-5)
        npt.assert_allclose((linrisk.conf_int()[0][1], linrisk.conf_int()[1][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_e(self, sdata):
        sas_w_sum = 151.2335
        sas_rd = -0.090875986
        sas_rd_ci = -0.180169444, -0.001582527
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', standardize='exposed', stabilized=False)
        ipt.regression_models(model)
        ipt.fit()
        sdata['iptw'] = ipt.Weight
        npt.assert_allclose(np.sum(sdata.dropna()['iptw']), sas_w_sum, rtol=1e-4)

        # Estimating GEE
        ind = sm.cov_struct.Independence()
        f = sm.families.family.Binomial(sm.families.links.identity)
        linrisk = smf.gee('dead ~ art', sdata['id'], sdata, cov_struct=ind, family=f, weights=sdata['iptw']).fit()
        npt.assert_allclose(linrisk.params[1], sas_rd, rtol=1e-5)
        npt.assert_allclose((linrisk.conf_int()[0][1], linrisk.conf_int()[1][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_u(self, sdata):
        sas_w_sum = 886.8178
        sas_rd = -0.080048197
        sas_rd_ci = -0.153567335, -0.006529058
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', standardize='unexposed', stabilized=False)
        ipt.regression_models(model)
        ipt.fit()
        sdata['iptw'] = ipt.Weight
        npt.assert_allclose(np.sum(sdata.dropna()['iptw']), sas_w_sum, rtol=1e-4)

        # Estimating GEE
        ind = sm.cov_struct.Independence()
        f = sm.families.family.Binomial(sm.families.links.identity)
        linrisk = smf.gee('dead ~ art', sdata['id'], sdata, cov_struct=ind, family=f, weights=sdata['iptw']).fit()
        npt.assert_allclose(linrisk.params[1], sas_rd, rtol=1e-5)
        npt.assert_allclose((linrisk.conf_int()[0][1], linrisk.conf_int()[1][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_e_stabilized(self, sdata):
        sas_rd = -0.090875986
        sas_rd_ci = -0.180169444, -0.001582527
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', standardize='exposed', stabilized=True)
        ipt.regression_models(model)
        ipt.fit()
        sdata['iptw'] = ipt.Weight

        # Estimating GEE
        ind = sm.cov_struct.Independence()
        f = sm.families.family.Binomial(sm.families.links.identity)
        linrisk = smf.gee('dead ~ art', sdata['id'], sdata, cov_struct=ind, family=f, weights=sdata['iptw']).fit()
        npt.assert_allclose(linrisk.params[1], sas_rd, rtol=1e-5)
        npt.assert_allclose((linrisk.conf_int()[0][1], linrisk.conf_int()[1][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_u_stabilized(self, sdata):
        sas_rd = -0.080048197
        sas_rd_ci = -0.153567335, -0.006529058
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', standardize='unexposed', stabilized=True)
        ipt.regression_models(model)
        ipt.fit()
        sdata['iptw'] = ipt.Weight

        # Estimating GEE
        ind = sm.cov_struct.Independence()
        f = sm.families.family.Binomial(sm.families.links.identity)
        linrisk = smf.gee('dead ~ art', sdata['id'], sdata, cov_struct=ind, family=f, weights=sdata['iptw']).fit()
        npt.assert_allclose(linrisk.params[1], sas_rd, rtol=1e-5)
        npt.assert_allclose((linrisk.conf_int()[0][1], linrisk.conf_int()[1][1]), sas_rd_ci, rtol=1e-4)

    # TODO add standardized differences check (after adding the plot functionality)


class TestIPMW:

    @pytest.fixture
    def mdata(self):
        df = pd.DataFrame()
        df['A'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        df['L'] = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
        df['M'] = [1, np.nan, 1, 0, np.nan, 0, 1, np.nan, np.nan, 1]
        return df

    def test_error_for_non_nan(self, mdata):
        with pytest.raises(ValueError):
            IPMW(mdata, missing_variable='L', stabilized=True)

    # TODO add tests after update from Docs_overhaul


class TestIPCW:

    @pytest.fixture
    def edata(self):
        df = pd.DataFrame()
        df['id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        df['A'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        df['Y'] = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
        df['t'] = [1, 1, 5, 2, 3, 1, np.nan, 1, 1, 5]
        return df

    @pytest.fixture
    def edata2(self):
        df = pd.DataFrame()
        df['id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        df['A'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        df['Y'] = [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
        df['t'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return df

    @pytest.fixture
    def flat_data(self):
        df = pd.DataFrame()
        df['id'] = [1, 2, 3, 4, 5]
        df['t'] = [1, 2, 3, 3, 2]
        df['Y'] = [0, 0, 1, 0, 1]
        df['A'] = [1, 1, 0, 0, 0]
        return df

    @pytest.fixture
    def flat_data2(self):
        df = pd.DataFrame()
        df['id'] = [1, 2, 3, 4, 5]
        df['enter'] = [0, 0, 2, 0, 1]
        df['t'] = [1, 2, 3, 3, 2]
        df['Y'] = [0, 0, 1, 0, 1]
        df['A'] = [1, 1, 0, 0, 0]
        return df

    def test_error_missing_times(self, edata):
        with pytest.raises(ValueError):
            IPCW(edata, idvar='id', time='t', event='Y')

    def test_error_time_is_only_one(self, edata2):
        with pytest.raises(ValueError):
            IPCW(edata2, idvar='id', time='t', event='Y')

    def test_data_conversion_warning(self, flat_data):
        with pytest.warns(UserWarning):
            IPCW(flat_data, idvar='id', time='t', event='Y', flat_df=True)

    def test_data_conversion(self, flat_data):
        ipc = IPCW(flat_data, idvar='id', time='t', event='Y', flat_df=True)
        expected_data = pd.DataFrame.from_records([{'id': 1, 't_enter': 0, 't_out': 1, 'A': 1, 'Y': 0.0,
                                                    '__uncensored__': 0},
                                                   {'id': 2, 't_enter': 0, 't_out': 1, 'A': 1, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 2, 't_enter': 1, 't_out': 2, 'A': 1, 'Y': 0.0,
                                                    '__uncensored__': 0},
                                                   {'id': 3, 't_enter': 0, 't_out': 1, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 3, 't_enter': 1, 't_out': 2, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 3, 't_enter': 2, 't_out': 3, 'A': 0, 'Y': 1.0,
                                                    '__uncensored__': 1},
                                                   {'id': 4, 't_enter': 0, 't_out': 1, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 4, 't_enter': 1, 't_out': 2, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 4, 't_enter': 2, 't_out': 3, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 5, 't_enter': 0, 't_out': 1, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 5, 't_enter': 1, 't_out': 2, 'A': 0, 'Y': 1.0,
                                                    '__uncensored__': 1}]
                                                  )
        pdt.assert_frame_equal(ipc.df[['id', 'A', 'Y', 't_enter', 't_out', '__uncensored__']],
                               expected_data[['id', 'A', 'Y', 't_enter', 't_out', '__uncensored__']],
                               check_dtype=False, check_index_type=False, check_like=True)

    def test_data_conversion_late_entry(self, flat_data2):
        ipc = IPCW(flat_data2, idvar='id', time='t', event='Y', enter='enter', flat_df=True)
        expected_data = pd.DataFrame.from_records([{'id': 1, 't_enter': 0, 't_out': 1, 'A': 1, 'Y': 0.0,
                                                    '__uncensored__': 0},
                                                   {'id': 2, 't_enter': 0, 't_out': 1, 'A': 1, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 2, 't_enter': 1, 't_out': 2, 'A': 1, 'Y': 0.0,
                                                    '__uncensored__': 0},
                                                   {'id': 3, 't_enter': 2, 't_out': 3, 'A': 0, 'Y': 1.0,
                                                    '__uncensored__': 1},
                                                   {'id': 4, 't_enter': 0, 't_out': 1, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 4, 't_enter': 1, 't_out': 2, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 4, 't_enter': 2, 't_out': 3, 'A': 0, 'Y': 0.0,
                                                    '__uncensored__': 1},
                                                   {'id': 5, 't_enter': 1, 't_out': 2, 'A': 0, 'Y': 1.0,
                                                    '__uncensored__': 1}]
                                                  )
        pdt.assert_frame_equal(ipc.df[['id', 'A', 'Y', 't_enter', 't_out', '__uncensored__']],
                               expected_data[['id', 'A', 'Y', 't_enter', 't_out', '__uncensored__']],
                               check_dtype=False, check_index_type=False, check_like=True)

    def test_match_sas_weights(self):
        sas_w_mean = 0.9993069
        sas_w_max = 1.7980410
        sas_w_min = 0.8986452
        df = load_sample_data(timevary=True)
        df['cd40_q'] = df['cd40'] ** 2
        df['cd40_c'] = df['cd40'] ** 3
        df['cd4_q'] = df['cd4'] ** 2
        df['cd4_c'] = df['cd4'] ** 3
        df['enter_q'] = df['enter'] ** 2
        df['enter_c'] = df['enter'] ** 3
        df['age0_q'] = df['age0'] ** 2
        df['age0_c'] = df['age0'] ** 3
        ipc = IPCW(df, idvar='id', time='enter', event='dead')
        cmodeln = 'enter + enter_q + enter_c'
        cmodeld = '''enter + enter_q + enter_c + male + age0 + age0_q + age0_c + dvl0 + cd40 +
                     cd40_q + cd40_c + dvl + cd4 + cd4_q + cd4_c'''
        ipc.regression_models(model_denominator=cmodeld, model_numerator=cmodeln)
        ipc.fit()
        cw = ipc.Weight
        npt.assert_allclose(np.mean(cw), sas_w_mean)
        npt.assert_allclose(np.max(cw), sas_w_max)
        npt.assert_allclose(np.min(cw), sas_w_min)
