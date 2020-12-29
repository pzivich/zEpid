import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family, links
from sklearn.linear_model import LogisticRegression

from zepid import load_sample_data, load_monotone_missing_data, spline
from zepid.causal.ipw import IPTW, IPMW, IPCW, StochasticIPTW


@pytest.fixture
def sdata():
    df = load_sample_data(False)
    df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
    df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)
    return df.drop(columns=['cd4_wk45'])

@pytest.fixture
def cdata():
    df = load_sample_data(False)
    df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
    df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)
    return df.drop(columns=['dead'])


class TestIPTW:

    @pytest.fixture
    def data(self):
        df = pd.DataFrame()
        df['A'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        df['Y'] = [1, 0, 0, 0, 1, 1, 1, 0, 0, 1]
        df['L'] = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
        return df

    def test_unstabilized_weights(self, data):
        ipt = IPTW(data, treatment='A', outcome='Y')
        ipt.treatment_model(model_denominator='L', stabilized=False, print_results=False)
        ipt.marginal_structural_model('A')
        ipt.fit()
        npt.assert_allclose(ipt.iptw, [3, 3, 4/3, 4/3, 4/3, 1.5, 1.5, 1.5, 1.5, 4])

    def test_stabilized_weights(self, data):
        ipt = IPTW(data, treatment='A', outcome='Y')
        ipt.treatment_model(model_denominator='L', print_results=False)
        ipt.marginal_structural_model('A')
        ipt.fit()
        npt.assert_allclose(ipt.iptw, [1.5, 1.5, 2/3, 2/3, 2/3, 3/4, 3/4, 3/4, 3/4, 2])

    def test_positivity_calculator(self, data):
        ipt = IPTW(data, treatment='A', outcome='Y')
        ipt.treatment_model(model_denominator='L', print_results=False)
        ipt.marginal_structural_model('A')
        ipt.fit()
        ipt.positivity()
        npt.assert_allclose(ipt._pos_avg, 1)
        npt.assert_allclose(ipt._pos_sd, 0.456435, rtol=1e-5)
        npt.assert_allclose(ipt._pos_min, 2/3)
        npt.assert_allclose(ipt._pos_max, 2)

    def test_match_sas_unstabilized(self, sdata):
        sas_w_sum = 1086.250
        sas_rd = -0.081519085
        sas_rd_ci = -0.156199938, -0.006838231
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead')
        ipt.treatment_model(model_denominator=model, stabilized=False, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(np.sum(ipt.iptw), sas_w_sum, rtol=1e-4)
        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_stabilized(self, sdata):
        sas_w_sum = 546.0858419
        sas_rd = -0.081519085
        sas_rd_ci = -0.156199938, -0.006838231
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead')
        ipt.treatment_model(model_denominator=model, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(np.sum(ipt.iptw), sas_w_sum, rtol=1e-4)
        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_e(self, sdata):
        sas_w_sum = 158.288404
        sas_rd = -0.090875986
        sas_rd_ci = -0.180169444, -0.001582527
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead', standardize='exposed')
        ipt.treatment_model(model_denominator=model, stabilized=False, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(np.sum(ipt.iptw), sas_w_sum, rtol=1e-4)
        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_u(self, sdata):
        sas_w_sum = 927.9618034
        sas_rd = -0.080048197
        sas_rd_ci = -0.153567335, -0.006529058
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead', standardize='unexposed')
        ipt.treatment_model(model_denominator=model, stabilized=False, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(np.sum(ipt.iptw), sas_w_sum, rtol=1e-4)
        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_e_stabilized(self, sdata):
        sas_rd = -0.090875986
        sas_rd_ci = -0.180169444, -0.001582527
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead', standardize='exposed')
        ipt.treatment_model(model_denominator=model, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci, rtol=1e-4)

    def test_match_sas_smr_u_stabilized(self, sdata):
        sas_rd = -0.080048197
        sas_rd_ci = -0.153567335, -0.006529058
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead', standardize='unexposed')
        ipt.treatment_model(model_denominator=model, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci, rtol=1e-4)

    def test_standardized_differences(self, sdata):
        ipt = IPTW(sdata, treatment='art', outcome='dead')
        ipt.treatment_model(model_denominator='male + age0 + cd40 + dvl0', print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()
        smd = ipt.standardized_mean_differences()

        npt.assert_allclose(np.array(smd['smd_u']),
                            np.array([-0.015684, 0.022311, -0.4867, -0.015729]),
                            rtol=1e-4)  # for unweighted
        # TODO find R package to test these weighted SMD's
        npt.assert_allclose(np.array(smd['smd_w']),
                            np.array([-0.097789, -0.012395, -0.018591, 0.050719]),
                            rtol=1e-4)  # for weighted

    def test_match_r_stddiff(self):
        # Simulated data for variable detection and standardized differences
        df = pd.DataFrame()
        df['y'] = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
        df['treat'] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        df['bin'] = [0, 1, 0, np.nan, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        df['con'] = [0.1, 0.0, 1.0, 1.1, 2.2, 1.3, 0.1, 0.5, 0.9, 0.5, 0.3, 0.2, 0.7, 0.9, 1.4]
        df['dis'] = [0, 1, 3, 2, 1, 0, 0, 0, 0, 0, 1, 3, 2, 2, 1]
        df['cat'] = [1, 2, 3, 1, 1, 2, 3, 1, 3, 2, 1, 2, 3, 2, 1]

        ipt = IPTW(df, treatment='treat', outcome='y')
        ipt.treatment_model(model_denominator='bin + con + dis + C(cat)', print_results=False)
        ipt.marginal_structural_model('treat')
        ipt.fit()
        smd = ipt.standardized_mean_differences()

        npt.assert_allclose(np.array(smd['smd_u']),
                            np.array([0.342997, 0.0, 0.06668, -0.513553]),
                            rtol=1e-4)  # for unweighted
        # TODO need to find an R package or something that calculates weighted SMD
        # currently compares to my own calculations
        npt.assert_allclose(np.array(smd['smd_w']),
                            np.array([0.206072, -0.148404,  0.035683,  0.085775]),
                            rtol=1e-4)  # for weighted

    def test_match_sas_gbound(self, sdata):
        sas_rd = -0.075562451
        sas_rd_ci = -0.151482078, 0.000357175
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead')
        ipt.treatment_model(model_denominator=model, print_results=False, bound=0.1)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci,
                            atol=1e-4, rtol=1e-4)

    def test_match_sas_gbound2(self, sdata):
        sas_rd = -0.050924398
        sas_rd_ci = -0.133182382, 0.031333585
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead')
        ipt.treatment_model(model_denominator=model, print_results=False, bound=[0.2, 0.9])
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci,
                            atol=1e-4, rtol=1e-4)

    def test_match_sas_gbound3(self, sdata):
        sas_rd = -0.045129870
        sas_rd_ci = -0.128184899, 0.037925158
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        ipt = IPTW(sdata, treatment='art', outcome='dead')
        ipt.treatment_model(model_denominator=model, print_results=False, bound=0.5)
        ipt.marginal_structural_model('art')
        ipt.fit()

        npt.assert_allclose(ipt.risk_difference['RD'][1], sas_rd, rtol=1e-5)
        npt.assert_allclose((ipt.risk_difference['95%LCL'][1], ipt.risk_difference['95%UCL'][1]), sas_rd_ci,
                            atol=1e-4, rtol=1e-4)

    def test_iptw_w_censor(self, sdata):
        iptw = IPTW(sdata, treatment='art', outcome='dead')
        iptw.treatment_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        iptw.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        iptw.marginal_structural_model('art')
        iptw.fit()

        npt.assert_allclose(iptw.risk_difference['RD'][1], -0.08092, rtol=1e-5)
        npt.assert_allclose((iptw.risk_difference['95%LCL'][1], iptw.risk_difference['95%UCL'][1]),
                            (-0.15641, -0.00543), atol=1e-4, rtol=1e-4)

    def test_iptw_w_censor2(self, cdata):
        iptw = IPTW(cdata, treatment='art', outcome='cd4_wk45')
        iptw.treatment_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', print_results=False)
        iptw.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                           print_results=False)
        iptw.marginal_structural_model('art')
        iptw.fit()

        npt.assert_allclose(iptw.average_treatment_effect['ATE'][1], 205.11238, rtol=1e-5)
        npt.assert_allclose((iptw.average_treatment_effect['95%LCL'][1], iptw.average_treatment_effect['95%UCL'][1]),
                            (96.88535, 313.33941), atol=1e-4, rtol=1e-4)


class TestStochasticIPTW:

    def test_error_no_model(self, sdata):
        sipw = StochasticIPTW(sdata.dropna(), treatment='art', outcome='dead')
        with pytest.raises(ValueError):
            sipw.fit(p=0.8)

    def test_error_p_oob(self, sdata):
        sipw = StochasticIPTW(sdata.dropna(), treatment='art', outcome='dead')
        with pytest.raises(ValueError):
            sipw.fit(p=1.8)

        with pytest.raises(ValueError):
            sipw.fit(p=-0.8)

    def test_error_summary(self, sdata):
        sipw = StochasticIPTW(sdata.dropna(), treatment='art', outcome='dead')
        with pytest.raises(ValueError):
            sipw.summary()

    def test_error_conditional(self, sdata):
        sipw = StochasticIPTW(sdata.dropna(), treatment='art', outcome='dead')
        sipw.treatment_model(model='male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                             print_results=False)
        with pytest.raises(ValueError):
            sipw.fit(p=[0.8, 0.1, 0.1], conditional=["df['male']==1", "df['male']==0"])

        with pytest.raises(ValueError):
            sipw.fit(p=[0.8], conditional=["df['male']==1", "df['male']==0"])

    def test_drop_missing(self, cdata):
        sipw = StochasticIPTW(cdata, treatment='art', outcome='cd4_wk45')
        assert sipw.df.shape[0] == cdata.dropna().shape[0]

    def test_uncond_treatment(self, sdata):
        r_pred = 0.1165162207

        sipw = StochasticIPTW(sdata.dropna(), treatment='art', outcome='dead')
        sipw.treatment_model(model='male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                             print_results=False)
        sipw.fit(p=0.8)
        r = sipw.marginal_outcome

        npt.assert_allclose(r_pred, r, atol=1e-7)

    def test_cond_treatment(self, sdata):
        r_pred = 0.117340102
        sipw = StochasticIPTW(sdata.dropna(), treatment='art', outcome='dead')
        sipw.treatment_model(model='male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                             print_results=False)
        sipw.fit(p=[0.75, 0.90], conditional=["df['male']==1", "df['male']==0"])
        r = sipw.marginal_outcome

        npt.assert_allclose(r_pred, r, atol=1e-7)

    def test_uncond_treatment_continuous(self, cdata):
        r_pred = 1249.4477406809

        sipw = StochasticIPTW(cdata.dropna(), treatment='art', outcome='cd4_wk45')
        sipw.treatment_model(model='male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                             print_results=False)
        sipw.fit(p=0.8)
        r = sipw.marginal_outcome

        npt.assert_allclose(r_pred, r, atol=1e-5)

    def test_cond_treatment_continuous(self, cdata):
        r_pred = 1246.4285662061
        sipw = StochasticIPTW(cdata.dropna(), treatment='art', outcome='cd4_wk45')
        sipw.treatment_model(model='male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0',
                             print_results=False)
        sipw.fit(p=[0.75, 0.90], conditional=["df['male']==1", "df['male']==0"])
        r = sipw.marginal_outcome

        npt.assert_allclose(r_pred, r, atol=1e-5)

    def test_match_iptw(self, sdata):
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        sdata = sdata.dropna().copy()

        # Estimating Marginal Structural Model
        ipt = IPTW(sdata, treatment='art', outcome='dead')
        ipt.treatment_model(model_denominator=model, stabilized=False, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        # Estimating 'Stochastic Treatment'
        sipw = StochasticIPTW(sdata, treatment='art', outcome='dead')
        sipw.treatment_model(model=model, print_results=False)
        sipw.fit(p=1.0)
        r_all = sipw.marginal_outcome
        sipw.fit(p=0.0)
        r_non = sipw.marginal_outcome

        npt.assert_allclose(ipt.risk_difference['RD'][1], r_all - r_non, atol=1e-7)

    def test_match_iptw_continuous(self, cdata):
        model = 'male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
        cdata = cdata.dropna().copy()

        # Estimating Marginal Structural Model
        ipt = IPTW(cdata, treatment='art', outcome='cd4_wk45')
        ipt.treatment_model(model_denominator=model, stabilized=False, print_results=False)
        ipt.marginal_structural_model('art')
        ipt.fit()

        # Estimating 'Stochastic Treatment'
        sipw = StochasticIPTW(cdata, treatment='art', outcome='cd4_wk45')
        sipw.treatment_model(model=model, print_results=False)
        sipw.fit(p=1.0)
        r_all = sipw.marginal_outcome
        sipw.fit(p=0.0)
        r_non = sipw.marginal_outcome

        npt.assert_allclose(ipt.average_treatment_effect['ATE'][1], r_all - r_non, atol=1e-4)

    # TODO add test for weights


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

    def test_missing_count(self, mdata):
        ipm = IPMW(mdata, missing_variable='M', stabilized=True)
        ipm.regression_models(model_denominator='A')
        assert 6 == np.sum(ipm.df['_observed_indicator_'])
        assert 4 == np.sum(1 - ipm.df['_observed_indicator_'])

    def test_missing_count2(self):
        df = load_sample_data(False)
        ipm = IPMW(df, missing_variable='dead', stabilized=True)
        ipm.regression_models(model_denominator='art')
        assert 517 == np.sum(ipm.df['_observed_indicator_'])
        assert 30 == np.sum(1 - ipm.df['_observed_indicator_'])

    def test_error_numerator_with_unstabilized(self):
        df = load_sample_data(False)
        ipm = IPMW(df, missing_variable='dead', stabilized=False)
        with pytest.raises(ValueError):
            ipm.regression_models(model_denominator='male + age0 + dvl0 + cd40', model_numerator='male')

    def test_unstabilized_weights(self):
        df = load_sample_data(False)
        ipm = IPMW(df, missing_variable='dead', stabilized=False)
        ipm.regression_models(model_denominator='male + age0 + dvl0 + cd40')
        ipm.fit()
        npt.assert_allclose(np.mean(ipm.Weight), 1.0579602715)
        npt.assert_allclose(np.std(ipm.Weight, ddof=1), 0.021019729152)

    def test_stabilized_weights(self):
        df = load_sample_data(False)
        ipm = IPMW(df, missing_variable='dead', stabilized=True)
        ipm.regression_models(model_denominator='male + age0 + dvl0 + cd40')
        ipm.fit()
        npt.assert_allclose(np.mean(ipm.Weight), 0.99993685627)
        npt.assert_allclose(np.std(ipm.Weight, ddof=1), 0.019866910369)

    def test_error_too_many_model(self):
        df = load_sample_data(False)
        ipm = IPMW(df, missing_variable='dead')
        with pytest.raises(ValueError):
            ipm.regression_models(model_denominator=['male + age0', 'male + age0 + dvl0'])

    # testing monotone missing data features
    def test_error_for_non_nan2(self):
        df = pd.DataFrame()
        df['a'] = [0, 0, 1]
        df['b'] = [0, 0, np.nan]
        with pytest.raises(ValueError):
            IPMW(df, missing_variable=['a', 'b'], stabilized=True)

    def test_nonmonotone_detection(self):
        df = pd.DataFrame()
        df['a'] = [0, 0, np.nan]
        df['b'] = [np.nan, 0, 1]
        ipm = IPMW(df, missing_variable=['a', 'b'])
        with pytest.raises(ValueError):
            ipm.regression_models(model_denominator=['b', 'a'])

    def test_check_overall_uniform(self):
        df = pd.DataFrame()
        df['a'] = [0, 0, 1, np.nan]
        df['b'] = [1, 0, 0, np.nan]
        df['c'] = [1, 0, 0, np.nan]
        df['d'] = [1, np.nan, np.nan, np.nan]
        ipm = IPMW(df, missing_variable=['a', 'b', 'c'])

        # Should be uniform
        assert ipm._check_overall_uniform(df, miss_vars=['a', 'b', 'c'])[1]

        # Not uniform
        assert not ipm._check_overall_uniform(df, miss_vars=['a', 'b', 'c', 'd'])[1]

    def test_check_uniform(self):
        df = pd.DataFrame()
        df['a'] = [0, 0, 1, np.nan]
        df['b'] = [1, 0, 0, np.nan]
        df['c'] = [1, np.nan, np.nan, np.nan]
        ipm = IPMW(df, missing_variable=['a', 'b', 'c'])

        # Should be uniform
        assert ipm._check_uniform(df, miss1='a', miss2='b')

        # Not uniform
        assert not ipm._check_uniform(df, miss1='a', miss2='c')

    def test_single_model_does_not_break(self):
        df = load_monotone_missing_data()
        ipm = IPMW(df, missing_variable=['B', 'C'], stabilized=False, monotone=True)
        ipm.regression_models(model_denominator='L')
        ipm.fit()
        x = ipm.Weight

    def test_monotone_example(self):
        # TODO find R or SAS to test against
        df = load_monotone_missing_data()
        ipm = IPMW(df, missing_variable=['B', 'C'], stabilized=False, monotone=True)
        ipm.regression_models(model_denominator=['L + A', 'L + B'])
        ipm.fit()
        df['w'] = ipm.Weight
        dfs = df.dropna(subset=['w'])
        npt.assert_allclose(np.average(dfs['B'], weights=dfs['w']), 0.41877344861340654)
        npt.assert_allclose(np.average(dfs['C'], weights=dfs['w']), 0.5637116735464095)


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
        with pytest.raises(ValueError):
            IPCW(flat_data2, idvar='id', time='t', event='Y', enter='enter', flat_df=True)

    def test_data_late_entry_error(self):
        df = pd.DataFrame()
        df['id'] = [1, 1, 2, 2, 3, 4, 4, 5, 5]
        df['t'] = [1, 2, 1, 2, 2, 1, 2, 1, 2]
        df['Y'] = [0, 0, 0, 1, 0, 0, 1, 0, 1]

        with pytest.raises(ValueError):
            IPCW(df, idvar='id', time='t', event='Y', flat_df=False)

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
