import pytest
import pandas as pd
import numpy.testing as npt

import zepid as ze
from zepid.causal.generalize import IPSW, GTransportFormula, AIPSW
from zepid.causal.ipw import IPTW


@pytest.fixture
def df_r():
    df = ze.load_generalize_data(False)
    df['W_sq'] = df['W'] ** 2
    return df


@pytest.fixture
def df_c():
    df = ze.load_generalize_data(True)
    df['W_sq'] = df['W'] ** 2
    return df


@pytest.fixture
def df_iptw(df_c):
    dfs = df_c.loc[df_c['S'] == 1].copy()

    ipt = IPTW(dfs, treatment='A', outcome='Y')
    ipt.treatment_model('L', stabilized=True, print_results=False)
    dfs['iptw'] = ipt.iptw
    return pd.concat([dfs, df_c.loc[df_c['S'] == 0]], ignore_index=True, sort=False)


class TestIPSW:

    def test_stabilize_error(self, df_c):
        ipsw = IPSW(df_c, exposure='A', outcome='Y', selection='S', stabilized=False)
        with pytest.raises(ValueError):
            ipsw.regression_models('L + W_sq', model_numerator='W', print_results=False)

    def test_no_model_error(self, df_c):
        ipsw = IPSW(df_c, exposure='A', outcome='Y', selection='S', generalize=True)
        with pytest.raises(ValueError):
            ipsw.fit()

    def test_generalize_unstabilized(self, df_r):
        ipsw = IPSW(df_r, exposure='A', outcome='Y', selection='S', stabilized=False)
        ipsw.regression_models('L + W_sq', print_results=False)
        ipsw.fit()
        npt.assert_allclose(ipsw.risk_difference, 0.046809, atol=1e-5)
        npt.assert_allclose(ipsw.risk_ratio, 1.13905, atol=1e-4)

    def test_generalize_stabilized(self, df_r):
        ipsw = IPSW(df_r, exposure='A', outcome='Y', selection='S', stabilized=True)
        ipsw.regression_models('L + W_sq', print_results=False)
        ipsw.fit()
        npt.assert_allclose(ipsw.risk_difference, 0.046809, atol=1e-5)
        npt.assert_allclose(ipsw.risk_ratio, 1.13905, atol=1e-4)

    def test_transport_unstabilized(self, df_r):
        ipsw = IPSW(df_r, exposure='A', outcome='Y', selection='S', stabilized=False, generalize=False)
        ipsw.regression_models('L + W_sq', print_results=False)
        ipsw.fit()
        npt.assert_allclose(ipsw.risk_difference, 0.034896, atol=1e-5)
        npt.assert_allclose(ipsw.risk_ratio, 1.097139, atol=1e-4)

    def test_transport_stabilized(self, df_r):
        ipsw = IPSW(df_r, exposure='A', outcome='Y', selection='S', stabilized=True, generalize=False)
        ipsw.regression_models('L + W_sq', print_results=False)
        ipsw.fit()
        npt.assert_allclose(ipsw.risk_difference, 0.034896, atol=1e-5)
        npt.assert_allclose(ipsw.risk_ratio, 1.097139, atol=1e-4)

    def test_generalize_iptw(self, df_iptw):
        ipsw = IPSW(df_iptw, exposure='A', outcome='Y', selection='S', generalize=True, weights='iptw')
        ipsw.regression_models('L + W + W_sq', print_results=False)
        ipsw.fit()
        npt.assert_allclose(ipsw.risk_difference, 0.055034, atol=1e-5)
        npt.assert_allclose(ipsw.risk_ratio, 1.167213, atol=1e-4)

    def test_transport_iptw(self, df_iptw):
        ipsw = IPSW(df_iptw, exposure='A', outcome='Y', selection='S', generalize=False, weights='iptw')
        ipsw.regression_models('L + W + W_sq', print_results=False)
        ipsw.fit()
        npt.assert_allclose(ipsw.risk_difference, 0.047296, atol=1e-5)
        npt.assert_allclose(ipsw.risk_ratio, 1.1372, atol=1e-4)


class TestGTransport:

    def test_no_model_error(self, df_c):
        gtf = GTransportFormula(df_c, exposure='A', outcome='Y', selection='S', generalize=True)
        with pytest.raises(ValueError):
            gtf.fit()

    def test_generalize(self, df_r):
        gtf = GTransportFormula(df_r, exposure='A', outcome='Y', selection='S', generalize=True)
        gtf.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        gtf.fit()
        npt.assert_allclose(gtf.risk_difference, 0.064038, atol=1e-5)
        npt.assert_allclose(gtf.risk_ratio, 1.203057, atol=1e-4)

    def test_transport(self, df_r):
        gtf = GTransportFormula(df_r, exposure='A', outcome='Y', selection='S', generalize=False)
        gtf.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        gtf.fit()
        npt.assert_allclose(gtf.risk_difference, 0.058573, atol=1e-5)
        npt.assert_allclose(gtf.risk_ratio, 1.176615, atol=1e-4)

    def test_generalize_conf(self, df_c):
        gtf = GTransportFormula(df_c, exposure='A', outcome='Y', selection='S', generalize=True)
        gtf.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        gtf.fit()
        npt.assert_allclose(gtf.risk_difference, 0.048949, atol=1e-5)
        npt.assert_allclose(gtf.risk_ratio, 1.149556, atol=1e-4)

    def test_transport_conf(self, df_c):
        gtf = GTransportFormula(df_c, exposure='A', outcome='Y', selection='S', generalize=False)
        gtf.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        gtf.fit()
        npt.assert_allclose(gtf.risk_difference, 0.042574, atol=1e-5)
        npt.assert_allclose(gtf.risk_ratio, 1.124257, atol=1e-4)


class TestAIPSW:

    def test_no_model_error(self, df_c):
        aipw = AIPSW(df_c, exposure='A', outcome='Y', selection='S', generalize=True)
        with pytest.raises(ValueError):
            aipw.fit()

        aipw.weight_model('L', print_results=False)
        with pytest.raises(ValueError):
            aipw.fit()

        aipw = AIPSW(df_c, exposure='A', outcome='Y', selection='S', generalize=True)
        aipw.outcome_model('A + L')
        with pytest.raises(ValueError):
            aipw.fit()

    def test_generalize(self, df_r):
        aipw = AIPSW(df_r, exposure='A', outcome='Y', selection='S', generalize=True)
        aipw.weight_model('L + W_sq', print_results=False)
        aipw.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_difference, 0.061382, atol=1e-5)
        npt.assert_allclose(aipw.risk_ratio, 1.193161, atol=1e-4)

    def test_transport(self, df_r):
        aipw = AIPSW(df_r, exposure='A', outcome='Y', selection='S', generalize=False)
        aipw.weight_model('L + W_sq', print_results=False)
        aipw.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_difference, 0.05479, atol=1e-5)
        npt.assert_allclose(aipw.risk_ratio, 1.16352, atol=1e-4)

    def test_generalize_conf(self, df_iptw):
        aipw = AIPSW(df_iptw, exposure='A', outcome='Y', selection='S', generalize=True, weights='iptw')
        aipw.weight_model('L + W_sq', print_results=False)
        aipw.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_difference, 0.048129, atol=1e-5)
        npt.assert_allclose(aipw.risk_ratio, 1.146787, atol=1e-4)

    def test_transport_conf(self, df_iptw):
        aipw = AIPSW(df_iptw, exposure='A', outcome='Y', selection='S', generalize=False, weights='iptw')
        aipw.weight_model('L + W_sq', print_results=False)
        aipw.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        aipw.fit()
        npt.assert_allclose(aipw.risk_difference, 0.041407, atol=1e-5)
        npt.assert_allclose(aipw.risk_ratio, 1.120556, atol=1e-4)
