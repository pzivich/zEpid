import pytest
import pandas as pd
import numpy.testing as npt

import zepid as ze
from zepid.causal.generalize import IPSW, GTransportFormula
from zepid.causal.ipw import IPTW


class TestIPSW:

    @pytest.fixture
    def df_r(self):
        df = ze.load_generalize_data(False)
        df['W_sq'] = df['W']**2
        return df

    @pytest.fixture
    def df_c(self):
        df = ze.load_generalize_data(True)
        df['W_sq'] = df['W']**2
        return df

    @pytest.fixture
    def df_iptw(self, df_c):
        dfs = df_c.loc[df_c['S'] == 1].copy()

        ipt = IPTW(dfs, treatment='A')
        ipt.regression_models('L', print_results=False)
        ipt.fit()
        dfs['iptw'] = ipt.Weight
        return pd.concat([dfs, df_c.loc[df_c['S'] == 0]], ignore_index=True, sort=False)

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

    @pytest.fixture
    def df_r(self):
        df = ze.load_generalize_data(False)
        df['W_sq'] = df['W']**2
        return df

    @pytest.fixture
    def df_c(self):
        df = ze.load_generalize_data(True)
        df['W_sq'] = df['W']**2
        return df

    def test_no_model_error(self, df_c):
        gtf = GTransportFormula(df_c, exposure='A', outcome='Y', selection='S', generalize=True)
        with pytest.raises(ValueError):
            gtf.fit()

    def test_generalize_stabilized(self, df_r):
        gtf = GTransportFormula(df_r, exposure='A', outcome='Y', selection='S', generalize=True)
        gtf.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        gtf.fit()
        npt.assert_allclose(gtf.risk_difference, 0.064038, atol=1e-5)
        npt.assert_allclose(gtf.risk_ratio, 1.203057, atol=1e-4)

    def test_transport_unstabilized(self, df_r):
        gtf = GTransportFormula(df_r, exposure='A', outcome='Y', selection='S', generalize=False)
        gtf.outcome_model('A + L + L:A + W_sq + W_sq:A + W_sq:A:L', print_results=False)
        gtf.fit()
        npt.assert_allclose(gtf.risk_difference, 0.058573, atol=1e-5)
        npt.assert_allclose(gtf.risk_ratio, 1.176615, atol=1e-4)
