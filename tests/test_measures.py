import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import matplotlib.pyplot as plt

import zepid as ze
from zepid import RiskRatio, RiskDifference, OddsRatio, NNT, IncidenceRateRatio, IncidenceRateDifference


@pytest.fixture
def data_set():
    df = pd.DataFrame()
    df['exp'] = [1 for i in range(50)] + [0 for i in range(50)]
    df['dis'] = [1 for i in range(25)] + [0 for i in range(25)] + [1 for i in range(25)] + [0 for i in range(25)]
    return df


@pytest.fixture
def multi_exposures():
    df = pd.DataFrame()
    df['exp'] = [1 for i in range(50)] + [0 for i in range(50)] + [2 for i in range(50)]
    df['dis'] = ([1 for i in range(25)] + [0 for i in range(25)] + [1 for i in range(25)] + [0 for i in range(25)] +
                 [1 for i in range(25)] + [0 for i in range(25)])
    return df


@pytest.fixture
def time_data():
    df = pd.DataFrame()
    df['exp'] = [1 for i in range(50)] + [0 for i in range(50)]
    df['dis'] = [1 for i in range(6)] + [0 for i in range(44)] + [1 for i in range(14)] + [0 for i in range(50-14)]
    df['t'] = [2 for i in range(50)] + [8 for i in range(50)]
    return df


class TestRiskRatio:

    def test_risk_ratio_reference_equal_to_1(self, data_set):
        rr = RiskRatio()
        rr.fit(data_set, exposure='exp', outcome='dis')
        assert rr.risk_ratio[0] == 1

    def test_risk_ratio_equal_to_1(self, data_set):
        rr = RiskRatio()
        rr.fit(data_set, exposure='exp', outcome='dis')
        assert rr.risk_ratio[1] == 1

    def test_multiple_exposures(self, multi_exposures):
        rr = RiskRatio()
        rr.fit(multi_exposures, exposure='exp', outcome='dis')
        assert rr.results.shape[0] == 3
        assert list(rr.results.index) == ['Ref:0', '1', '2']

    def test_match_sas_ci(self, data_set):
        sas_ci = 0.6757, 1.4799
        rr = RiskRatio()
        rr.fit(data_set, exposure='exp', outcome='dis')
        df = rr.results
        npt.assert_allclose(np.round(df.loc[df.index == '1'][['RR_LCL', 'RR_UCL']], 4), [sas_ci])

    def test_match_sas_sampledata(self):
        sas_rd = 0.742118331
        sas_se = 0.312612740
        sas_ci = 0.402139480, 1.369523870
        df = ze.load_sample_data(False)
        rr = RiskRatio()
        rr.fit(df, exposure='art', outcome='dead')
        npt.assert_allclose(rr.risk_ratio[1], sas_rd, rtol=1e-5)
        rf = rr.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['RR_LCL', 'RR_UCL']], [sas_ci], rtol=1e-5)
        npt.assert_allclose(rf.loc[rf.index == '1'][['SD(RR)']], sas_se, rtol=1e-5)

    def test_plot_returns_axes(self, data_set):
        rr = RiskRatio()
        rr.fit(data_set, exposure='exp', outcome='dis')
        assert isinstance(rr.plot(), type(plt.gca()))


class TestRiskDifference:

    def test_risk_difference_reference_equal_to_0(self, data_set):
        rd = RiskDifference()
        rd.fit(data_set, exposure='exp', outcome='dis')
        assert rd.risk_difference[0] == 0

    def test_risk_difference_equal_to_0(self, data_set):
        rd = RiskDifference()
        rd.fit(data_set, exposure='exp', outcome='dis')
        assert rd.risk_difference[1] == 0

    def test_multiple_exposures(self, multi_exposures):
        rd = RiskDifference()
        rd.fit(multi_exposures, exposure='exp', outcome='dis')
        assert rd.results.shape[0] == 3
        assert list(rd.results.index) == ['Ref:0', '1', '2']

    def test_match_sas_ci(self, data_set):
        sas_ci = -0.195996398, 0.195996398
        rd = RiskDifference()
        rd.fit(data_set, exposure='exp', outcome='dis')
        df = rd.results
        npt.assert_allclose(df.loc[df.index == '1'][['RD_LCL', 'RD_UCL']], [sas_ci])

    def test_match_sas_se(self, data_set):
        sas_se = 0.1
        rd = RiskDifference()
        rd.fit(data_set, exposure='exp', outcome='dis')
        df = rd.results
        npt.assert_allclose(df.loc[df.index == '1'][['SD(RD)']], sas_se)

    def test_match_sas_sampledata(self):
        sas_rr = -0.045129870
        sas_se = 0.042375793
        sas_ci = -0.128184899, 0.037925158
        df = ze.load_sample_data(False)
        rd = RiskDifference()
        rd.fit(df, exposure='art', outcome='dead')
        npt.assert_allclose(rd.risk_difference[1], sas_rr)
        rf = rd.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['RD_LCL', 'RD_UCL']], [sas_ci])
        npt.assert_allclose(rf.loc[rf.index == '1'][['SD(RD)']], sas_se)

    def test_plot_returns_axes(self, data_set):
        rd = RiskDifference()
        rd.fit(data_set, exposure='exp', outcome='dis')
        assert isinstance(rd.plot(), type(plt.gca()))


class TestOddsRatio:

    def test_odds_ratio_reference_equal_to_1(self, data_set):
        ord = OddsRatio()
        ord.fit(data_set, exposure='exp', outcome='dis')
        assert ord.odds_ratio[0] == 1

    def test_odds_ratio_equal_to_1(self, data_set):
        ord = OddsRatio()
        ord.fit(data_set, exposure='exp', outcome='dis')
        assert ord.odds_ratio[1] == 1

    def test_multiple_exposures(self, multi_exposures):
        ord = OddsRatio()
        ord.fit(multi_exposures, exposure='exp', outcome='dis')
        assert ord.results.shape[0] == 3
        assert list(ord.results.index) == ['Ref:0', '1', '2']

    def test_match_sas_ci(self, data_set):
        sas_ci = 0.4566, 2.1902
        ord = OddsRatio()
        ord.fit(data_set, exposure='exp', outcome='dis')
        df = ord.results
        npt.assert_allclose(df.loc[df.index == '1'][['OR_LCL', 'OR_UCL']], [sas_ci], rtol=1e-4)

    def test_match_sas_sampledata(self):
        sas_or = 0.7036
        sas_se = 0.361479191
        sas_ci = 0.3465, 1.4290
        df = ze.load_sample_data(False)
        ord = OddsRatio()
        ord.fit(df, exposure='art', outcome='dead')
        npt.assert_allclose(ord.odds_ratio[1], sas_or, rtol=1e-4)
        rf = ord.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['OR_LCL', 'OR_UCL']], [sas_ci], rtol=1e-3)
        npt.assert_allclose(rf.loc[rf.index == '1'][['SD(OR)']], sas_se, rtol=1e-4)

    def test_plot_returns_axes(self, data_set):
        ord = OddsRatio()
        ord.fit(data_set, exposure='exp', outcome='dis')
        assert isinstance(ord.plot(), type(plt.gca()))


class TestNNT:

    def test_return_infinity(self, data_set):
        nnt = NNT()
        nnt.fit(data_set, exposure='exp', outcome='dis')
        assert np.isinf(nnt.number_needed_to_treat[1])

    def test_match_inverse_of_risk_difference(self):
        df = ze.load_sample_data(False)

        rd = RiskDifference()
        rd.fit(df, exposure='art', outcome='dead')

        nnt = NNT()
        nnt.fit(df, exposure='art', outcome='dead')

        npt.assert_allclose(nnt.number_needed_to_treat[1], 1/rd.risk_difference[1])
        rf = rd.results
        nf = nnt.results
        npt.assert_allclose(nf.loc[nf.index == '1'][['NNT_LCL', 'NNT_UCL']],
                            1 / rf.loc[rf.index == '1'][['RD_LCL', 'RD_UCL']])
        npt.assert_allclose(nf.loc[nf.index == '1'][['SD(RD)']], rf.loc[rf.index == '1'][['SD(RD)']])

    def test_multiple_exposures(self, multi_exposures):
        nnt = NNT()
        nnt.fit(multi_exposures, exposure='exp', outcome='dis')
        assert nnt.results.shape[0] == 3
        assert list(nnt.results.index) == ['Ref:0', '1', '2']


class TestIncidenceRateRatio:

    def test_incidence_rate_ratio_reference_equal_to_1(self, time_data):
        irr = IncidenceRateRatio()
        irr.fit(time_data, exposure='exp', outcome='dis', time='t')
        assert irr.incidence_rate_ratio[0] == 1

    def test_incidence_rate_ratio_equal_to_expected(self, time_data):
        sas_irr = 1.714285714
        sas_se = 0.487950036
        sas_ci = 0.658778447, 4.460946657
        irr = IncidenceRateRatio()
        irr.fit(time_data, exposure='exp', outcome='dis', time='t')
        npt.assert_allclose(irr.incidence_rate_ratio[1], sas_irr, rtol=1e-4)
        rf = irr.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['IRR_LCL', 'IRR_UCL']], [sas_ci], rtol=1e-4)
        npt.assert_allclose(rf.loc[rf.index == '1'][['SD(IRR)']], sas_se, rtol=1e-4)

    def test_multiple_exposures(self):
        df = pd.DataFrame()
        df['exp'] = [1 for i in range(50)] + [0 for i in range(50)] + [2 for i in range(50)]
        df['dis'] = ([1 for i in range(25)] + [0 for i in range(25)] + [1 for i in range(25)] + [0 for i in range(25)] +
                     [1 for i in range(25)] + [0 for i in range(25)])
        df['t'] = 2
        irr = IncidenceRateRatio()
        irr.fit(df, exposure='exp', outcome='dis', time='t')
        assert irr.results.shape[0] == 3
        assert list(irr.results.index) == ['Ref:0', '1', '2']

    def test_match_sas_sampledata(self):
        sas_irr = 0.740062626
        sas_se = 0.336135409
        sas_ci = 0.382956543, 1.430169300
        df = ze.load_sample_data(False)
        irr = IncidenceRateRatio()
        irr.fit(df, exposure='art', outcome='dead', time='t')
        npt.assert_allclose(irr.incidence_rate_ratio[1], sas_irr, rtol=1e-5)
        rf = irr.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['IRR_LCL', 'IRR_UCL']], [sas_ci], rtol=1e-5)
        npt.assert_allclose(rf.loc[rf.index == '1'][['SD(IRR)']], sas_se, rtol=1e-5)

    def test_plot_returns_axes(self, time_data):
        irr = IncidenceRateRatio()
        irr.fit(time_data, exposure='exp', outcome='dis', time='t')
        assert isinstance(irr.plot(), type(plt.gca()))


class TestIncidenceRateDifference:

    def test_incidence_rate_difference_reference_equal_to_0(self, time_data):
        ird = IncidenceRateDifference()
        ird.fit(time_data, exposure='exp', outcome='dis', time='t')
        assert ird.incidence_rate_difference[0] == 0

    def test_multiple_exposures(self):
        df = pd.DataFrame()
        df['exp'] = [1 for i in range(50)] + [0 for i in range(50)] + [2 for i in range(50)]
        df['dis'] = ([1 for i in range(25)] + [0 for i in range(25)] + [1 for i in range(25)] + [0 for i in range(25)] +
                     [1 for i in range(25)] + [0 for i in range(25)])
        df['t'] = 2
        ird = IncidenceRateDifference()
        ird.fit(df, exposure='exp', outcome='dis', time='t')
        assert ird.results.shape[0] == 3
        assert list(ird.results.index) == ['Ref:0', '1', '2']

    def test_match_openepi_sampledata(self):
        oe_irr = -0.0008614
        oe_ci = -0.002552, 0.0008291
        df = ze.load_sample_data(False)
        ird = IncidenceRateDifference()
        ird.fit(df, exposure='art', outcome='dead', time='t')
        npt.assert_allclose(ird.incidence_rate_difference[1], oe_irr, rtol=1e-4)
        rf = ird.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['IRD_LCL', 'IRD_UCL']], [oe_ci], rtol=1e-3)

    def test_plot_returns_axes(self, time_data):
        irr = IncidenceRateRatio()
        irr.fit(time_data, exposure='exp', outcome='dis', time='t')
        assert isinstance(irr.plot(), type(plt.gca()))

# TODO test for sensitivity, specificity, splines, table 1 generator(?)
