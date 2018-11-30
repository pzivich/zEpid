import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import matplotlib.pyplot as plt

from zepid import RiskRatio, RiskDifference


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


# TODO add test that incorporates the simulated data
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

    def test_plot_returns_axes(self, data_set):
        rr = RiskRatio()
        rr.fit(data_set, exposure='exp', outcome='dis')
        assert isinstance(rr.plot(), type(plt.gca()))


class TestRiskDifference:

    def test_risk_difference_reference_equal_to_1(self, data_set):
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
        print(df.columns)
        npt.assert_allclose(df.loc[df.index == '1'][['SD(RD)']], sas_se)

    def test_plot_returns_axes(self, data_set):
        rd = RiskDifference()
        rd.fit(data_set, exposure='exp', outcome='dis')
        assert isinstance(rd.plot(), type(plt.gca()))
