import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
from scipy.stats import logistic

import zepid as ze
from zepid import (RiskRatio, RiskDifference, OddsRatio, NNT, IncidenceRateRatio, IncidenceRateDifference,
                   Sensitivity, Specificity, Diagnostics, interaction_contrast, interaction_contrast_ratio, spline,
                   table1_generator)
from zepid.calc import sensitivity, specificity


@pytest.fixture
def data_set():
    df = pd.DataFrame()
    df['exp'] = [1]*50 + [0]*50
    df['dis'] = [1]*25 + [0]*25 + [1]*25 + [0]*25
    return df


@pytest.fixture
def multi_exposures():
    df = pd.DataFrame()
    df['exp'] = [1]*50 + [0]*50 + [2]*50
    df['dis'] = [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*25
    return df


@pytest.fixture
def time_data():
    df = pd.DataFrame()
    df['exp'] = [1]*50 + [0]*50
    df['dis'] = [1]*6 + [0]*44 + [1]*14 + [0]*36
    df['t'] = [2]*50 + [8]*50
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

    def test_frechet_bounds(self):
        df = ze.load_sample_data(False)
        rd = RiskDifference()
        rd.fit(df, exposure='art', outcome='dead')
        npt.assert_allclose(rd.results['UpperBound'][1] - rd.results['LowerBound'][1], 1.0000)

    def test_frechet_bounds2(self, multi_exposures):
        rd = RiskDifference()
        rd.fit(multi_exposures, exposure='exp', outcome='dis')
        npt.assert_allclose(rd.results['UpperBound'][1:] - rd.results['LowerBound'][1:], [1.0000, 1.0000])


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
        df['exp'] = [1]*50 + [0]*50 + [2]*50
        df['dis'] = [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*25
        df['t'] = 2
        irr = IncidenceRateRatio()
        irr.fit(df, exposure='exp', outcome='dis', time='t')
        assert irr.results.shape[0] == 3
        assert list(irr.results.index) == ['Ref:0', '1', '2']

    def test_match_sas_sampledata(self):
        sas_irr = 0.753956
        sas_se = 0.336135409
        sas_ci = 0.390146, 1.457017
        df = ze.load_sample_data(False)
        irr = IncidenceRateRatio()
        irr.fit(df, exposure='art', outcome='dead', time='t')
        npt.assert_allclose(irr.incidence_rate_ratio[1], sas_irr, rtol=1e-5)
        rf = irr.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['IRR_LCL', 'IRR_UCL']], [sas_ci], rtol=1e-5)
        npt.assert_allclose(rf.loc[rf.index == '1'][['SD(IRR)']], sas_se, rtol=1e-5)


class TestIncidenceRateDifference:

    def test_incidence_rate_difference_reference_equal_to_0(self, time_data):
        ird = IncidenceRateDifference()
        ird.fit(time_data, exposure='exp', outcome='dis', time='t')
        assert ird.incidence_rate_difference[0] == 0

    def test_multiple_exposures(self):
        df = pd.DataFrame()
        df['exp'] = [1]*50 + [0]*50 + [2]*50
        df['dis'] = [1]*25 + [0]*25 + [1]*25 + [0]*25 + [1]*25 + [0]*25
        df['t'] = 2
        ird = IncidenceRateDifference()
        ird.fit(df, exposure='exp', outcome='dis', time='t')
        assert ird.results.shape[0] == 3
        assert list(ird.results.index) == ['Ref:0', '1', '2']

    def test_match_openepi_sampledata(self):
        oe_irr = -0.001055
        oe_ci = -0.003275, 0.001166
        df = ze.load_sample_data(False)
        ird = IncidenceRateDifference()
        ird.fit(df, exposure='art', outcome='dead', time='t')
        npt.assert_allclose(ird.incidence_rate_difference[1], oe_irr, atol=1e-5)
        rf = ird.results
        npt.assert_allclose(rf.loc[rf.index == '1'][['IRD_LCL', 'IRD_UCL']], [oe_ci], atol=1e-5)


class TestDiagnostics:

    @pytest.fixture
    def test_data(self):
        df = pd.DataFrame()
        df['test'] = [1]*50 + [0]*50
        df['case'] = [1]*40 + [0]*10 + [1]*15 + [0]*35
        return df

    def test_sensitivity_same_as_calc(self, test_data):
        se = Sensitivity()
        se.fit(test_data, test='test', disease='case')
        sens = sensitivity(40, 50)
        npt.assert_allclose(se.sensitivity, sens[0])

    def test_specificity_same_as_calc(self, test_data):
        sp = Specificity()
        sp.fit(test_data, test='test', disease='case')
        spec = specificity(15, 50)
        npt.assert_allclose(sp.specificity, spec[0])

    def test_diagnostic_same_as_compositions(self, test_data):
        se = Sensitivity()
        se.fit(test_data, test='test', disease='case')

        sp = Specificity()
        sp.fit(test_data, test='test', disease='case')

        diag = Diagnostics()
        diag.fit(test_data, test='test', disease='case')

        npt.assert_allclose(diag.sensitivity.sensitivity, se.sensitivity)
        npt.assert_allclose(diag.specificity.specificity, sp.specificity)

    def test_match_sas_sensitivity_ci(self, test_data):
        sas_ci = [0.689127694, 0.910872306]
        diag = Diagnostics()
        diag.fit(test_data, test='test', disease='case')
        npt.assert_allclose(diag.sensitivity.results[['Se_LCL', 'Se_UCL']], [sas_ci])

    def test_match_sas_specificity_ci(self, test_data):
        sas_ci = [0.572979816, 0.827020184]
        diag = Diagnostics()
        diag.fit(test_data, test='test', disease='case')
        npt.assert_allclose(diag.specificity.results[['Sp_LCL', 'Sp_UCL']], [sas_ci])


class TestInteractionContrasts:

    @pytest.fixture
    def data_ic(self, n=10000):
        df = pd.DataFrame()
        np.random.seed(111)
        df['exp'] = np.random.binomial(1, 0.5, size=n)
        df['mod'] = np.random.binomial(1, 0.5, size=n)
        df['y'] = np.random.binomial(1, size=n, p=logistic.cdf(0.1 + 0.2*df['exp'] + 0.3*df['mod'] -
                                                               0.4*df['mod']*df['exp']))
        # Note: IC will not be equal to ICR
        return df

    def test_interaction_contrast(self, data_ic):
        ic = interaction_contrast(data_ic, exposure='exp', outcome='y', modifier='mod', print_results=False)
        npt.assert_allclose(np.round(ic[0], 4), -0.1009)

    def test_interaction_contrast_ci(self, data_ic):
        ic = interaction_contrast(data_ic, exposure='exp', outcome='y', modifier='mod', print_results=False)
        assert ic[1] < -0.1009 < ic[2]

    def test_interaction_contrast_ratio(self, data_ic):
        icr = interaction_contrast_ratio(data_ic, exposure='exp', outcome='y', modifier='mod', print_results=False)
        npt.assert_allclose(np.round(icr[0], 4), -0.4908)

    def test_interaction_contrast_ratio_delta_ci(self, data_ic):
        icr = interaction_contrast_ratio(data_ic, exposure='exp', outcome='y', modifier='mod', print_results=False)
        assert icr[1] < -0.4908 < icr[2]

    def test_interaction_contrast_ratio_bootstrap_ci(self, data_ic):
        icr = interaction_contrast_ratio(data_ic, exposure='exp', outcome='y', modifier='mod',
                                         ci='bootstrap', print_results=False)
        assert icr[1] < -0.4908 < icr[2]


class TestSplines:

    @pytest.fixture
    def spline_data(self):
        df = pd.DataFrame()
        df['v'] = [1, 5, 10, 15, 20]
        return df

    def test_error_for_bad_nknots(self, spline_data):
        with pytest.raises(ValueError):
            spline_data['sp'] = spline(spline_data, 'v', n_knots=1.5)
        with pytest.raises(ValueError):
            spline_data['sp'] = spline(spline_data, 'v', n_knots=0)
        with pytest.raises(ValueError):
            spline_data['sp'] = spline(spline_data, 'v', n_knots=-1)
        with pytest.raises(ValueError):
            spline_data['sp'] = spline(spline_data, 'v', n_knots=8)

    def test_error_for_unequal_numbers(self, spline_data):
        with pytest.raises(ValueError):
            spline_data['sp'] = spline(spline_data, 'v', n_knots=1, knots=[1, 3])
        with pytest.raises(ValueError):
            spline_data['sp'] = spline(spline_data, 'v', n_knots=3, knots=[1, 3])

    def test_error_for_bad_order(self, spline_data):
        with pytest.raises(ValueError):
            spline_data['sp'] = spline(spline_data, 'v', n_knots=3, knots=[3, 1, 2])

    def test_auto_knots1(self, spline_data):
        spline_data['sp'] = spline(spline_data, 'v', n_knots=1, restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 5.0},
                                                      {'sp': 10.0}])
        pdt.assert_series_equal(spline_data['sp'], expected_splines['sp'])

    def test_auto_knots2(self, spline_data):
        spline_data[['sp1', 'sp2']] = spline(spline_data, 'v', n_knots=2, restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 10 - 20/3, 'sp2': 0.0},
                                                      {'sp1': 15 - 20/3, 'sp2': 15 - 40/3},
                                                      {'sp1': 20 - 20/3, 'sp2': 20 - 40/3}])
        pdt.assert_frame_equal(spline_data[['sp1', 'sp2']], expected_splines[['sp1', 'sp2']])

    def test_user_knots1(self, spline_data):
        spline_data['sp'] = spline(spline_data, 'v', n_knots=1, knots=[16], restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 4.0}])
        pdt.assert_series_equal(spline_data['sp'], expected_splines['sp'])

    def test_user_knots2(self, spline_data):
        spline_data[['sp1', 'sp2']] = spline(spline_data, 'v', n_knots=2, knots=[10, 16], restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 5.0, 'sp2': 0.0},
                                                      {'sp1': 10.0, 'sp2': 4.0}])
        pdt.assert_frame_equal(spline_data[['sp1', 'sp2']], expected_splines[['sp1', 'sp2']])

    def test_quadratic_spline1(self, spline_data):
        spline_data['sp'] = spline(spline_data, 'v', n_knots=1, knots=[16], term=2, restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 4.0**2}])
        pdt.assert_series_equal(spline_data['sp'], expected_splines['sp'])

    def test_quadratic_spline2(self, spline_data):
        spline_data[['sp1', 'sp2']] = spline(spline_data, 'v', n_knots=2, knots=[10, 16], term=2, restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 5.0**2, 'sp2': 0.0},
                                                      {'sp1': 10.0**2, 'sp2': 4.0**2}])
        pdt.assert_frame_equal(spline_data[['sp1', 'sp2']], expected_splines[['sp1', 'sp2']])

    def test_cubic_spline1(self, spline_data):
        spline_data['sp'] = spline(spline_data, 'v', n_knots=1, knots=[16], term=3, restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 0.0},
                                                      {'sp': 4.0**3}])
        pdt.assert_series_equal(spline_data['sp'], expected_splines['sp'])

    def test_cubic_spline2(self, spline_data):
        spline_data[['sp1', 'sp2']] = spline(spline_data, 'v', n_knots=2, knots=[10, 16], term=3, restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 5.0**3, 'sp2': 0.0},
                                                      {'sp1': 10.0**3, 'sp2': 4.0**3}])
        pdt.assert_frame_equal(spline_data[['sp1', 'sp2']], expected_splines[['sp1', 'sp2']])

    def test_higher_order_spline(self, spline_data):
        spline_data[['sp1', 'sp2']] = spline(spline_data, 'v', n_knots=2, knots=[10, 16], term=3.7, restricted=False)
        expected_splines = pd.DataFrame.from_records([{'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 0.0, 'sp2': 0.0},
                                                      {'sp1': 5.0**3.7, 'sp2': 0.0},
                                                      {'sp1': 10.0**3.7, 'sp2': 4.0**3.7}])
        pdt.assert_frame_equal(spline_data[['sp1', 'sp2']], expected_splines[['sp1', 'sp2']])

    def test_restricted_spline1(self, spline_data):
        spline_data['rsp'] = spline(spline_data, 'v', n_knots=2, knots=[10, 16], restricted=True)
        expected_splines = pd.DataFrame.from_records([{'rsp': 0.0},
                                                      {'rsp': 0.0},
                                                      {'rsp': 0.0},
                                                      {'rsp': 5.0},
                                                      {'rsp': 6.0}])
        pdt.assert_series_equal(spline_data['rsp'], expected_splines['rsp'])

    def test_restricted_spline2(self, spline_data):
        spline_data['rsp'] = spline(spline_data, 'v', n_knots=2, knots=[5, 16], restricted=True)
        expected_splines = pd.DataFrame.from_records([{'rsp': 0.0},
                                                      {'rsp': 0.0},
                                                      {'rsp': 10.0 - 5.0},
                                                      {'rsp': 15.0 - 5.0},
                                                      {'rsp': (20.0 - 5.0) - (20.0 - 16.0)}])
        pdt.assert_series_equal(spline_data['rsp'], expected_splines['rsp'])

    def test_restricted_spline3(self, spline_data):
        spline_data['rsp'] = spline(spline_data, 'v', n_knots=2, knots=[5, 16], term=2, restricted=True)
        expected_splines = pd.DataFrame.from_records([{'rsp': 0.0},
                                                      {'rsp': 0.0},
                                                      {'rsp': (10.0 - 5.0)**2 - 0},
                                                      {'rsp': (15.0 - 5.0)**2 - 0},
                                                      {'rsp': (20.0 - 5.0)**2 - (20.0 - 16.0)**2}])
        pdt.assert_series_equal(spline_data['rsp'], expected_splines['rsp'])


class TestTable1:

    @pytest.fixture
    def data(self, n=1000):
        df = pd.DataFrame()
        np.random.seed(111)
        df['exp'] = np.random.binomial(1, 0.5, size=n)
        df['mod'] = np.random.binomial(1, 0.5, size=n)
        df['y'] = np.random.binomial(1, size=n, p=logistic.cdf(0.1 + 0.2*df['exp'] + 0.3*df['mod'] -
                                                               0.4*df['mod']*df['exp']))
        df['continuous'] = np.random.normal(size=n)
        return df

    def test_unstratified_median(self, data):
        t = table1_generator(data, cols=['exp', 'mod', 'y', 'continuous'],
                             variable_type=['category', 'category', 'category', 'continuous'])
        assert isinstance(t, type(pd.DataFrame()))

    def test_unstratified_mean(self, data):
        t = table1_generator(data, cols=['exp', 'mod', 'y', 'continuous'],
                             variable_type=['category', 'category', 'category', 'continuous'],
                             continuous_measure='mean')
        assert isinstance(t, type(pd.DataFrame()))

    def test_stratified_median(self, data):
        t = table1_generator(data, cols=['mod', 'y', 'continuous'],
                             variable_type=['category', 'category', 'continuous'], strat_by='exp')
        assert isinstance(t, type(pd.DataFrame()))

    def test_stratified_mean(self, data):
        t = table1_generator(data, cols=['mod', 'y', 'continuous'],
                             variable_type=['category', 'category', 'continuous'],
                             continuous_measure='mean', strat_by='exp')
        assert isinstance(t, type(pd.DataFrame()))

    def test_catch_different_lengths(self, data):
        with pytest.raises(ValueError):
            table1_generator(data, cols=['exp', 'mod', 'y', 'continuous'],
                             variable_type=['category', 'category', 'continuous'],
                             continuous_measure='A')

    def test_wrong_continuous_measure_error(self, data):
        with pytest.raises(ValueError):
            table1_generator(data, cols=['exp', 'mod', 'y', 'continuous'],
                             variable_type=['category', 'category', 'category', 'continuous'],
                             continuous_measure='A')
        with pytest.raises(ValueError):
            table1_generator(data, cols=['mod', 'y', 'continuous'],
                             variable_type=['category', 'category', 'continuous'],
                             continuous_measure='A', strat_by='exp')
