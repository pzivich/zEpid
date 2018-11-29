import pytest
import numpy as np
import numpy.testing as npt

from zepid.calc import risk_ci, incidence_rate_ci, risk_ratio, risk_difference, number_needed_to_treat, odds_ratio


class TestRisks:

    def test_correct_risk(self):
        r = risk_ci(25, 50)
        assert r[0] == 0.5

    def test_match_sas_ci(self):
        sas_ci = (0.361409618, 0.638590382)
        r = risk_ci(25, 50, confint='wald')
        npt.assert_allclose(r[1:3], sas_ci)

    def test_match_sas_se(self):
        sas_se = 0.070710678
        r = risk_ci(25, 50, confint='wald')
        npt.assert_allclose(r[3], sas_se)


class TestIncidenceRate:

    def test_incidencerate(self):
        events = 5
        time = 100
        i = incidence_rate_ci(events, time)
        npt.assert_allclose(i[0], events/time)


class TestRiskRatio:

    @pytest.fixture
    def counts_1(self):
        return 25, 25, 25, 25

    def test_risk_ratio_equal_to_1(self, counts_1):
        rr = risk_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        assert rr[0] == 1

    def test_risk_ratio_equal_to_2(self):
        rr = risk_ratio(50, 50, 25, 75)
        assert rr[0] == 2

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            risk_ratio(-5, 1, 1, 1)

    def test_match_sas_ci(self, counts_1):
        sas_se = 0.6757, 1.4799
        rr = risk_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        npt.assert_allclose(np.round(rr[1:3], 4), sas_se)


class TestRiskDifference:

    @pytest.fixture
    def counts_1(self):
        return 25, 25, 25, 25

    def test_risk_difference_equal_to_0(self, counts_1):
        rd = risk_difference(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        assert rd[0] == 0

    def test_risk_difference_equal_to_half(self):
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(rd[0], 0.25)

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            risk_difference(-5, 1, 1, 1)

    def test_match_sas_ci(self, counts_1):
        sas_ci = -0.195996398, 0.195996398
        rd = risk_difference(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        npt.assert_allclose(rd[1:3], sas_ci)

    def test_match_sas_se(self, counts_1):
        sas_se = 0.1
        rd = risk_difference(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        npt.assert_allclose(rd[3], sas_se)


class TestNumberNeededtoTreat:

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            number_needed_to_treat(-5, 1, 1, 1)

    def test_match_risk_difference(self):
        nnt = number_needed_to_treat(50, 50, 25, 75)
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(nnt[0], 1/rd[0])

    def test_match_rd_ci(self):
        nnt = number_needed_to_treat(50, 50, 25, 75)
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(nnt[1:3], [1/i for i in rd[1:3]])

    def test_match_rd_se(self):
        nnt = number_needed_to_treat(50, 50, 25, 75)
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(nnt[3], rd[3])

    def test_rd_of_zero_is_nnt_inf(self):
        nnt = number_needed_to_treat(25, 25, 25, 25)
        assert np.isinf(nnt[0])


class TestOddsRatio:

    @pytest.fixture
    def counts_1(self):
        return 25, 25, 25, 25

    def test_odds_ratio_equal_to_1(self, counts_1):
        odr = odds_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        assert odr[0] == 1

    def test_odds_ratio_greater_than_risk_ratio(self):
        odr = odds_ratio(50, 50, 25, 75)
        rr = risk_ratio(50, 50, 25, 75)
        assert odr[0] > rr[0]

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            odds_ratio(-5, 1, 1, 1)

    def test_match_sas_ci(self, counts_1):
        sas_se = 0.4566, 2.1902
        odr = odds_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        npt.assert_allclose(np.round(odr[1:3], 4), sas_se)
