import pytest
import math
import numpy as np
import numpy.testing as npt

from zepid.calc import (risk_ci, incidence_rate_ci, risk_ratio, risk_difference, number_needed_to_treat, odds_ratio,
                        incidence_rate_ratio, incidence_rate_difference, odds_to_probability, probability_to_odds,
                        semibayes, attributable_community_risk, population_attributable_fraction, sensitivity,
                        specificity, npv_converter, ppv_converter, rubins_rules, s_value)


@pytest.fixture
def counts_1():
    return 25, 25, 25, 25


# Tests for Basic Measures

class TestRisks:

    def test_correct_risk(self):
        r = risk_ci(25, 50)
        assert r.point_estimate == 0.5
        assert r[0] == 0.5

    def test_match_sas_ci(self):
        sas_ci = (0.361409618, 0.638590382)
        r = risk_ci(25, 50, confint='wald')
        npt.assert_allclose(r[1:3], sas_ci)
        npt.assert_allclose([r.lower_bound, r.upper_bound], sas_ci)

    def test_match_sas_se1(self):
        sas_se = 0.070710678
        r = risk_ci(25, 50, confint='wald')
        npt.assert_allclose(r[3], sas_se)
        npt.assert_allclose(r.standard_error, sas_se)

    def test_match_sas_se2(self):
        sas_se = 0.070710678
        r = risk_ci(25, 50, confint='wald')
        npt.assert_allclose(r[3], sas_se)
        npt.assert_allclose(r.standard_error, sas_se)


class TestIncidenceRate:

    def test_incidence_rate(self):
        irc = incidence_rate_ci(14, 400)
        npt.assert_allclose(irc.point_estimate, 0.035)

    def test_match_sas_se(self):
        sas_se = 0.009354
        irc = incidence_rate_ci(14, 400)
        npt.assert_allclose(irc.standard_error, sas_se, atol=1e-5)

    def test_match_normalapprox_ci(self):
        # Because incidence rate CI's have no agreed convention, I use the normal approx. This is not the same as SAS
        sas_ci = 0.01667, 0.05333
        i = incidence_rate_ci(14, 400)
        npt.assert_allclose(np.round(i[1:3], 5), sas_ci, atol=1e-5)


class TestRiskRatio:

    def test_risk_ratio_equal_to_1(self, counts_1):
        rr = risk_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        assert rr.point_estimate == 1

    def test_risk_ratio_equal_to_2(self):
        rr = risk_ratio(50, 50, 25, 75)
        assert rr.point_estimate == 2

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            risk_ratio(-5, 1, 1, 1)

    def test_match_sas_ci(self, counts_1):
        sas_se = 0.6757, 1.4799
        rr = risk_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        npt.assert_allclose(np.round(rr[1:3], 4), sas_se)


class TestRiskDifference:

    def test_risk_difference_equal_to_0(self, counts_1):
        rd = risk_difference(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        assert rd.point_estimate == 0

    def test_risk_difference_equal_to_half(self):
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(rd.point_estimate, 0.25)

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
        npt.assert_allclose(rd.standard_error, sas_se)

    def test_raises_warning_if_small_cells(self):
        with pytest.warns(UserWarning, match='confidence interval approximation is invalid'):
            rd = risk_difference(1, 10, 10, 10)


class TestNumberNeededtoTreat:

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            number_needed_to_treat(-5, 1, 1, 1)

    def test_match_risk_difference(self):
        nnt = number_needed_to_treat(50, 50, 25, 75)
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(nnt.point_estimate, 1/rd[0])

    def test_match_rd_ci(self):
        nnt = number_needed_to_treat(50, 50, 25, 75)
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(nnt[1:3], [1/i for i in rd[1:3]])

    def test_match_rd_se(self):
        nnt = number_needed_to_treat(50, 50, 25, 75)
        rd = risk_difference(50, 50, 25, 75)
        npt.assert_allclose(nnt.standard_error, rd[3])

    def test_rd_of_zero_is_nnt_inf(self):
        nnt = number_needed_to_treat(25, 25, 25, 25)
        assert np.isinf(nnt[0])


class TestOddsRatio:

    def test_odds_ratio_equal_to_1(self, counts_1):
        odr = odds_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        assert odr.point_estimate == 1

    def test_odds_ratio_greater_than_risk_ratio(self):
        odr = odds_ratio(50, 50, 25, 75)
        rr = risk_ratio(50, 50, 25, 75)
        assert odr.point_estimate > rr.point_estimate

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            odds_ratio(-5, 1, 1, 1)

    def test_match_sas_ci(self, counts_1):
        sas_se = 0.4566, 2.1902
        odr = odds_ratio(counts_1[0], counts_1[1], counts_1[2], counts_1[3])
        npt.assert_allclose(np.round(odr[1:3], 4), sas_se)


class TestIncidenceRateRatio:

    def test_incidence_rate_ratio_equal_to_1(self):
        irr = incidence_rate_ratio(6, 6, 100, 100)
        assert irr.point_estimate == 1

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            incidence_rate_ratio(-5, 1, 1, 1)

    def test_match_sas_ci(self):
        sas_ci = 0.658778447, 4.460946657
        irr = incidence_rate_ratio(6, 14, 100, 400)
        npt.assert_allclose(irr[1:3], sas_ci)

    def test_match_sas_se(self):
        sas_se = 0.487950036
        irr = incidence_rate_ratio(6, 14, 100, 400)
        npt.assert_allclose(irr.standard_error, sas_se)


class TestIncidenceRateDiff:

    def test_incidence_rate_difference_equal_to_1(self):
        ird = incidence_rate_difference(6, 6, 100, 100)
        assert ird.point_estimate == 0

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            incidence_rate_difference(-5, 1, 1, 1)

    def test_match_sas(self):
        sas_ird = 0.025
        ird = incidence_rate_difference(6, 14, 100, 400)
        npt.assert_allclose(ird.point_estimate, sas_ird)

    def test_correct_ci(self):
        # SAS does not provide CI's for IRD easily. Instead comparing to OpenEpi calculator
        oe_ci = -0.02639, 0.07639
        ird = incidence_rate_difference(6, 14, 100, 400)
        npt.assert_allclose(ird.point_estimate, 0.025)
        npt.assert_allclose(ird[1:3], oe_ci, atol=1e-6)


class TestACR:

    def test_acr_is_zero(self):
        acr = attributable_community_risk(25, 25, 25, 25)
        assert acr == 0

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            attributable_community_risk(-5, 1, 1, 1)

    def test_compare_to_formula(self):
        acr_formula = (25 + 10) / 100 - 10 / 50
        acr = attributable_community_risk(25, 25, 10, 40)
        npt.assert_allclose(acr, acr_formula)


class TestPAF:

    def test_paf_is_zero(self):
        paf = population_attributable_fraction(25, 25, 25, 25)
        assert paf == 0

    def test_value_error_for_negative_counts(self):
        with pytest.raises(ValueError):
            population_attributable_fraction(-5, 1, 1, 1)

    def test_compare_to_formula(self):
        paf_formula = ((25 + 10) / 100 - 10 / 50) / ((25 + 10) / 100)
        paf = population_attributable_fraction(25, 25, 10, 40)
        npt.assert_allclose(paf, paf_formula)


# Test testing measures
class TestPredictiveFunctions:

    def test_correct_sensitivity(self):
        r = sensitivity(25, 50)
        assert r[0] == 0.5

    def test_sensitivity_match_sas_ci(self):
        sas_ci = (0.361409618, 0.638590382)
        r = sensitivity(25, 50, confint='wald')
        npt.assert_allclose(r[1:3], sas_ci)

    def test_sensitivity_match_sas_se(self):
        sas_se = 0.070710678
        r = sensitivity(25, 50, confint='wald')
        npt.assert_allclose(r[3], sas_se)

    def test_correct_specificity(self):
        r = specificity(25, 50)
        assert r[0] == 0.5

    def test_specificity_match_sas_ci(self):
        sas_ci = (0.361409618, 0.638590382)
        r = specificity(25, 50, confint='wald')
        npt.assert_allclose(r[1:3], sas_ci)

    def test_specificity_match_sas_se(self):
        sas_se = 0.070710678
        r = specificity(25, 50, confint='wald')
        npt.assert_allclose(r[3], sas_se)

    def test_ppv_conversion(self):
        sens = 0.8
        spec = 0.8
        prev = 0.1
        ppv_formula = (sens*prev) / ((sens*prev) + ((1-spec) * (1-prev)))
        ppv = ppv_converter(sens, spec, prev)
        npt.assert_allclose(ppv, ppv_formula)

    def test_npv_conversion(self):
        sens = 0.8
        spec = 0.8
        prev = 0.1
        npv_formula = (spec*0.9) / ((spec*0.9) + ((1-sens) * (prev)))
        npv = npv_converter(sens, spec, prev)
        npt.assert_allclose(npv, npv_formula)


# Test other calculators
class TestOddsProbabilityConverter:

    def test_odds_to_probability(self):
        pr = odds_to_probability(1.1)
        npt.assert_allclose(pr, 1.1/2.1)

    def test_probability_to_odds(self):
        od = probability_to_odds(0.5)
        assert od == 1

    def test_back_and_forth_conversions(self):
        original = 0.12
        odd = probability_to_odds(original)
        pr = odds_to_probability(odd)
        npt.assert_allclose(original, pr)

    def test_forth_and_back_conversions(self):
        original = 1.1
        pr = odds_to_probability(original)
        odd = probability_to_odds(pr)
        npt.assert_allclose(original, odd)


class TestsSemiBayes:

    def test_compare_to_modernepi3(self):
        # Compares to Modern Epidemiology 3 example on page 334-335
        posterior_rr = math.exp(math.log(3.51)/0.569 / (1/0.5 + 1/0.569))
        posterior_ci = math.exp(0.587 - 1.96*(0.266**0.5)), math.exp(0.587 + 1.96*(0.266**0.5))
        sb = semibayes(prior_mean=1, prior_lcl=0.25, prior_ucl=4, mean=3.51, lcl=0.80, ucl=15.4,
                       ln_transform=True, print_results=False)
        npt.assert_allclose(sb[0], posterior_rr, atol=1e-3)
        npt.assert_allclose(sb[1:], posterior_ci, rtol=1e-3)


class TestRubinsRules:

    def test_error_wrong_len(self):
        rr_est = [1, 1, 3]
        rr_std = [0.05, 0.05]
        with pytest.raises(ValueError):
            rubins_rules(rr_est, rr_std)

    def test_match_sas1(self):
        # points
        rr_est = [0.52, 0.31, -0.04]
        rr_var = [0.075, 0.083, 0.065]

        # SAS calculations via PROC MIANALYZE
        est_sas = 0.26333333
        std_sas = 0.33509816

        b = rubins_rules(rr_est, rr_var)
        npt.assert_allclose(b[0], est_sas)
        npt.assert_allclose(b[1], std_sas)

    def test_match_sas2(self):
        # points
        rr_est = [-0.52, -0.31, -0.04, -0.8, 0.01, -0.12, -0.34]
        rr_var = [0.035, 0.043, 0.025, 0.045, 0.023, 0.001, 0.021]

        # SAS calculations via PROC MIANALYZE
        est_sas = -0.30285714
        std_sas = 0.30896574

        b = rubins_rules(rr_est, rr_var)
        npt.assert_allclose(b[0], est_sas)
        npt.assert_allclose(b[1], std_sas)


class TestSValues:

    def test_svalue1(self):
        npt.assert_allclose(4.3219280949, s_value(0.05))

    def test_svalue2(self):
        npt.assert_allclose(0.2009126939, s_value(0.87))

    def test_svalue3(self):
        npt.assert_allclose([4.3219280949, 0.2009126939], s_value([0.05, 0.87]))
