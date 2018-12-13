import pytest
import numpy as np
import numpy.testing as npt

from zepid.sensitivity_analysis import trapezoidal, MonteCarloRR


class TestTrapezoidalDistribution:

    def test_minimum(self):
        n = trapezoidal(mini=0.9, mode1=1.1, mode2=1.7, maxi=1.8, size=10000)
        assert np.min(n) >= 0.9

    def test_maximum(self):
        n = trapezoidal(mini=0.9, mode1=1.1, mode2=1.7, maxi=1.8, size=10000)
        assert np.max(n) <= 1.8


class TestMonteCarloBiasAnalysis:

    @pytest.fixture
    def mcba(self):
        np.random.seed(101)
        mcrr = MonteCarloRR(observed_RR=0.73322, sample=10000)
        mcrr.confounder_RR_distribution(trapezoidal(mini=0.9, mode1=1.1, mode2=1.7, maxi=1.8, size=10000))
        mcrr.prop_confounder_exposed(trapezoidal(mini=0.25, mode1=0.28, mode2=0.32, maxi=0.35, size=10000))
        mcrr.prop_confounder_unexposed(trapezoidal(mini=0.55, mode1=0.58, mode2=0.62, maxi=0.65, size=10000))
        mcrr.fit()
        return mcrr

    def test_mean_mcba_rr(self, mcba):
        m = np.mean(mcba.corrected_RR)
        npt.assert_allclose(m, 0.804864, rtol=1e-5)

    def test_median_mcba_rr(self, mcba):
        m = np.median(mcba.corrected_RR)
        npt.assert_allclose(m, 0.806774, rtol=1e-5)

    def test_percentiles_rr(self, mcba):
        m = np.percentile(mcba.corrected_RR, q=[2.5, 97.5])
        npt.assert_allclose(m, [0.729893, 0.875549], rtol=1e-5)
