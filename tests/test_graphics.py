import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zepid import load_sample_data
from zepid.graphics import EffectMeasurePlot, pvalue_plot, functional_form_plot, spaghetti_plot, roc, dynamic_risk_plot


class TestForestPlot:  # referred to as EffectMeasurePlot in zepid

    @pytest.fixture
    def plotter(self):
        return EffectMeasurePlot

    @pytest.fixture
    def data(self):
        labs = ['Overall', 'Adjusted', '', '2012-2013', 'Adjusted', '', '2013-2014', 'Adjusted', '', '2014-2015',
                'Adjusted']
        measure = [np.nan, 0.94, np.nan, np.nan, 1.22, np.nan, np.nan, 0.59, np.nan, np.nan, 1.09]
        lower = [np.nan, 0.77, np.nan, np.nan, '0.80', np.nan, np.nan, '0.40', np.nan, np.nan, 0.83]
        upper = [np.nan, 1.15, np.nan, np.nan, 1.84, np.nan, np.nan, 0.85, np.nan, np.nan, 1.44]
        return labs, measure, lower, upper

    def test_accepts_formatted_data(self, plotter, data):
        p = plotter(label=data[0], effect_measure=data[1], lcl=data[2], ucl=data[3])

    def test_change_plot_labels(self, plotter, data):
        p = plotter(label=data[0], effect_measure=data[1], lcl=data[2], ucl=data[3])
        changes = ('RR', '90% CI', 'log', 1)
        p.labels(effectmeasure=changes[0], conf_int=changes[1], scale=changes[2], center=changes[3])
        assert p.em == changes[0]
        assert p.ci == changes[1]
        assert p.scale == changes[2]
        assert p.center == changes[3]

    def test_change_plot_colors(self, plotter, data):
        p = plotter(label=data[0], effect_measure=data[1], lcl=data[2], ucl=data[3])
        changes = ('red', 'triangle', 'blue', 'purple')
        p.colors(errorbarcolor=changes[0], pointshape=changes[1], linecolor=changes[2], pointcolor=changes[3])
        assert p.errc == changes[0]
        assert p.shape == changes[1]
        assert p.linec == changes[2]
        assert p.pc == changes[3]

    def test_return_axes(self, plotter, data):
        p = plotter(label=data[0], effect_measure=data[1], lcl=data[2], ucl=data[3])
        assert isinstance(p.plot(), type(plt.gca()))


class TestFunctionalFormPlot:

    @pytest.fixture
    def data(self):
        df = load_sample_data(False)
        df['age_sq'] = df['age0']**2
        return df

    def test_error_outcome_type(self, data):
        with pytest.raises(ValueError):
            functional_form_plot(data, 'cd40', var='age0', outcome_type='categorical')

    def test_basic_functional_form(self, data):
        f = functional_form_plot(data, 'dead', var='age0', loess=False)
        assert isinstance(f, type(plt.gca()))

    def test_discrete_comparison(self, data):
        f = functional_form_plot(data, 'dead', var='age0', discrete=True, loess=False)
        assert isinstance(f, type(plt.gca()))

    def test_continuous_outcome(self, data):
        f = functional_form_plot(data, 'cd40', var='age0', outcome_type='continuous', loess=False)
        assert isinstance(f, type(plt.gca()))

    def test_add_points(self, data):
        f = functional_form_plot(data, 'dead', var='age0', points=True, loess=False)
        assert isinstance(f, type(plt.gca()))

    def test_loess(self, data):
        f = functional_form_plot(data, 'dead', var='age0', loess=True, loess_value=0.25, discrete=True)
        assert isinstance(f, type(plt.gca()))

    def test_specified_functional_form(self, data):
        f = functional_form_plot(data, 'dead', var='age0', f_form='age0 + age_sq', loess=False)
        assert isinstance(f, type(plt.gca()))


class TestPValuePlot:

    def test_pvalue_plot(self):
        p = pvalue_plot(point=0.23, sd=0.1)
        assert isinstance(p, type(plt.gca()))

    def test_change_fill(self):
        p = pvalue_plot(point=0.23, sd=0.1, fill=False)
        assert isinstance(p, type(plt.gca()))

    def test_change_color(self):
        p = pvalue_plot(point=0.23, sd=0.1, color='r')
        assert isinstance(p, type(plt.gca()))

    def test_change_null(self):
        p = pvalue_plot(point=0.23, sd=0.1, null=0.05)
        assert isinstance(p, type(plt.gca()))

    def test_set_alpha(self):
        p = pvalue_plot(point=0.23, sd=0.1, alpha=0.05)
        assert isinstance(p, type(plt.gca()))


class TestSpaghettiPlot:

    def test_spaghetti_plot(self):
        df = load_sample_data(timevary=True)
        s = spaghetti_plot(df, idvar='id', variable='cd4', time='enter')
        assert isinstance(s, type(plt.gca()))


class TestReceiverOperatorCurve:

    @pytest.fixture
    def data(self):
        df = pd.DataFrame()
        df['d'] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        df['p'] = [0.1, 0.15, 0.1, 0.7, 0.5, 0.9, 0.95, 0.5, 0.4, 0.8, 0.99, 0.99, 0.89, 0.95]
        return df

    def test_generate_roc(self, data):
        r = roc(data, true='d', threshold='p', youden_index=False)
        assert isinstance(r, type(plt.gca()))

    def test_calculate_youden_index(self, data):
        r = roc(data, true='d', threshold='p', youden_index=True)
        assert isinstance(r, type(plt.gca()))


class TestDynamicRiskPlot:

    @pytest.fixture
    def data(self):
        a = pd.DataFrame([[0, 0], [1, 0.15], [2, 0.25], [4, 0.345]], columns=['timeline', 'riske']).set_index(
            'timeline')
        b = pd.DataFrame([[0, 0], [1, 0.2], [1.5, 0.31], [3, 0.345]], columns=['timeline', 'riske']).set_index(
            'timeline')
        return a, b

    def test_risk_difference(self, data):
        d = dynamic_risk_plot(data[0], data[1], loess=False)
        assert isinstance(d, type(plt.gca()))

    def test_risk_ratio_linear(self, data):
        d = dynamic_risk_plot(data[0], data[1], measure='RR', loess=False)
        assert isinstance(d, type(plt.gca()))

    def test_risk_ratio_logtransform(self, data):
        d = dynamic_risk_plot(data[0], data[1], measure='RR', scale='log-transform', loess=False)
        assert isinstance(d, type(plt.gca()))

    def test_risk_ratio_log(self, data):
        d = dynamic_risk_plot(data[0], data[1], measure='RR', scale='log', loess=False)
        assert isinstance(d, type(plt.gca()))

    def test_loess(self, data):
        d = dynamic_risk_plot(data[0], data[1], loess=True, loess_value=0.4)
        assert isinstance(d, type(plt.gca()))

    def test_change_colors(self, data):
        d = dynamic_risk_plot(data[0], data[1], loess=False, point_color='green', line_color='green')
        assert isinstance(d, type(plt.gca()))
