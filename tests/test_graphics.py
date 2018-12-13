import pytest
import numpy as np

from zepid import load_sample_data
from zepid.graphics import EffectMeasurePlot, functional_form_plot


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


class TestFunctionalFormPlot:

    @pytest.fixture
    def data(self):
        df = load_sample_data(False)
        df['age_sq'] = df['age0']**2
        return df

    def test_error_outcome_type(self, data):
        with pytest.raises(ValueError):
            functional_form_plot(data, 'cd40', var='age0', outcome_type='categorical')
