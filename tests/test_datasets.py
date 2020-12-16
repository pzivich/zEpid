import pandas as pd

import zepid as ze


class TestSampleData:

    def test_return_pandas(self):
        df = ze.load_sample_data(False)
        assert isinstance(df, type(pd.DataFrame()))

    def test_correct_nobs_timefixed(self):
        df = ze.load_sample_data(False)
        assert df.shape[0] == 547

    def test_correct_ncols_timefixed(self):
        df = ze.load_sample_data(False)
        assert df.shape[1] == 9

    def test_missing_in_timefixed(self):
        df = ze.load_sample_data(False)
        assert df.shape[0] != df.dropna().shape[0]

    def test_correct_nobs_timevary(self):
        df = ze.load_sample_data(True)
        assert df.shape[0] == 27382

    def test_correct_ncols_timevary(self):
        df = ze.load_sample_data(True)
        assert df.shape[1] == 12


class TestEwingSarcoma:

    def test_return_pandas(self):
        df = ze.load_ewing_sarcoma_data()
        assert isinstance(df, type(pd.DataFrame()))

    def test_correct_nobs(self):
        df = ze.load_ewing_sarcoma_data()
        assert df.shape[0] == 76

    def test_correct_ncols(self):
        df = ze.load_ewing_sarcoma_data()
        assert df.shape[1] == 4

    def test_no_missing(self):
        df = ze.load_ewing_sarcoma_data()
        assert df.shape[0] == df.dropna().shape[0]


class TestGvHD:

    def test_return_pandas(self):
        df = ze.load_gvhd_data()
        assert isinstance(df, type(pd.DataFrame()))

    def test_correct_nobs(self):
        df = ze.load_gvhd_data()
        assert df.shape[0] == 108714

    def test_correct_ncols(self):
        df = ze.load_gvhd_data()
        assert df.shape[1] == 30

    def test_no_missing(self):
        df = ze.load_gvhd_data()
        assert df.shape[0] == df.dropna().shape[0]


class TestSciatica:

    def test_return_pandas(self):
        df = ze.load_sciatica_data()
        assert isinstance(df, type(pd.DataFrame()))

    def test_correct_nobs(self):
        df = ze.load_sciatica_data()
        assert df.shape[0] == 1240

    def test_correct_ncols(self):
        df = ze.load_sciatica_data()
        assert df.shape[1] == 17

    def test_no_missing(self):
        df = ze.load_sciatica_data()
        assert df.shape[0] == df.dropna().shape[0]


class TestLeukemia:

    def test_return_pandas(self):
        df = ze.load_leukemia_data()
        assert isinstance(df, type(pd.DataFrame()))

    def test_correct_nobs(self):
        df = ze.load_leukemia_data()
        assert df.shape[0] == 42

    def test_correct_ncols(self):
        df = ze.load_leukemia_data()
        assert df.shape[1] == 5

    def test_no_missing(self):
        df = ze.load_leukemia_data()
        assert df.shape[0] == df.dropna().shape[0]


class TestLongitudinal:

    def test_return_pandas(self):
        df = ze.load_longitudinal_data()
        assert isinstance(df, type(pd.DataFrame()))


class TestBingeData:

    def test_return_pandas(self):
        df = ze.load_binge_drinking_data()
        assert isinstance(df, type(pd.DataFrame()))

    def test_correct_ncols(self):
        df = ze.load_binge_drinking_data()
        assert df.shape[1] == 8

    def test_correct_nobs(self):
        df = ze.load_binge_drinking_data()
        assert df.shape[0] == 4000


class TestCaseControl:

    def test_return_pandas(self):
        df = ze.load_case_control_data()
        assert isinstance(df, type(pd.DataFrame()))

    def test_correct_ncols(self):
        df = ze.load_case_control_data()
        assert df.shape[1] == 9

    def test_correct_nobs(self):
        df = ze.load_case_control_data()
        assert df.shape[0] == 11


class TestGeneralize:

    def test_return_pandas_rct(self):
        df = ze.load_generalize_data(False)
        assert isinstance(df, type(pd.DataFrame()))

    def test_return_pandas_conf(self):
        df = ze.load_generalize_data(True)
        assert isinstance(df, type(pd.DataFrame()))


class TestZivichBreskin:

    def test_correct_ncols(self):
        df = ze.load_zivich_breskin_data()
        assert df.shape[1] == 7

    def test_correct_nobs(self):
        df = ze.load_zivich_breskin_data()
        assert df.shape[0] == 3000
