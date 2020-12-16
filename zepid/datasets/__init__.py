import numpy as np
import pandas as pd
from scipy.stats import logistic
from pkg_resources import resource_filename


def load_sample_data(timevary):
    """Load data that is part of the zepid package. This data set comes from simulated data from Jessie Edwards
    (thanks Jess!). This data is used for examples on zepid.readthedocs

    Parameters
    --------------
    timevary : bool
        Whether to return the time-varying data set or the time fixed. If True then returns data set with repeated
        visits. If False then a data set with single observation per subject representing the 45-week risk is returned

    Notes
    -------------

    For the time-varying data set, the following variables are returned;
        * id - participant unique ID
        * enter - start of follow-up period
        * out - end of time period
        * male - indicator variable for male (1 = yes)
        * age0 - age at enter = 0
        * cd40 - CD4 T cell count at enter = 0
        * dvl0 - detectable viral load data at enter = 0
        * cd4 - CD4 T cell count at enter = t
        * dvl - viral load at enter = t
        * art - indicator of whether ART was prescribed at enter = t
        * drop - indicator of whether individual dropped out of the study at enter = t (1 = yes)
        * dead - indicator for death at out = t (1 = yes)

    For the time-fixed data set, the following variables are returned
        * id - participant unique ID
        * male - indicator variable for male (1 = yes)
        * age0 - age at enter = 0
        * cd40 - CD4 T cell count at enter = 0
        * dvl0 - detectable viral load data at enter = 0
        * art - indicator of whether ART was prescribed at enter = 0
        * t - total time contributed

    Returns
    ----------
    DataFrame
        Returns either a time-varying or time-fixed pandas DataFrame

    Examples
    --------
    Load the time-fixed exposure data set

    >>> from zepid import load_sample_data
    >>> load_sample_data(timevary=False)

    Load the time-varying exposure data set

    >>> load_sample_data(timevary=True)
    """
    cols = ['id', 'enter', 'out', 'male', 'age0', 'cd40', 'dvl0', 'cd4', 'dvl', 'art', 'drop', 'dead']
    df = pd.read_csv(resource_filename('zepid', 'datasets/data.dat'),
                     delim_whitespace=True, header=None, names=cols, index_col=False)
    df.sort_values(by=['id', 'enter'], inplace=True)
    if timevary:
        return df
    else:
        dfi = df.loc[df.groupby('id').cumcount() == 0, ['id', 'male', 'age0', 'cd40', 'dvl0', 'art']].copy()
        dfo = df.loc[df.id != df.id.shift(-1)][['id', 'dead', 'drop', 'out']].copy()
        dfo.loc[(dfo['drop'] == 1) & (dfo['out'] <= 45), 'dead'] = np.nan
        dfo['dead'] = np.where((dfo['dead'] == 1) & (dfo['out'] > 45), 0, dfo['dead'])
        dff = pd.merge(dfi, dfo, left_on='id', right_on='id')
        dff.rename(columns={'out': 't'}, inplace=True)
        dff.drop('drop', axis=1, inplace=True)

        # Adding last CD4 T-cell count for continuous outcome data
        dfl = df.loc[df['out'] <= 45, ['id', 'cd4']].copy()
        dfll = dfl.loc[dfl.id != dfl.id.shift(-1)].copy()
        dfll.rename(columns={'cd4': 'cd4_wk45'}, inplace=True)
        dff = pd.merge(dff, dfll, left_on='id', right_on='id')
        dff['cd4_wk45'] = np.where(dff['dead'] == 1, np.nan, dff['cd4_wk45'])
        dff['t'] = np.where(dff['t'] >= 45, 45, dff['t'])

        return dff


def load_ewing_sarcoma_data():
    """Loads Ewing's Sarcoma survival data from Glaubiger DL, Makuch R, Schwarz J, Levine AS, Johnson RE. Determination
    of prognostic factors and their influence on therapeutic results in patients with Ewing's sarcoma. Cancer.
    1980;45(8):2213-9. In total, data for 76 patients is loaded. Further descriptions of the 76 patients and the data
    can be found in Makuch RW. Adjusted survival curve estimation using covariates. J Chronic Dis. 1982;35(6):437-43.
    or Cole SR, Hernán MA. Adjusted survival curves with inverse probability weights. Comput Methods Programs
    Biomed. 2004;75(1):45-9.

    Notes
    -----------
    Variables within the dataset are
        * treat - treatment (1 is novel treatment; 0 is one of three standard treatments)
        * ldh - pre-treatment serum lactic acid dehydrogenase (LDH) (1 is >= 200 international units; 0 is <200)
        * time - days till recurrence or censoring (continuous)
        * outcome - sarcoma recurrence (1 is recurrence; 0 is censored)

    Returns
    ----------
    DataFrame
        Returns pandas DataFrame
    """
    return pd.read_csv(resource_filename('zepid', 'datasets/ewing.dat'), index_col=False)


def load_gvhd_data():
    """Loads bone marrow transplant recipient data from Keil AP, Edwards JK, Richardson DB, Naimi AI, Cole SR. The
    parametric g-formula for time-to-event data: intuition and a worked example. Epidemiology. 2014;25(6):889-97.
    Patients were followed until death or administrative censoring at 5-years.

    Notes
    ---------------
    Variables are formatted exactly as described in Keil et al. 2014
        * id: unique ID for each participant
        * age: participant baseline age
        * agesq: squared baseline age
        * agecurs1: restricted cubic spline knot 1 for baseline age
        * agecurs2: restricted cubic spline knot 2 for basline age
        * male: participant gender (1 is male, 0 is female)
        * cmv: cytomegalovirus baseline immune status (1 is yes, 0 is no)
        * all: at this time, I am unsure what this variable indicates (1, 0)
        * wait: wait time from diagnosis to transplantation (months)
        * day: day since transplantation
        * daysq: squared day since transplantation
        * daycu: cubic day since transplantation
        * daycurs1: restricted cubic spline knot 1 for days since transplantation
        * daycurs2: restricted cubic spline knot 2 for days since transplantation
        * yesterday: previous day
        * tomorrow: day after
        * gvhd: indicator for Graph-versus-Host Disease (1 is yes, 0 is no)
        * d: indicator of death (1 is yes, 0 is no)
        * relapse: indicator for relapse (1 is yes, 0 is no)
        * platnorm: indicator for normal platelet count (1 is yes, 0 is no)
        * censlost: indicator for censoring due to loss-to-follow-up (1 is yes, 0 is no)
        * gvhdm1: indicator for previous day diagnosis of GvHD (1 is yes, 0 is no)
        * relapsem1: indicator for previous day relapse (1 is yes, 0 is no)
        * platnormm1: indicator for previous day normal platelet count (1 is yes, 0 is no)
        * daysnogvhd: number of consecutive days without a GvHD diagnosis
        * daysnorelapse: number of consecutive days without relapse
        * daysnoplatnorm: number of consecutive days without normal platelet count
        * daysgvhd: number of consecutive days with GvHD
        * daysrelapse: number of consecutive days after relapse
        * daysplatnorm: number of consecutive days with normal platelet count

    Returns
    ----------
    DataFrame
        Returns pandas DataFrame
    """
    df = pd.read_csv(resource_filename('zepid', 'datasets/gvhd.dat'), delim_whitespace=True, index_col=False)

    # Coding variables to match Keil et al. exactly
    df['wait'] = df['waitdays'] / 30.5
    df['agesq'] = df['age']**2
    df['agecurs1'] = ((df['age'] > 17.0) * (df['age'] - 17.0) ** 3 -
                      ((df['age'] > 30.0) * (df['age'] - 30.0) ** 3) * (41.4 - 17.0) / (41.4 - 30.0))
    df['agecurs2'] = ((df['age'] > 25.4) * (df['age'] - 25.4) ** 3 -
                      ((df['age'] > 41.4) * (df['age'] - 41.4) ** 3) * (41.4 - 25.4) / (41.4 - 30.0))

    # Expanding data into person-time periods
    df['t_int'] = df['t'].astype(int)
    lf = pd.DataFrame(np.repeat(df.values, df['t_int'] + 1, axis=0), columns=df.columns)
    lf['yesterday'] = lf.groupby('id')['t'].cumcount()
    lf['day'] = lf['yesterday'] + 1
    lf['tomorrow'] = lf['day'] + 1
    lf['daysq'] = lf['day']**2
    lf['daycu'] = lf['day']**3
    lf['daycurs1'] = (((lf['day'] > 63) * ((lf['day'] - 63) / 63) ** 3) +
                      ((lf['day'] > 716) * ((lf['day'] - 716) / 63) ** 3) * (350.0 - 63) -
                      ((lf['day'] > 350) * ((lf['day'] - 350) / 63) ** 3) * (716 - 63) / (716 - 350))
    lf['daycurs2'] = (((lf['day'] > 168) * ((lf['day'] - 168) / 63) ** 3) +
                      ((lf['day'] > 716) * ((lf['day'] - 716) / 63) ** 3) * (350 - 168) -
                      ((lf['day'] > 350) * ((lf['day'] - 350) / 63) ** 3) * (716 - 168) / (716 - 350))
    lf['tdiff'] = lf['t'] - lf['yesterday']
    lf = lf.loc[lf['tdiff'] != 0].copy()  # Removes redundant first observation

    # Recoding variables for the expanded data
    lf['d'] = (lf['day'] >= lf['t']) * lf['d_dea']
    lf['gvhd'] = (lf['day'] > lf['t_gvhd']) * 1
    lf['relapse'] = (lf['day'] > lf['t_rel']) * 1
    lf['platnorm'] = (lf['day'] > lf['t_pla']) * 1
    lf['gvhdm1'] = (lf['yesterday'] > lf['t_gvhd']) * 1
    lf['relapsem1'] = (lf['yesterday'] > lf['t_rel']) * 1
    lf['platnormm1'] = (lf['yesterday'] > lf['t_pla']) * 1

    # Censoring time point
    lf['censeof'] = np.where((lf['day'] == lf['t']) & (lf['d'] == 0) & (lf['day'] == 1825), 1, 0)
    lf['censlost'] = np.where((lf['day'] == lf['t']) & (lf['d'] == 0) & (lf['day'] != 1825), 1, 0)

    # Setting initial values for days from X
    lf['daysnorelapse'] = np.where(lf['relapse'] == 0, lf['day'], np.nan)
    lf['daysnorelapse'] = lf['daysnorelapse'].fillna(method='ffill')
    lf['daysnoplatnorm'] = np.where(lf['id'] != lf['id'].shift(1), 0, np.nan)
    lf['daysnoplatnorm'] = np.where(lf['platnorm'] == 0, lf['day'], lf['daysnoplatnorm'])
    lf['daysnoplatnorm'] = lf['daysnoplatnorm'].fillna(method='ffill')
    lf['daysnogvhd'] = np.where(lf['gvhd'] == 0, lf['day'], np.nan)
    lf['daysnogvhd'] = lf['daysnogvhd'].fillna(method='ffill')

    lf['daysrelapse'] = lf.groupby('id')['relapse'].cumsum()
    lf['daysplatnorm'] = lf.groupby('id')['platnorm'].cumsum()
    lf['daysgvhd'] = lf.groupby('id')['gvhd'].cumsum()
    return lf[['id', 'age', 'agesq', 'agecurs1', 'agecurs2', 'male', 'cmv', 'all', 'wait', 'yesterday', 'tomorrow',
               'day', 'daysq', 'daycu', 'daycurs1', 'daycurs2', 'd', 'gvhd', 'relapse', 'platnorm', 'gvhdm1',
               'relapsem1', 'platnormm1', 'censlost', 'daysnorelapse', 'daysnoplatnorm', 'daysnogvhd',
               'daysrelapse', 'daysplatnorm', 'daysgvhd']]


def load_sciatica_data():
    """Loads the Sciatica Trial data published in; Mertens, BJA, Jacobs, WCH, Brand, R, and Peul, WC. Assessment of
    patient-specific surgery effect based on weighted estimation and propensity scoring in the re-analysis of the
    Sciatica Trial. PLOS One 2014. Details of the original Sciatica Trial are available in; Peul WC, van Houwelingen HC,
    van den Hout WB, et al. Surgery versus Prolonged Conservative Treatment for Sciatica. NEJM 2007
    DOI: 10.1177/0962280214545529

    Notes
    -------------
    Variables included are
        * id: unique identifier for patient
        * tpoints: follow-up time period
        * time: follow-up time
        * age_b: age at follow-up time
        * age_t: age at randomization
        * vas1_t: VAS score
        * vas2_t: VAS score
        * roland_t: Roland score
        * likert_t: Likert score
        * vas1_b: VAS1 at baseline
        * vas2_b: VAS2 at baseline
        * roland_b: Roland score at baseline
        * likert_b: Likert score at baseline
        * male: participant gender (1 is male, 0 is female)
        * weight: participant weight in kilograms
        * height: participant height in meters
        * surgery: whether participant received the surgery (1 is surgery, 0 is not yet surgery)

    Returns
    ----------
    DataFrame
        Returns pandas DataFrame
    """
    df = pd.read_csv(resource_filename('zepid', 'datasets/sciatica.dat'), delim_whitespace=True, index_col=False,
                     header=None, names=['id', 'tpoints', 'time', 'age_b', 'age_t', 'vas1_t', 'vas2_t', 'roland_t',
                                         'likert_t', 'vas1_b', 'vas2_b', 'roland_b', 'likert_b', 'male', 'weight',
                                         'height', 'surgery'])
    return df


def load_leukemia_data():
    """Loads data from Freireich EJ et al., "The Effect of 6-Mercaptopurine on the Duration of Steriod-induced
    Remissions in Acute Leukemia: A Model for Evaluation of Other Potentially Useful Therapy" Blood 1963

    Notes
    ------------
    Variables included are
        t: time
        status: event indicator (0: censored, 1: relapsed)
        sex: male, female
        logwbc: log-transformed white blood cell count
        treat: treatment indicator

    Returns
    ----------
    DataFrame
        Returns pandas DataFrame
    """
    df = pd.read_csv(resource_filename('zepid', 'datasets/leukemia.dat'), delim_whitespace=True, index_col=False)
    return df


def load_binge_drinking_data():
    """Loads data from Ahren J et al., "Predicting the Population Health Impacts of Community Interventions: The Case
    of Alcohol Outlets and Binge Drinking" AJPH 2016. Below is some notes taken from the supplementary materials
    detailed by the paper authors;

    "The data provided for use with this sample code are simulated. The data are designed to be similar to the real
    data and associations examined in the main paper. There are 4000 observations representing individuals who are
    nested in 44 communities (variable name: neighborhood_id). The exposure of interest is neighborhood alcohol outlet
    density (alc_outlet_density), with values ranging from 39 to 168. The outcome of interest is a  binary indicator of
    binge drinking (binge_drink), and covariates include gender (male), age (age_categorical), marital status
    (married), education (education_categorical), and race/ethnicity (race_categorical). Alcohol outlet density and
    binge drinking were simulated as simple linear functions of the covariates. Thus, unlike the applied example, the
    relation of outlet density with binge drinking has a linear shape."

    Notes
    -----
    Variables included are
        * male: gender (0: female, 1: male)
        * age_categorical: age groups (not clearly defined as to what they refer to)
        * married: marital status
        * education_categorical: categories of education levels
        * race_categorical: categories of race
        * alc_outlet_density: density of alcohol outlets in the neighborhood (continuous)
        * binge_drink: whether individual binge drinks (1: yes, 0: no)
        * neighborhood_id: identifier for groups that individuals are nested in

    Returns
    -------
    DataFrame
        Returns pandas DataFrame
    """
    df = pd.read_csv(resource_filename('zepid', 'datasets/binge.dat'), index_col=False)
    return df


def load_longitudinal_data():
    """Loads simulated longitudinal data. This longitudinal data is used to demonstrate the sequential regression
    time-varying g-formula, and the longitudinal targeted maximum likelihood estimator. The format of the returned
    file is a long data set

    Notes
    -----
    Variables included are
       * A: treatment of interest
       * Y: outcome of interest
       * t: time-point
       * W: baseline variable
       * L: time-varying variable
       * id: unique identifier for each subject

    Returns
    -------
    DataFrame
        Returns pandas DataFrame
    """
    df = pd.read_csv(resource_filename('zepid', 'datasets/longitudinal.dat'), index_col=False)
    return df


def load_case_control_data():
    """Loads example case-control data, meant to reflect observed data. The goal of this case-control is to try to
    determine the causative agent / route

    Notes
    -----
    Variables included are:
        * id: id number
        * case: whether individual is a case or control (0: control, 1:case)
        * breakfast: whether individual ate breakfast (0: no, 1: yes)
        * lunch: lunch eaten by individual (0: no lunch, 1: salad, 2: chicken)
        * snack: whether individual ate snack (0: no, 1:yes)
        * dinner: whether individual ate dinner (0: no, 1:yes)
        * sleep: hours of sleep last night (0: 6-8 hours, 1: 8+ hours)
        * sunscreen: sunscreen usage (0: no, 1: yes)
        * hand_wash: hand hygiene habits (1: none, 2: poor, 3: average, 4: high, 5: optimal)

    Returns
    -------
    DataFrame
        Returns pandas DataFrame
    """
    df = pd.read_csv(resource_filename('zepid', 'datasets/case_control.dat'), delim_whitespace=True, index_col=False)
    return df


def load_monotone_missing_data():
    """Loads simulated data for monotone missing. This data is for demonstration with inverse probability of missing
    weights with data that is has a monotone missing pattern.

    Notes
    -----
    Variables included are:
        * id: id number
        * A: variable with no missing data
        * B: variable with missing data, conditional on A and L
        * C: variable with missing data, conditional on B and L
        * L: variable related to the missingness mechanism

    The above data is monotone missing with all individuals missing C are also missing B.

    True means in the observed data:
        * B: 0.4258
        * C: 0.5343

    Returns
    -------
    DataFrame
        Returns pandas DataFrame
    """
    df = pd.read_csv(resource_filename('zepid', 'datasets/monotone.dat'), index_col=False)
    return df[['id', 'A', 'B', 'C', 'L']]


def load_generalize_data(confounding):
    """Loads simulated data for demonstration of generalizability/transportability methods.

    Notes
    -----
    For the data, the true effect measures are
        Generalizability
            RD: 0.04651
            RR: 1.147
        Transportability
            RD: 0.02887
            RR: 1.085

    Parameters
    ----------
    confounding : bool
        Whether to load data with a confounder (True) or data from a hypothetical RCT (False). If the confounder is
        loaded, then the method used will need to account for confounding

    Returns
    -------
    DataFrame
        Returns pandas DataFrame
    """
    if confounding:
        df = pd.read_csv(resource_filename('zepid', 'datasets/generalize_conf.dat'), index_col=False)
    else:
        df = pd.read_csv(resource_filename('zepid', 'datasets/generalize_rct.dat'), index_col=False)
    df['id'] = df.index
    return df[['id', 'Y', 'A', 'S', 'L', 'W']]


def load_zivich_breskin_data():
    """Loads the singular simulation data set from Zivich PN & Breskin 2021.

    Notes
    -----------
    Variables within the dataset are
        * Y - outcome (atherosclerotic cardiovascular disease)
        * statin - treatment (1 is given statins; 0 is not)
        * age - Age
        * ldl_log - log-transformed LDL
        * diabetes - diabetes indicator
        * risk_score - calculated risk score between 0 and 1

    Returns
    ----------
    DataFrame
        Returns pandas DataFrame
    """
    return pd.read_csv(resource_filename('zepid', 'datasets/zivich_breskin_sim.csv'), index_col=False)
