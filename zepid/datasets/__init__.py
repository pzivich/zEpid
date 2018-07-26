import numpy as np
import pandas as pd
from pkg_resources import resource_filename

def load_sample_data(timevary):
    '''Load data that is part of the zepid package. This data set comes
    from simulated data from Jessie Edwards (thanks Jess!). This data
    is used for all examples on zepid.readthedocs.io and part of my
    other repository Python-for-Epidemiologists

    timevary:
        True    -produces a repeated follow-up data set
        False   -produces a single observation per subject

    Variables:
    -id: participant unique ID (multiple observations per person)
    -enter: start of time period
    -out: end of time period
    -male: indicator variable for male (1 = yes)
    -age0: baseline age (at enter = 0)
    -cd40: baseline CD4 T cell count (at enter = 0)
    -dvl0: baseline viral load data (at enter = 0)
    -cd4: CD4 T cell count at that follow-up visit
    -dvl: viral load at that follow-up visit
    -art: indicator of whether ART (antiretroviral treatment) was
          prescribed at that time
    -drop: indicator of whether individual dropped out of the study (1 = yes)
    -dead: indicator for death at end of follow-up period (1 = yes)
    -t: total time contributed
    '''
    cols = ['id', 'enter', 'out', 'male', 'age0', 'cd40', 'dvl0', 'cd4', 'dvl', 'art', 'drop', 'dead']
    df = pd.read_csv(resource_filename('zepid', 'datasets/data.txt'),
                     delim_whitespace=True, header=None, names=cols, index_col=False)
    df.sort_values(by=['id', 'enter'], inplace=True)
    if timevary is True:
        return df
    else:
        dfi = df.loc[df.groupby('id').cumcount()==0][['id', 'male', 'age0', 'cd40', 'cd4', 'dvl0', 'art']].copy()
        dfo = df.loc[df.id != df.id.shift(-1)][['id', 'dead', 'drop', 'out']].copy()
        dfo.loc[dfo['drop'] == 1, 'dead'] = np.nan
        dff = pd.merge(dfi, dfo, left_on='id', right_on='id')
        dff.rename(columns={'out':'t'}, inplace=True)
        dff.drop(columns=['drop'], inplace=True)
        return dff


