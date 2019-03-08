import numpy as np
import pandas as pd


class IPSW:
    def __init__(self, df_sample, df_target, exposure, outcome, generalize=True, weights=None):
        """Calculate inverse probability of sampling weights through logistic regression.
        """
        self.sample = df_sample.copy()
        self.target = df_target.copy()

        self.generalize = generalize  # important for weight construction
        # IPSW for generalizability
        # IOSW for transportability

        self.a = exposure
        self.d = outcome
        self.weight = weights
