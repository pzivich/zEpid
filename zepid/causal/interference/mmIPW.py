#########################################
# Mixed Effect Model IPW
#########################################
from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Below is for SM GLIMMIX
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import warnings
import patsy


# TODO have it work
# TODO have different estimators (important for households of various sizes Basse & Feller 2018)
# TODO validate with R data
# TODO variance estimation
# TODO diagnostics


class InterferenceIPW:
    """IPW for interference settings. Described in Tchetgen Tchetgen and VanderWeele 2012, and Perez-Heydrich et al.
    2014.
    """
    print('InterferenceIPW is not supported yet')

    def __init__(self, stabilized=True, coverage_comparison=None, estimator='individual', alpha=0.05):
        """stuff"""
        self._estimator = estimator
        self._stabilized = stabilized
        self._coverage_ref = coverage_comparison
        self.alpha = alpha

    def fit(self, df, idvar, group, treatment, outcome, model, allocations):
        df = df.copy()
        if self._coverage_ref is None:
            reference = np.mean(df[treatment])
        else:
            reference = self._coverage_ref

        # Setting up allocations as needed
        comparison_allocation_j = self._pi_calc(alpha=reference, exclude_j=False)
        comparison_allocation_n = self._pi_calc(alpha=reference, exclude_j=True)

        # Getting predicted probabilities as needed
        mm = BinomialBayesMixedGLM.from_formula(outcome+' ~ '+model,  # TODO once 0.9.0 released, switch this
                                                vc_formulas={'cluster': group}, data=df).fit()
        # mm = sm.BinomialBayesMixedGLM.from_formula(outcome+' ~ art + male',
        #                                           vc_formulas={'label': 'age0'}, data=self.df).fit_map()
        df['pred'] = mm.predict()

        # estimating each of the corresponding functions
        for a in allocations:
            print(a)

    def _pi_calc(self, alpha, exclude_j=False):
        """
        Calculates the part of the numerator for Y^ipw_i as designated by pi(A; alpha) in Perez-Heydrich et al. 2014

        For estimating Y(alpha)

            pi(A_i ; alpha) = cumprod_j=1(alpha^A * (1-alpha)^(1-A))

        For estimating Y(a;alpha) need to exclude j from the calculation such that;

            pi(A_i,-j ; alpha) = cumprod_k=1,k!=j(alpha^A * (1-alpha)^(1-A))

        This is done by using that same process as before, but now we divide by the individual level (to cancel out that
        cumulative product including individual j
        """
        pf = self.df.copy()
        pf['i_alpha'] = alpha ** (pf['treatment']) * (1 - alpha) ** (1 - pf['treatment'])
        pf.groupby(self.group)['i_alpha'].prod().reset_index()
        pf = pd.merge(pf, pf.groupby(self.group)['i_alpha'].prod().rename('g_alpha').reset_index(),
                      left_on=self.group, right_on=self.group)

        if exclude_j:
            pi = pf['g_alpha'] / pf['i_alpha']
        else:
            pi = pf['g_alpha']
        return pi

    def _yipw_calc(self, denom, a=None):
        """
        This calculates the corresponding Y^ipw for each potential combination. The version used depends on if 'a' is
        specified

        Y^ipw_i(a; alpha) = (sum(pi(A_i,-j; alpha) * I(A=a) * Y^ij) / Pr(A_i|X_i; phi) * n_i

        Y^ipw_i(alpha) = (sum(pi(A_i; alpha) * Y^ij) / Pr(A_i|X_i; phi) * n_i
        """
        ni = self.df.groupby(self.group)[self.id].count()
        if a is None:
            self.df['raw_numerator'] = self.df['pi2'] * self.df[self.ou]
            numer = self.df.groupby(self.group)['raw_numerator'].sum()
            yipw = numer / (denom * ni)
        else:
            self.df['raw_numerator'] = self.df['pi1'] * np.where(self.df[self.ex]==treat, 1, 0) * self.df[self.ou]
            numer = self.df.groupby(self.group)['raw_numerator'].sum()
            yipw = numer / (denom * ni)
        return yipw


#################################################
# Statsmodels GLIMMIX purely for my testing
# here until new statsmodels version released
#################################################
glw = [
    [0.2955242247147529, -0.1488743389816312],
    [0.2955242247147529, 0.1488743389816312],
    [0.2692667193099963, -0.4333953941292472],
    [0.2692667193099963, 0.4333953941292472],
    [0.2190863625159820, -0.6794095682990244],
    [0.2190863625159820, 0.6794095682990244],
    [0.1494513491505806, -0.8650633666889845],
    [0.1494513491505806, 0.8650633666889845],
    [0.0666713443086881, -0.9739065285171717],
    [0.0666713443086881, 0.9739065285171717],
]


class _BayesMixedGLM(base.Model):
    def __init__(self,
                 endog,
                 exog,
                 exog_vc=None,
                 ident=None,
                 family=None,
                 vcp_p=1,
                 fe_p=2,
                 fep_names=None,
                 vcp_names=None,
                 vc_names=None,
                 **kwargs):

        if len(ident) != exog_vc.shape[1]:
            msg = "len(ident) should match the number of columns of exog_vc"
            raise ValueError(msg)

        # Get the fixed effects parameter names
        if fep_names is None:
            if hasattr(exog, "columns"):
                fep_names = exog.columns.tolist()
            else:
                fep_names = ["FE_%d" % (k + 1) for k in range(exog.shape[1])]

        # Get the variance parameter names
        if vcp_names is None:
            vcp_names = ["VC_%d" % (k + 1) for k in range(int(max(ident)) + 1)]
        else:
            if len(vcp_names) != len(set(ident)):
                msg = "The lengths of vcp_names and ident should be the same"
                raise ValueError(msg)

        if not sparse.issparse(exog_vc):
            exog_vc = sparse.csr_matrix(exog_vc)

        ident = ident.astype(np.int)
        vcp_p = float(vcp_p)
        fe_p = float(fe_p)

        # Number of fixed effects parameters
        if exog is None:
            k_fep = 0
        else:
            k_fep = exog.shape[1]

        # Number of variance component structure parameters and
        # variance component realizations.
        if exog_vc is None:
            k_vc = 0
            k_vcp = 0
        else:
            k_vc = exog_vc.shape[1]
            k_vcp = max(ident) + 1

        # power would be better but not available in older scipy
        exog_vc2 = exog_vc.multiply(exog_vc)

        super(_BayesMixedGLM, self).__init__(endog, exog, **kwargs)

        self.exog_vc = exog_vc
        self.exog_vc2 = exog_vc2
        self.ident = ident
        self.family = family
        self.k_fep = k_fep
        self.k_vc = k_vc
        self.k_vcp = k_vcp
        self.fep_names = fep_names
        self.vcp_names = vcp_names
        self.vc_names = vc_names
        self.fe_p = fe_p
        self.vcp_p = vcp_p
        self.names = fep_names + vcp_names
        if vc_names is not None:
            self.names += vc_names

    def _unpack(self, vec):

        ii = 0

        # Fixed effects parameters
        fep = vec[:ii + self.k_fep]
        ii += self.k_fep

        # Variance component structure parameters (standard
        # deviations).  These are on the log scale.  The standard
        # deviation for random effect j is exp(vcp[ident[j]]).
        vcp = vec[ii:ii + self.k_vcp]
        ii += self.k_vcp

        # Random effect realizations
        vc = vec[ii:]

        return fep, vcp, vc

    def logposterior(self, params):
        """
        The overall log-density: log p(y, fe, vc, vcp).
        This differs by an additive constant from the log posterior
        log p(fe, vc, vcp | y).
        """

        fep, vcp, vc = self._unpack(params)

        # Contributions from p(y | x, vc)
        lp = 0
        if self.k_fep > 0:
            lp += np.dot(self.exog, fep)
        if self.k_vc > 0:
            lp += self.exog_vc.dot(vc)

        mu = self.family.link.inverse(lp)
        ll = self.family.loglike(self.endog, mu)

        if self.k_vc > 0:

            # Contributions from p(vc | vcp)
            vcp0 = vcp[self.ident]
            s = np.exp(vcp0)
            ll -= 0.5 * np.sum(vc**2 / s**2) + np.sum(vcp0)

            # Contributions from p(vc)
            ll -= 0.5 * np.sum(vcp**2 / self.vcp_p**2)

        # Contributions from p(fep)
        if self.k_fep > 0:
            ll -= 0.5 * np.sum(fep**2 / self.fe_p**2)

        return ll

    def logposterior_grad(self, params):
        """
        The gradient of the log posterior.
        """

        fep, vcp, vc = self._unpack(params)

        lp = 0
        if self.k_fep > 0:
            lp += np.dot(self.exog, fep)
        if self.k_vc > 0:
            lp += self.exog_vc.dot(vc)

        mu = self.family.link.inverse(lp)

        score_factor = (self.endog - mu) / self.family.link.deriv(mu)
        score_factor /= self.family.variance(mu)

        te = [None, None, None]

        # Contributions from p(y | x, z, vc)
        if self.k_fep > 0:
            te[0] = np.dot(score_factor, self.exog)
        if self.k_vc > 0:
            te[2] = self.exog_vc.transpose().dot(score_factor)

        if self.k_vc > 0:
            # Contributions from p(vc | vcp)
            # vcp0 = vcp[self.ident]
            # s = np.exp(vcp0)
            # ll -= 0.5 * np.sum(vc**2 / s**2) + np.sum(vcp0)
            vcp0 = vcp[self.ident]
            s = np.exp(vcp0)
            u = vc**2 / s**2 - 1
            te[1] = np.bincount(self.ident, weights=u)
            te[2] -= vc / s**2

            # Contributions from p(vcp)
            # ll -= 0.5 * np.sum(vcp**2 / self.vcp_p**2)
            te[1] -= vcp / self.vcp_p**2

        # Contributions from p(fep)
        if self.k_fep > 0:
            te[0] -= fep / self.fe_p**2

        te = [x for x in te if x is not None]

        return np.concatenate(te)

    def _get_start(self):
        start_fep = np.zeros(self.k_fep)
        start_vcp = np.ones(self.k_vcp)
        start_vc = np.random.normal(size=self.k_vc)
        start = np.concatenate((start_fep, start_vcp, start_vc))
        return start

    @classmethod
    def from_formula(cls,
                     formula,
                     vc_formulas,
                     data,
                     family=None,
                     vcp_p=1,
                     fe_p=2):
        """
        Fit a BayesMixedGLM using a formula.
        Parameters
        ----------
        formula : string
            Formula for the endog and fixed effects terms (use ~ to separate
            dependent and independent expressions).
        vc_formulas : dictionary
            vc_formulas[name] is a one-sided formula that creates one
            collection of random effects with a common variance
            prameter.  If using a categorical expression to produce
            variance components, note that generally `0 + ...` should
            be used so that an intercept is not included.
        data : data frame
            The data to which the formulas are applied.
        family : genmod.families instance
            A GLM family.
        vcp_p : float
            The prior standard deviation for the logarithms of the standard
            deviations of the random effects.
        fe_p : float
            The prior standard deviation for the fixed effects parameters.
        """

        ident = []
        exog_vc = []
        vcp_names = []
        j = 0
        for na, fml in vc_formulas.items():
            mat = patsy.dmatrix(fml, data, return_type='dataframe')
            exog_vc.append(mat)
            vcp_names.append(na)
            ident.append(j * np.ones(mat.shape[1]))
            j += 1
        exog_vc = pd.concat(exog_vc, axis=1)
        vc_names = exog_vc.columns.tolist()

        ident = np.concatenate(ident)

        model = super(_BayesMixedGLM, cls).from_formula(
            formula,
            data=data,
            family=family,
            subset=None,
            exog_vc=exog_vc,
            ident=ident,
            vc_names=vc_names,
            vcp_names=vcp_names,
            fe_p=fe_p,
            vcp_p=vcp_p)

        return model

    def fit(self, method="BFGS", minim_opts=None):
        """
        fit is equivalent to fit_map for this class.
        See fit_map for parameter information.
        """
        self.fit_map(method, minim_opts)

    def fit_map(self, method="BFGS", minim_opts=None, scale_fe=False):
        """
        Construct the Laplace approximation to the posterior
        distribution.
        Parameters
        ----------
        method : string
            Optimization method for finding the posterior mode.
        minim_opts : dict-like
            Options passed to scipy.minimize.
        scale_fe : bool
            If true, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.
        Returns
        -------
        BayesMixedGLMResults instance.
        """

        if scale_fe:
            mn = self.exog.mean(0)
            sc = self.exog.std(0)
            self._exog_save = self.exog
            self.exog = self.exog.copy()
            ixs = np.flatnonzero(sc > 1e-8)
            self.exog[:, ixs] -= mn[ixs]
            self.exog[:, ixs] /= sc[ixs]

        def fun(params):
            return -self.logposterior(params)

        def grad(params):
            return -self.logposterior_grad(params)

        start = self._get_start()

        r = minimize(fun, start, method=method, jac=grad, options=minim_opts)
        if not r.success:
            msg = ("Laplace fitting did not converge, |gradient|=%.6f" %
                   np.sqrt(np.sum(r.jac**2)))
            warnings.warn(msg)

        from statsmodels.tools.numdiff import approx_fprime
        hess = approx_fprime(r.x, grad)
        cov = np.linalg.inv(hess)

        params = r.x

        if scale_fe:
            self.exog = self._exog_save
            del self._exog_save
            params[ixs] /= sc[ixs]
            cov[ixs, :][:, ixs] /= np.outer(sc[ixs], sc[ixs])

        return BayesMixedGLMResults(self, params, cov, optim_retvals=r)

    def predict(self, params, exog=None, linear=False):
        """
        Return the fitted mean structure.
        Parameters
        ----------
        params : array-like
            The parameter vector, may be the full parameter vector, or may
            be truncated to include only the mean parameters.
        exog : array-like
            The design matrix for the mean structure.  If omitted, use the
            model's design matrix.
        linear : bool
            If True, return the linear predictor without passing through the
            link function.
        Result
        ------
        A 1-dimensional array of predicted values
        """

        if exog is None:
            exog = self.exog

        q = exog.shape[1]
        pr = np.dot(exog, params[0:q])

        if not linear:
            pr = self.family.link.inverse(pr)

        return pr


class _VariationalBayesMixedGLM(object):
    """
    A mixin providing generic (not family-specific) methods for
    variational Bayes mean field fitting.
    """

    # Integration range (from -rng to +rng).  The integrals are with
    # respect to a standard Gaussian distribution so (-5, 5) will be
    # sufficient in many cases.
    rng = 5

    verbose = False

    # Returns the mean and variance of the linear predictor under the
    # given distribution parameters.
    def _lp_stats(self, fep_mean, fep_sd, vc_mean, vc_sd):

        tm = np.dot(self.exog, fep_mean)
        tv = np.dot(self.exog**2, fep_sd**2)
        tm += self.exog_vc.dot(vc_mean)
        tv += self.exog_vc2.dot(vc_sd**2)

        return tm, tv

    def vb_elbo_base(self, h, tm, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd,
                     vc_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.
        This function calculates the family-specific ELBO function
        based on information provided from a subclass.
        Parameters
        ----------
        h : function mapping 1d vector to 1d vector
            The contribution of the model to the ELBO function can be
            expressed as y_i*lp_i + Eh_i(z), where y_i and lp_i are
            the response and linear predictor for observation i, and z
            is a standard normal rangom variable.  This formulation
            can be achieved for any GLM with a canonical link
            function.
        """

        # p(y | vc) contributions
        iv = 0
        for w in glw:
            z = self.rng * w[1]
            iv += w[0] * h(z) * np.exp(-z**2 / 2)
        iv /= np.sqrt(2 * np.pi)
        iv *= self.rng
        iv += self.endog * tm
        iv = iv.sum()

        # p(vc | vcp) * p(vcp) * p(fep) contributions
        iv += self._elbo_common(fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean,
                                vc_sd)

        r = (iv + np.sum(np.log(fep_sd)) + np.sum(np.log(vcp_sd)) + np.sum(
            np.log(vc_sd)))

        return r

    def vb_elbo_grad_base(self, h, tm, tv, fep_mean, vcp_mean, vc_mean, fep_sd,
                          vcp_sd, vc_sd):
        """
        Return the gradient of the ELBO function.
        See vb_elbo_base for parameters.
        """

        fep_mean_grad = 0.
        fep_sd_grad = 0.
        vcp_mean_grad = 0.
        vcp_sd_grad = 0.
        vc_mean_grad = 0.
        vc_sd_grad = 0.

        # p(y | vc) contributions
        for w in glw:
            z = self.rng * w[1]
            u = h(z) * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
            r = u / np.sqrt(tv)
            fep_mean_grad += w[0] * np.dot(u, self.exog)
            vc_mean_grad += w[0] * self.exog_vc.transpose().dot(u)
            fep_sd_grad += w[0] * z * np.dot(r, self.exog**2 * fep_sd)
            v = self.exog_vc2.multiply(vc_sd).transpose().dot(r)
            v = np.squeeze(np.asarray(v))
            vc_sd_grad += w[0] * z * v

        fep_mean_grad *= self.rng
        vc_mean_grad *= self.rng
        fep_sd_grad *= self.rng
        vc_sd_grad *= self.rng
        fep_mean_grad += np.dot(self.endog, self.exog)
        vc_mean_grad += self.exog_vc.transpose().dot(self.endog)

        (fep_mean_grad_i, fep_sd_grad_i, vcp_mean_grad_i, vcp_sd_grad_i,
         vc_mean_grad_i, vc_sd_grad_i) = self._elbo_grad_common(
             fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd)

        fep_mean_grad += fep_mean_grad_i
        fep_sd_grad += fep_sd_grad_i
        vcp_mean_grad += vcp_mean_grad_i
        vcp_sd_grad += vcp_sd_grad_i
        vc_mean_grad += vc_mean_grad_i
        vc_sd_grad += vc_sd_grad_i

        fep_sd_grad += 1 / fep_sd
        vcp_sd_grad += 1 / vcp_sd
        vc_sd_grad += 1 / vc_sd

        mean_grad = np.concatenate((fep_mean_grad, vcp_mean_grad,
                                    vc_mean_grad))
        sd_grad = np.concatenate((fep_sd_grad, vcp_sd_grad, vc_sd_grad))

        if self.verbose:
            print(
                "|G|=%f" % np.sqrt(np.sum(mean_grad**2) + np.sum(sd_grad**2)))

        return mean_grad, sd_grad

    def fit_vb(self,
               mean=None,
               sd=None,
               fit_method="BFGS",
               minim_opts=None,
               scale_fe=False,
               verbose=False):
        """
        Fit a model using the variational Bayes mean field approximation.
        Parameters
        ----------
        mean : array-like
            Starting value for VB mean vector
        sd : array-like
            Starting value for VB standard deviation vector
        fit_method : string
            Algorithm for scipy.minimize
        minim_opts : dict-like
            Options passed to scipy.minimize
        scale_fe : bool
            If true, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.
        verbose : bool
            If True, print the gradient norm to the screen each time
            it is calculated.
        Notes
        -----
        The goal is to find a factored Gaussian approximation
        q1*q2*...  to the posterior distribution, approximately
        minimizing the KL divergence from the factored approximation
        to the actual posterior.  The KL divergence, or ELBO function
        has the form
            E* log p(y, fe, vcp, vc) - E* log q
        where E* is expectation with respect to the product of qj.
        References
        ----------
        Blei, Kucukelbir, McAuliffe (2017).  Variational Inference: A
        review for Statisticians
        https://arxiv.org/pdf/1601.00670.pdf
        """

        self.verbose = verbose

        if scale_fe:
            mn = self.exog.mean(0)
            sc = self.exog.std(0)
            self._exog_save = self.exog
            self.exog = self.exog.copy()
            ixs = np.flatnonzero(sc > 1e-8)
            self.exog[:, ixs] -= mn[ixs]
            self.exog[:, ixs] /= sc[ixs]

        n = self.k_fep + self.k_vcp + self.k_vc
        ml = self.k_fep + self.k_vcp + self.k_vc
        if mean is None:
            m = np.zeros(n)
        else:
            if len(mean) != ml:
                raise ValueError(
                    "mean has incorrect length, %d != %d" % (len(mean), ml))
            m = mean.copy()
        if sd is None:
            s = -0.5 + 0.1 * np.random.normal(size=n)
        else:
            if len(sd) != ml:
                raise ValueError(
                    "sd has incorrect length, %d != %d" % (len(sd), ml))

            # s is parameterized on the log-scale internally when
            # optimizing the ELBO function (this is transparent to the
            # caller)
            s = np.log(sd)

        # Don't allow the variance parameter starting mean values to
        # be too small.
        i1, i2 = self.k_fep, self.k_fep + self.k_vcp
        m[i1:i2] = np.where(m[i1:i2] < -1, -1, m[i1:i2])

        # Don't allow the posterior standard deviation starting values
        # to be too small.
        s = np.where(s < -1, -1, s)

        def elbo(x):
            n = len(x) // 2
            return -self.vb_elbo(x[:n], np.exp(x[n:]))

        def elbo_grad(x):
            n = len(x) // 2
            gm, gs = self.vb_elbo_grad(x[:n], np.exp(x[n:]))
            gs *= np.exp(x[n:])
            return -np.concatenate((gm, gs))

        start = np.concatenate((m, s))
        mm = minimize(
            elbo, start, jac=elbo_grad, method=fit_method, options=minim_opts)
        if not mm.success:
            warnings.warn("VB fitting did not converge")

        n = len(mm.x) // 2
        params = mm.x[0:n]
        va = np.exp(2 * mm.x[n:])

        if scale_fe:
            self.exog = self._exog_save
            del self._exog_save
            params[ixs] /= sc[ixs]
            va[ixs] /= sc[ixs]**2

        return BayesMixedGLMResults(self, params, va, mm)

    # Handle terms in the ELBO that are common to all models.
    def _elbo_common(self, fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd):

        iv = 0

        # p(vc | vcp) contributions
        m = vcp_mean[self.ident]
        s = vcp_sd[self.ident]
        iv -= np.sum((vc_mean**2 + vc_sd**2) * np.exp(2 * (s**2 - m))) / 2
        iv -= np.sum(m)

        # p(vcp) contributions
        iv -= 0.5 * (vcp_mean**2 + vcp_sd**2).sum() / self.vcp_p**2

        # p(b) contributions
        iv -= 0.5 * (fep_mean**2 + fep_sd**2).sum() / self.fe_p**2

        return iv

    def _elbo_grad_common(self, fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean,
                          vc_sd):

        # p(vc | vcp) contributions
        m = vcp_mean[self.ident]
        s = vcp_sd[self.ident]
        u = vc_mean**2 + vc_sd**2
        ve = np.exp(2 * (s**2 - m))
        dm = u * ve - 1
        ds = -2 * u * ve * s
        vcp_mean_grad = np.bincount(self.ident, weights=dm)
        vcp_sd_grad = np.bincount(self.ident, weights=ds)

        vc_mean_grad = -vc_mean.copy() * ve
        vc_sd_grad = -vc_sd.copy() * ve

        # p(vcp) contributions
        vcp_mean_grad -= vcp_mean / self.vcp_p**2
        vcp_sd_grad -= vcp_sd / self.vcp_p**2

        # p(b) contributions
        fep_mean_grad = -fep_mean.copy() / self.fe_p**2
        fep_sd_grad = -fep_sd.copy() / self.fe_p**2

        return (fep_mean_grad, fep_sd_grad, vcp_mean_grad, vcp_sd_grad,
                vc_mean_grad, vc_sd_grad)


class BayesMixedGLMResults(object):
    """
    Attributes
    ----------
    fe_mean : array-like
        Posterior mean of the fixed effects coefficients.
    fe_sd : array-like
        Posterior standard deviation of the fixed effects coefficients
    vcp_mean : array-like
        Posterior mean of the logged variance component standard
        deviations.
    vcp_sd : array-like
        Posterior standard deviation of the logged variance component
        standard deviations.
    vc_mean : array-like
        Posterior mean of the random coefficients
    vc_sd : array-like
        Posterior standard deviation of the random coefficients
    """

    def __init__(self, model, params, cov_params, optim_retvals=None):

        self.model = model
        self.params = params
        self._cov_params = cov_params
        self.optim_retvals = optim_retvals

        self.fe_mean, self.vcp_mean, self.vc_mean = (model._unpack(params))

        if cov_params.ndim == 2:
            cp = np.diag(cov_params)
        else:
            cp = cov_params
        self.fe_sd, self.vcp_sd, self.vc_sd = model._unpack(cp)
        self.fe_sd = np.sqrt(self.fe_sd)
        self.vcp_sd = np.sqrt(self.vcp_sd)
        self.vc_sd = np.sqrt(self.vc_sd)

    def cov_params(self):

        if hasattr(self.model.data, "frame"):
            # Return the covariance matrix as a dataframe or series
            na = (self.model.fep_names + self.model.vcp_names +
                  self.model.vc_names)
            if self._cov_params.ndim == 2:
                return pd.DataFrame(self._cov_params, index=na, columns=na)
            else:
                return pd.Series(self._cov_params, index=na)

        # Return the covariance matrix as a ndarray
        return self._cov_params

    def summary(self):

        df = pd.DataFrame()
        m = self.model.k_fep + self.model.k_vcp
        df["Type"] = (["M" for k in range(self.model.k_fep)] +
                      ["V" for k in range(self.model.k_vcp)])

        df["Post. Mean"] = self.params[0:m]

        if self._cov_params.ndim == 2:
            v = np.diag(self._cov_params)[0:m]
            df["Post. SD"] = np.sqrt(v)
        else:
            df["Post. SD"] = np.sqrt(self._cov_params[0:m])

        # Convert variance parameters to natural scale
        df["SD"] = np.exp(df["Post. Mean"])
        df["SD (LB)"] = np.exp(df["Post. Mean"] - 2 * df["Post. SD"])
        df["SD (UB)"] = np.exp(df["Post. Mean"] + 2 * df["Post. SD"])
        df["SD"] = ["%.3f" % x for x in df.SD]
        df["SD (LB)"] = ["%.3f" % x for x in df["SD (LB)"]]
        df["SD (UB)"] = ["%.3f" % x for x in df["SD (UB)"]]
        df.loc[df.index < self.model.k_fep, "SD"] = ""
        df.loc[df.index < self.model.k_fep, "SD (LB)"] = ""
        df.loc[df.index < self.model.k_fep, "SD (UB)"] = ""

        df.index = self.model.fep_names + self.model.vcp_names

        summ = summary2.Summary()
        summ.add_title(self.model.family.__class__.__name__ +
                       " Mixed GLM Results")
        summ.add_df(df)

        summ.add_text("Parameter types are mean structure (M) and variance structure (V)")
        summ.add_text("Variance parameters are modeled as log standard deviations")

        return summ

    def random_effects(self, term=None):
        """
        Posterior mean and standard deviation of random effects.
        Parameters
        ----------
        term : int or None
            If None, results for all random effects are returned.  If
            an integer, returns results for a given set of random
            effects.  The value of `term` refers to an element of the
            `ident` vector, or to a position in the `vc_formulas`
            list.
        Returns
        -------
        Data frame of posterior means and posterior standard
        deviations of random effects.
        """

        z = self.vc_mean
        s = self.vc_sd
        na = self.model.vc_names

        if term is not None:
            termix = self.model.vcp_names.index(term)
            ii = np.flatnonzero(self.model.ident == termix)
            z = z[ii]
            s = s[ii]
            na = [na[i] for i in ii]

        x = pd.DataFrame({"Mean": z, "SD": s})

        if na is not None:
            x.index = na

        return x

    def predict(self, exog=None, linear=False):
        """
        Return predicted values for the mean structure.
        Parameters
        ----------
        exog : array-like
            The design matrix for the mean structure.  If None,
            use the model's design matrix.
        linear : bool
            If True, returns the linear predictor, otherwise
            transform the linear predictor using the link function.
        Returns
        -------
        A one-dimensional array of fitted values.
        """

        return self.model.predict(self.params, exog, linear)


class BinomialBayesMixedGLM(_VariationalBayesMixedGLM, _BayesMixedGLM):
    def __init__(self,
                 endog,
                 exog,
                 exog_vc,
                 ident,
                 vcp_p=1,
                 fe_p=2,
                 fep_names=None,
                 vcp_names=None,
                 vc_names=None):

        super(BinomialBayesMixedGLM, self).__init__(
            endog,
            exog,
            exog_vc=exog_vc,
            ident=ident,
            vcp_p=vcp_p,
            fe_p=fe_p,
            family=families.Binomial(),
            fep_names=fep_names,
            vcp_names=vcp_names,
            vc_names=vc_names)

    @classmethod
    def from_formula(cls, formula, vc_formulas, data, vcp_p=1, fe_p=2):

        fam = families.Binomial()
        x = _BayesMixedGLM.from_formula(
            formula, vc_formulas, data, family=fam, vcp_p=vcp_p, fe_p=fe_p)

        # Copy over to the intended class structure
        mod = BinomialBayesMixedGLM(
            x.endog,
            x.exog,
            exog_vc=x.exog_vc,
            ident=x.ident,
            vcp_p=x.vcp_p,
            fe_p=x.fe_p,
            fep_names=x.fep_names,
            vcp_names=x.vcp_names,
            vc_names=x.vc_names)
        mod.data = x.data

        return mod

    def vb_elbo(self, vb_mean, vb_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.
        """

        fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
        fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)
        tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

        def h(z):
            return -np.log(1 + np.exp(tm + np.sqrt(tv) * z))

        return self.vb_elbo_base(h, tm, fep_mean, vcp_mean, vc_mean, fep_sd,
                                 vcp_sd, vc_sd)

    def vb_elbo_grad(self, vb_mean, vb_sd):
        """
        Returns the gradient of the model's evidence lower bound (ELBO).
        """

        fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
        fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)
        tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

        def h(z):
            u = tm + np.sqrt(tv) * z
            x = np.zeros_like(u)
            ii = np.flatnonzero(u > 0)
            uu = u[ii]
            x[ii] = 1 / (1 + np.exp(-uu))
            ii = np.flatnonzero(u <= 0)
            uu = u[ii]
            x[ii] = np.exp(uu) / (1 + np.exp(uu))
            return -x

        return self.vb_elbo_grad_base(h, tm, tv, fep_mean, vcp_mean, vc_mean,
                                      fep_sd, vcp_sd, vc_sd)
