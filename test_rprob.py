import unittest
import numpy as np
from rprob import (
    rnorm, dnorm, pnorm, qnorm,
    runif, dunif, punif, qunif,
    rbinom, dbinom, pbinom, qbinom,
    rpois, dpois, ppois, qpois,
    rexp, dexp, pexp, qexp,
    rgamma, dgamma, pgamma, qgamma,
    rbeta, dbeta, pbeta, qbeta,
    rt, dt, pt, qt,
    rchisq, dchisq, pchisq, qchisq,
    rf, df_, pf, qf,
    rgeom, dgeom, pgeom, qgeom,
    rnbinom, dnbinom, pnbinom, qnbinom,
    rhyper, dhyper, phyper, qhyper,
    rweibull, dweibull, pweibull, qweibull,
    rlogis, dlogis, plogis, qlogis,
    rcauchy, dcauchy, pcauchy, qcauchy,
    rlaplace, dlaplace, plaplace, qlaplace,
    rlnorm, dlnorm, plnorm, qlnorm,
    rbern, dbern, pbern, qbern
)

class TestRProb(unittest.TestCase):

    def test_rnorm(self):
        result = rnorm(10, mean=0, sd=1)
        self.assertEqual(result.shape, (10,))

    def test_dnorm(self):
        result = dnorm([0, 1, 2], mean=0, sd=1)
        self.assertEqual(result.shape, (3,))

    def test_pnorm(self):
        result = pnorm([0, 1, 2], mean=0, sd=1)
        self.assertEqual(result.shape, (3,))

    def test_qnorm(self):
        result = qnorm([0.25, 0.5, 0.75], mean=0, sd=1)
        self.assertEqual(result.shape, (3,))

    def test_runif(self):
        result = runif(10, min=0, max=1)
        self.assertEqual(result.shape, (10,))

    def test_dunif(self):
        result = dunif([0.1, 0.5, 0.9], min=0, max=1)
        self.assertEqual(result.shape, (3,))

    def test_punif(self):
        result = punif([0.1, 0.5, 0.9], min=0, max=1)
        self.assertEqual(result.shape, (3,))

    def test_qunif(self):
        result = qunif([0.1, 0.5, 0.9], min=0, max=1)
        self.assertEqual(result.shape, (3,))

    def test_rbinom(self):
        result = rbinom(10, size=5, prob=0.5)
        self.assertEqual(result.shape, (10,))

    def test_dbinom(self):
        result = dbinom([0, 1, 2], size=5, prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_pbinom(self):
        result = pbinom([0, 1, 2], size=5, prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_qbinom(self):
        result = qbinom([0.1, 0.5, 0.9], size=5, prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_rpois(self):
        result = rpois(10, lam=3)
        self.assertEqual(result.shape, (10,))

    def test_dpois(self):
        result = dpois([0, 1, 2], lam=3)
        self.assertEqual(result.shape, (3,))

    def test_ppois(self):
        result = ppois([0, 1, 2], lam=3)
        self.assertEqual(result.shape, (3,))

    def test_qpois(self):
        result = qpois([0.1, 0.5, 0.9], lam=3)
        self.assertEqual(result.shape, (3,))

    def test_rexp(self):
        result = rexp(10, rate=1)
        self.assertEqual(result.shape, (10,))

    def test_dexp(self):
        result = dexp([0.1, 0.5, 0.9], rate=1)
        self.assertEqual(result.shape, (3,))

    def test_pexp(self):
        result = pexp([0.1, 0.5, 0.9], rate=1)
        self.assertEqual(result.shape, (3,))

    def test_qexp(self):
        result = qexp([0.1, 0.5, 0.9], rate=1)
        self.assertEqual(result.shape, (3,))

    def test_rgamma(self):
        result = rgamma(10, shape=2, scale=1)
        self.assertEqual(result.shape, (10,))

    def test_dgamma(self):
        result = dgamma([0.1, 0.5, 0.9], shape=2, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_pgamma(self):
        result = pgamma([0.1, 0.5, 0.9], shape=2, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_qgamma(self):
        result = qgamma([0.1, 0.5, 0.9], shape=2, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_rbeta(self):
        result = rbeta(10, shape1=2, shape2=2)
        self.assertEqual(result.shape, (10,))

    def test_dbeta(self):
        result = dbeta([0.1, 0.5, 0.9], shape1=2, shape2=2)
        self.assertEqual(result.shape, (3,))

    def test_pbeta(self):
        result = pbeta([0.1, 0.5, 0.9], shape1=2, shape2=2)
        self.assertEqual(result.shape, (3,))

    def test_qbeta(self):
        result = qbeta([0.1, 0.5, 0.9], shape1=2, shape2=2)
        self.assertEqual(result.shape, (3,))

    def test_rt(self):
        result = rt(10, df=5)
        self.assertEqual(result.shape, (10,))

    def test_dt(self):
        result = dt([0.1, 0.5, 0.9], df=5)
        self.assertEqual(result.shape, (3,))

    def test_pt(self):
        result = pt([0.1, 0.5, 0.9], df=5)
        self.assertEqual(result.shape, (3,))

    def test_qt(self):
        result = qt([0.1, 0.5, 0.9], df=5)
        self.assertEqual(result.shape, (3,))

    def test_rchisq(self):
        result = rchisq(10, df=5)
        self.assertEqual(result.shape, (10,))

    def test_dchisq(self):
        result = dchisq([0.1, 0.5, 0.9], df=5)
        self.assertEqual(result.shape, (3,))

    def test_pchisq(self):
        result = pchisq([0.1, 0.5, 0.9], df=5)
        self.assertEqual(result.shape, (3,))

    def test_qchisq(self):
        result = qchisq([0.1, 0.5, 0.9], df=5)
        self.assertEqual(result.shape, (3,))

    def test_rf(self):
        result = rf(10, df1=5, df2=2)
        self.assertEqual(result.shape, (10,))

    def test_df_(self):
        result = df_([0.1, 0.5, 0.9], df1=5, df2=2)
        self.assertEqual(result.shape, (3,))

    def test_pf(self):
        result = pf([0.1, 0.5, 0.9], df1=5, df2=2)
        self.assertEqual(result.shape, (3,))

    def test_qf(self):
        result = qf([0.1, 0.5, 0.9], df1=5, df2=2)
        self.assertEqual(result.shape, (3,))

    def test_rgeom(self):
        result = rgeom(10, prob=0.5)
        self.assertEqual(result.shape, (10,))

    def test_dgeom(self):
        result = dgeom([0, 1, 2], prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_pgeom(self):
        result = pgeom([0, 1, 2], prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_qgeom(self):
        result = qgeom([0.1, 0.5, 0.9], prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_rnbinom(self):
        result = rnbinom(10, size=5, prob=0.5)
        self.assertEqual(result.shape, (10,))

    def test_dnbinom(self):
        result = dnbinom([0, 1, 2], size=5, prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_pnbinom(self):
        result = pnbinom([0, 1, 2], size=5, prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_qnbinom(self):
        result = qnbinom([0.1, 0.5, 0.9], size=5, prob=0.5)
        self.assertEqual(result.shape, (3,))

    def test_rhyper(self):
        result = rhyper(10, m=5, n=5, k=5)
        self.assertEqual(result.shape, (10,))

    def test_dhyper(self):
        result = dhyper([0, 1, 2], m=5, n=5, k=5)
        self.assertEqual(result.shape, (3,))

    def test_phyper(self):
        result = phyper([0, 1, 2], m=5, n=5, k=5)
        self.assertEqual(result.shape, (3,))

    def test_qhyper(self):
        result = qhyper([0.1, 0.5, 0.9], m=5, n=5, k=5)
        self.assertEqual(result.shape, (3,))

    def test_rweibull(self):
        result = rweibull(10, shape=2, scale=1)
        self.assertEqual(result.shape, (10,))

    def test_dweibull(self):
        result = dweibull([0.1, 0.5, 0.9], shape=2, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_pweibull(self):
        result = pweibull([0.1, 0.5, 0.9], shape=2, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_qweibull(self):
        result = qweibull([0.1, 0.5, 0.9], shape=2, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_rlogis(self):
        result = rlogis(10, location=0, scale=1)
        self.assertEqual(result.shape, (10,))

    def test_dlogis(self):
        result = dlogis([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_plogis(self):
        result = plogis([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_qlogis(self):
        result = qlogis([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_rcauchy(self):
        result = rcauchy(10, location=0, scale=1)
        self.assertEqual(result.shape, (10,))

    def test_dcauchy(self):
        result = dcauchy([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_pcauchy(self):
        result = pcauchy([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_qcauchy(self):
        result = qcauchy([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_rlaplace(self):
        result = rlaplace(10, location=0, scale=1)
        self.assertEqual(result.shape, (10,))

    def test_dlaplace(self):
        result = dlaplace([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_plaplace(self):
        result = plaplace([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_qlaplace(self):
        result = qlaplace([0.1, 0.5, 0.9], location=0, scale=1)
        self.assertEqual(result.shape, (3,))

    def test_rlnorm(self):
        result = rlnorm(10, meanlog=0, sdlog=1)
        self.assertEqual(result.shape, (10,))

    def test_dlnorm(self):
        result = dlnorm([0.1, 0.5, 0.9], meanlog=0, sdlog=1)
        self.assertEqual(result.shape, (3,))

    def test_plnorm(self):
        result = plnorm([0.1, 0.5, 0.9], meanlog=0, sdlog=1)
        self.assertEqual(result.shape, (3,))

    def test_qlnorm(self):
        result = qlnorm([0.1, 0.5, 0.9], meanlog=0, sdlog=1)
        self.assertEqual(result.shape, (3,))

    def test_rbern(self):
        result = rbern(10, prob=0.5)
        self.assertEqual(result.shape, (10,))

    def test_dbern(self):
        result = dbern([0, 1], prob=0.5)
        self.assertEqual(result.shape, (2,))

    def test_pbern(self):
        result = pbern([0, 1], prob=0.5)
        self.assertEqual(result.shape, (2,))

    def test_qbern(self):
        result = qbern([0.1, 0.5, 0.9], prob=0.5)
        self.assertEqual(result.shape, (3,))

if __name__ == '__main__':
    unittest.main()
