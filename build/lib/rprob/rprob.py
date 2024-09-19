# rprob.py

import numpy as np
from scipy.stats import (
    norm, uniform, binom, poisson, expon, gamma, beta, t, chi2, f, geom,
    nbinom, hypergeom, weibull_min, logistic, cauchy, laplace, lognorm,
    multinomial, multivariate_normal
)
from typing import Union, List

# Normal distribution functions
def rnorm(n: int, mean: float = 0, sd: float = 1) -> np.ndarray:
    return norm.rvs(loc=mean, scale=sd, size=n)

def dnorm(x: Union[float, List[float], np.ndarray], mean: float = 0, sd: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return norm.pdf(x, loc=mean, scale=sd)

def pnorm(q: Union[float, List[float], np.ndarray], mean: float = 0, sd: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = norm.cdf(q, loc=mean, scale=sd)
    return cdf if lower_tail else 1 - cdf

def qnorm(p: Union[float, List[float], np.ndarray], mean: float = 0, sd: float = 1, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return norm.ppf(p, loc=mean, scale=sd)

# Uniform distribution functions
def runif(n: int, min: float = 0, max: float = 1) -> np.ndarray:
    return uniform.rvs(loc=min, scale=max - min, size=n)

def dunif(x: Union[float, List[float], np.ndarray], min: float = 0, max: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return uniform.pdf(x, loc=min, scale=max - min)

def punif(q: Union[float, List[float], np.ndarray], min: float = 0, max: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = uniform.cdf(q, loc=min, scale=max - min)
    return cdf if lower_tail else 1 - cdf

def qunif(p: Union[float, List[float], np.ndarray], min: float = 0, max: float = 1, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return uniform.ppf(p, loc=min, scale=max - min)

# Binomial distribution functions
def rbinom(n: int, size: int, prob: float) -> np.ndarray:
    return binom.rvs(n=size, p=prob, size=n)

def dbinom(x: Union[int, List[int], np.ndarray], size: int, prob: float) -> np.ndarray:
    x = np.asarray(x)
    return binom.pmf(x, n=size, p=prob)

def pbinom(q: Union[int, List[int], np.ndarray], size: int, prob: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = binom.cdf(q, n=size, p=prob)
    return cdf if lower_tail else 1 - cdf

def qbinom(p: Union[float, List[float], np.ndarray], size: int, prob: float, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return binom.ppf(p, n=size, p=prob)

# Poisson distribution functions
def rpois(n: int, lam: float) -> np.ndarray:
    return poisson.rvs(mu=lam, size=n)

def dpois(x: Union[int, List[int], np.ndarray], lam: float) -> np.ndarray:
    x = np.asarray(x)
    return poisson.pmf(x, mu=lam)

def ppois(q: Union[int, List[int], np.ndarray], lam: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = poisson.cdf(q, mu=lam)
    return cdf if lower_tail else 1 - cdf

def qpois(p: Union[float, List[float], np.ndarray], lam: float, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return poisson.ppf(p, mu=lam)

# Exponential distribution functions
def rexp(n: int, rate: float = 1) -> np.ndarray:
    scale = 1 / rate
    return expon.rvs(scale=scale, size=n)

def dexp(x: Union[float, List[float], np.ndarray], rate: float = 1) -> np.ndarray:
    x = np.asarray(x)
    scale = 1 / rate
    return expon.pdf(x, scale=scale)

def pexp(q: Union[float, List[float], np.ndarray], rate: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    scale = 1 / rate
    cdf = expon.cdf(q, scale=scale)
    return cdf if lower_tail else 1 - cdf

def qexp(p: Union[float, List[float], np.ndarray], rate: float = 1, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    scale = 1 / rate
    return expon.ppf(p, scale=scale)

# Gamma distribution functions
def rgamma(n: int, shape: float, scale: float = 1) -> np.ndarray:
    return gamma.rvs(a=shape, scale=scale, size=n)

def dgamma(x: Union[float, List[float], np.ndarray], shape: float, scale: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return gamma.pdf(x, a=shape, scale=scale)

def pgamma(q: Union[float, List[float], np.ndarray], shape: float, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = gamma.cdf(q, a=shape, scale=scale)
    return cdf if lower_tail else 1 - cdf

def qgamma(p: Union[float, List[float], np.ndarray], shape: float, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return gamma.ppf(p, a=shape, scale=scale)

# Beta distribution functions
def rbeta(n: int, shape1: float, shape2: float) -> np.ndarray:
    return beta.rvs(a=shape1, b=shape2, size=n)

def dbeta(x: Union[float, List[float], np.ndarray], shape1: float, shape2: float) -> np.ndarray:
    x = np.asarray(x)
    return beta.pdf(x, a=shape1, b=shape2)

def pbeta(q: Union[float, List[float], np.ndarray], shape1: float, shape2: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = beta.cdf(q, a=shape1, b=shape2)
    return cdf if lower_tail else 1 - cdf

def qbeta(p: Union[float, List[float], np.ndarray], shape1: float, shape2: float, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return beta.ppf(p, a=shape1, b=shape2)

# t-distribution functions
def rt(n: int, df: float) -> np.ndarray:
    return t.rvs(df=df, size=n)

def dt(x: Union[float, List[float], np.ndarray], df: float) -> np.ndarray:
    x = np.asarray(x)
    return t.pdf(x, df=df)

def pt(q: Union[float, List[float], np.ndarray], df: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = t.cdf(q, df=df)
    return cdf if lower_tail else 1 - cdf

def qt(p: Union[float, List[float], np.ndarray], df: float, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return t.ppf(p, df=df)

# Chi-squared distribution functions
def rchisq(n: int, df: float) -> np.ndarray:
    return chi2.rvs(df=df, size=n)

def dchisq(x: Union[float, List[float], np.ndarray], df: float) -> np.ndarray:
    x = np.asarray(x)
    return chi2.pdf(x, df=df)

def pchisq(q: Union[float, List[float], np.ndarray], df: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = chi2.cdf(q, df=df)
    return cdf if lower_tail else 1 - cdf

def qchisq(p: Union[float, List[float], np.ndarray], df: float, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return chi2.ppf(p, df=df)

# F-distribution functions
def rf(n: int, df1: float, df2: float) -> np.ndarray:
    return f.rvs(dfn=df1, dfd=df2, size=n)

def df_(x: Union[float, List[float], np.ndarray], df1: float, df2: float) -> np.ndarray:  # df는 예약어이므로 df_로 사용
    x = np.asarray(x)
    return f.pdf(x, dfn=df1, dfd=df2)

def pf(q: Union[float, List[float], np.ndarray], df1: float, df2: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = f.cdf(q, dfn=df1, dfd=df2)
    return cdf if lower_tail else 1 - cdf

def qf(p: Union[float, List[float], np.ndarray], df1: float, df2: float, lower_tail: bool = True) -> np.ndarray:
    p = np.asarray(p)
    if not lower_tail:
        p = 1 - p
    return f.ppf(p, dfn=df1, dfd=df2)

# Geometric distribution functions
def rgeom(n: int, prob: float) -> np.ndarray:
    return geom.rvs(p=prob, size=n) - 1  # R과 인덱스 차이 보정

def dgeom(x: Union[int, List[int], np.ndarray], prob: float) -> np.ndarray:
    x = np.asarray(x) + 1  # R과 인덱스 차이 보정
    return geom.pmf(x, p=prob)

def pgeom(q: Union[int, List[int], np.ndarray], prob: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q) + 1  # R과 인덱스 차이 보정
    cdf = geom.cdf(q, p=prob)
    return cdf if lower_tail else 1 - cdf

def qgeom(p: Union[float, List[float], np.ndarray], prob: float, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    quantile = geom.ppf(p, p=prob)
    return quantile - 1  # R과 인덱스 차이 보정

# Negative binomial distribution functions
def rnbinom(n: int, size: int, prob: float) -> np.ndarray:
    return nbinom.rvs(n=size, p=prob, size=n)

def dnbinom(x: Union[int, List[int], np.ndarray], size: int, prob: float) -> np.ndarray:
    x = np.asarray(x)
    return nbinom.pmf(x, n=size, p=prob)

def pnbinom(q: Union[int, List[int], np.ndarray], size: int, prob: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = nbinom.cdf(q, n=size, p=prob)
    return cdf if lower_tail else 1 - cdf

def qnbinom(p: Union[float, List[float], np.ndarray], size: int, prob: float, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return nbinom.ppf(p, n=size, p=prob)

# Hypergeometric distribution functions
def rhyper(nn: int, m: int, n: int, k: int) -> np.ndarray:
    return hypergeom.rvs(M=m + n, n=m, N=k, size=nn)

def dhyper(x: Union[int, List[int], np.ndarray], m: int, n: int, k: int) -> np.ndarray:
    x = np.asarray(x)
    return hypergeom.pmf(x, M=m + n, n=m, N=k)

def phyper(q: Union[int, List[int], np.ndarray], m: int, n: int, k: int, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = hypergeom.cdf(q, M=m + n, n=m, N=k)
    return cdf if lower_tail else 1 - cdf

def qhyper(p: Union[float, List[float], np.ndarray], m: int, n: int, k: int, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return hypergeom.ppf(p, M=m + n, n=m, N=k)

# Weibull distribution functions
def rweibull(n: int, shape: float, scale: float = 1) -> np.ndarray:
    return weibull_min.rvs(c=shape, scale=scale, size=n)

def dweibull(x: Union[float, List[float], np.ndarray], shape: float, scale: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return weibull_min.pdf(x, c=shape, scale=scale)

def pweibull(q: Union[float, List[float], np.ndarray], shape: float, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = weibull_min.cdf(q, c=shape, scale=scale)
    return cdf if lower_tail else 1 - cdf

def qweibull(p: Union[float, List[float], np.ndarray], shape: float, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return weibull_min.ppf(p, c=shape, scale=scale)

# Logistic distribution functions
def rlogis(n: int, location: float = 0, scale: float = 1) -> np.ndarray:
    return logistic.rvs(loc=location, scale=scale, size=n)

def dlogis(x: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return logistic.pdf(x, loc=location, scale=scale)

def plogis(q: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = logistic.cdf(q, loc=location, scale=scale)
    return cdf if lower_tail else 1 - cdf

def qlogis(p: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return logistic.ppf(p, loc=location, scale=scale)

# Cauchy distribution functions
def rcauchy(n: int, location: float = 0, scale: float = 1) -> np.ndarray:
    return cauchy.rvs(loc=location, scale=scale, size=n)

def dcauchy(x: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return cauchy.pdf(x, loc=location, scale=scale)

def pcauchy(q: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = cauchy.cdf(q, loc=location, scale=scale)
    return cdf if lower_tail else 1 - cdf

def qcauchy(p: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return cauchy.ppf(p, loc=location, scale=scale)

# Laplace distribution functions (Double exponential distribution)
def rlaplace(n: int, location: float = 0, scale: float = 1) -> np.ndarray:
    return laplace.rvs(loc=location, scale=scale, size=n)

def dlaplace(x: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return laplace.pdf(x, loc=location, scale=scale)

def plaplace(q: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = laplace.cdf(q, loc=location, scale=scale)
    return cdf if lower_tail else 1 - cdf

def qlaplace(p: Union[float, List[float], np.ndarray], location: float = 0, scale: float = 1, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return laplace.ppf(p, loc=location, scale=scale)

# Log-normal distribution functions
def rlnorm(n: int, meanlog: float = 0, sdlog: float = 1) -> np.ndarray:
    return lognorm.rvs(s=sdlog, scale=np.exp(meanlog), size=n)

def dlnorm(x: Union[float, List[float], np.ndarray], meanlog: float = 0, sdlog: float = 1) -> np.ndarray:
    x = np.asarray(x)
    return lognorm.pdf(x, s=sdlog, scale=np.exp(meanlog))

def plnorm(q: Union[float, List[float], np.ndarray], meanlog: float = 0, sdlog: float = 1, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = lognorm.cdf(q, s=sdlog, scale=np.exp(meanlog))
    return cdf if lower_tail else 1 - cdf

def qlnorm(p: Union[float, List[float], np.ndarray], meanlog: float = 0, sdlog: float = 1, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return lognorm.ppf(p, s=sdlog, scale=np.exp(meanlog))

# Multinomial distribution functions
def rmultinom(n: int, size: int, prob: List[float]) -> np.ndarray:
    return multinomial.rvs(n=size, p=prob, size=n)

def dmultinom(x: Union[List[int], np.ndarray], size: int, prob: List[float]) -> np.ndarray:
    x = np.asarray(x)
    return multinomial.pmf(x, n=size, p=prob)

# Multivariate normal distribution functions
def rmvnorm(n: int, mean: List[float], cov: np.ndarray) -> np.ndarray:
    return multivariate_normal.rvs(mean=mean, cov=cov, size=n)

def dmvnorm(x: np.ndarray, mean: List[float], cov: np.ndarray) -> np.ndarray:
    return multivariate_normal.pdf(x, mean=mean, cov=cov)

# Bernoulli distribution functions
def rbern(n: int, prob: float) -> np.ndarray:
    return bernoulli.rvs(p=prob, size=n)

def dbern(x: Union[int, List[int], np.ndarray], prob: float) -> np.ndarray:
    x = np.asarray(x)
    return bernoulli.pmf(x, p=prob)

def pbern(q: Union[int, List[int], np.ndarray], prob: float, lower_tail: bool = True) -> np.ndarray:
    q = np.asarray(q)
    cdf = bernoulli.cdf(q, p=prob)
    return cdf if lower_tail else 1 - cdf

def qbern(p: Union[float, List[float], np.ndarray], prob: float, lower_tail: bool = True) -> np.ndarray:
    if not lower_tail:
        p = 1 - p
    return bernoulli.ppf(p, p=prob)


