from functools import partialmethod

from pandas.core.indexing import convert_from_missing_indexer_tuple
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = ["uncondExact2x2", "uncondExact2x2DF"]

import pandas as pd
from typing import Dict, Optional


def uncondExact2x2(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    parmtype: str = "difference",
    nullparm: Optional[float] = None,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    method: str = "FisherAdj",
    tsmethod: str = "central",
    midp: bool = False,
    gamma: float = 0.0,
    EplusM: bool = False,
    tiebreak: bool = False,
    conf_int:bool = False
) -> Dict:
    """
          x1: number of events in group 1

          n1: sample size in group 1

          x2: number of events in group 2

          n2: sample size in group 2

    parmtype: type of parameter of interest, one of "difference", "ratio"
              or "oddsratio" (see details)

    nullparm: value of the parameter of interest at null hypothesis, NULL
              defaults to 0 for parmtype='difference' and 1 for
              parmtype='ratio' or 'oddsratio'

    alternative: alternative hypothesis, one of "two.sided", "less", or
              "greater", default is "two.sided" (see details)

    conf.int: logical, calculate confidence interval?

    conf.level: confidence level

      method: method type, one of "FisherAdj" (default), "simple",
              "simpleTB", "wald-pooled", "wald-unpooled", "score", "user",
              or "user-fixed" (see details)

    tsmethod: two-sided method, either "central" or "square" (see details)

        midp: logical. Use mid-p-value method?

       gamma: Beger-Boos adjustment parameter. 0 means no adjustment. (see
              details).

      EplusM: logical, do the E+M adjustment? (see details)

    tiebreak: logical, do tiebreak adjustment? (see details)


    Details:

         The ‘uncondExact2x2’ function gives unconditional exact tests and
         confidence intervals for two independent binomial observations.
         The ‘uncondExact2x2Pvals’ function repeatedly calls
         ‘uncondExact2x2’ to get the p-values for the entire sample space.

         Let X1 be binomial(n1,theta1) and X2 be binomial(n2,theta2). The
         parmtype determines the parameter of interest: `difference' is
         theta2 - theta1, 'ratio' is theta2/theta1, and `oddsratio' is
         (theta2*(1-theta1))/(theta1*(1-theta2)).

         The options ‘method’, ‘parmtype’, ‘tsmethod’, ‘alternative’,
         ‘EplusM’, and ‘tiebreak’ define some built-in test statistic
         function, Tstat, that is used to order the sample space, using
         ‘pickTstat’ and ‘calcTall’. The first 5 arguments of Tstat must be
         ‘Tstat(X1,N1,X2,N2, delta0)’, where X1 and X2 must allow vectors,
         and delta0 is the null parameter value (but delta0 does not need
         to be used in the ordering).  Ordering when ‘parmtype="ratio"’ or
         ‘parmtype="oddsratio"’ is only used when there is information
         about the parameter. So the ordering function value is not used
         for ordering when x1=0 and x2=0 for ‘parmtype="ratio"’, and it is
         not used when (x1=0 and x2=0) or (x1=n1 and x2=n2) for
         ‘parmtype="oddsratio"’.

         We describe the ordering functions first for the basic case, the
         case when ‘tsmethod="central"’ or ‘alternative!="two.sided"’,
         ‘EplusM=FALSE’, and ‘tiebreak=FALSE’. In this basic case the
         ordering function, Tstat, is determined by ‘method’ and
         ‘parmtype’:

            • method='simple' - Tstat essentially replaces theta1 with
              x1/n1 and theta2 with x2/n2 in the parameter definition. If
              parmtype=`difference' then ‘Tstat(X1,N1,X2,N2,delta0)’
              returns ‘X2/N2-X1/N1-delta0’. If parmtype='ratio' then the
              Tstat function returns ‘log(X2/N2) - log(X1/N1) -
              log(delta0)’. If parmtype='oddsratio' we get ‘log(
              X2*(N1-X1)/(delta0*X1*(N2-X2)))’.

            • method='wald-pooled' - Tstat is a Z statistic on the
              difference using the pooled variance (not allowed if
              ‘parmtype!="difference"’)

            • method='wald-unpooled' - Tstat is a Z statistics on the
              difference using unpooled variance (not allowed if
              ‘parmtype!="difference"’)

            • method='score' - Tstat is a Z statistic formed using score
              statistics, where the parameter is defined by parmtype, and
              the constrained maximum likelihood estimates of the parameter
              are calculated by ‘constrMLE.difference’, ‘constrMLE.ratio’,
              or ‘constrMLE.oddsratio’.

            • method='FisherAdj' - Tstat is a one-sided Fisher's 'exact'
              mid p-value. The mid p-value is an adjustment for ties that
              technically removes the 'exactness' of the Fisher's
              p-value...BUT, here we are only using it to order the sample
              space, so the results of the resulting unconditional test
              will still be exact.

         In the basic case, if ‘alternative="two.sided"’, the argument
         ‘tsmethod’="central" gives the two-sided central method. The
         p-value is just twice the minimum of the one-sided p-values (or 1
         if the doubling is greater than 1).

         Now consider cases other than the basic case.  The
         ‘tsmethod="square"’ option gives the square of the test statistic
         (when method="simple", "score", "wald-pooled", or "wald-unpooled")
         and larger values suggest rejection in either direction (unless
         method='user', then the user supplies any test statistic for which
         larger values suggest rejection).

         The ‘tiebreak=TRUE’ option breaks ties in a reasonable way when
         ‘method="simple"’ (see 'details' section of ‘calcTall’). The
         ‘EplusM=TRUE’ option performs Lloyd's (2008) E+M ordering on Tstat
         (see 'details' section of ‘calcTall’).

         If ‘tiebreak=TRUE’ and ‘EplusM=TRUE’, the tiebreak calculations
         are always done first.

         Berger and Boos (1994) developed a very general method for
         calculating p-values when a nuisance parameter is present. First,
         calculate a (1-gamma) confidence interval for the nuisance
         parameter, check for the supremum over the union of the null
         hypothesis parameter space and that confidence interval, then add
         back gamma to the p-value. This adjustment is valid (in other
         words, applied to exact tests it still gives an adjustment that is
         exact). The Berger-Boos adjustment is applied when ‘gamma’>0.

         When method='simple' or method='user-fixed' does a simple grid
         search algorithm using ‘unirootGrid’. No checks are done on the
         Tstat function when method='user-fixed' to make sure the simple
         grid search will converge to the proper answer. So
         method='user-fixed' should be used by advanced users only.

         When ‘midp=TRUE’ the mid p-value is calculated (and the associated
         confidence interval if ‘conf.int=TRUE’) instead of the standard
         p-value. Loosely speaking, the standard p-value calculates the
         probability of observing equal or more extreme responses, while
         the mid p-value calculates the probability of more extreme
         responses plus 1/2 the probability of equally extreme responses.
         The tests and confidence intervals when ‘midp=TRUE’ are not exact,
         but give type I error rates and coverage of confidence intervals
         closer to the nominal values. The mid p-value was studied by
         Lancaster (1961), see vignette on mid p-values for details.

    Value:

         The function ‘uncondExact2x2Pvals’ returns a (n1+1) by (n2+1)
         matrix of p-values for all possible x1 and x2 values, while
         ‘uncondExact2x2’ returns a list of class 'htest' with elements:

    statistic: proportion in sample 1

    parameter: proportion in sample 2

     p.value: p-value from test

    conf.int: confidence interval on parameter given by parmtype

    estimate: MLE estimate of parameter given by parmtype

    null.value: null hypothesis value of parameter given by parmtype

    alternative: alternative hypothesis

      method: description of test

    data.name: description of data

    Warning:

         The algorithm for calculating the p-values and confidence
         intervals is based on a series of grid searches. Because the grid
         searches are often trying to optimize non-monotonic functions, the
         algorithm is not guaranteed to give the correct answer. At the
         cost of increasing computation time, better accuracy can be
         obtained by increasing control$nPgrid, and less often by
         increasing control$nCIgrid.

    Author(s):

         Michael P. Fay, Sally A. Hunsberger

    References:

         Berger, R. L. and Boos, D. D. (1994). P values maximized over a
         confidence set for the nuisance parameter. Journal of the American
         Statistical Association 89 1012-1016.

         Lancaster, H.O. (1961). Significance tests in discrete
         distributions. JASA 56: 223-234.

         Lloyd, C. J. (2008). Exact p-values for discrete models obtained
         by estimation and maximization. Australian & New Zealand Journal
         of Statistics 50 329-345.

    See Also:

         See ‘boschloo’ for unconditional exact tests with ordering
         function based on Fisher's exact p-values.
    """
    from logging import warning

    assert x1 <= n1
    assert x2 <= n2
    if nullparm is None:
        if parmtype == "difference":
            nullparm = 0.0
        else:
            nullparm = 1.0

    if method != "simple" and tiebreak:
        warning("Ignoring tiebreak, since %s != simple", method)
        tiebreak = False
    from rpy2.robjects.packages import importr

    
    exact2x2 = importr("exact2x2")

    res = exact2x2.uncondExact2x2(
        x1,
        n1,
        x2,
        n2,
        parmtype,
        nullparm,
        alternative,
        conf_int,
        conf_level,
        method,
        tsmethod,
        midp,
        gamma,
        EplusM,
        tiebreak,
    )

    res_d = {}
    for k, v in res.items():
        res_d[k] = v[0]
    return res_d


def boschloo(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    alternative: str = "two.sided",
    OR: float = 1.0,
    conf_int: bool = False,
    conf_level: float = 0.95,
    midp=False,
    tsmethod="central",
):
    from rpy2.robjects.packages import importr

    conf_int = False
    exact2x2 = importr("exact2x2")

    res = exact2x2.boschloo(
        x1, n1, x2, n2, alternative, OR, conf_int, conf_level, midp, tsmethod
    )

    res_d = {}
    for k, v in res.items():
        res_d[k] = v[0]
    return res_d


def uncondExact2x2DF(df: pd.DataFrame, **kwargs) -> pd.Series:
    from rpy2.robjects.packages import importr

    assert df.shape == (2, 2), "Input dataframe must be of shape 2x2"
    exact2x2 = importr("exact2x2")
    c1 = int(df.iloc[0, 0])
    c2 = int(df.iloc[1, 0])
    n1, n2 = [int(x) for x in df.sum(axis=1)]

    res_d = uncondExact2x2(c1, n1, c2, n2, **kwargs)

    return pd.Series(res_d)
