# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import pytest
from logging import info
from sys import float_info
import pyrexact2x2
from hypothesis import given, strategies as st,settings,Verbosity,example
from scipy.stats import fisher_exact

MAX_OBSERVED_N = 30

@st.composite
def sub_pairs(draw, max_values=100,min_values=2):
    "Return pair of int:s (x,n) where 0 <= x <= 3 <= n < max_values"
    n = draw(st.integers(min_value=min_values, max_value=max_values))
    x = draw(st.integers(min_value=0, max_value=n))
    return x, n

@st.composite
def alpha_gamma(draw, min_alpha=0.9,max_alpha=1.0-float_info.epsilon):
    """Returns conf_level and gamma

    Args: draw ([type]): Hypothesis draw object min_alpha (float, optional): Minimum confidence
        level. Defaults to 0.9. max_alpha (float, optional): Maximum confidence level. Defaults to
        1.0-float_info.epsilon.

    Returns: tuple: (alpha,gamma) with alpha between min_alpha and max_alpha and gamma between zero
        and 1-alpha with resolution of float epsilon.
    """

    #"Return pair of floats:s (alphax,n) where 0 <= x <= 3 <= n < max_values"
    alpha = draw(st.floats(min_value=min_alpha, max_value=max_alpha))

    max_gamma = 1-alpha - 2*float_info.epsilon
    if max_gamma <= float_info.epsilon:
        gamma=0.0
    else:
        gamma_factor = draw(st.integers(min_value=0, max_value=int(max_gamma/float_info.epsilon)))
        gamma = gamma_factor * float_info.epsilon
    return alpha,gamma 

@st.composite
def alt_tsmehod(draw):
    "return (alternative,tsmethod) that are compatile "
    alternative = draw(st.sampled_from(["two.sided", "less", "greater"]))
    if alternative == "two.sided":
        #tsmethod = draw(st.sampled_from(["central", "square"]))
        tsmethod = "central"
    else:
        tsmethod = "central"
    return alternative, tsmethod


@pytest.mark.skip(reason="Way too slow")
@settings(max_examples=20,deadline=500,verbosity=Verbosity.verbose)
@given(
    xn1=sub_pairs(MAX_OBSERVED_N),
    xn2=sub_pairs(MAX_OBSERVED_N),
    paramtype=st.sampled_from(["difference", "ratio", "oddsratio"]),
    nullparm=st.one_of(st.none(), st.floats(min_value=-1.0, max_value=1.0)),
    alternative_tsmethod=alt_tsmehod(),
    alpha_gamma=alpha_gamma(min_alpha=0.9),
    method=st.sampled_from(
        ["FisherAdj", "simple", "wald-pooled", "wald-unpooled", "score"]
    ),
    midp=st.booleans(),
    EplusM=st.booleans(),
    tiebreak=st.booleans(),
)
def test_uncondExact2x2(
    xn1,
    xn2,
    paramtype,
    nullparm,
    alternative_tsmethod,
    alpha_gamma,
    method,
    midp,
    EplusM,
    tiebreak,
):
    
    
    x1, n1 = xn1
    x2, n2 = xn2
    alternative, tsmethod = alternative_tsmethod
    conf_level,gamma = alpha_gamma

    ret= pyrexact2x2.uncondExact2x2(
        x1=x1,
        n1=n1,
        x2=x2,
        n2=n2,
        parmtype=paramtype,
        nullparm=nullparm,
        alternative=alternative,
        conf_level=conf_level,
        method=method,
        tsmethod=tsmethod,
        midp=midp,
        gamma=gamma,
        EplusM=EplusM,
        tiebreak=tiebreak,
    )
    fisher_or,fisher_pvalue = fisher_exact([[x1,n1-x1],[x2,n2-x2]],
            alternative=alternative.replace(".","-"))
    

    assert ret["p.value"] > 0
    assert ret["p.value"] <= 1 


@settings(max_examples=20,deadline=1000)
@given(
    xn1=sub_pairs(MAX_OBSERVED_N),
    xn2=sub_pairs(MAX_OBSERVED_N),
    paramtype=st.sampled_from([ "ratio", "oddsratio"]),
    nullparm=st.one_of(st.none(), st.floats(min_value=float_info.epsilon, max_value=10.0)),
    alternative_tsmethod=alt_tsmehod(),
    alpha_gamma=alpha_gamma(min_alpha=0.9),
    method=st.sampled_from(
        ["FisherAdj", "simple", "wald-pooled", "wald-unpooled", "score"]
    ),
    midp=st.booleans(),
    EplusM=st.booleans(),
    tiebreak=st.booleans(),
)
def test_uncondExact2x2_ratios(
    xn1,
    xn2,
    paramtype,
    nullparm,
    alternative_tsmethod,
    alpha_gamma,
    method,
    midp,
    EplusM,
    tiebreak,
):
    
    
    x1, n1 = xn1
    x2, n2 = xn2
    alternative, tsmethod = alternative_tsmethod
    conf_level,gamma = alpha_gamma

    ret= pyrexact2x2.uncondExact2x2(
        x1=x1,
        n1=n1,
        x2=x2,
        n2=n2,
        parmtype=paramtype,
        nullparm=nullparm,
        alternative=alternative,
        conf_level=conf_level,
        method=method,
        tsmethod=tsmethod,
        midp=midp,
        gamma=gamma,
        EplusM=EplusM,
        tiebreak=tiebreak,
    )
    fisher_or,fisher_pvalue = fisher_exact([[x1,n1-x1],[x2,n2-x2]],
            alternative=alternative.replace(".","-"))
    
    # The below doesn't hold for all configurations.
    #assert paramtype!="oddsratio" or alternative!="two.sided" or  ret["p.value"] <= fisher_pvalue,"Unconditional test should be more powerful"
    assert ret["p.value"] > 0
    assert ret["p.value"] <= 1 
    


@settings(max_examples=20,deadline=1000)
@given(
    xn1=sub_pairs(MAX_OBSERVED_N),
    xn2=sub_pairs(MAX_OBSERVED_N),
    nullparm=st.one_of(st.none(), st.floats(min_value=-1+float_info.epsilon, max_value=1-float_info.epsilon)),
    alternative_tsmethod=alt_tsmehod(),
    alpha_gamma=alpha_gamma(min_alpha=0.9),
    method=st.sampled_from(
        ["FisherAdj", "simple", "wald-pooled", "wald-unpooled", "score"]
    ),
    midp=st.booleans(),
    EplusM=st.booleans(),
    tiebreak=st.booleans(),
)
def test_uncondExact2x2_difference(
    xn1,
    xn2,
    nullparm,
    alternative_tsmethod,
    alpha_gamma,
    method,
    midp,
    EplusM,
    tiebreak,
):
    
    
    x1, n1 = xn1
    x2, n2 = xn2
    alternative, tsmethod = alternative_tsmethod
    conf_level,gamma = alpha_gamma

    ret= pyrexact2x2.uncondExact2x2(
        x1=x1,
        n1=n1,
        x2=x2,
        n2=n2,
        parmtype="difference",
        nullparm=nullparm,
        alternative=alternative,
        conf_level=conf_level,
        method=method,
        tsmethod=tsmethod,
        midp=midp,
        gamma=gamma,
        EplusM=EplusM,
        tiebreak=tiebreak,
    )
    fisher_or,fisher_pvalue = fisher_exact([[x1,n1-x1],[x2,n2-x2]],
            alternative=alternative.replace(".","-"))
    
    assert ret["p.value"] > 0
    assert ret["p.value"] <= 1 
    



@settings(max_examples=20,deadline=1000)
@given(
    xn1=sub_pairs(MAX_OBSERVED_N),
    xn2=sub_pairs(MAX_OBSERVED_N),
    paramtype=st.sampled_from(["difference", "ratio", "oddsratio"]),
    alternative_tsmethod=alt_tsmehod(),
    conf_level=st.floats(min_value=0.9, max_value=1.0 - 1e-5),
    method=st.sampled_from(
        ["FisherAdj", "simple", "wald-pooled", "wald-unpooled", "score"]
    ),
    midp=st.booleans()
)
@example(xn1=(1,6),xn2=(1,5),paramtype="difference",alternative_tsmethod=("two.sided","central"),
    conf_level=0.9,method="FisherAdj",midp=True)
def test_uncondExact2x2_alt1(
    xn1,
    xn2,
    paramtype,
    alternative_tsmethod,
    conf_level,
    method,
    midp
):
    
    
    x1, n1 = xn1
    x2, n2 = xn2
    alternative, tsmethod = alternative_tsmethod
    
    ret= pyrexact2x2.uncondExact2x2(
        x1=x1,
        n1=n1,
        x2=x2,
        n2=n2,
        parmtype=paramtype,
        alternative=alternative,
        conf_level=conf_level,
        method=method,
        tsmethod=tsmethod,
        midp=midp,

    )
    fisher_or,fisher_pvalue = fisher_exact([[x1,n1-x1],[x2,n2-x2]],
            alternative=alternative.replace(".","-"))
    
    assert paramtype!="oddsratio" or alternative!="two.sided" or  ret["p.value"] <= fisher_pvalue,"Unconditional test should be more powerful "+str(ret)
    assert ret["p.value"] > 0
    assert ret["p.value"] <= 1



   
@settings(deadline=2000)
@given(
    xn1=sub_pairs(40),
    xn2=sub_pairs(40)
)
@example(xn1=(1, 5), xn2=(0, 6))
def test_boschloo(
    xn1,
    xn2
):
    x1, n1 = xn1
    x2, n2 = xn2

    
    ret= pyrexact2x2.boschloo(
        x1=x1,
        n1=n1,
        x2=x2,
        n2=n2,
        tsmethod="minlike",midp=False
    )
    fisher_or,fisher_pvalue = fisher_exact([[x1,n1-x1],[x2,n2-x2]])
    
    assert  ret["p.value"] <= fisher_pvalue,"Unconditional test should be more powerful "+str(ret)
    assert ret["p.value"] > 0
    

def test_uncondExact2x2DF():
    import pandas as pd
    
    df = pd.DataFrame(
        [[28, 99], [17, 78]], columns=["colA", "colB"], index=["rowA", "rowB"]
    )
    ret = pyrexact2x2.uncondExact2x2DF(df=df, parmtype="odds")
    assert ret["p.value"] < 0.48 and ret["p.value"] > 0.47
    assert ret["p.value"]<= fisher_exact(df)[1]
    assert ret["p.value"] > 0
    return ret


if __name__ == "__main__":
    r = test_uncondExact2x2DF() 
    print(r)