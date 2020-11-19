from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = ["uncondExact2x2"]

from collections import namedtuple
import pandas as pd

UncondResult = namedtuple("UncondResult",("difference","p1","p2","pvalue"))
def uncondExact2x2(df:pd.DataFrame) -> UncondResult:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    from rpy2.robjects.conversion import localconverter
    
    exact2x2 = importr('exact2x2')
    base = importr('base')

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(df)
        
        res = exact2x2.uncondExact2x2(r_from_pd_df)
        r_summary = base.summary(res)
        pd_from_r_df = ro.conversion.rpy2py(r_summary)

    return pd_from_r_df
