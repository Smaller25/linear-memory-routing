"""Public paper-faithful SSC API.

The previous file mixed DSC's previous-chunk router and recurrent re-scan into
Memory Caching.  Those operations are not in the MC paper.  Import the common
SSC core here and use the explicit GDN adapter in ``dsc.mc_gdn1`` or
``dsc.mc_gdn2``.
"""

from .mc_ssc import MCSSC, SSCOutput, SparseSelectiveCaching

__all__ = ["MCSSC", "SSCOutput", "SparseSelectiveCaching"]
