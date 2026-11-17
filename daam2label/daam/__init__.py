from ._version import __version__
from .experiment import *
from .heatmap import *
from .utils import *
try:
    from .hook import *
    from .trace import *
except ImportError:
    # diffusers >= 0.19 removed the legacy CrossAttention API that hook/trace
    # monkey-patch (SD 1.4/1.5 path). On newer diffusers use daam.trace_xl (SDXL).
    pass
