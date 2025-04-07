from .Donut_Detailer import NODE_CLASS_MAPPINGS as m1
from .Donut_Detailer_2 import NODE_CLASS_MAPPINGS as m2
from .Donut_Detailer_4 import NODE_CLASS_MAPPINGS as m3
from .Donut_Detailer_LoRA_6 import NODE_CLASS_MAPPINGS as m4

NODE_CLASS_MAPPINGS = {**m1, **m2, **m3, **m4}
__all__ = ["NODE_CLASS_MAPPINGS"]
