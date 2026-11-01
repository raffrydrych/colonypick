"""colonypick: morphology-guided colony picking that maximises the expected number of distinct taxa.

Built on CAMII (Huang et al., Nat Biotechnol 2023, https://github.com/hym0405/CAMII).
"""
from .detection import DetectionConfig, detect_plate, detect_folder, FEATURES
from .model import ColonyModel
from .picking import expected_novelty_order, camii_maxmin_select

__version__ = "0.1.0"
__all__ = ["DetectionConfig", "detect_plate", "detect_folder", "FEATURES", "ColonyModel",
           "expected_novelty_order", "camii_maxmin_select", "__version__"]
