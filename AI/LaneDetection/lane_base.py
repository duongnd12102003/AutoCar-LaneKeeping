from typing import Dict, Any
import numpy as np

class BaseLaneModel:
    def step(self, frame_bgr: np.ndarray, want_overlay=True) -> Dict[str, Any]:
        raise NotImplementedError
    def name(self) -> str:
        raise NotImplementedError
