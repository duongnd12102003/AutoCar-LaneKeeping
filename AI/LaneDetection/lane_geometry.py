import numpy as np
import cv2

class BEVProjector:
    def __init__(self):
        self.SRC_RATIOS = ((0.20, 0.58), (0.10, 0.90), (0.90, 0.90), (0.80, 0.58))
        self.DST_RATIOS = ((0.25, 0.00), (0.25, 1.00), (0.75, 1.00), (0.75, 0.00))
        self.M = None
        self.M_inv = None
        self._wh = None

    def _ensure_H(self, W, H):
        if self._wh == (W, H) and self.M is not None:
            return
        src = np.float32([(x*W, y*H) for x, y in self.SRC_RATIOS])
        dst = np.float32([(x*W, y*H) for x, y in self.DST_RATIOS])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        self._wh = (W, H)

    def warp(self, mask01: np.ndarray) -> np.ndarray:
        H, W = mask01.shape
        self._ensure_H(W, H)
        bev = cv2.warpPerspective((mask01.astype(np.uint8)*255), self.M, (W, H), flags=cv2.INTER_NEAREST)
        return (bev > 0).astype(np.uint8)
    