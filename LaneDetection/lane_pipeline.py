from typing import Dict, Any, Optional
import time
from collections import deque
import numpy as np
import cv2

from .lane_geometry import BEVProjector
from .lane_controller import LaneController
from .lane_overlay import draw_overlay
from .common import meters_per_pixel, heading_deg_at_ratio
from .common import refine_mask01, apply_roi

# ==== scoring hyper params ====
WIN, W_STAB, W_HEAD, W_LAT, EPS = 15, 2.0, 0.1, 0.05, 1e-6

class LanePipeline:
    def __init__(
        self,
        backend,
        ratios=(0.98, 0.92, 0.82, 0.72),
        dy=30,
        lane_width_m=0.20,
        bottom_ratio=0.88,
        show_overlay=True,
        ema_alpha=0.30, 
    ):
        self.backend = backend
        self.ratios = list(ratios)
        self.draw_points = True
        self.dy = int(dy)
        self.lane_width_m = float(lane_width_m)
        self.bottom_ratio = float(bottom_ratio)
        self.show_overlay = bool(show_overlay)

        # ROI
        self.ROI_POLY = np.float32([[0.03,0.58],[0.97,0.58],[0.97,0.99],[0.03,0.99]])

        # BEV
        self.proj = BEVProjector()
        self.ctrl = LaneController()

        # History Buffer(multi ratio)
        self.hist_pos  = {r: deque(maxlen=WIN) for r in self.ratios}
        self.hist_head = {r: deque(maxlen=WIN) for r in self.ratios}
        self.hist_ok   = {r: deque(maxlen=WIN) for r in self.ratios}
        self.primary_idx = 0

        # EMA
        self.ema_alpha = float(ema_alpha)
        self.ema_pos   = None
        self.ema_head  = None
        
        # FPS Counter
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._fps_val = 0.0
        self._wh = None

    def name(self) -> str:
        return getattr(self.backend, "__class__", type("B", (), {})).__name__

    def _fps_update(self) -> float:
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t0 >= 1.0:
            self._fps_val = self._fps_n / (now - self._fps_t0)
            self._fps_t0 = now
            self._fps_n = 0
        return self._fps_val

    def _measure_with_history(self, bev01: np.ndarray) -> Dict[str, Any]:
        H, W = bev01.shape
        xm = meters_per_pixel(bev01, self.lane_width_m)

        poss_m, heads_deg, oks, centers_px = [], [], [], []
        for r in self.ratios:
            head_deg, x_px = heading_deg_at_ratio(bev01, r, dy_px=self.dy)
            pos_m = (W / 2.0 - x_px) * xm
            valid = 0.0 if np.isnan(x_px) else 1.0

            self.hist_pos[r].append(float(pos_m))
            self.hist_head[r].append(float(head_deg))
            self.hist_ok[r].append(valid)

            poss_m.append(pos_m)
            heads_deg.append(head_deg)
            centers_px.append(float(x_px))
            oks.append(valid)

        # Adaptive Selection
        best_score, best_i = None, self.primary_idx
        for i, r in enumerate(self.ratios):
            if len(self.hist_pos[r]) < max(5, WIN // 3):
                continue
            std_pos   = float(np.std(self.hist_pos[r]))
            mean_head = float(np.mean(self.hist_head[r]))
            coverage  = float(np.mean(self.hist_ok[r]))
            score = coverage*( W_STAB/(std_pos+EPS) - W_HEAD*abs(mean_head) ) - W_LAT*(1.0 - r)
            if best_score is None or score > best_score:
                best_score, best_i = score, i
        self.primary_idx = best_i

        pos_m    = float(poss_m[self.primary_idx])
        head_deg = float(heads_deg[self.primary_idx])
        r_star   = float(self.ratios[self.primary_idx])

        # lane_ok
        band = bev01[int(self.bottom_ratio * H):, :]
        cov = float(band.mean())
        xs = np.where(bev01[H - 1, :] > 0)[0]
        width_ok = (xs.size >= 2 and (xs[-1] - xs[0]) >= 14)
        lane_ok = 1 if (cov >= 0.02 and width_ok) else 0

        return {
            "pos_m": pos_m,
            "head_deg": head_deg,
            "r_star": r_star,
            "lane_ok": int(lane_ok),
            "centers_px": centers_px,
        }

    def step(self, frame_bgr, fps_hint: Optional[float] = None) -> Dict[str, Any]:
        # 1. PERCEPTION: Deep Learning Backend -> Semantic Mask
        mask01 = self.backend.infer_mask01(frame_bgr)
        # 2. MASK REFINEMENT: ROI/BEV & Morphology
        if mask01 is None or mask01.size == 0:
            bev01 = np.zeros(frame_bgr.shape[:2], np.uint8)
        else:
            mask01 = (mask01 > 0).astype(np.uint8)
            mask01 = apply_roi(mask01, self.ROI_POLY)
            mask01 = refine_mask01(mask01)
            bev01 = self.proj.warp(mask01)

        # 3. GEOMETRY: Multi-Ratio Measurement
        meas = self._measure_with_history(bev01)

        # 4. STABILIZATION: EMA Filter
        pos_raw, head_raw = meas["pos_m"], meas["head_deg"]
        if self.ema_pos is None:
            self.ema_pos  = pos_raw
            self.ema_head = head_raw
        else:
            a = self.ema_alpha
            self.ema_pos  = (1.0 - a) * self.ema_pos  + a * pos_raw
            self.ema_head = (1.0 - a) * self.ema_head + a * head_raw

        # 5) FPS
        fps = self._fps_update() if fps_hint is None else float(fps_hint)

        # 6) CONTROL: Adaptive Logic
        self.ctrl.set_fps(fps)
        self.ctrl.update_ema(self.ema_pos, self.ema_head)
        dir_u8, speed_u8, alpha_i8 = self.ctrl.decide(int(meas["lane_ok"]))

        out: Dict[str, Any] = {
            "dir": int(dir_u8),
            "speed": int(speed_u8),
            "alpha": int(alpha_i8),
            "ema_pos": float(self.ema_pos),
            "ema_head": float(self.ema_head),
            "raw_pos": float(pos_raw),
            "raw_head": float(head_raw),
            "r_star": float(meas["r_star"]),
            "lane_ok": int(meas["lane_ok"]),
            "fps": float(fps),
        }

        sign_head  = np.sign(out["ema_head"])
        sign_alpha = np.sign(out["alpha"])
        out["sign_check"] = 0 if (sign_head == 0 or sign_alpha == 0) else (1 if sign_head == sign_alpha else -1)
        # 7. VISUALIZATION
        if self.show_overlay:
            should_draw = getattr(self, "draw_points", True)
            draw = draw_overlay(
                frame_bgr, bev01, self.proj.M_inv,
                out["ema_pos"], out["ema_head"], out["fps"],
                ratios = self.ratios if should_draw else None,
                centers_px=meas["centers_px"],
                primary_idx=self.primary_idx
            )
            out["overlay"] = draw

        return out
