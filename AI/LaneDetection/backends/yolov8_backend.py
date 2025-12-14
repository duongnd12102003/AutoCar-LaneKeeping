import numpy as np
import cv2
import torch
from ultralytics import YOLO

class YoloV8Backend:
    def __init__(self, weights: str, device="0", imgsz=640, conf=0.18):
        print(f"[YOLOv8] Initializing from {weights}...")
        self.model = YOLO(str(weights))
        self.device = str(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        self.has_cuda = torch.cuda.is_available()
        self.fp16 = bool(self.has_cuda and self.device != "cpu")

        if self.has_cuda and self.device != "cpu":
            self.model.to("cuda")
            try:
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        try:
            print("[YOLOv8] Fusing layers for faster inference...")
            self.model.fuse()
        except Exception:
            pass

        # 3. Warmup with FP16 check
        try:
            print("[YOLOv8] Warming up...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self.model.predict(
                    dummy, imgsz=self.imgsz, conf=self.conf,
                    device=self.device, verbose=False, half=self.fp16
                )
        except Exception as e:
            print(f"[YOLOv8] Warmup warning: {e}")

    def infer_mask01(self, frame_bgr: np.ndarray) -> np.ndarray:
        H, W = frame_bgr.shape[:2]
        
        with torch.inference_mode():
            res = self.model.predict(
                frame_bgr, imgsz=self.imgsz, conf=self.conf,
                device=self.device, verbose=False, half=self.fp16
            )

        result = res[0]
        if result.masks is None or result.masks.data is None:
            return np.zeros((H, W), np.uint8)
        m_np = result.masks.data.detach().cpu().numpy().astype(np.uint8)
        
        if m_np.shape[0] == 0:
             return np.zeros((H, W), np.uint8)
        merged = np.zeros((H, W), dtype=np.uint8)
        
        for k in range(m_np.shape[0]):
            mk = cv2.resize(m_np[k], (W, H), interpolation=cv2.INTER_NEAREST)
            merged = cv2.bitwise_or(merged, mk)

        return merged

    def name(self) -> str:
        return "YOLOv8Backend"