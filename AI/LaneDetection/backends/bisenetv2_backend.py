import cv2
import numpy as np
import torch
from .bisenet_impl.model import BiSeNetV2Lane

class BiseNetV2Backend:
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        input_h: int = 256,
        input_w: int = 256,
        num_classes: int = 2,
    ):
        # 1. Device Configuration
        if device in ("cpu", "cuda", "cuda:0", "cuda:1"):
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_classes = int(num_classes)

        # 2. Load Model Architecture & Weights
        print(f"[BiSeNetV2] Loading weights: {weights}...")
        self.model = BiSeNetV2Lane(num_classes=self.num_classes)
        
        try:
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"[BiSeNetV2] Failed to load weights: {e}")

        self.model.to(self.device).eval()

        # 3. CUDA Optimization
        if self.device.type == 'cuda':
            try:
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # 4. Warmup Strategy
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, self.input_h, self.input_w, device=self.device)
                _ = self.model(dummy)
        except Exception as e:
            print(f"[BiSeNetV2] Warmup warning: {e}")

    def infer_mask01(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((1, 1), np.uint8)

        h0, w0 = frame_bgr.shape[:2]
        # Preprocessing: Resize & Normalize
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(
            img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        x = img_res.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        # Inference
        with torch.inference_mode():
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        mask_small = (pred == 1).astype(np.uint8)
        mask_full = cv2.resize(
            mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        return mask_full

    def name(self) -> str:
        return "BiSeNetV2Backend"