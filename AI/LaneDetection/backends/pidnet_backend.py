import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .pidnet_impl.pidnet import get_pred_model

class PIDNetBackend:
    def __init__(self, weights: str, device="0", input_h=320, input_w=416, thr=0.50, arch="pidnet_small"):
        use_cuda = (torch.cuda.is_available() and str(device) != "cpu")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.input_size = (int(input_h), int(input_w))
        self.thr = float(thr)
        self.arch = str(arch)

        # 1. Build & Load Model
        print(f"[PIDNet] Loading architecture: {self.arch}...")
        self.model = get_pred_model(self.arch, num_classes=2).to(self.device).eval()

        try:
            ckpt = torch.load(str(weights), map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            fixed = {}
            for k, v in ckpt.items():
                kk = k[7:] if k.startswith("module.") else (k[6:] if k.startswith("model.") else k)
                fixed[kk] = v
            
            missing, unexpected = self.model.load_state_dict(fixed, strict=False)
            print(f"[PIDNet] Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            raise RuntimeError(f"[PIDNet] Failed to load weights: {e}")

        # 2. CUDA Optimization
        if use_cuda:
            try:
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # 3. Warmup
        try:
            with torch.inference_mode():
                h, w = self.input_size
                dummy = torch.zeros(1, 3, h, w, device=self.device)
                _ = self.model(dummy)
        except Exception:
            pass

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        h, w = self.input_size
        img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0)
        return x.to(self.device, non_blocking=True)

    def infer_mask01(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None: return np.zeros((1,1), dtype=np.uint8)
        H, W = frame_bgr.shape[:2]
        
        with torch.inference_mode():
            x = self._preprocess(frame_bgr)
            logits = self.model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            
            h_in, w_in = self.input_size
            logits = F.interpolate(logits, size=(h_in, w_in), mode="bilinear", align_corners=True)
            if logits.shape[1] == 1:
                prob = torch.sigmoid(logits[:, 0])
            else:
                prob = torch.softmax(logits, dim=1)[:, 1]
            
            mask = (prob[0].detach().cpu().numpy() > self.thr).astype(np.uint8)
            
        return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    def name(self) -> str:
        return f"PIDNet_{self.arch}"