import cv2
import numpy as np
import torch
import inspect

def _build_twinlite(num_classes):
    from .twinlite_impl.TwinLite import TwinLiteNet
    sig = inspect.signature(TwinLiteNet)
    if "num_classes" in sig.parameters:
        return TwinLiteNet(num_classes=num_classes)
    return TwinLiteNet()

def _pick_lane_output(out):
    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            return out[1] if out[1] is not None else out[0]
        return out[0]
    if isinstance(out, dict):
        if 'll' in out and out['ll'] is not None: return out['ll']
        if 'da' in out and out['da'] is not None: return out['da']
    return out

class TwinLiteBackend:
    def __init__(self, weights: str, device="0", input_h=320, input_w=416, thr=0.5, num_classes=2):
        use_cuda = (torch.cuda.is_available() and str(device) != "cpu")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.in_h, self.in_w = int(input_h), int(input_w)
        self.thr = float(thr)

        print(f"[TwinLite] Loading model on {self.device}...")
        self.model = _build_twinlite(num_classes).to(self.device).eval()
        try:
            sd = torch.load(str(weights), map_location="cpu")
            sd = sd.get("state_dict", sd)
            model_sd = self.model.state_dict()
            
            fixed = {}
            for k, v in sd.items():
                kk = k[7:] if k.startswith("module.") else k
                if kk in model_sd and model_sd[kk].shape == v.shape:
                    fixed[kk] = v
            
            missing, unexpected = self.model.load_state_dict(fixed, strict=False)
            print(f"[TwinLite] Loaded. Matching layers: {len(fixed)}")
        except Exception as e:
            print(f"[TwinLite] Warning: Weights loading issue: {e}")

        # CUDA Optimization
        if use_cuda:
            try:
                torch.backends.cudnn.benchmark = True
            except: pass

        # Warmup
        try:
            with torch.inference_mode():
                d = torch.zeros(1, 3, self.in_h, self.in_w, device=self.device)
                _ = self.model(d)
        except: pass

    def name(self) -> str:
        return "TwinLiteNet"

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        
        x = cv2.resize(bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0)
        return x.to(self.device, non_blocking=True)

    def infer_mask01(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((480, 640), np.uint8)

        H, W = frame_bgr.shape[:2]
        with torch.inference_mode():
            x = self._preprocess(frame_bgr)
            out = self.model(x)
            out = _pick_lane_output(out)
            
            if out.shape[1] == 1:
                prob = torch.sigmoid(out[:, 0])
            else:
                prob = torch.softmax(out, dim=1)[:, 1]
                
            m = (prob[0].detach().cpu().numpy() > self.thr).astype(np.uint8)
            
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)