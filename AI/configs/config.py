from pathlib import Path

# ===== UDP / NETWORK =====
LISTEN_IP   = "192.168.1.114"
LISTEN_PORT = 3000
ESP_IP      = "192.168.1.167"
ESP_PORT    = 3001
END_MARKER  = b"\xFF\xD9"

BUFFER_SIZE = 2048
RBUF_BYTES  = 8 * 1024 * 1024
NO_PACKET_TIMEOUT_S = 5.0
NO_FRAME_TIMEOUT_S  = 4.0
MAX_ACCUM_BYTES     = 3 * 1024 * 1024

# ===== PATHS =====
ROOT = Path(__file__).resolve().parents[2]
LOG_DIR    = ROOT / "AI" / "logs"
CALIB_PATH = ROOT / "AI" / "camera_intrinsics.npz"

# "yolov8" | "pidnet" | "twinlite" | "bisenet" |
LANE_MODEL = "pidnet"
LANE_WEIGHTS = ROOT / "AI" / "LaneDetection" / "Lane_weight" / {
    "yolov8":  "Yolo_v8/best.pt",
    "pidnet":  "PIDNet/best.pt",
    "twinlite":"TwinLite/best.pth",
    "bisenet":"BiseNet/best.pth"
}[LANE_MODEL.lower()]

# ===== RUNTIME =====
DEVICE = "0"
SHOW   = True
WIN_NAME = "ACE LANE"
WIN_W, WIN_H = 1280, 720

# ===== YOLOv8 =====
IMGSZ = 640
CONF  = 0.18

# ===== PIDNet =====
PIDNET_H   = 320
PIDNET_W   = 416
PIDNET_THR = 0.50
PIDNET_ARCH = "pidnet_small"

# ===== TwinLiteNet =====
TWIN_H   = 320
TWIN_W   = 416
TWIN_THR = 0.50
TWIN_NUM_CLASSES = 2 

# ===== BiseNetV2 =====
BISENET_H = 256
BISENET_W = 256
BISENET_NUM_CLASSES = 2
