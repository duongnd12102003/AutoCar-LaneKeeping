import time
import os
import cv2
import numpy as np
from typing import Optional
from configs import config as C
from utils.udp import make_rx, make_tx
from utils.calib import load_undistort_map
from utils.logger import BenchmarkLogger
from utils.frame_receiver import FrameReceiver
from LaneDetection.lane_pipeline import LanePipeline
from LaneDetection.backends.yolov8_backend import YoloV8Backend
from LaneDetection.backends.pidnet_backend import PIDNetBackend
from LaneDetection.backends.twinlite_backend import TwinLiteBackend
from LaneDetection.backends.bisenetv2_backend import BiseNetV2Backend

def build_lane_pipeline() -> LanePipeline:
    m = C.LANE_MODEL.lower()
    print(f"[System] Model {m.upper()}")

    if m == "yolov8":
        backend = YoloV8Backend(
            weights=C.LANE_WEIGHTS,
            device=C.DEVICE,
            imgsz=C.IMGSZ,
            conf=C.CONF
        )

    elif m == "pidnet":
        backend = PIDNetBackend(
            weights=C.LANE_WEIGHTS,
            device=C.DEVICE,
            input_h=C.PIDNET_H,
            input_w=C.PIDNET_W,
            thr=C.PIDNET_THR,
            arch=C.PIDNET_ARCH
        )

    elif m == "twinlite":
        backend = TwinLiteBackend(
            weights=C.LANE_WEIGHTS,
            device=C.DEVICE,
            input_h=C.TWIN_H,
            input_w=C.TWIN_W,
            thr=C.TWIN_THR,
            num_classes=C.TWIN_NUM_CLASSES
        )

    elif m == "bisenet":
        backend = BiseNetV2Backend(
            weights=C.LANE_WEIGHTS,
            device=C.DEVICE,
            input_h=C.BISENET_H,
            input_w=C.BISENET_W,
            num_classes=C.BISENET_NUM_CLASSES
        )

    else:
        raise ValueError(f"Unknown LANE_MODEL = {C.LANE_MODEL}")

    return LanePipeline(backend, show_overlay=C.SHOW)


# ============================================================
#  MAIN LOOP
# ============================================================

def main():
    # --- A. UDP Setup ---
    print(f"[Network] Listening: {C.LISTEN_IP}:{C.LISTEN_PORT}")
    rx = make_rx(C.LISTEN_IP, C.LISTEN_PORT, C.RBUF_BYTES)
    rx.settimeout(C.NO_PACKET_TIMEOUT_S)

    tx = make_tx()
    esp_addr = (C.ESP_IP, C.ESP_PORT)
    print(f"[Network] ESP32 Target: {esp_addr}")

    # --- B. Build AI Pipeline + Logger ---
    lane = build_lane_pipeline()
    logger = BenchmarkLogger(C.LOG_DIR)
    if C.SHOW:
        cv2.namedWindow(C.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(C.WIN_NAME, C.WIN_W, C.WIN_H)

    # --- C. Start receiver thread ---
    print("[System] Starting FrameReceiver...")
    frame_q = FrameReceiver(
        rx, C.END_MARKER, C.MAX_ACCUM_BYTES, None, qsize=2
    ).start()

    print("[System] Waiting for camera stream...")
    first_frame = None
    while first_frame is None:
        try:
            first_frame = frame_q.get(timeout=2.0)
        except:
            print("  ... waiting camera ...")

    H_img, W_img = first_frame.shape[:2]
    print(f"[Camera] Connected: {W_img}×{H_img}")

    if os.path.exists(str(C.CALIB_PATH)):
        undist = load_undistort_map(str(C.CALIB_PATH), W_img, H_img)
        if undist is not None:
            frame_q.undist = undist
            print("[Calib] Undistortion applied.")
    else:
        print("[Calib] No calibration file → raw mode.")
    print("    >>> AI READY — STARTING DRIVE <<<")
    fps_display = 0.0
    fps_n = 0
    t_fps0 = time.time()
    last_cmd = None

    try:
        while True:
            try:
                frame = frame_q.get(timeout=1.0)
            except:
                continue

            t0 = time.time()

            status = lane.step(frame, fps_display if fps_display > 0 else None)
            latency = time.time() - t0

            dir_u8   = status["dir"]
            speed_u8 = status["speed"]
            alpha_i8 = status["alpha"]
            overlay  = status.get("overlay", frame)

            # SEND COMMAND
            cmd = f"{dir_u8} {speed_u8} {alpha_i8}\n"
            if cmd != last_cmd:
                tx.sendto(cmd.encode("ascii"), esp_addr)
                last_cmd = cmd
            fps_n += 1
            t_now = time.time()
            if t_now - t_fps0 >= 1.0:
                fps_display = fps_n / (t_now - t_fps0)
                fps_n = 0
                t_fps0 = t_now
            logger.write_from_lane(lane.name(), status, fps_display, latency)
            if C.SHOW:
                cv2.imshow(C.WIN_NAME, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("p"):
                    lane.draw_points = not lane.draw_points

    except KeyboardInterrupt:
        print("\n[System] Interrupted.")

    finally:
        print("[System] Cleanup...")
        logger.close()
        try: cv2.destroyAllWindows()
        except: pass
        try:
            tx.sendto(b"10 0 0\n", esp_addr)
        except:
            pass

        print("[System] Bye.")


if __name__ == "__main__":
    main()
