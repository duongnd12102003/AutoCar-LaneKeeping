import csv
import time
import psutil
from pathlib import Path

class BenchmarkLogger:
    def __init__(self, log_dir: Path):
        self.dir = log_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"benchmark_{timestamp}.csv"

        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self.flush_every = 20
        self.count = 0
        self._w.writerow([
            "timestamp", "model", "fps", "latency_ms", "cpu_percent", "ram_mb",
            "ema_pos", "ema_head", 
            "raw_pos", "raw_head", "r_star",
            "dir", "speed", "alpha", "lane_ok", "sign_check"
        ])
        print(f"[Logger] Saving to: {self.path}")

    def write_from_lane(self, model_name, lane_status, fps, latency_s):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.Process().memory_info().rss / (1024 * 1024) 

        self._w.writerow([
            f"{time.time():.4f}",
            model_name,
            f"{float(fps):.2f}",
            f"{latency_s * 1000:.2f}",
            f"{cpu_usage:.1f}",
            f"{ram_usage:.1f}",
            f"{float(lane_status.get('ema_pos', 0.0)):.4f}",
            f"{float(lane_status.get('ema_head', 0.0)):.2f}",
            f"{float(lane_status.get('raw_pos', 0.0)):.4f}",
            f"{float(lane_status.get('raw_head', 0.0)):.2f}",
            f"{float(lane_status.get('r_star', 0.0)):.2f}",            
            int(lane_status.get("dir", 1)),
            int(lane_status.get("speed", 0)),
            int(lane_status.get("alpha", 0)),
            int(lane_status.get("lane_ok", 0)),
            int(lane_status.get("sign_check", 0)),
        ])

        self.count += 1
        if self.count % self.flush_every == 0:
            self._f.flush()
    def close(self):
        self._f.flush()
        self._f.close()