import os, numpy as np, cv2
from typing import Optional, Tuple

def load_undistort_map(calib_path: str, W: int, H: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not calib_path or not os.path.exists(calib_path):
        return None
    try:
        data = np.load(calib_path)
        mtx, dist = data["mtx"], data["dist"]
        newK = cv2.getOptimalNewCameraMatrix(mtx, dist, (W, H), 0)[0]
        map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newK, (W, H), cv2.CV_16SC2)
        return map1, map2
    except Exception:
        return None
