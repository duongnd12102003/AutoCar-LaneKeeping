import numpy as np, cv2

def refine_mask01(mask01):
    m = (mask01 > 0).astype(np.uint8)
    ker_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker_vertical)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(m)
    c_max = max(contours, key=cv2.contourArea)
    out = np.zeros_like(m)
    cv2.drawContours(out, [c_max], -1, 1, thickness=cv2.FILLED)
    return out

def meters_per_pixel(mask_bev, lane_width_m=0.20):
    H, W = mask_bev.shape
    row = mask_bev[H-1, :]
    xs = np.where(row > 0)[0]
    if xs.size >= 2:
        width_px = float(max(16.0, xs[-1] - xs[0]))
        return lane_width_m / width_px
    return lane_width_m / max(W, 1.0)

def center_x(mask_bev, y_ratio):
    H, W = mask_bev.shape
    y = int(np.clip(y_ratio * H, 0, H - 1))
    xs = np.where(mask_bev[y, :] > 0)[0]
    if xs.size < 2:
        return W/2.0
    return 0.5 * (xs[0] + xs[-1])

def heading_deg_at_ratio(mask_bev, y_ratio, dy_px=30):
    H, _ = mask_bev.shape
    yb = int(np.clip(y_ratio * H, 0, H - 1))
    yt = int(np.clip(yb - dy_px, 0, H - 1))
    xb = center_x(mask_bev, yb / H)
    xu = center_x(mask_bev, yt / H)
    dy = (yb - yt) + 1e-6
    dx = xb - xu
    return float(np.degrees(np.arctan2(dx, dy))), float(xb)

def apply_roi(mask, roi_poly):
    H, W = mask.shape
    poly = (roi_poly * np.array([W, H], np.float32)).astype(np.int32)
    roi = np.zeros_like(mask, np.uint8)
    cv2.fillPoly(roi, [poly], 1)
    return (mask & roi)

