import cv2
import numpy as np

def draw_overlay(
    frame_bgr,
    bev01,
    M_inv,
    ema_pos,
    ema_head,
    fps,
    ratios=None,        
    centers_px=None,    
    primary_idx=0,    
    theta_thresh=2.5,
):
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    draw = frame_bgr.copy()
    H, W = draw.shape[:2]

    if bev01 is not None and M_inv is not None:
        mask_warp = cv2.warpPerspective(
            (bev01 * 255).astype(np.uint8), 
            M_inv, (W, H), 
            flags=cv2.INTER_NEAREST
        )
        roi = mask_warp > 0
        if roi.any():
            roi_bg = draw[roi].astype(np.uint16)
            draw[roi] = (roi_bg // 2 + np.array([0, 128, 0], dtype=np.uint16)).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    
    color_shadow = (0, 0, 0)
    color_green  = (0, 255, 0)
    color_white  = (255, 255, 255)
    color_yellow = (0, 255, 255)
    color_red    = (0, 0, 255)

    x0, y0 = 10, 30
    line_h = 25
    txt1 = f"Pos: {ema_pos:.3f}m | Head: {ema_head:+.1f} deg"
    cv2.putText(draw, txt1, (x0+1, y0+1), font, scale, color_shadow, thick+1, cv2.LINE_AA)
    cv2.putText(draw, txt1, (x0, y0),     font, scale, color_green,  thick,   cv2.LINE_AA)

    dir_text = "STRAIGHT"
    dir_color = color_white
    if ema_head > +theta_thresh: 
        dir_text = "<< LEFT"
        dir_color = color_red
    elif ema_head < -theta_thresh: 
        dir_text = "RIGHT >>"
        dir_color = color_red

    y1 = y0 + line_h
    cv2.putText(draw, f"Dir: {dir_text}", (x0+1, y1+1), font, scale, color_shadow, thick+1, cv2.LINE_AA)
    cv2.putText(draw, f"Dir: {dir_text}", (x0, y1),     font, scale, dir_color,    thick,   cv2.LINE_AA)

    y2 = y1 + line_h
    fps_text = f"FPS: {float(fps):.1f}"
    cv2.putText(draw, fps_text, (x0+1, y2+1), font, scale, color_shadow, thick+1, cv2.LINE_AA)
    cv2.putText(draw, fps_text, (x0, y2),     font, scale, color_yellow, thick,   cv2.LINE_AA)
    if ratios is not None and centers_px is not None and M_inv is not None:
        if len(centers_px) == len(ratios):
            pts_bev = []
            valid_indices = []
            
            for i, r in enumerate(ratios):
                cx = centers_px[i]
                if not np.isnan(cx):
                    py = r * H      
                    px = cx
                    pts_bev.append([px, py])
                    valid_indices.append(i)
            
            if len(pts_bev) > 0:
                pts_bev = np.array(pts_bev, dtype=np.float32).reshape(-1, 1, 2)
                pts_persp = cv2.perspectiveTransform(pts_bev, M_inv)
                for k, idx in enumerate(valid_indices):
                    pt_x, pt_y = pts_persp[k][0]
                    if 0 <= pt_x < W and 0 <= pt_y < H:
                        color_pt = color_red if idx == int(primary_idx) else color_yellow
                        radius = 5 if idx == int(primary_idx) else 3
                        
                        cv2.circle(draw, (int(pt_x), int(pt_y)), radius, color_pt, -1)

    return draw