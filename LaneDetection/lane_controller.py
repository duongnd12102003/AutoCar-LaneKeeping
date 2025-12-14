import numpy as np

class LaneController:
    def __init__(self):
        # ==== DEADBAND & THRESHOLDS ====
        self.theta_thresh       = 2.5
        self.pos_deadband_m     = 0.016
        self.head_deadband_deg  = 0.75
        
        # ==== SERVO LIMITS & PROTECTION ====
        self.alpha_limit        = 54
        self.alpha_limit_left   = 52 
        self.alpha_limit_right  = 52 
        self.alpha_filter       = 0.38
        self.alpha_slew         = 10.0
        self.min_abs_alpha      = 4.0
        
        # ==== GAIN SCHEDULING ====
        self.left_boost         = 1.36
        self.right_boost        = 1.32
        
        # ==== PID PARAMETERS ====
        self.pos_i_gain         = 5.0
        self.pos_i_forget       = 0.92
        self.pos_i_clip_deg     = 5.0
        self.pos_d_gain_R       = 60.0 
        self.pos_d_forget       = 0.22  
        
        # ==== STEERING TRIM ====
        self.steer_trim_deg     = +0.5
        self.trim_beta          = 0.0025
        self.trim_sm            = float(self.steer_trim_deg)
        
        # ==== SPEED CONTROL ====
        self.base_speed         = 150
        
        # ==== INTERNAL STATE ====
        self.alpha_sm           = None
        self._alpha_out_prev    = 0.0
        self._prev_head         = 0.0
        self._curveL_cnt        = 0
        self._curveR_cnt        = 0
        self._pos_prev          = None
        self._pos_d_ema         = 0.0
        self.pos_i_state        = 0.0
        self.fps_val            = 20.0
        
        # ==== INPUT EMA ====
        self.ema_pos            = None
        self.ema_head           = None

    def set_fps(self, fps: float):
        self.fps_val = max(8.0, float(fps) if fps and fps > 0 else 20.0)

    def update_ema(self, pos_m: float, head_deg: float):
        self.ema_pos  = float(pos_m)
        self.ema_head = float(head_deg)

    @staticmethod
    def _soft_db(x: float, db: float) -> float:
        if abs(x) <= db: return 0.0
        return np.sign(x) * (abs(x) - db)

    def decide(self, lane_ok: int):
        # ---------------------------------------------------------
        # 1. SIGNAL CONDITIONING (Bias Correction & Deadband)
        # ---------------------------------------------------------
        ema_pos  = float(self.ema_pos  if self.ema_pos  is not None else 0.0)
        ema_head = float(self.ema_head if self.ema_head is not None else 0.0)
        
        # Bias correction for turning logic
        bias_mag_L = 0.006
        bias_mag_R = 0.010
        bias = bias_mag_L if ema_head > 0 else (bias_mag_R if ema_head < 0 else 0.0)
        pos_bias = np.sign(-ema_head) * abs(bias)

        pos_err  = self._soft_db(ema_pos - pos_bias, self.pos_deadband_m)
        head_err = self._soft_db(ema_head,          self.head_deadband_deg)
        d_head   = head_err - self._prev_head
        self._prev_head = head_err

        # ---------------------------------------------------------
        # 2. STATE ESTIMATION (Curve Persistence Counter)
        # ---------------------------------------------------------
        if head_err > +self.theta_thresh:
            self._curveL_cnt = min(self._curveL_cnt + 1, 12); self._curveR_cnt = max(self._curveR_cnt - 1, 0)
        elif head_err < -self.theta_thresh:
            self._curveR_cnt = min(self._curveR_cnt + 1, 12); self._curveL_cnt = max(self._curveL_cnt - 1, 0)
        else:
            self._curveL_cnt = max(self._curveL_cnt - 1, 0); self._curveR_cnt = max(self._curveR_cnt - 1, 0)

        # ---------------------------------------------------------
        # 3. AUTO-TRIM LEARNING
        # ---------------------------------------------------------
        near_straight = abs(head_err) < 1.2
        if near_straight and lane_ok:
            self.trim_sm = (1.0 - self.trim_beta)*self.trim_sm + self.trim_beta * np.clip(ema_pos*120.0, -2.0, 2.0)
        else:
            self.trim_sm = 0.999*self.trim_sm + 0.001*self.steer_trim_deg

        # ---------------------------------------------------------
        # 4. PHYSICS-BASED DERIVATIVE (FPS Normalized)
        # ---------------------------------------------------------
        if self._pos_prev is None:
            self._pos_prev = ema_pos
        d_pos_inst  = float(ema_pos - self._pos_prev)
        self._pos_prev = ema_pos
        d_pos_per_s = d_pos_inst * self.fps_val
        self._pos_d_ema = (1.0 - self.pos_d_forget)*self._pos_d_ema + self.pos_d_forget*d_pos_per_s

        # ---------------------------------------------------------
        # 5. ASYMMETRIC GAIN SCHEDULING
        # ---------------------------------------------------------
        if head_err < -self.theta_thresh:  # RIGHT TURN
            K_head = 1.22 * self.right_boost * (1.0 + 0.05*self._curveR_cnt)
            K_pos  = 16.0 * self.right_boost
            K_d    = 0.28
            limit  = self.alpha_limit_right
            local_filter = min(0.40, self.alpha_filter + 0.08) 
            local_slew   = max(8.0, 0.85 * self.alpha_slew)  
            
        elif head_err > +self.theta_thresh: # LEFT TURN
            K_head = 1.36 * self.left_boost  * (1.0 + 0.06*self._curveL_cnt)
            K_pos  = 20.0 * self.left_boost
            K_d    = 0.30
            limit  = self.alpha_limit_left
            local_filter = min(0.40, self.alpha_filter + 0.08)
            local_slew   = self.alpha_slew 
            
        else: # STRAIGHT
            K_head, K_pos, K_d = 1.15, 15.0, 0.22
            limit        = self.alpha_limit
            local_filter = self.alpha_filter
            local_slew   = self.alpha_slew

        # ---------------------------------------------------------
        # 6. CONDITIONAL INTEGRAL ACTION
        # ---------------------------------------------------------
        if near_straight and lane_ok:
            self.pos_i_state = self.pos_i_forget*self.pos_i_state + (1.0 - self.pos_i_forget)*pos_err
        else:
            self.pos_i_state *= 0.98
        u_i = np.clip(self.pos_i_gain * self.pos_i_state, -self.pos_i_clip_deg, self.pos_i_clip_deg)

        # ---------------------------------------------------------
        # 7. CONTROL LAW & SHAPING
        # ---------------------------------------------------------
        u = K_head*head_err + K_pos*pos_err + K_d*d_head + u_i
        u_abs = abs(u)
        u_shaped = 0.75*u + 0.25*np.sign(u)*(u_abs*u_abs/20.0)
        
        if head_err < -self.theta_thresh:  
            u_shaped *= 0.90
        alpha_cmd = u_shaped + self.trim_sm
        if 1e-3 < abs(alpha_cmd) < self.min_abs_alpha:
            alpha_cmd = np.sign(alpha_cmd) * self.min_abs_alpha
        if alpha_cmd < 0:
            alpha_cmd = max(alpha_cmd, -limit)
        else:
            alpha_cmd = min(alpha_cmd, +limit)

        sat_right = False
        if head_err < -self.theta_thresh and alpha_cmd > 0:
            if self._curveR_cnt >= 8:
                alpha_cmd = min(alpha_cmd, 0.92*limit); sat_right = True
            elif self._curveR_cnt >= 5 and abs(self._alpha_out_prev) > 0.85*limit:
                alpha_cmd = min(alpha_cmd, 0.96*limit); sat_right = True
        if sat_right:
            self.pos_i_state *= 0.85
            
        if (head_err < -self.theta_thresh) and (pos_err < -0.030):
            alpha_cmd = max(alpha_cmd, -0.85 * limit)

        # ---------------------------------------------------------
        # 8. OUTPUT SMOOTHING & SLEW RATE LIMITING
        # ---------------------------------------------------------
        if self.alpha_sm is None:
            self.alpha_sm = alpha_cmd
        else:
            self.alpha_sm = (1 - local_filter)*self.alpha_sm + local_filter*alpha_cmd

        if not lane_ok:
            local_slew = max(6.0, 0.7*local_slew)
            if alpha_cmd < 0:
                self.alpha_sm = max(self.alpha_sm, -min(limit, 35))
            else:
                self.alpha_sm = min(self.alpha_sm,  +min(limit, 35))
                
        deg_per_s_cap = 240.0  
        local_slew = min(local_slew, deg_per_s_cap / max(8.0, self.fps_val))
        step = np.clip(self.alpha_sm - self._alpha_out_prev, -local_slew, local_slew)
        alpha_out = self._alpha_out_prev + step
        self._alpha_out_prev = alpha_out

        curve_cnt = max(self._curveL_cnt, self._curveR_cnt)
        speed = self.base_speed - 1.4*abs(head_err) - 0.6*max(0, curve_cnt - 6)
        
        if head_err > +self.theta_thresh:
            speed -= 2
        if head_err < -self.theta_thresh:
            speed -= 6
        if not lane_ok:
            speed = min(speed, 120)
            
        speed = max(110, min(185, speed))

        return 1, int(speed), int(np.clip(round(alpha_out), -128, 127))