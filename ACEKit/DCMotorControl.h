#pragma once
#include <Arduino.h>
#define pwmFrequency 800 
#define pwmResolution 8 
#define STRAIGHT 1
#define SLOW     2
#define LEFT     3
#define RIGHT    4
#define TURN_LEFT  5
#define TURN_RIGHT 6 
#define STANDBY  7
#define STOP     8

class DCMotorControl {
public:
  DCMotorControl(uint8_t pinPWNMotorA, uint8_t pinPWNMotorB);
  void CarMovementControl(uint8_t direction, uint8_t speed, int8_t alpha);
  void SettingMotor(uint8_t speedMotorA, uint8_t speedMotorB);

  void setAlphaGain(float k)                   { _alphaGain = k; }
  void setSideGains(float leftG, float rightG) { _leftGain = leftG; _rightGain = rightG; }
  void setTrim(int8_t trimA, int8_t trimB)     { _trimA = trimA; _trimB = trimB; }
  void setPwmLimits(uint8_t mn, uint8_t mx)    { _pwmMin = mn; _pwmMax = mx; }
  void setDeadband(uint8_t db)                 { _alphaDB = db; }
  void setDeadbandKick(bool en)                { _dbKick = en; }
  void setSlowScale(float s)                   { _slowScale = s; }
  void setInvertSteer(bool en)                 { _invertSteer = en; }

private:
  uint8_t _pinA;
  uint8_t _pinB;
  float   _alphaGain = 1.00f;
  float   _leftGain  = 1.00f;
  float   _rightGain = 1.00f;
  int8_t  _trimA     = 0;
  int8_t  _trimB     = 0;

  uint8_t _pwmMin    = 40;
  uint8_t _pwmMax    = 255;
  uint8_t _alphaDB   = 6;  
  bool    _dbKick    = true; 
  float   _slowScale = 0.70f; 

  bool    _invertSteer = false;

  static inline int16_t clamp16(int16_t v, int16_t lo, int16_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  }
};
