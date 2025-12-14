#include <Arduino.h>
#include "DCMotorControl.h"

DCMotorControl::DCMotorControl(uint8_t pinPWNMotorA, uint8_t pinPWNMotorB)
: _pinA(pinPWNMotorA), _pinB(pinPWNMotorB)
{
  ledcAttachChannel(_pinA, pwmFrequency, pwmResolution, 0);
  ledcAttachChannel(_pinB, pwmFrequency, pwmResolution, 1);
  SettingMotor(0, 0);
}

void DCMotorControl::SettingMotor(uint8_t speedMotorA, uint8_t speedMotorB) {
  ledcWrite(_pinA, speedMotorA);
  ledcWrite(_pinB, speedMotorB);
}

void DCMotorControl::CarMovementControl(uint8_t direction, uint8_t speed, int8_t alpha) {
  int16_t d = (int16_t)((float)alpha * _alphaGain);
  if (_invertSteer) d = -d;
  int16_t pwmA = (int16_t)speed + (int16_t)(_leftGain  * d) + _trimA;
  int16_t pwmB = (int16_t)speed - (int16_t)(_rightGain * d) + _trimB;
  switch (direction) {
    case SLOW:
      pwmA = (int16_t)(pwmA * _slowScale);
      pwmB = (int16_t)(pwmB * _slowScale);
      break;
    case LEFT:
      pwmA += 8; pwmB -= 8;
      break;
    case RIGHT:
      pwmA -= 8; pwmB += 8;
      break;
    case STANDBY:
    case STOP:
      SettingMotor(0, 0);
      return;
    case STRAIGHT:
    default:
      break;
  }

  if (_dbKick && alpha != 0 && (uint8_t)abs(alpha) < _alphaDB) {
    int bump = _alphaDB - abs(alpha);
    if (alpha > 0) { pwmA += bump; pwmB -= bump; }
    else           { pwmA -= bump; pwmB += bump; }
  }

  if (speed > 0) {
    pwmA = clamp16(pwmA, _pwmMin, _pwmMax);
    pwmB = clamp16(pwmB, _pwmMin, _pwmMax);
  } else {
    pwmA = 0; pwmB = 0;
  }

  SettingMotor((uint8_t)pwmA, (uint8_t)pwmB);
}
  