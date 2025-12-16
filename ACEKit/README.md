# AutoCar-Kit Firmware

## Source Code Origin

This firmware is based on the open-source **AutoCar-Kit** project developed by a previous student cohort at FPT University.

* **Original Repository:** [https://github.com/nohope-n3/ACE_v2.3.git](https://github.com/nohope-n3/ACE_v2.3.git)

The original code provides the ESP32-S3 camera configuration, basic JPEG image capture, UDP image transmission, and motor control structure for the AutoCar-Kit platform.

## Modifications

The current team adapted the firmware to support the AI-based lane keeping system in this project, with the following adjustments:

* **Continuous Control:** Replaced discrete, rule-based motor commands with continuous steering values received from the Python-based control pipeline.
* **Dynamics Tuning:** Added basic gain, limit, and trim handling to improve steering smoothness and stability.
* **Real-time Optimization:** Kept the original FreeRTOS task structure while removing blocking delays to ensure real-time operation.

These changes retain the original platform design while enabling tighter integration with the external AI processing system.
