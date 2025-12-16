# AutoCar-Kit Firmware

## Source Code Origin
This firmware is based on the open-source **AutoCar-Kit** project.
* **Original Repository:** [🔗 https://github.com/nohope-n3/ACE_v2.3.git](https://github.com/nohope-n3/ACE_v2.3.git)

## Modifications
We have adapted this codebase to work with our **AI Lane Keeping System**:
1.  **UDP Streaming:** Lightweight raw UDP streaming mechanism to reduce latency and support real-time deep learning inference on the host computer.
2.  **Control Logic:** Updated command parsing to receive steering values from the Python backend.

*Use this firmware with the ESP32-S3 hardware configuration defined in the original repository.*
