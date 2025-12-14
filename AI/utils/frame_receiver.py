import socket
import cv2
import numpy as np
import threading
from queue import Queue

class FrameReceiver:
    def __init__(self, rx, end_marker, max_accum_bytes, undist_map, qsize=2):
        self.rx = rx
        self.end_marker = end_marker
        self.max_accum = max_accum_bytes
        self.buffer = bytearray()
        self.undist = undist_map 
        self.queue = Queue(maxsize=qsize)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self.queue

    def _loop(self):
        while True:
            try:
                data, _ = self.rx.recvfrom(65535)
            except socket.timeout:
                self.buffer = bytearray()
                continue
            except Exception:
                continue
            if data == b"HB" or data == b"HB\n":
                continue
            self.buffer.extend(data)
            if len(self.buffer) > self.max_accum:
                self.buffer.clear()
                continue
            pos = self.buffer.find(self.end_marker)
            if pos == -1:
                continue
            pos += 2 
            img_data = self.buffer[:pos]
            self.buffer = self.buffer[pos:]
            try:
                frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), 1)
            except:
                frame = None

            if frame is None:
                continue
            if self.undist is not None:
                try:
                    map1, map2 = self.undist
                    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
                except Exception:
                    pass
            if self.queue.full():
                try: self.queue.get_nowait()
                except: pass

            self.queue.put(frame)