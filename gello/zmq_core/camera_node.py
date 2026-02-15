# gello/zmq_core/camera_node.py
import pickle
import threading
import time
from typing import Optional, Tuple

import numpy as np
import zmq

from gello.cameras.camera import CameraDriver


class ZMQClientCamera(CameraDriver):
    def __init__(
        self,
        port: int,
        host: str,
        camera_name: str,
        dummy_shape_rgb: Tuple[int, int, int] = (720, 1280, 3),
        dummy_shape_depth: Tuple[int, int, int] = (720, 1280, 1),
    ):
        print(
            f"ZMQClientCamera ({camera_name}): "
            f"Subscribing (SUB) [tcp://{host}:{port}]"
        )
        self.camera_name = camera_name
        self.host = host
        self.port = port
        self._context = zmq.Context()
        dummy_ts = 0.0

        # Both cameras now use 1280x720
        dummy_img = np.zeros(dummy_shape_rgb, dtype=np.uint8)
        dummy_depth = np.zeros(dummy_shape_depth, dtype=np.uint16)
        print(f"   (dummy frames set to {dummy_shape_rgb})")

        self.latest_payload = (dummy_ts, dummy_img, dummy_depth)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(
            target=self._update_loop,
            args=(self._context,),
            daemon=True,
        )
        self.thread.start()
        print(f"ZMQClientCamera ({camera_name}): " f"Background listener started.")

    def _update_loop(self, context: zmq.Context):
        """Background thread: subscribe and listen for camera frames."""
        socket = None
        try:
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.CONFLATE, 1)  # Keep only latest
            socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2s timeout
            socket.connect(f"tcp://{self.host}:{self.port}")

            # Subscribe to all messages
            socket.setsockopt(zmq.SUBSCRIBE, b"")

            while self.running:
                try:
                    payload_data = socket.recv()
                    payload = pickle.loads(payload_data)

                    with self.lock:
                        self.latest_payload = payload
                except zmq.Again:
                    continue  # Timeout is normal

        except zmq.ContextTerminated:
            pass  # Normal shutdown
        except Exception as e:
            print(f"ZMQClientCamera ({self.camera_name}) " f"thread error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if socket:
                socket.close()

    def read(
        self, img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Non-blocking read. Returns latest (timestamp, image, depth)."""
        ts, img, dep = 0.0, None, None
        with self.lock:
            ts, img, dep = self.latest_payload

        if ts > 0 and (time.time() - ts) > 2.0:
            print(
                f"Warning: {self.camera_name} camera data stale "
                f"(delay: {time.time() - ts:.2f}s)"
            )

        return ts, img, dep

    def close(self):
        """Close thread and terminate ZMQ context."""
        self.running = False
        self.thread.join(timeout=1)
        self._context.term()
