# gello/zmq_core/camera_node.py (最终方案 B + 简化消息)
import pickle
import threading
from typing import Optional, Tuple
import numpy as np
import zmq
import time

from gello.cameras.camera import CameraDriver

class ZMQClientCamera(CameraDriver):
    def __init__(
        self,
        port: int,
        host: str,
        camera_name: str, 
        # [修改] RealSense 目标 (H, W, C)
        dummy_shape_rgb: Tuple[int, int, int] = (240, 424, 3), 
        dummy_shape_depth: Tuple[int, int, int] = (240, 424, 1), # <--- [修改] RealSense Depth 现在也对齐到 424
    ):
        print(f"ZMQClientCamera ({camera_name}): 正在准备订阅 (SUB, 简单消息) [tcp://{host}:{port}]")
        self.camera_name = camera_name
        self.host = host
        self.port = port
        self._context = zmq.Context()
        dummy_ts = 0.0
        
        # [修改] 如果是 base 相机，使用新的 OAK-D 尺寸 (416)
        if camera_name == 'base':
             dummy_img = np.zeros((240, 416, 3), dtype=np.uint8)
             dummy_depth = np.zeros((240, 416, 1), dtype=np.uint16)
             print(f"   (OAK-D dummy 帧设为 416x240)")
        else: # 默认为 wrist (RealSense)
             dummy_img = np.zeros(dummy_shape_rgb, dtype=np.uint8)
             dummy_depth = np.zeros(dummy_shape_depth, dtype=np.uint16)
             print(f"   (RealSense dummy 帧设为 424x240)")


        self.latest_payload = (dummy_ts, dummy_img, dummy_depth)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(
            target=self._update_loop, args=(self._context,), daemon=True
        )
        self.thread.start()
        print(f"✅ ZMQClientCamera ({camera_name}): 已启动后台监听线程。")



    def _update_loop(self, context: zmq.Context):
        """
        后台线程：创建自己的 Socket 并监听简单消息。
        """
        socket = None
        try:
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.CONFLATE, 1) # 只保留最新消息
            socket.setsockopt(zmq.RCVTIMEO, 2000) # 2 秒超时
            socket.connect(f"tcp://{self.host}:{self.port}")
            
            # [简化] 订阅所有消息 (空字符串)
            socket.setsockopt(zmq.SUBSCRIBE, b'') 
            
            while self.running:
                try:
                    # [简化] 直接接收 payload 数据
                    payload_data = socket.recv()
                    payload = pickle.loads(payload_data)
                    
                    with self.lock:
                        self.latest_payload = payload
                except zmq.Again:
                    continue # 超时是正常的

        except zmq.ContextTerminated:
            pass # 正常关闭
        except Exception as e:
            print(f"❌ ZMQClientCamera ({self.camera_name}) 线程错误: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误
        finally:
            if socket:
                socket.close()

    def read(
        self, img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        非阻塞读取。立即返回最新的 (timestamp, image, depth)。
        """
        ts, img, dep = 0.0, None, None
        with self.lock:
            ts, img, dep = self.latest_payload
            
        if ts > 0 and (time.time() - ts) > 2.0: # 2 秒没有新帧
             print(f"⚠️ 警告: {self.camera_name} 相机数据已过期 (延迟: {time.time() - ts:.2f}s)")
        
        return ts, img, dep

    def close(self):
        """关闭线程并终止 ZMQ context。"""
        self.running = False
        self.thread.join(timeout=1)
        self._context.term()

# (ZMQServerCamera 在此方案中不再被使用)


