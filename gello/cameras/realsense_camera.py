import os
import time
from typing import List, Optional, Tuple


import numpy as np
import cv2 # <--- 确保导入 cv2


from gello.cameras.camera import CameraDriver




def get_device_ids() -> List[str]:
    import pyrealsense2 as rs


    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        # (可选：重置可能有助于解决启动问题)
        # try:
        #     dev.hardware_reset()
        #     time.sleep(2.0) # 重置需要时间
        # except Exception as e:
        #     print(f"警告: 无法重置设备 {dev.get_info(rs.camera_info.serial_number)}: {e}")
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    
    # (如果 hardware_reset 被使用, 这个 sleep 可能是必要的)
    # time.sleep(5) 
    return device_ids




class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id}, align=True)"


    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        import pyrealsense2 as rs


        self._device_id = device_id
        self._pipeline = rs.pipeline()
        config = rs.config()


        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            if not devices: raise RuntimeError("未找到 RealSense 设备。")
            print("未指定 device_id，使用找到的第一个 RealSense 设备。")
        else:
            config.enable_device(device_id)


        # [关键修改] RGB: 424x240 @ 30Hz, Depth: 480x270 @ 30Hz
        rgb_width, rgb_height = 424, 240
        dep_width, dep_height = 480, 270
        fps = 30


        try:
            print(f"尝试配置 RealSense: Depth {dep_width}x{dep_height} @ {fps}Hz, Color {rgb_width}x{rgb_height} @ {fps}Hz")
            # 尝试 Depth
            config.enable_stream(rs.stream.depth, dep_width, dep_height, rs.format.z16, fps)
            # 尝试 Color
            config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)
            
            # [新] 存储目标 RGB 尺寸 (H, W)，用于 dummy 帧
            self._target_rgb_size = (rgb_height, rgb_width) 


        except RuntimeError as e:
            print(f"❌ 无法设置 RealSense 为指定模式: {e}")
            print("   请检查你的相机型号是否支持此组合。")
            raise # 重新抛出错误


        # 启动 pipeline
        pipeline_profile = self._pipeline.start(config)


        # [关键修改] 创建一个 Align 对象
        # 我们告诉它将所有流对齐到 Color 流
        self._align = rs.align(rs.stream.color)
        print("✅ RealSense Aligner 已创建 (Depth -> Color)。")


        self._flip = flip
        time.sleep(1.0) # 等待流稳定


    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None, # img_size is (H, W)
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        frames = None
        try:
            frames = self._pipeline.wait_for_frames(5000) # 5 秒超时
        except RuntimeError as e:
             print(f"❌ RealSense 错误: {e}. 可能是 'Frame didn't arrive'.")
             
        if not frames:
            print("警告: RealSense 未接收到帧集 (frameset)。返回 dummy 帧。")
            dummy_color = np.zeros((self._target_rgb_size[0], self._target_rgb_size[1], 3), dtype=np.uint8)
            dummy_depth = np.zeros((self._target_rgb_size[0], self._target_rgb_size[1], 1), dtype=np.uint16) # <--- 注意：dummy depth 现在也匹配 RGB 尺寸
            return dummy_color, dummy_depth


        # [关键修改] 执行对齐
        aligned_frames = self._align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame() # <--- 获取对齐后的深度帧
        
        if not aligned_depth_frame or not color_frame:
            print("警告: RealSense 未接收到 color/aligned_depth 帧。返回 dummy 帧。")
            dummy_color = np.zeros((self._target_rgb_size[0], self._target_rgb_size[1], 3), dtype=np.uint8)
            dummy_depth = np.zeros((self._target_rgb_size[0], self._target_rgb_size[1], 1), dtype=np.uint16) # <--- 尺寸 (240, 424, 1)
            return dummy_color, dummy_depth


        # [关键修改] 两个图像现在都是 (240, 424) 尺寸
        color_image = np.asanyarray(color_frame.get_data()) # Shape: (240, 424, 3) BGR
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) # Shape: (240, 424) uint16


        # (img_size resize 逻辑保持不变，现在它将对两个相同尺寸的图像进行操作)
        if img_size is not None and (img_size[0] != color_image.shape[0] or img_size[1] != color_image.shape[1]):
            # (注意：img_size 是 H, W, 但 cv2.resize 期望 W, H)
            image_resized = cv2.resize(color_image, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)[:, :, ::-1] # BGR to RGB
        else:
            image_resized = color_image[:, :, ::-1] # BGR to RGB
            
        if img_size is not None and (img_size[0] != depth_image.shape[0] or img_size[1] != depth_image.shape[1]):
             depth_resized = cv2.resize(depth_image, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
             depth_resized = depth_image


        image_final = image_resized
        depth_final = depth_resized


        if self._flip:
            image_final = cv2.rotate(image_final, cv2.ROTATE_180)
            depth_final = cv2.rotate(depth_final, cv2.ROTATE_180)
            
        # 确保深度形状为 HxWx1
        if depth_final.ndim == 2:
            depth_final = depth_final[:, :, None]


        return image_final, depth_final


    def __del__(self):
        """确保 pipeline 被停止"""
        if hasattr(self, '_pipeline'):
            try:
                self._pipeline.stop()
            except RuntimeError as e:
                pass # 忽略退出时的错误


def _debug_read(camera, save_datastream=False):
    import cv2
    # [修改] 两个窗口现在应该显示相同分辨率的图像
    cv2.namedWindow("image (RGB)")
    cv2.namedWindow("depth (Aligned)")
    counter = 0
    if not os.path.exists("images"): os.makedirs("images")
    if save_datastream and not os.path.exists("stream"): os.makedirs("stream")
    
    print("按 's' 保存, 按 'ESC' 退出...")
    
    while True:
        # time.sleep(0.033) # 模拟 30Hz
        image, depth = camera.read() # (返回 424x240, 424x240)
        
        # 可视化深度图
        depth_viz = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)


        # 检查分辨率是否匹配
        if image.shape[:2] != depth.shape[:2]:
             print(f"!!! 警告: 分辨率不匹配! RGB: {image.shape}, Depth: {depth.shape} !!!")
             # (尝试 resize 以便显示)
             depth_viz_color = cv2.resize(depth_viz_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


        cv2.imshow("image (RGB)", image[:, :, ::-1]) # CV2 需要 BGR
        cv2.imshow("depth (Aligned)", depth_viz_color)
        
        key = cv2.waitKey(1)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth) # 保存原始 uint16 深度
            print(f"已保存 image_{counter}.png, depth_{counter}.png")
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break
    cv2.destroyAllWindows()




if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    if device_ids:
        rs = RealSenseCamera(flip=False, device_id=device_ids[0])
        im, depth = rs.read()
        print(f"--- 首次读取 ---")
        print(f"RGB 图像 shape: {im.shape}, dtype: {im.dtype}")
        print(f"Depth 图像 shape: {depth.shape}, dtype: {depth.dtype}")
        print(f"-----------------")
        _debug_read(rs, save_datastream=True)
    else:
        print("未找到 RealSense 设备，无法启动调试。")



