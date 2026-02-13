# gello/cameras/oakd_camera.py (最终版 - 修复 16 倍数宽度)
import time
from typing import List, Optional, Tuple


import numpy as np
import cv2 # <--- 确保导入 cv2


from gello.cameras.camera import CameraDriver




def get_device_ids() -> List[str]:
    """获取所有连接的 OAK-D 设备的 MXID (序列号)"""
    import depthai as dai
    device_infos = dai.Device.getAllAvailableDevices()
    device_ids = []
    for info in device_infos:
        device_ids.append(info.getMxId())
    return device_ids




class OAKDCamera(CameraDriver):
    """
    OAK-D Pro camera driver using DepthAI.
    
    [已修改] 返回 (在 img_size=None 时):
        color: (240, 416, 3) uint8, RGB
        depth: (240, 416, 1) uint16, in millimeters (已对齐到 RGB)
    """


    def __repr__(self) -> str:
        # [修改] 更新 repr
        return f"OAKDCamera(device_id={self._device_id}, res=({self._target_rgb_size[1]},{self._target_rgb_size[0]}), fps={self._fps}, align=True)"


    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        fps: int = 30, # 目标帧率
        depth_millimeters: bool = True,
    ):
        import depthai as dai


        self._device_id = device_id
        self._flip = flip
        self._fps = fps 
        self._depth_in_mm = depth_millimeters
        
        # [关键修复] RGB 宽度 424 不是 16 的倍数。改为 416。
        rgb_width, rgb_height = 416, 240
        # (Depth 的原始计算分辨率 W, H 保持不变)
        dep_width, dep_height = 480, 270
        
        # [修改] 两个流的最终输出尺寸都将是 416x240
        self._target_rgb_size = (rgb_width, rgb_height) # (W, H)
        self._target_depth_size = (rgb_width, rgb_height) # (W, H) 


        pipeline = dai.Pipeline()


        # Color camera (RGB)
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        # [关键修复] 设置预览 (硬件缩放) 尺寸为 416x240
        cam_rgb.setPreviewSize(rgb_width, rgb_height)
        cam_rgb.setFps(self._fps)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB) 
        ctrl = cam_rgb.initialControl
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        ctrl.setAutoExposureEnable()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)


        # Mono cameras (for Stereo Depth)
        mono_res = dai.MonoCameraProperties.SensorResolution.THE_400_P
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(mono_res)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_left.setFps(self._fps)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(mono_res)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_right.setFps(self._fps)


        # Stereo depth node
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        
        # [关键修复] 启用硬件对齐
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        
        # [关键修复] 强制深度输出分辨率匹配 RGB 的分辨率 (416x240)
        stereo.setOutputSize(rgb_width, rgb_height)


        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)


        # XLink outputs (现在是已对齐的)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input) # 输出 416x240


        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input) # [修改] 现在输出 416x240 (对齐后)


        # Create device (保持线程安全修复)
        try:
            if self._device_id:
                found, dev_info = dai.Device.getDeviceByMxId(self._device_id)
                if not found: raise RuntimeError(f"Device ID {self._device_id} not found")
                self._device = dai.Device(pipeline, dev_info, dai.UsbSpeed.SUPER_PLUS)
            else:
                self._device = dai.Device(pipeline)
        except RuntimeError as e:
            print(f"❌ 无法启动 OAK-D 设备 (ID: {self._device_id}). 错误: {e}")
            raise


        # Output queues
        self._q_rgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self._q_depth = self._device.getOutputQueue(name="depth", maxSize=4, blocking=False)


        time.sleep(1.0) # 增加等待时间


    def read(self, img_size: Optional[Tuple[int, int]] = None): # img_size is (H, W)
        rgb_packet = self._q_rgb.tryGet()
        if rgb_packet is None: rgb_packet = self._q_rgb.get()
        depth_packet = self._q_depth.tryGet()
        if depth_packet is None: depth_packet = self._q_depth.get()


        # [修改] 两个帧现在都具有相同的 (240, 416) 尺寸
        image = rgb_packet.getCvFrame()  # (240, 416, 3) RGB
        depth_frame = depth_packet.getFrame() # (240, 416) uint16
        depth = depth_frame


        # (img_size resize 逻辑保持不变)
        if img_size is not None and (img_size[0] != image.shape[0] or img_size[1] != image.shape[1]):
            image_resized = cv2.resize(image, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        else:
            image_resized = image
            
        if img_size is not None and (img_size[0] != depth.shape[0] or img_size[1] != depth.shape[1]):
             depth_resized = cv2.resize(depth, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
             depth_resized = depth


        image_final = image_resized
        depth_final = depth_resized
        
        if self._flip:
            image_final = cv2.rotate(image_final, cv2.ROTATE_180)
            depth_final = cv2.rotate(depth_final, cv2.ROTATE_180)


        if depth_final.ndim == 2:
            depth_final = depth_final[:, :, None] # 确保 HxWx1


        return image_final, depth_final




    def __del__(self):
        """确保在对象销毁时关闭设备"""
        if hasattr(self, "_device"):
            self._device.close()


#
# (这里的 _debug_read 和 __main__ 函数保持不变)
# (当你运行 __main__ 时, 它现在会打印 416x240 的 shape)
#
def _debug_read(camera: CameraDriver, save_datastream: bool = False):
    import cv2
    import os
    cv2.namedWindow("image (RGB)")
    cv2.namedWindow("depth (Aligned)")
    print("--- 开始调试 ---")
    print(" 按 's' 保存图像 (保存到 'stream_oakd/' 目录下)")
    print(" 按 'ESC' 退出")
    counter = 0
    save_dir = "stream_oakd"
    if (save_datastream or 's') and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建目录: {save_dir}")
    while True:
        try:
            image, depth = camera.read()
            image_bgr_display = image[:, :, ::-1] # CV2 需要 BGR
            depth_viz = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            if image.shape[:2] != depth.shape[:2]:
                 print(f"!!! 警告: 分辨率不匹配! RGB: {image.shape}, Depth: {depth.shape} !!!")
                 depth_viz_color = cv2.resize(depth_viz_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("image (RGB)", image_bgr_display)
            cv2.imshow("depth (Aligned)", depth_viz_color)
            key = cv2.waitKey(1)
            if key == ord("s"):
                img_path = os.path.join(save_dir, f"image_{counter}.png")
                cv2.imwrite(img_path, image_bgr_display)
                depth_path = os.path.join(save_dir, f"depth_{counter}.png")
                cv2.imwrite(depth_path, depth) # 保存原始 uint16 深度
                print(f"已保存: {img_path} 和 {depth_path}")
                counter += 1
            if save_datastream:
                 img_path = os.path.join(save_dir, f"image_{counter}.png")
                 cv2.imwrite(img_path, image_bgr_display)
                 depth_path = os.path.join(save_dir, f"depth_{counter}.png")
                 cv2.imwrite(depth_path, depth)
                 counter += 1
            if key == 27: print("--- 退出调试 ---"); break
        except KeyboardInterrupt: print("--- 退出调试 ---"); break
    cv2.destroyAllWindows()




if __name__ == "__main__":
    import os
    import cv2
    device_ids = get_device_ids()
    print(f"发现 {len(device_ids)} 个 OAK-D 设备:")
    print(device_ids)
    if len(device_ids) > 0:
        device_to_use = device_ids[0]
        print(f"正在连接到设备: {device_to_use}")
        oak_cam = OAKDCamera(device_id=device_to_use, flip=True)
        try:
            im, depth = oak_cam.read()
            print("\n--- 首次读取成功 ---")
            print(f"图像分辨率 (H, W, C): {im.shape}")   # <--- 应该显示 (240, 416, 3)
            print(f"深度图分辨率 (H, W, C): {depth.shape}") # <--- 应该显示 (240, 416, 1)
            print(f"深度数据类型: {depth.dtype}")  
            print("----------------------\n")
            _debug_read(oak_cam, save_datastream=False)
        except Exception as e:
            print(f"读取相机时出错: {e}")
    else:
        print("未发现 OAK-D 设备。请检查连接。")



