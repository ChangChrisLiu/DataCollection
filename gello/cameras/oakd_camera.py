# gello/cameras/oakd_camera.py
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gello.cameras.camera import CameraDriver


def get_device_ids() -> List[str]:
    """Get all connected OAK-D device MXIDs (serial numbers)."""
    import depthai as dai

    device_infos = dai.Device.getAllAvailableDevices()
    device_ids = []
    for info in device_infos:
        device_ids.append(info.getMxId())
    return device_ids


class OAKDCamera(CameraDriver):
    """OAK-D Pro camera driver using DepthAI.

    Returns (when img_size=None):
        color: (240, 416, 3) uint8, RGB
        depth: (240, 416, 1) uint16, in millimeters (aligned to RGB)
    """

    def __repr__(self) -> str:
        w, h = self._target_rgb_size
        return (
            f"OAKDCamera(device_id={self._device_id}, "
            f"res=({w},{h}), fps={self._fps}, align=True)"
        )

    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        fps: int = 30,
        depth_millimeters: bool = True,
    ):
        import depthai as dai

        self._device_id = device_id
        self._flip = flip
        self._fps = fps
        self._depth_in_mm = depth_millimeters

        # RGB width 424 is not a multiple of 16. Use 416 instead.
        rgb_width, rgb_height = 416, 240
        dep_width, dep_height = 480, 270

        # Both streams output at 416x240
        self._target_rgb_size = (rgb_width, rgb_height)  # (W, H)
        self._target_depth_size = (rgb_width, rgb_height)  # (W, H)

        pipeline = dai.Pipeline()

        # Color camera (RGB)
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_12_MP
        )
        cam_rgb.setPreviewSize(rgb_width, rgb_height)
        cam_rgb.setFps(self._fps)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(
            dai.ColorCameraProperties.ColorOrder.RGB
        )
        ctrl = cam_rgb.initialControl
        ctrl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.OFF
        )
        ctrl.setAutoExposureEnable()
        ctrl.setAutoWhiteBalanceMode(
            dai.CameraControl.AutoWhiteBalanceMode.AUTO
        )

        # Mono cameras (for Stereo Depth)
        mono_res = (
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
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
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.DEFAULT
        )
        stereo.initialConfig.setMedianFilter(
            dai.MedianFilter.KERNEL_5x5
        )
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)

        # Enable hardware alignment
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Force depth output to match RGB resolution (416x240)
        stereo.setOutputSize(rgb_width, rgb_height)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # XLink outputs (aligned)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)  # 416x240

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)  # 416x240 (aligned)

        # Create device
        try:
            if self._device_id:
                found, dev_info = dai.Device.getDeviceByMxId(
                    self._device_id
                )
                if not found:
                    raise RuntimeError(
                        f"Device ID {self._device_id} not found"
                    )
                self._device = dai.Device(
                    pipeline, dev_info, dai.UsbSpeed.SUPER_PLUS
                )
            else:
                self._device = dai.Device(pipeline)
        except RuntimeError as e:
            print(
                f"Failed to start OAK-D device "
                f"(ID: {self._device_id}). Error: {e}"
            )
            raise

        # Output queues
        self._q_rgb = self._device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False
        )
        self._q_depth = self._device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )

        time.sleep(1.0)  # Wait for streams to stabilize

    def read(
        self, img_size: Optional[Tuple[int, int]] = None
    ):  # img_size is (H, W)
        rgb_packet = self._q_rgb.tryGet()
        if rgb_packet is None:
            rgb_packet = self._q_rgb.get()
        depth_packet = self._q_depth.tryGet()
        if depth_packet is None:
            depth_packet = self._q_depth.get()

        # Both frames now have the same (240, 416) size
        image = rgb_packet.getCvFrame()  # (240, 416, 3) RGB
        depth_frame = depth_packet.getFrame()  # (240, 416) uint16
        depth = depth_frame

        if img_size is not None and (
            img_size[0] != image.shape[0]
            or img_size[1] != image.shape[1]
        ):
            image_resized = cv2.resize(
                image,
                (img_size[1], img_size[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            image_resized = image

        if img_size is not None and (
            img_size[0] != depth.shape[0]
            or img_size[1] != depth.shape[1]
        ):
            depth_resized = cv2.resize(
                depth,
                (img_size[1], img_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            depth_resized = depth

        image_final = image_resized
        depth_final = depth_resized

        if self._flip:
            image_final = cv2.rotate(image_final, cv2.ROTATE_180)
            depth_final = cv2.rotate(depth_final, cv2.ROTATE_180)

        if depth_final.ndim == 2:
            depth_final = depth_final[:, :, None]  # Ensure HxWx1

        return image_final, depth_final

    def __del__(self):
        """Ensure device is closed on destruction."""
        if hasattr(self, "_device"):
            self._device.close()


def _debug_read(
    camera: CameraDriver, save_datastream: bool = False
):
    import os

    import cv2

    cv2.namedWindow("image (RGB)")
    cv2.namedWindow("depth (Aligned)")
    print("--- Debug started ---")
    print(" Press 's' to save (to 'stream_oakd/' directory)")
    print(" Press 'ESC' to exit")
    counter = 0
    save_dir = "stream_oakd"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    while True:
        try:
            image, depth = camera.read()
            image_bgr_display = image[:, :, ::-1]  # CV2 needs BGR
            depth_viz = cv2.normalize(
                depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
            )
            depth_viz_color = cv2.applyColorMap(
                depth_viz, cv2.COLORMAP_JET
            )
            if image.shape[:2] != depth.shape[:2]:
                print(
                    f"WARNING: Resolution mismatch! "
                    f"RGB: {image.shape}, Depth: {depth.shape}"
                )
                depth_viz_color = cv2.resize(
                    depth_viz_color,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow("image (RGB)", image_bgr_display)
            cv2.imshow("depth (Aligned)", depth_viz_color)
            key = cv2.waitKey(1)
            if key == ord("s"):
                img_path = os.path.join(
                    save_dir, f"image_{counter}.png"
                )
                cv2.imwrite(img_path, image_bgr_display)
                depth_path = os.path.join(
                    save_dir, f"depth_{counter}.png"
                )
                cv2.imwrite(depth_path, depth)
                print(f"Saved: {img_path} and {depth_path}")
                counter += 1
            if save_datastream:
                img_path = os.path.join(
                    save_dir, f"image_{counter}.png"
                )
                cv2.imwrite(img_path, image_bgr_display)
                depth_path = os.path.join(
                    save_dir, f"depth_{counter}.png"
                )
                cv2.imwrite(depth_path, depth)
                counter += 1
            if key == 27:
                print("--- Debug exited ---")
                break
        except KeyboardInterrupt:
            print("--- Debug exited ---")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import os

    import cv2

    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} OAK-D devices:")
    print(device_ids)
    if len(device_ids) > 0:
        device_to_use = device_ids[0]
        print(f"Connecting to device: {device_to_use}")
        oak_cam = OAKDCamera(device_id=device_to_use, flip=True)
        try:
            im, depth = oak_cam.read()
            print("\n--- First read successful ---")
            print(f"Image resolution (H, W, C): {im.shape}")
            print(f"Depth resolution (H, W, C): {depth.shape}")
            print(f"Depth dtype: {depth.dtype}")
            print("----------------------\n")
            _debug_read(oak_cam, save_datastream=False)
        except Exception as e:
            print(f"Error reading camera: {e}")
    else:
        print("No OAK-D devices found. Check connections.")
