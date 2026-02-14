import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gello.cameras.camera import CameraDriver


def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        device_ids.append(dev.get_info(rs.camera_info.serial_number))

    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id}, align=True)"

    def __init__(
        self, device_id: Optional[str] = None, flip: bool = False
    ):
        import pyrealsense2 as rs

        self._device_id = device_id
        self._pipeline = rs.pipeline()
        config = rs.config()

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            if not devices:
                raise RuntimeError("No RealSense device found.")
            print("No device_id specified, using first RealSense device.")
        else:
            config.enable_device(device_id)

        # RGB: 424x240 @ 30Hz, Depth: 480x270 @ 30Hz
        rgb_width, rgb_height = 424, 240
        dep_width, dep_height = 480, 270
        fps = 30

        try:
            print(
                f"Configuring RealSense: Depth {dep_width}x{dep_height} "
                f"@ {fps}Hz, Color {rgb_width}x{rgb_height} @ {fps}Hz"
            )
            config.enable_stream(
                rs.stream.depth, dep_width, dep_height, rs.format.z16, fps
            )
            config.enable_stream(
                rs.stream.color,
                rgb_width,
                rgb_height,
                rs.format.bgr8,
                fps,
            )

            # Store target RGB size (H, W) for dummy frames
            self._target_rgb_size = (rgb_height, rgb_width)

        except RuntimeError as e:
            print(f"Failed to configure RealSense: {e}")
            raise

        # Start pipeline
        pipeline_profile = self._pipeline.start(config)

        # Create Align object (align depth to color stream)
        self._align = rs.align(rs.stream.color)
        print("RealSense Aligner created (Depth -> Color).")

        self._flip = flip
        time.sleep(1.0)  # Wait for streams to stabilize

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        frames = None
        try:
            frames = self._pipeline.wait_for_frames(5000)
        except RuntimeError as e:
            print(f"RealSense error: {e}. Possibly 'Frame didn't arrive'.")

        if not frames:
            print("Warning: RealSense received no frameset. Returning dummy.")
            h, w = self._target_rgb_size
            dummy_color = np.zeros((h, w, 3), dtype=np.uint8)
            dummy_depth = np.zeros((h, w, 1), dtype=np.uint16)
            return dummy_color, dummy_depth

        # Perform alignment
        aligned_frames = self._align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()

        if not aligned_depth_frame or not color_frame:
            print("Warning: Missing color/depth frame. Returning dummy.")
            h, w = self._target_rgb_size
            dummy_color = np.zeros((h, w, 3), dtype=np.uint8)
            dummy_depth = np.zeros((h, w, 1), dtype=np.uint16)
            return dummy_color, dummy_depth

        # Both images are now (240, 424) after alignment
        color_image = np.asanyarray(color_frame.get_data())  # BGR
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # uint16

        if img_size is not None and (
            img_size[0] != color_image.shape[0]
            or img_size[1] != color_image.shape[1]
        ):
            image_resized = cv2.resize(
                color_image,
                (img_size[1], img_size[0]),
                interpolation=cv2.INTER_AREA,
            )[:, :, ::-1]  # BGR to RGB
        else:
            image_resized = color_image[:, :, ::-1]  # BGR to RGB

        if img_size is not None and (
            img_size[0] != depth_image.shape[0]
            or img_size[1] != depth_image.shape[1]
        ):
            depth_resized = cv2.resize(
                depth_image,
                (img_size[1], img_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            depth_resized = depth_image

        image_final = image_resized
        depth_final = depth_resized

        if self._flip:
            image_final = cv2.rotate(image_final, cv2.ROTATE_180)
            depth_final = cv2.rotate(depth_final, cv2.ROTATE_180)

        # Ensure depth shape is HxWx1
        if depth_final.ndim == 2:
            depth_final = depth_final[:, :, None]

        return image_final, depth_final

    def __del__(self):
        """Ensure pipeline is stopped."""
        if hasattr(self, "_pipeline"):
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image (RGB)")
    cv2.namedWindow("depth (Aligned)")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")

    print("Press 's' to save, 'ESC' to exit...")

    while True:
        image, depth = camera.read()

        # Visualize depth
        depth_viz = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

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

        cv2.imshow("image (RGB)", image[:, :, ::-1])
        cv2.imshow("depth (Aligned)", depth_viz_color)

        key = cv2.waitKey(1)
        if key == ord("s"):
            cv2.imwrite(
                f"images/image_{counter}.png", image[:, :, ::-1]
            )
            cv2.imwrite(f"images/depth_{counter}.png", depth)
            print(f"Saved image_{counter}.png, depth_{counter}.png")
        if save_datastream:
            cv2.imwrite(
                f"stream/image_{counter}.png", image[:, :, ::-1]
            )
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
        print("--- First read ---")
        print(f"RGB shape: {im.shape}, dtype: {im.dtype}")
        print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
        print("-----------------")
        _debug_read(rs, save_datastream=True)
    else:
        print("No RealSense device found.")
