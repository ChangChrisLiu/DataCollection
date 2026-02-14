# launch_camera_nodes.py
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import List

import tyro

from gello.cameras.oakd_camera import get_device_ids as get_oakd_ids
from gello.cameras.realsense_camera import (
    get_device_ids as get_realsense_ids,
)


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    wrist_port: int = 5000
    base_port: int = 5001


def launch_publisher_server_with_init(
    camera_type: str,
    device_id: str,
    port: int,
    host: str,
    camera_name: str,
    flip: bool,
):
    """Async PUB/SUB camera server.

    Initializes camera in subprocess, publishes pickled
    (timestamp, image, depth) payloads at ~30Hz.
    """
    import pickle
    import time

    import zmq

    camera_driver = None
    try:
        if camera_type == "realsense":
            from gello.cameras.realsense_camera import (
                RealSenseCamera,
            )

            camera_driver = RealSenseCamera(
                device_id=device_id, flip=flip
            )
        elif camera_type == "oakd":
            from gello.cameras.oakd_camera import OAKDCamera

            camera_driver = OAKDCamera(
                device_id=device_id, flip=flip
            )
        else:
            raise ValueError(
                f"Unknown camera type: {camera_type}"
            )
    except Exception as e:
        print(
            f"{camera_name} ({camera_type}) "
            f"initialization failed: {e}"
        )
        return

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    addr = f"tcp://{host}:{port}"
    socket.bind(addr)
    print(
        f"Starting Async Camera PUBLISHER: "
        f"{camera_name} on {addr} (30Hz)"
    )

    target_sleep = 1.0 / 30.0

    while True:
        try:
            start_time = time.time()
            image, depth = camera_driver.read()
            timestamp = time.time()
            payload = (timestamp, image, depth)
            socket.send(pickle.dumps(payload))

            time_to_sleep = target_sleep - (
                time.time() - start_time
            )
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(
                f"Camera PUBLISHER Error ({camera_name}): {e}"
            )
            time.sleep(1)


def main(args: Args):
    processes: List[mp.Process] = []
    print("--- Starting async (PUB/SUB) camera servers ---")

    # Find RealSense
    try:
        realsense_ids = get_realsense_ids()
        if realsense_ids:
            print(
                f"Found RealSense: {realsense_ids[0]}. "
                f"Assigned as 'wrist' on port {args.wrist_port}"
            )
            rs_process = mp.Process(
                target=launch_publisher_server_with_init,
                args=(
                    "realsense",
                    realsense_ids[0],
                    args.wrist_port,
                    args.hostname,
                    "wrist",
                    False,
                ),
            )
            processes.append(rs_process)
        else:
            print("No RealSense device found.")
    except Exception as e:
        print(f"RealSense lookup failed: {e}")

    time.sleep(1)

    # Find OAK-D
    try:
        oakd_ids = get_oakd_ids()
        if oakd_ids:
            print(
                f"Found OAK-D: {oakd_ids[0]}. "
                f"Assigned as 'base' on port {args.base_port}"
            )
            oak_process = mp.Process(
                target=launch_publisher_server_with_init,
                args=(
                    "oakd",
                    oakd_ids[0],
                    args.base_port,
                    args.hostname,
                    "base",
                    True,
                ),
            )
            processes.append(oak_process)
        else:
            print("No OAK-D device found.")
    except Exception as e:
        print(f"OAK-D lookup failed: {e}")

    if not processes:
        print("No camera servers to start. Exiting.")
        return

    print(
        f"\nStarting {len(processes)} camera PUBLISHER processes..."
    )
    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down all camera processes...")
        for p in processes:
            p.terminate()
            p.join(timeout=1)
        print("All camera servers shut down.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("... Set multiprocessing start method to 'spawn' ...")
    except RuntimeError:
        pass

    main(tyro.cli(Args))
