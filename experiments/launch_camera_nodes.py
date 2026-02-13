# launch_camera_nodes.py (最终方案 B + 简化消息)
import time
from dataclasses import dataclass
from typing import List
import tyro
import multiprocessing as mp 

# (我们只需要在父进程中导入这些来获取 ID)
from gello.cameras.realsense_camera import get_device_ids as get_realsense_ids
from gello.cameras.oakd_camera import get_device_ids as get_oakd_ids

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
    flip: bool
):
    """
    一个异步的 PUB/SUB 服务器 (简化消息)。
    它在子进程内部初始化相机，并直接发送 pickle 后的 payload。
    """
    import zmq
    import pickle
    import time
    
    camera_driver = None
    try:
        # (相机初始化逻辑不变)
        if camera_type == "realsense":
            from gello.cameras.realsense_camera import RealSenseCamera
            camera_driver = RealSenseCamera(device_id=device_id, flip=flip)
        elif camera_type == "oakd":
            from gello.cameras.oakd_camera import OAKDCamera
            camera_driver = OAKDCamera(device_id=device_id, flip=flip)
        else: raise ValueError(f"Unknown camera type: {camera_type}")
    except Exception as e:
        print(f"❌ {camera_name} ({camera_type}) initialization failed: {e}")
        return 

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    addr = f"tcp://{host}:{port}"
    socket.bind(addr)
    # [关键修改] 更新打印信息
    print(f"✅ Starting Async Camera PUBLISHER: {camera_name} on {addr} (30Hz, Simple Msg)")

    # [关键修改] 更新目标 sleep 时间为 30Hz
    target_sleep = 1.0 / 30.0 

    while True:
        try:
            start_time = time.time()
            image, depth = camera_driver.read()
            timestamp = time.time()
            payload = (timestamp, image, depth)
            socket.send(pickle.dumps(payload))
            
            time_to_sleep = target_sleep - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Camera PUBLISHER Error ({camera_name}): {e}")
            time.sleep(1) 


def main(args: Args):
    processes: List[mp.Process] = []
    print("--- 正在启动 异步 (PUB/SUB, 简单消息) 相机服务器 ---")

    # (查找 RealSense ID 的逻辑保持不变)
    try:
        realsense_ids = get_realsense_ids()
        if realsense_ids:
            print(f"🔍 找到 RealSense: {realsense_ids[0]}. 分配为 'wrist' on port {args.wrist_port}")
            rs_process = mp.Process(
                target=launch_publisher_server_with_init,
                args=( "realsense", realsense_ids[0], args.wrist_port,
                       args.hostname, "wrist", False ),
            )
            processes.append(rs_process)
        else: print("⚠️ 未找到 RealSense 设备。")
    except Exception as e: print(f"❌ 查找 RealSense 失败: {e}")

    time.sleep(1) 

    # (查找 OAK-D ID 的逻辑保持不变)
    try:
        oakd_ids = get_oakd_ids()
        if oakd_ids:
            print(f"🔍 找到 OAK-D: {oakd_ids[0]}. 分配为 'base' on port {args.base_port}")
            oak_process = mp.Process(
                target=launch_publisher_server_with_init,
                args=( "oakd", oakd_ids[0], args.base_port,
                       args.hostname, "base", True ),
            )
            processes.append(oak_process)
        else: print("⚠️ 未找到 OAK-D 设备。")
    except Exception as e: print(f"❌ 查找 OAK-D 失败: {e}")

    if not processes:
        print("❌ 没有可启动的相机服务器。退出。")
        return
    
    print(f"\n🚀 正在启动 {len(processes)} 个相机 PUBLISHER 进程...")
    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n🛑 正在关闭所有相机进程...")
        for p in processes:
            p.terminate()
            p.join(timeout=1)
        print("✅ 所有相机服务器已关闭。")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("... 已将多进程启动模式设置为 'spawn' ...")
    except RuntimeError: pass
    
    main(tyro.cli(Args))


