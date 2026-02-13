# experiments/launch_gello_node.py (最终 7-DOF 版)
import glob
from dataclasses import dataclass
import tyro
import time
import pickle
import zmq
from gello.agents.gello_agent import GelloAgent

@dataclass
class Args:
    gello_port: str = None
    hostname: str = "127.0.0.1"
    gello_server_port: int = 6000

def launch_gello_server(args: Args):
    gello_port = args.gello_port
    if gello_port is None:
        usb_ports = glob.glob("/dev/serial/by-id/*FTDI_USB__-__Serial_Converter*")
        print(f"Found {len(usb_ports)} FTDI ports")
        if len(usb_ports) > 0: gello_port = usb_ports[0]
        else: raise ValueError("No Gello port found.")
        print(f"Using Gello port: {gello_port}")

    # GelloAgent 默认是 7-DOF
    agent = GelloAgent(port=gello_port)
    print("✅ Gello Agent 硬件已初始化 (7-DOF)。")
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    addr = f"tcp://{args.hostname}:{args.gello_server_port}"
    socket.bind(addr)
    print(f"✅ Gello REQ/REP Server 正在监听: {addr}")

    while True:
        try:
            message = socket.recv()
            obs = pickle.loads(message)
            
            # [关键修复] 运行并发送完整的 7-DOF 动作
            action_7dof = agent.act(obs)
            socket.send(pickle.dumps(action_7dof))

        except KeyboardInterrupt: 
            print("\n🛑 正在关闭 Gello Server...")
            break
        except Exception as e:
            print(f"❌ Gello Server 错误: {e}")
            socket.send(pickle.dumps(None))

if __name__ == "__main__":
    launch_gello_server(tyro.cli(Args))


