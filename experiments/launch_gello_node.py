# experiments/launch_gello_node.py
"""Agent server (T3) — wraps a GELLO device as a ZMQ REQ/REP server.

The control loop (T4) sends observations and receives joint-angle actions.
Currently only GELLO requires a dedicated server process (joystick and
spacemouse run in-process inside T4).
"""

import glob
from dataclasses import dataclass

import pickle
import zmq
import tyro

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
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
        else:
            raise ValueError(
                "No Gello port found. Plug in your GELLO or specify --gello-port."
            )
        print(f"Using Gello port: {gello_port}")

    agent = GelloAgent(port=gello_port)
    print(f"Gello Agent initialized ({agent.num_dofs}-DOF).")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    addr = f"tcp://{args.hostname}:{args.gello_server_port}"
    socket.bind(addr)
    print(f"Gello REQ/REP Server listening on: {addr}")

    while True:
        try:
            message = socket.recv()
            obs = pickle.loads(message)
            action = agent.act(obs)
            socket.send(pickle.dumps(action))
        except KeyboardInterrupt:
            print("\nShutting down Gello Server...")
            break
        except Exception as e:
            print(f"Gello Server error: {e}")
            try:
                socket.send(pickle.dumps(None))
            except Exception:
                pass  # socket may be in bad state


if __name__ == "__main__":
    launch_gello_server(tyro.cli(Args))
