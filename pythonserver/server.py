import asyncio
import os
import sys
import threading
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import prediction
from servo_control.foosball_serial_bridge import FoosballSerialBridge
from servo_control.bridge_config import REDIS_HOST, REDIS_PORT

# Set DRY_RUN=0 to send real commands; default is dry-run for safety
_DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"
bridge = FoosballSerialBridge(redis_host=REDIS_HOST, redis_port=REDIS_PORT, dry_run=_DRY_RUN)

clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None
_loop_ready = threading.Event()


def on_tracker_message(message: dict) -> None:
    """Called from the main thread — schedules a broadcast on the server event loop."""
    _loop_ready.wait()
    future = asyncio.run_coroutine_threadsafe(broadcast_to_clients(message), _loop)

    def _log_future_error(fut):
        try:
            fut.result()
        except Exception as e:
            print(f"Broadcast scheduling error: {e}")

    future.add_done_callback(_log_future_error)


app = FastAPI()


async def client_heartbeat_loop(websocket: WebSocket) -> None:
    seq = 0
    while True:
        await websocket.send_text(json.dumps({"type": "heartbeat", "seq": seq}))
        seq += 1
        await asyncio.sleep(1.0)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("Client connected")
    await websocket.send_text(json.dumps({"type": "server_hello", "message": "connected"}))
    heartbeat_task = asyncio.create_task(client_heartbeat_loop(websocket))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
                print(f"Received command: {command}")

                command_type = command.get("type")
                if not command_type:
                    print("Invalid command format: missing type")
                    continue

                if command_type == "slide":
                    if command.get("rod") is None or command.get("position") is None:
                        print("Invalid slide command format")
                        continue
                    handle_slide_command(command)
                elif command_type == "kick":
                    if command.get("rod") is None or command.get("level") is None:
                        print("Invalid kick command format")
                        continue
                    handle_kick_command(command)
                elif command_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": command.get("timestamp")}))
                else:
                    print(f"Unknown command type: {command_type}")

            except json.JSONDecodeError as e:
                print(f"Parse error: {e}")

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        heartbeat_task.cancel()
        clients.discard(websocket)


def handle_slide_command(command: dict) -> None:
    print(f"Moving rod {command['rod']} to position {command['position']}")
    # sliding


def handle_kick_command(command: dict) -> None:
    print(f"Kicking rod {command['rod']} with level {command['level']}, direction: {command['direction']}")
    # kick


async def broadcast_to_clients(message: dict) -> None:
    disconnected = set()
    for client in list(clients):
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            disconnected.add(client)
    clients.difference_update(disconnected)


def start_server() -> None:
    async def serve():
        global _loop
        _loop = asyncio.get_running_loop()
        _loop_ready.set()
        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        server.install_signal_handlers = False  # required when not on the main thread
        await server.serve()

    asyncio.run(serve())


if __name__ == "__main__":
    print("WebSocket server running on ws://localhost:8000/ws")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    prediction.run(on_tracker_message, bridge=bridge)
