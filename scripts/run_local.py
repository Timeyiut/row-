import asyncio
import csv
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from bleak import BleakClient
import websockets

# ===================== Config =====================
BLE_DEVICE_ADDRESS = "90:f1:57:ea:f0:45"
BLE_HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
OUTPUT_CSV = "rowing_data_complete.csv"
CAMERA_INDEX = 1
HR_MAX_AGE_S = 2.0

WS_HOST = "0.0.0.0"
WS_PORT = 8765
WS_PATH = "/ws"

# ===================== Shared State =====================
@dataclass
class HRState:
    hr_ble: int | None = None
    ts_ble: float = 0.0
    ble_connected: bool = False

    hr_ant: int | None = None
    ts_ant: float = 0.0
    ant_connected: bool = False

    def update_ble(self, hr: int):
        self.hr_ble = hr
        self.ts_ble = time.time()

    def update_ant(self, hr: int):
        self.hr_ant = hr
        self.ts_ant = time.time()

    def pick(self, max_age_s: float = HR_MAX_AGE_S) -> tuple[int, str]:
        now = time.time()
        if self.hr_ble is not None and (now - self.ts_ble) <= max_age_s:
            return self.hr_ble, "BLE"
        if self.hr_ant is not None and (now - self.ts_ant) <= max_age_s:
            return self.hr_ant, "ANT"
        return 0, "NONE"


# ===================== Vision =====================
class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def calculate_angle(a, b, c) -> int:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return int(angle)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)

        angles = {"knee": 0, "hip": 0, "elbow": 0}
        phase = "Ready"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            def pt(i):
                return [lm[i].x * w, lm[i].y * h]

            # MediaPipe indices: 11 L-shoulder, 13 L-elbow, 15 L-wrist, 23 L-hip, 25 L-knee, 27 L-ankle
            angles["knee"] = self.calculate_angle(pt(23), pt(25), pt(27))
            angles["hip"] = self.calculate_angle(pt(11), pt(23), pt(25))
            angles["elbow"] = self.calculate_angle(pt(11), pt(13), pt(15))

            # MVP phase rule (可後續換成狀態機 + 峰谷)
            if angles["knee"] < 90:
                phase = "Catch"
            elif angles["knee"] > 150 and angles["elbow"] < 100:
                phase = "Finish"
            else:
                phase = "Drive"

        return results, angles, phase


# ===================== BLE HR =====================
class BleHR:
    def __init__(self, state: HRState):
        self.state = state

    def handler(self, sender, data: bytearray):
        flag = data[0]
        hr = data[1] if (flag & 1 == 0) else int.from_bytes(data[1:3], "little")
        self.state.update_ble(int(hr))

async def ble_manager(state: HRState):
    ble = BleHR(state)
    while True:
        if not state.ble_connected:
            try:
                async with BleakClient(BLE_DEVICE_ADDRESS, timeout=10.0) as client:
                    state.ble_connected = True
                    await client.start_notify(BLE_HR_UUID, ble.handler)
                    while client.is_connected:
                        await asyncio.sleep(1)
                    state.ble_connected = False
            except Exception:
                state.ble_connected = False
                await asyncio.sleep(2)
        await asyncio.sleep(1)


# ===================== ANT+ HR (Optional) =====================
def ant_worker(state: HRState):
    """Optional ANT+ HR reader in a background thread.

    Requires:
      - ANT+ USB dongle
      - Compatible Python library (example shown uses openant)
    If missing, the worker will exit and ANT will remain disconnected.
    """
    try:
        from ant.easy.node import Node
        from ant.devices.heart_rate import HeartRate

        node = Node()
        # NOTE: network key config varies by library; keep as placeholder.
        node.set_network_key(0x00, bytes.fromhex("B9A521FBBD72C345"))
        hrm = HeartRate(node)

        def on_data(page, dev: HeartRate):
            if dev.heart_rate is not None:
                state.update_ant(int(dev.heart_rate))
                state.ant_connected = True

        hrm.on_device_data = on_data
        node.start()
        node.join()
    except Exception:
        state.ant_connected = False

async def ant_manager(state: HRState):
    await asyncio.to_thread(ant_worker, state)


# ===================== WebSocket Broadcast =====================
_clients: set[websockets.WebSocketServerProtocol] = set()

async def ws_handler(websocket):
    _clients.add(websocket)
    try:
        # keepalive
        async for _ in websocket:
            pass
    finally:
        _clients.discard(websocket)

async def ws_broadcast(topic: str, data: dict):
    if not _clients:
        return
    payload = json.dumps({"topic": topic, "data": data}, ensure_ascii=False)
    dead = []
    for ws in _clients:
        try:
            await ws.send(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


# ===================== Main =====================
async def main():
    state = HRState()
    analyzer = PoseAnalyzer()

    asyncio.create_task(ble_manager(state))
    asyncio.create_task(ant_manager(state))

    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"WebSocket server listening on ws://localhost:{WS_PORT}{WS_PATH} (path not enforced in MVP)")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("無法開啟攝影機")
        ws_server.close()
        await ws_server.wait_closed()
        return

    csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp", "HeartRate", "HR_Source", "Knee", "Hip", "Elbow", "Phase"])

    print("系統啟動（BLE + ANT+）... 按 q 離開")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ts = time.time()
            results, angles, phase = analyzer.process(frame)

            hr, hr_src = state.pick()
            writer.writerow([ts, hr, hr_src, angles["knee"], angles["hip"], angles["elbow"], phase])

            # Broadcast fused snapshot
            await ws_broadcast("fusion/live", {
                "ts": ts,
                "hr": hr,
                "hr_source": hr_src,
                "phase": phase,
                "angles": angles,
                "ble_connected": state.ble_connected,
                "ant_connected": state.ant_connected,
            })

            if results.pose_landmarks:
                analyzer.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, analyzer.mp_pose.POSE_CONNECTIONS
                )

            # Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (300, 240), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            ble_color = (0, 255, 0) if state.ble_connected else (0, 0, 255)
            ant_color = (0, 255, 0) if state.ant_connected else (0, 0, 255)

            cv2.putText(frame, f"HR({hr_src}): {hr} bpm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"BLE: {'OK' if state.ble_connected else '...'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ble_color, 2)
            cv2.putText(frame, f"ANT: {'OK' if state.ant_connected else '...'}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ant_color, 2)

            cv2.putText(frame, f"Knee:  {angles['knee']}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Hip:   {angles['hip']}",  (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Elbow: {angles['elbow']}",(10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Phase: {phase}",          (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("Rowing Analysis System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.01)

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        ws_server.close()
        await ws_server.wait_closed()
        print("程式結束")

if __name__ == "__main__":
    asyncio.run(main())
