import asyncio
import csv
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from bleak import BleakClient

# ===================== 設定 =====================
DEVICE_ADDRESS = "90:f1:57:ea:f0:45"
HEART_RATE_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

CAMERA_INDEX = 1
OUTPUT_CSV = "rowing_data_complete.csv"
RR_CSV = "rr_events.csv"

# ===================== 心率狀態 =====================
@dataclass
class HRState:
    hr: int = 0
    connected: bool = False
    rr_events: deque = deque(maxlen=10000)

    def update(self, hr, rr_list=None):
        self.hr = hr
        if rr_list:
            ts = time.time()
            for rr in rr_list:
                self.rr_events.append((ts, rr))

# ===================== BLE HR + RR =====================
class BleHRMonitor:
    def __init__(self, state: HRState):
        self.state = state

    def handler(self, sender, data):
        flags = data[0]
        hr_16bit = flags & 0x01
        rr_present = flags & 0x10

        idx = 1
        if hr_16bit:
            hr = int.from_bytes(data[idx:idx+2], "little")
            idx += 2
        else:
            hr = data[idx]
            idx += 1

        rr_list = []
        if rr_present:
            while idx + 1 < len(data):
                rr_1024 = int.from_bytes(data[idx:idx+2], "little")
                rr_ms = rr_1024 * 1000 / 1024
                rr_list.append(rr_ms)
                idx += 2

        self.state.update(hr, rr_list)

async def ble_task(state: HRState):
    monitor = BleHRMonitor(state)
    while True:
        try:
            async with BleakClient(DEVICE_ADDRESS) as client:
                state.connected = True
                await client.start_notify(HEART_RATE_UUID, monitor.handler)
                while client.is_connected:
                    await asyncio.sleep(1)
        except Exception:
            state.connected = False
            await asyncio.sleep(2)

# ===================== 姿態分析 =====================
class PoseAnalyzer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()
        self.draw = mp.solutions.drawing_utils

    @staticmethod
    def angle(a, b, c):
        a, b, c = map(np.array, (a, b, c))
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        deg = abs(rad * 180 / np.pi)
        return int(360 - deg if deg > 180 else deg)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        angles = {"knee": 0, "hip": 0, "elbow": 0}
        phase = "Ready"

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            h, w, _ = frame.shape
            pt = lambda i: (lm[i].x * w, lm[i].y * h)

            angles["knee"] = self.angle(pt(23), pt(25), pt(27))
            angles["hip"] = self.angle(pt(11), pt(23), pt(25))
            angles["elbow"] = self.angle(pt(11), pt(13), pt(15))

            if angles["knee"] < 90:
                phase = "Catch"
            elif angles["knee"] > 150:
                phase = "Finish"
            else:
                phase = "Drive"

        return res, angles, phase

# ===================== 主程式 =====================
async def main():
    hr_state = HRState()
    analyzer = PoseAnalyzer()

    asyncio.create_task(ble_task(hr_state))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    csv_main = open(OUTPUT_CSV, "w", newline="")
    csv_rr = open(RR_CSV, "w", newline="")

    writer = csv.writer(csv_main)
    rr_writer = csv.writer(csv_rr)

    writer.writerow(["ts", "hr", "knee", "hip", "elbow", "phase"])
    rr_writer.writerow(["ts", "rr_ms"])

    last_rr_len = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts = time.time()
        res, angles, phase = analyzer.process(frame)

        writer.writerow([ts, hr_state.hr,
                         angles["knee"], angles["hip"],
                         angles["elbow"], phase])

        if len(hr_state.rr_events) > last_rr_len:
            for rr in list(hr_state.rr_events)[last_rr_len:]:
                rr_writer.writerow(rr)
            last_rr_len = len(hr_state.rr_events)

        if res.pose_landmarks:
            analyzer.draw.draw_landmarks(
                frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        cv2.putText(frame, f"HR: {hr_state.hr}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Rowing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        await asyncio.sleep(0.01)

    cap.release()
    csv_main.close()
    csv_rr.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
