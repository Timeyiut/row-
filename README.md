# medippi-rowing (Rowing Analysis System)

本專案為「medippi 划船運動表現分析系統」之開發原型：以手機/攝影機即時姿態辨識（MediaPipe）萃取關節角度與動作分期，並整合穿戴裝置心率（BLE + ANT+）與表現數據（可擴充功率/划頻/距離），透過 WebSocket 即時推送至大螢幕儀表板，並作為後續訓練處方生成基礎。

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .

python scripts/run_local.py
