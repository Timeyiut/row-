import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("analysis_out")
OUT.mkdir(exist_ok=True)

RR_FILE = Path("rr_events.csv")

def clean_rr(rr):
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 10:
        return rr
    med = np.median(rr)
    mad = np.median(np.abs(rr - med)) + 1e-6
    z = 0.6745 * (rr - med) / mad
    return rr[np.abs(z) < 5]

def hrv(rr):
    rr = clean_rr(rr)
    if len(rr) < 10:
        return {"available": False}

    diff = np.diff(rr)
    return {
        "available": True,
        "n": int(len(rr)),
        "mean_rr": float(rr.mean()),
        "mean_hr": float(60000 / rr.mean()),
        "sdnn": float(rr.std(ddof=1)),
        "rmssd": float(np.sqrt(np.mean(diff**2))),
        "pnn50": float(np.mean(np.abs(diff) > 50))
    }

def main():
    df = pd.read_csv(RR_FILE)
    rr = df["rr_ms"].values

    result = hrv(rr)
    out = OUT / "hrv_summary.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("產生:", out)

if __name__ == "__main__":
    main()
