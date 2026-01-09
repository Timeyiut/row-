import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

# ---------------------------
# Config
# ---------------------------
INPUT_CSV = "rowing_data_complete.csv"
OUT_DIR = "analysis_out"

# 若你有真實 HRmax，請改成實測；暫用估算也可
DEFAULT_HR_MAX = 190

@dataclass
class AnalysisConfig:
    hr_max: int = DEFAULT_HR_MAX
    sg_window: int = 11     # 建議奇數：9/11/13
    sg_poly: int = 2
    resample_hz: float = 30.0  # 將資料重取樣到固定頻率（與 camera fps 接近）


# ---------------------------
# Helpers
# ---------------------------
def robust_clip(series: pd.Series, lo: float, hi: float) -> pd.Series:
    s = series.copy()
    s[(s < lo) | (s > hi)] = np.nan
    return s.interpolate(limit=10).bfill().ffill()

def smooth(series: pd.Series, window: int, poly: int) -> pd.Series:
    x = series.to_numpy()
    if len(x) < window:
        return series
    # 補 NaN
    x = pd.Series(x).interpolate().bfill().ffill().to_numpy()
    y = savgol_filter(x, window_length=window, polyorder=poly)
    return pd.Series(y, index=series.index)

def add_derivative(df: pd.DataFrame, col: str, tcol: str = "Timestamp") -> pd.Series:
    t = df[tcol].to_numpy()
    x = df[col].to_numpy()
    dt = np.diff(t, prepend=t[0])
    dx = np.diff(x, prepend=x[0])
    dt[dt == 0] = np.nan
    d = dx / dt
    return pd.Series(d).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def hr_zones(hr: pd.Series, hr_max: int) -> dict:
    # 簡化 5 zones：50-60-70-80-90-100% HRmax
    bounds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    zones = {}
    for i in range(5):
        lo = bounds[i] * hr_max
        hi = bounds[i+1] * hr_max
        zones[f"Z{i+1}"] = ((hr >= lo) & (hr < hi)).mean()
    return zones

# ---------------------------
# Stroke detection (MVP)
# ---------------------------
def detect_strokes(df: pd.DataFrame, knee_col="Knee_smooth"):
    """
    以 Knee 角度的「谷值」近似 Catch（膝角最小）作為每槳邊界。
    - distance 用於限制最短週期（避免雜訊多抓）
    """
    knee = df[knee_col].to_numpy()
    # 膝角谷值 -> 找 peaks on -knee
    fps_est = 1.0 / np.median(np.diff(df["Timestamp"].to_numpy()))
    # 假設最短一槳 0.7s（可依實測調）
    min_dist = int(0.7 * fps_est)
    peaks, _ = find_peaks(-knee, distance=max(min_dist, 1), prominence=2.0)
    return peaks

def per_stroke_stats(df: pd.DataFrame, stroke_idx: np.ndarray):
    rows = []
    ts = df["Timestamp"].to_numpy()

    for i in range(len(stroke_idx) - 1):
        a = stroke_idx[i]
        b = stroke_idx[i+1]
        seg = df.iloc[a:b].copy()
        if len(seg) < 5:
            continue

        t0, t1 = ts[a], ts[b-1]
        dur = max(t1 - t0, 1e-6)
        spm = 60.0 / dur

        def stats(col):
            x = seg[col].to_numpy()
            return {
                "min": float(np.nanmin(x)),
                "max": float(np.nanmax(x)),
                "mean": float(np.nanmean(x)),
                "rom": float(np.nanmax(x) - np.nanmin(x)),
                "sd": float(np.nanstd(x)),
                "cv": float(np.nanstd(x) / max(np.nanmean(x), 1e-6)),
            }

        row = {
            "stroke_id": i,
            "t_start": float(t0),
            "t_end": float(t1),
            "duration_s": float(dur),
            "spm": float(spm),
        }
        row.update({f"knee_{k}": v for k, v in stats("Knee_smooth").items()})
        row.update({f"hip_{k}": v for k, v in stats("Hip_smooth").items()})
        row.update({f"elbow_{k}": v for k, v in stats("Elbow_smooth").items()})
        row["hr_mean"] = float(np.nanmean(seg["HR_smooth"].to_numpy()))
        rows.append(row)

    return pd.DataFrame(rows)

# ---------------------------
# Main analysis
# ---------------------------
def main():
    cfg = AnalysisConfig()
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    # 基本排序與去重
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp")

    # 清理：合理範圍裁切 + 插補
    df["HeartRate"] = robust_clip(df["HeartRate"], lo=30, hi=220)
    df["Knee"] = robust_clip(df["Knee"], lo=0, hi=180)
    df["Hip"] = robust_clip(df["Hip"], lo=0, hi=180)
    df["Elbow"] = robust_clip(df["Elbow"], lo=0, hi=180)

    # 重取樣到固定頻率（利於差分、頻域、比較）
    t0, t1 = df["Timestamp"].iloc[0], df["Timestamp"].iloc[-1]
    dt = 1.0 / cfg.resample_hz
    t_grid = np.arange(t0, t1, dt)
    df_rs = pd.DataFrame({"Timestamp": t_grid})

    for col in ["HeartRate", "Knee", "Hip", "Elbow"]:
        df_rs[col] = np.interp(t_grid, df["Timestamp"], df[col])

    # 平滑
    df_rs["HR_smooth"] = smooth(df_rs["HeartRate"], cfg.sg_window, cfg.sg_poly)
    df_rs["Knee_smooth"] = smooth(df_rs["Knee"], cfg.sg_window, cfg.sg_poly)
    df_rs["Hip_smooth"] = smooth(df_rs["Hip"], cfg.sg_window, cfg.sg_poly)
    df_rs["Elbow_smooth"] = smooth(df_rs["Elbow"], cfg.sg_window, cfg.sg_poly)

    # 角速度（可當爆發力/節奏 proxy）
    df_rs["dKnee_dt"] = add_derivative(df_rs, "Knee_smooth")
    df_rs["dHip_dt"] = add_derivative(df_rs, "Hip_smooth")
    df_rs["dElbow_dt"] = add_derivative(df_rs, "Elbow_smooth")
    df_rs["dHR_dt"] = add_derivative(df_rs, "HR_smooth")

    # phase（你原始資料有 phase，但重取樣後要對齊；MVP：用最近點）
    if "Phase" in df.columns:
        # 用 merge_asof 把 phase 貼到 df_rs
        df_phase = df[["Timestamp", "Phase"]].copy().sort_values("Timestamp")
        df_rs = pd.merge_asof(df_rs.sort_values("Timestamp"), df_phase, on="Timestamp", direction="nearest")

    # ---------------- summary: HR ----------------
    hr = df_rs["HR_smooth"]
    hr_summary = {
        "hr_mean": float(hr.mean()),
        "hr_max": float(hr.max()),
        "hr_min": float(hr.min()),
        "hr_sd": float(hr.std()),
        "hr_zones_ratio": hr_zones(hr, cfg.hr_max),
        "dhr_dt_mean": float(df_rs["dHR_dt"].abs().mean()),
    }

    # ---------------- summary: angles ----------------
    def angle_summary(col):
        x = df_rs[col]
        return {
            "mean": float(x.mean()),
            "min": float(x.min()),
            "max": float(x.max()),
            "rom": float(x.max() - x.min()),
            "sd": float(x.std()),
            "cv": float(x.std() / max(x.mean(), 1e-6)),
        }

    angles_summary = {
        "knee": angle_summary("Knee_smooth"),
        "hip": angle_summary("Hip_smooth"),
        "elbow": angle_summary("Elbow_smooth"),
        "knee_speed_abs_mean": float(df_rs["dKnee_dt"].abs().mean()),
        "hip_speed_abs_mean": float(df_rs["dHip_dt"].abs().mean()),
        "elbow_speed_abs_mean": float(df_rs["dElbow_dt"].abs().mean()),
    }

    # ---------------- phase summary ----------------
    phase_summary_df = None
    if "Phase" in df_rs.columns:
        g = df_rs.groupby("Phase")
        phase_summary_df = g.agg(
            hr_mean=("HR_smooth", "mean"),
            knee_mean=("Knee_smooth", "mean"),
            knee_rom=("Knee_smooth", lambda s: float(s.max()-s.min())),
            hip_mean=("Hip_smooth", "mean"),
            hip_rom=("Hip_smooth", lambda s: float(s.max()-s.min())),
            elbow_mean=("Elbow_smooth", "mean"),
            elbow_rom=("Elbow_smooth", lambda s: float(s.max()-s.min())),
            n=("Phase", "size"),
        ).reset_index()

    # ---------------- stroke detection + per-stroke ----------------
    stroke_idx = detect_strokes(df_rs, knee_col="Knee_smooth")
    stroke_df = per_stroke_stats(df_rs, stroke_idx)

    # 由每槳估計 SPM（更穩）
    spm_summary = {}
    if len(stroke_df) >= 3:
        spm_summary = {
            "spm_mean": float(stroke_df["spm"].mean()),
            "spm_sd": float(stroke_df["spm"].std()),
            "strokes_count": int(len(stroke_df)),
        }

    # ---------------- multimodal coupling (簡單關聯) ----------------
    # HR 與 SPM、角度穩定性（以每槳 CV）做關係（MVP：相關係數）
    coupling = {}
    if len(stroke_df) >= 10:
        coupling = {
            "corr_hr_vs_spm": float(stroke_df["hr_mean"].corr(stroke_df["spm"])),
            "corr_hr_vs_knee_rom": float(stroke_df["hr_mean"].corr(stroke_df["knee_rom"])),
            "corr_hr_vs_knee_cv": float(stroke_df["hr_mean"].corr(stroke_df["knee_cv"])),
        }

    # ---------------- export ----------------
    session_summary = {
        "config": cfg.__dict__,
        "duration_s": float(df_rs["Timestamp"].iloc[-1] - df_rs["Timestamp"].iloc[0]),
        "hr": hr_summary,
        "angles": angles_summary,
        "spm": spm_summary,
        "coupling": coupling,
    }

    (out / "session_summary.json").write_text(json.dumps(session_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df_rs.to_csv(out / "timeseries_resampled.csv", index=False, encoding="utf-8")

    if phase_summary_df is not None:
        phase_summary_df.to_csv(out / "phase_summary.csv", index=False, encoding="utf-8")

    if len(stroke_df) > 0:
        stroke_df.to_csv(out / "stroke_summary.csv", index=False, encoding="utf-8")

    print(f"[OK] analysis exported to: {out.resolve()}")
    print(f" - session_summary.json")
    print(f" - timeseries_resampled.csv")
    if phase_summary_df is not None:
        print(f" - phase_summary.csv")
    if len(stroke_df) > 0:
        print(f" - stroke_summary.csv")


if __name__ == "__main__":
    main()

