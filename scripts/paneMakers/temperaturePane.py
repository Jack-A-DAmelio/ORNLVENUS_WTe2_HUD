import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from matplotlib.widgets import RectangleSelector

# ==========================================================
# PATHS
# ==========================================================




N = 10






# ==========================================================
# TIME CLEANER
# ==========================================================

def clean_time_column(time_list):

    cleaned = []

    for t in time_list:
        if isinstance(t, (list, tuple, np.ndarray)):
            t = t[0]
        if isinstance(t, (bytes, bytearray)):
            t = t.decode("utf-8")

        t = str(t)
        t = t.replace("[", "").replace("]", "")
        t = t.replace("b'", "").replace("'", "")

        cleaned.append(t)

    return pd.to_datetime(cleaned)

# ==========================================================
# TIFF PARSER
# ==========================================================

def parse_run_range(filename):

    name = filename.replace(".tif", "")
    parts = name.split("_")

    try:
        start_run = int(parts[1])
        end_run = int(parts[2])
    except:
        return []

    return list(range(start_run, end_run + 1))

# ==========================================================
# DISPLAY NORMALIZATION (UNCHANGED)
# ==========================================================

def to_display(img):

    img = img.astype(np.float32)

    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)

    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)





# ==========================================================
# GLOBAL RECORD STORE (SYNC SOURCE)
# ==========================================================

records = []

# ==========================================================
# TEMPERATURE CONTEXT (UNCHANGED LOGIC)
# ==========================================================

def get_temperature_context(run_range, N=5):

    window_T = []
    window_time = []

    for r in run_range:
        if r in run_to_avgT and r in run_to_start:
            window_T.append(run_to_avgT[r])
            window_time.append(run_to_start[r])

    if len(window_T) == 0:
        return None

    indices = [run_to_idx[r] for r in run_range if r in run_to_idx]

    if len(indices) == 0:
        return None

    left = max(min(indices) - N, 0)
    right = min(max(indices) + N, len(run_list) - 1)

    context_T = []
    context_time = []

    for i in range(left, right + 1):
        r = run_list[i]
        if r in run_to_avgT and r in run_to_start:
            context_T.append(run_to_avgT[r])
            context_time.append(run_to_start[r])

    return {
        "window_time": window_time,
        "window_T": window_T,
        "context_time": context_time,
        "context_T": context_T
    }

# ==========================================================
# TEMPERATURE PANEL (RESTORED FULL VERSION)
# ==========================================================

def make_temperature_panel(data):

    window_time = np.array(clean_time_column(data["window_time"]))
    context_time = np.array(clean_time_column(data["context_time"]))

    window_T = np.array(data["window_T"])
    context_T = np.array(data["context_T"])

    order = np.argsort(window_time)

    window_time = window_time[order]
    window_T = window_T[order]

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    ax.plot(context_time, context_T, color="gray", alpha=0.3)
    ax.plot(window_time, window_T, color="orange", linewidth=2.5)

    ax.axvspan(window_time.min(), window_time.max(), color="orange", alpha=0.15)

    ax.set_xticks([])
    ax.set_ylabel("T")

    fig.canvas.draw()
    panel = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return panel



# ==========================================================
# MAIN LOOP (SYNCED RECORDS ADDED ONLY)
# ==========================================================

def prepare_temperaturePane(tif_folder, csv_file, destination_folder):
    # ==========================================================
    # LOAD CSV
    # ==========================================================

    df = pd.read_csv(csv_file)

    df["RunNum"] = (
        df["RunNum"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .astype(int)
    )

    df = df.sort_values("RunNum").reset_index(drop=True)

    run_list = df["RunNum"].tolist()

    run_to_avgT = dict(zip(df["RunNum"], df["AvgT"]))
    run_to_start = dict(zip(df["RunNum"], df["StartTime"]))
    run_to_idx = {r: i for i, r in enumerate(run_list)}

    global records
    tiff_files = sorted(list(tif_folder.glob("*.tif"))) #Load TIF Files
    for idx, tif in enumerate(tiff_files):

        run_range = parse_run_range(tif.name)

        if len(run_range) == 0:
            continue

        r = run_range[len(run_range)//2]

        if r not in run_to_avgT or r not in run_to_start:
            continue

        temp_data = get_temperature_context(run_range, N=N)

        if temp_data is None:
            continue

        print("Processing:", tif.name)

        temp_panel = make_temperature_panel(temp_data)
        Image.fromarray(combined.astype(np.uint8)).save(
            destination_folder / f"{tif.stem}.png"
        )

        # sync record (ONLY addition, no behavior change)
        records.append({
            "index": idx,
            "filename": tif.name,
            "run": r,
            "time": run_to_start[r],
            "temperature": run_to_avgT[r],
            "roi_mean": float(roi_means[idx])
        })

    # summaries + csv
    make_full_range_summaries_from_records(records, destination_folder)
    make_csv(records, destination_folder)
    return 0

# ==========================================================
# SUMMARY + CSV
# ==========================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def make_full_range_summaries_from_records(records, output_dir):

    # 🔥 ensure folder exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp = np.array([r["temperature"] for r in records], dtype=float)

    plt.figure()
    plt.plot(temp, color="black")

    plt.savefig(output_dir / "Temperature_FullRange.png", dpi=200)
    plt.close()


def make_csv(records, output_dir):

    pd.DataFrame(records).to_csv(
        output_dir / "TIFF_Temperature_Log.csv",
        index=False
    )

# ==========================================================
# RUN
# ==========================================================

#if __name__ == "__main__":
 #   process_all()