import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ==========================================================
# CONFIG
# ==========================================================

N = 10

# ==========================================================
# UTILITIES
# ==========================================================

def clean_time_column(time_list):

    cleaned = []

    for t in time_list:
        if isinstance(t, (list, tuple, np.ndarray)):
            t = t[0]
        if isinstance(t, (bytes, bytearray)):
            t = t.decode("utf-8")

        t = str(t).replace("[", "").replace("]", "").replace("b'", "").replace("'", "")
        cleaned.append(t)

    return pd.to_datetime(cleaned)


def parse_run_range(filename):

    name = filename.replace(".tif", "")
    parts = name.split("_")

    try:
        start_run = int(parts[1])
        end_run = int(parts[2])
        return list(range(start_run, end_run + 1))
    except:
        return []

# ==========================================================
# TEMPERATURE CONTEXT
# ==========================================================

def get_temperature_context(run_range, N,
                            run_to_avgT,
                            run_to_start,
                            run_list,
                            run_to_idx):

    window_T, window_time = [], []

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

    context_T, context_time = [], []

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
# PLOT (UNCHANGED STYLE)
def make_temperature_panel(data,
                           y_min,
                           y_max):

    window_time = np.array(clean_time_column(data["window_time"]))
    context_time = np.array(clean_time_column(data["context_time"]))

    window_T = np.array(data["window_T"])
    context_T = np.array(data["context_T"])

    order = np.argsort(window_time)

    window_time = window_time[order]
    window_T = window_T[order]

    avg_T = float(np.mean(window_T))

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    ax.plot(
        context_time,
        context_T,
        color="gray",
        alpha=0.3
    )

    ax.plot(
        window_time,
        window_T,
        color="orange",
        linewidth=2.5
    )

    ax.axvspan(
        window_time.min(),
        window_time.max(),
        color="orange",
        alpha=0.15
    )

    ax.text(
        0.02,
        0.95,
        f"Avg T: {avg_T:.2f} °C",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7),
        fontsize=10,
        va="top"
    )

    # FIXED SCALE FOR ENTIRE DATASET
    ax.set_ylim(y_min, y_max)

    ax.set_xticks([])
    ax.set_ylabel("T")

    fig.canvas.draw()

    panel = np.array(fig.canvas.buffer_rgba())[:, :, :3]

    plt.close(fig)

    return panel

# ==========================================================
# CORE PROCESSOR (NO GLOBALS)
# ==========================================================

def prepare_temperaturePane(tif_folder, csv_file, output_folder, N=20):

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    panelsMade = []
    # -------------------------
    # LOAD CSV
    # -------------------------
    df = pd.read_csv(csv_file)

    df["RunNum"] = (
        df["RunNum"].astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .astype(int)
    )

    df = df.sort_values("RunNum").reset_index(drop=True)

    run_list = df["RunNum"].tolist()
    run_to_avgT = dict(zip(df["RunNum"], df["AvgT"]))
    run_to_start = dict(zip(df["RunNum"], df["StartTime"]))
    run_to_idx = {r: i for i, r in enumerate(run_list)}
    
    # ----------------------------------------------------------
    # GLOBAL TEMPERATURE SCALE
    # ----------------------------------------------------------

    global_temp_min = float(df["AvgT"].min())
    global_temp_max = float(df["AvgT"].max())

    pad = 0.05 * (global_temp_max - global_temp_min)

    global_temp_min -= pad
    global_temp_max += pad
    # -------------------------
    # OUTPUT RECORDS (LOCAL ONLY)
    # -------------------------
    records = []

    tiff_files = sorted(Path(tif_folder).glob("*.tif"))
    print(len(tiff_files))
    for idx, tif in enumerate(tiff_files):

        run_range = parse_run_range(tif.name)

        if len(run_range) == 0:
            continue

        valid_runs = [r for r in run_range if r in run_to_avgT]
        if len(valid_runs) == 0:
            continue

        r = valid_runs[len(valid_runs) // 2]

        temp_data = get_temperature_context(
            run_range,
            N,
            run_to_avgT,
            run_to_start,
            run_list,
            run_to_idx
        )

        if temp_data is None:
            print("failed to get temp data")
            continue
       #print("here")
        # -------------------------
        # BUILD PANEL
        # -------------------------
        temp_panel = make_temperature_panel( temp_data,
        global_temp_min,
        global_temp_max
             )

        Image.fromarray(temp_panel.astype(np.uint8)).save(
            output_folder / f"{tif.stem}.png"
        )
        panelsMade.append(output_folder / f"{tif.stem}.png")

        # -------------------------
        # RECORD
        # -------------------------
        records.append({
            "index": idx,
            "filename": tif.name,
            "run": r,
            "time": run_to_start[r],
            "temperature": run_to_avgT[r]
        })
    save_summary(records, output_folder)
    return panelsMade

# ==========================================================
# SUMMARY + CSV
# ==========================================================

def save_summary(records, output_folder):

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if len(records) == 0:
        print("WARNING: no records to summarize")
        return

    temp = np.array([r["temperature"] for r in records], dtype=float)

    plt.figure()
    plt.plot(temp, color="black")
    plt.savefig(output_folder / "Temperature_FullRange.png", dpi=200)
    plt.close()

    pd.DataFrame(records).to_csv(
        output_folder / "TIFF_Temperature_Log.csv",
        index=False
    )

# ==========================================================
# ENTRY POINT
# ==========================================================

def run_pipeline(tif_folder, csv_file, output_folder, N=10):

    records = process_all(tif_folder, csv_file, output_folder, N=N)
    save_summary(records, output_folder)

    return records