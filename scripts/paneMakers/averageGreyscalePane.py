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

TIFF_FOLDER = Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/Nucleation/halfPercent/Crystal Images_rolling/30_roll/postProcess_Run_23139_LF99D_Diss_05at_0_416C_0_000AngsMin_0/")
CSV_FILE = "/SNS/VENUS/IPTS-36967/shared/Jack/runSummary_MayALL.csv"

HUD_DIR = TIFF_FOLDER / "3Panel HUD"
HUD_DIR.mkdir(exist_ok=True)

ROI_FILE = HUD_DIR / "rois.npy"

N = 10

# ==========================================================
# LOAD CSV
# ==========================================================

df = pd.read_csv(CSV_FILE)

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
# ROI SELECTION (UNCHANGED)
# ==========================================================

def select_roi(image):

    roi = {"box": None}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(to_display(image), cmap="gray")
    ax.set_title("Draw ROI Rectangle and Close Window")

    def onselect(eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        roi["box"] = (
            int(min(x1, x2)),
            int(max(x1, x2)),
            int(min(y1, y2)),
            int(max(y1, y2)),
        )

        print("ROI:", roi["box"])

    RectangleSelector(
        ax,
        onselect,
        useblit=False,
        button=[1],
        interactive=True
    )

    plt.show()

    return roi["box"]

# ==========================================================
# LOAD TIFFS
# ==========================================================

tiff_files = sorted(list(TIFF_FOLDER.glob("*.tif")))

# ==========================================================
# ROI LOAD
# ==========================================================

if ROI_FILE.exists():
    print("Loading ROI")
    x1, x2, y1, y2 = np.load(ROI_FILE)
else:
    img0 = np.array(Image.open(tiff_files[0]))
    x1, x2, y1, y2 = select_roi(img0)
    np.save(ROI_FILE, np.array([x1, x2, y1, y2]))

# ==========================================================
# ROI MEANS
# ==========================================================

roi_means = []

for tif in tiff_files:
    img = np.array(Image.open(tif))
    roi = img[y1:y2, x1:x2]
    roi_means.append(float(np.mean(roi)))

roi_means = np.array(roi_means)

ROI_YMIN = roi_means.min()
ROI_YMAX = roi_means.max()

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
# ROI PANEL (UNCHANGED STRUCTURE)
# ==========================================================

def make_roi_panel(idx):

    left = max(idx - N, 0)
    right = min(idx + N, len(roi_means) - 1)

    x = np.arange(left, right + 1)
    y = roi_means[left:right + 1]

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    ax.plot(x, y, color="gray", alpha=0.5)
    ax.axvline(idx, color="orange", linewidth=6)

    ax.scatter(0.96, 0.92, s=150, c="black", transform=ax.transAxes)
    ax.scatter(0.96, 0.08, s=150, facecolors="white",
               edgecolors="black", transform=ax.transAxes)

    ax.set_ylim(ROI_YMAX, ROI_YMIN)
    ax.set_xticks([])

    fig.canvas.draw()
    panel = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return panel

# ==========================================================
# IMAGE PANEL (UNCHANGED)
# ==========================================================

def make_image_panel(tif_path):

    img = np.array(Image.open(tif_path))
    disp = to_display(img)

    rgb = (disp * 255).astype(np.uint8)
    rgb = np.stack([rgb]*3, axis=-1)

    return np.array(Image.fromarray(rgb).resize((512, 512)))

# ==========================================================
# GLUE
# ==========================================================

def make_three_panel(a, b, c):
    return np.concatenate([a, b, c], axis=1)

# ==========================================================
# MAIN LOOP (SYNCED RECORDS ADDED ONLY)
# ==========================================================

def process_all():

    global records

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
        roi_panel = make_roi_panel(idx)
        image_panel = make_image_panel(tif)

        combined = make_three_panel(temp_panel, roi_panel, image_panel)

        Image.fromarray(combined.astype(np.uint8)).save(
            HUD_DIR / f"{tif.stem}.png"
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
    make_full_range_summaries_from_records(records, HUD_DIR)
    make_csv(records, HUD_DIR)

# ==========================================================
# SUMMARY + CSV
# ==========================================================

def make_full_range_summaries_from_records(records, output_dir):

    roi = np.array([r["roi_mean"] for r in records])
    temp = np.array([r["temperature"] for r in records])

    plt.figure()
    plt.plot(roi, color="black")
    plt.gca().invert_yaxis()
    plt.savefig(output_dir / "ROI_Mean_FullRange.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(temp, color="black")
    plt.savefig(output_dir / "Temperature_FullRange.png", dpi=200)
    plt.close()


def make_csv(records, output_dir):

    pd.DataFrame(records).to_csv(
        output_dir / "TIFF_ROI_Temperature_Log.csv",
        index=False
    )

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    process_all()