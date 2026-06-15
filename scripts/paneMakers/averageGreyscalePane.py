import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from pathlib import Path
from PIL import Image
from matplotlib.widgets import RectangleSelector


# ==========================================================
# ROI SELECTION
# ==========================================================

def select_roi(image):

    roi = {"box": None}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")

    ax.set_title("Draw ROI Rectangle Then Close Window")

    def onselect(eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        roi["box"] = (
            int(min(x1, x2)),
            int(max(x1, x2)),
            int(min(y1, y2)),
            int(max(y1, y2))
        )

        print("ROI:", roi["box"])

    RectangleSelector(ax, onselect, useblit=False, button=[1], interactive=True)

    plt.show()

    if roi["box"] is None:
        raise ValueError("No ROI selected")

    return roi["box"]


# ==========================================================
# RUN INDEX PARSER
# ==========================================================

def extract_run_index(filename):
    m = re.findall(r"\d+", filename)
    return int(m[-1]) if m else 0


# ==========================================================
# ROI COMPUTATION
# ==========================================================

def compute_roi_means(tiff_files, roi):

    x1, x2, y1, y2 = roi

    runs = []
    means = []

    for tif in tiff_files:

        img = np.array(Image.open(tif))

        roi_pixels = img[y1:y2, x1:x2]

        means.append(np.mean(roi_pixels))
        runs.append(extract_run_index(tif.name))

    return np.array(runs), np.array(means)


# ==========================================================
# SAFETY: ENSURE FLAT LIST (KEY FIX)
# ==========================================================

def ensure_flat_list(pane_paths):
    """
    Prevents accidental nesting like:
        [[a, b, c]]  -> [a, b, c]
    """

    if len(pane_paths) == 0:
        return pane_paths

    # detect nested structure
    if isinstance(pane_paths[0], (list, tuple)):
        print("[WARN] Nested pane list detected → flattening")
        flat = []
        for item in pane_paths:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    return pane_paths


# ==========================================================
# PANEL (FULL RANGE + POSITION INDICATOR)
# ==========================================================

def make_panel(runs, roi_means, idx):

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    # FULL DATASET
    ax.plot(runs, roi_means, color="black", linewidth=1)

    # CURRENT POSITION
    ax.axvline(runs[idx], color="orange", linewidth=3, alpha=0.9)

    ax.scatter(
        runs[idx],
        roi_means[idx],
        color="red",
        s=80,
        zorder=10
    )

    # TEXT BOX
    ax.text(
        0.02, 0.95,
        f"Run: {runs[idx]}\nROI: {roi_means[idx]:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # FIXED AXES
    ax.set_xlim(np.min(runs), np.max(runs))
    ax.set_ylim(np.min(roi_means), np.max(roi_means))

    # --------------------------------------------------
    # 🔥 NEW: invert y-axis
    # --------------------------------------------------
    ax.invert_yaxis()

    # --------------------------------------------------
    # 🔵 CORNER MARKERS (NEW)
    # --------------------------------------------------

    # top-right: black dot
    ax.scatter(
        0.96, 0.96,
        s=180,
        c="black",
        transform=ax.transAxes,
        zorder=20
    )

    # bottom-left: white dot with black outline
    ax.scatter(
        0.96, 0.04,
        s=180,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        transform=ax.transAxes,
        zorder=20
    )

    # labels
    ax.set_xlabel("Run index")
    ax.set_ylabel("ROI Mean")
    ax.set_title("Full Dataset View (Current Position Highlighted)")

    plt.tight_layout()

    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return img


# ==========================================================
# PIPELINE
# ==========================================================

def run_roi_pipeline(tif_folder, output_folder, roi):

    tif_folder = Path(tif_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(tif_folder.glob("*.tif"))

    if not tiff_files:
        raise ValueError("No TIFF files found")

    # compute data
    runs, roi_means = compute_roi_means(tiff_files, roi)

    pane_paths = []
    records = []

    # generate one panel per frame
    for idx, tif in enumerate(tiff_files):

        panel = make_panel(runs, roi_means, idx)

        out_path = output_folder / f"{tif.stem}.png"

        Image.fromarray(panel.astype(np.uint8)).save(out_path)

        pane_paths.append(out_path)

        records.append({
            "index": idx,
            "run": runs[idx],
            "roi_mean": roi_means[idx]
        })

    # ======================================================
    # SAFETY CHECK (THIS PREVENTS YOUR ORIGINAL ERROR)
    # ======================================================
    pane_paths = ensure_flat_list(pane_paths)

    # CSV
    pd.DataFrame(records).to_csv(
        output_folder / "ROI_log.csv",
        index=False
    )

    # summary plot
    plt.figure(figsize=(10, 4))
    plt.plot(runs, roi_means, color="black")
    plt.xlabel("Run index")
    plt.ylabel("ROI Mean")
    plt.title("Full ROI Trend")
    plt.tight_layout()

    plt.savefig(output_folder / "ROI_full.png", dpi=200)
    plt.close()

    return pane_paths