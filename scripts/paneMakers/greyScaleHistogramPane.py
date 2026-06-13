import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from matplotlib.widgets import RectangleSelector


# ==========================================================
# DISPLAY NORMALIZATION
# ==========================================================

def to_display(img):

    img = img.astype(np.float32)

    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)

    return np.clip(
        (img - lo) / (hi - lo + 1e-8),
        0,
        1
    )


# ==========================================================
# ROI SELECTION
# ==========================================================

def select_roi(image):

    roi = {"box": None}

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        to_display(image),
        cmap="gray"
    )

    ax.set_title(
        "Draw ROI Rectangle Then Close Window"
    )

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

    RectangleSelector(
        ax,
        onselect,
        useblit=False,
        button=[1],
        interactive=True
    )

    plt.show()

    if roi["box"] is None:
        raise ValueError("No ROI selected")

    return roi["box"]


# ==========================================================
# LOAD OR CREATE ROI
# ==========================================================

def load_or_create_roi(
    tiff_files,
    roi_file
):

    roi_file = Path(roi_file)

    if roi_file.exists():

        print("Loading ROI:", roi_file)

        return tuple(
            np.load(roi_file)
        )

    first_img = np.array(
        Image.open(tiff_files[0])
    )

    roi = select_roi(first_img)

    np.save(
        roi_file,
        np.array(roi)
    )

    print("Saved ROI:", roi_file)

    return roi


# ==========================================================
# ROI MEANS
# ==========================================================

def compute_roi_means(
    tiff_files,
    roi
):

    x1, x2, y1, y2 = roi

    roi_means = []

    for tif in tiff_files:

        img = np.array(
            Image.open(tif)
        )

        roi_pixels = img[
            y1:y2,
            x1:x2
        ]

        roi_means.append(
            float(
                np.mean(roi_pixels)
            )
        )

    return np.array(roi_means)


# ==========================================================
# ROI PANEL
# ==========================================================

def make_roi_panel(
    idx,
    roi_means,
    roi_ymin,
    roi_ymax,
    N=10
):

    left = max(idx - N, 0)
    right = min(idx + N, len(roi_means) - 1)

    x = np.arange(
        left,
        right + 1
    )

    y = roi_means[
        left:right + 1
    ]

    fig, ax = plt.subplots(
        figsize=(5.12, 5.12),
        dpi=100
    )

    ax.plot(
        x,
        y,
        color="gray",
        alpha=0.5
    )

    ax.axvline(
        idx,
        color="orange",
        linewidth=6,
        alpha=0.8
    )

    ax.scatter(
        0.96,
        0.92,
        s=150,
        c="black",
        transform=ax.transAxes,
        zorder=10
    )

    ax.scatter(
        0.96,
        0.08,
        s=150,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        transform=ax.transAxes,
        zorder=10
    )

    ax.text(
        0.02,
        0.95,
        f"Mean ROI:\n{roi_means[idx]:.2f}",
        transform=ax.transAxes,
        bbox=dict(
            facecolor="white",
            alpha=0.7
        ),
        fontsize=10,
        va="top"
    )

    # darker values at top
    ax.set_ylim(
        roi_ymax,
        roi_ymin
    )

    ax.set_xticks([])

    ax.set_xlabel(
        f"{2*N+1} images"
    )

    ax.set_ylabel(
        "ROI Mean"
    )

    plt.tight_layout()

    fig.canvas.draw()

    panel = np.array(
        fig.canvas.buffer_rgba()
    )[:, :, :3]

    plt.close(fig)

    return panel


# ==========================================================
# PREPARE ROI PANES
# ==========================================================

def prepare_roiPane(
    tif_folder,
    output_folder,
    roi_file,
    N=10
):

    tif_folder = Path(
        tif_folder
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    tiff_files = sorted(
        tif_folder.glob("*.tif")
    )

    if len(tiff_files) == 0:
        raise ValueError(
            "No TIFF files found"
        )

    roi = load_or_create_roi(
        tiff_files,
        roi_file
    )

    roi_means = compute_roi_means(
        tiff_files,
        roi
    )

    roi_ymin = float(
        roi_means.min()
    )

    roi_ymax = float(
        roi_means.max()
    )

    pane_paths = []
    records = []

    for idx, tif in enumerate(
        tiff_files
    ):

        print(
            "Processing:",
            tif.name
        )

        panel = make_roi_panel(
            idx,
            roi_means,
            roi_ymin,
            roi_ymax,
            N=N
        )

        out_path = (
            output_folder /
            f"{tif.stem}.png"
        )

        Image.fromarray(
            panel.astype(np.uint8)
        ).save(out_path)

        pane_paths.append(
            out_path
        )

        records.append({

            "index": idx,

            "filename": tif.name,

            "roi_mean":
                float(
                    roi_means[idx]
                )

        })

    return pane_paths, records


# ==========================================================
# SUMMARY PLOT + CSV
# ==========================================================

def save_roi_summary(
    records,
    output_folder
):

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    roi = np.array([
        r["roi_mean"]
        for r in records
    ])

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        roi,
        color="black"
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Image Index"
    )

    plt.ylabel(
        "Mean ROI"
    )

    plt.title(
        "ROI Mean Over Full Dataset"
    )

    plt.tight_layout()

    plt.savefig(
        output_folder /
        "ROI_Mean_FullRange.png",
        dpi=200
    )

    plt.close()

    pd.DataFrame(
        records
    ).to_csv(
        output_folder /
        "TIFF_ROI_Log.csv",
        index=False
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

def run_roi_pipeline(
    tif_folder,
    output_folder,
    roi_file,
    N=10
):

    pane_paths, records = prepare_roiPane(
        tif_folder,
        output_folder,
        roi_file,
        N=N
    )

    save_roi_summary(
        records,
        output_folder
    )

    return pane_paths


# ==========================================================
# EXAMPLE
# ==========================================================

if __name__ == "__main__":

    pane_paths, records = run_roi_pipeline(

        tif_folder="/path/to/tiffs",

        output_folder="/path/to/ROI_Panes",

        roi_file="/path/to/ROI_Panes/roi.npy",

        N=10
    )

    print(
        "Created",
        len(pane_paths),
        "ROI panes"
    )