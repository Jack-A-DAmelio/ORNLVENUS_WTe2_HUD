import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


# ==========================================================
# ROI MEANS
# ==========================================================

def compute_roi_means(tiff_files, roi):

    x1, x2, y1, y2 = roi

    roi_means = []

    for tif in tiff_files:

        img = np.array(Image.open(tif))

        roi_pixels = img[y1:y2, x1:x2]

        roi_means.append(float(np.mean(roi_pixels)))

    return np.array(roi_means)


# ==========================================================
# ROI PANEL RENDER
# ==========================================================

def make_roi_panel(idx, roi_means, roi_ymin, roi_ymax, N=10):

    left = max(idx - N, 0)
    right = min(idx + N, len(roi_means) - 1)

    x = np.arange(left, right + 1)
    y = roi_means[left:right + 1]

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    ax.plot(x, y, color="gray", alpha=0.5)
    ax.axvline(idx, color="orange", linewidth=6, alpha=0.8)

    ax.scatter(0.96, 0.92, s=150, c="black",
               transform=ax.transAxes, zorder=10)

    ax.scatter(0.96, 0.08, s=150,
               facecolors="white",
               edgecolors="black",
               linewidths=1.5,
               transform=ax.transAxes,
               zorder=10)

    ax.text(
        0.02, 0.95,
        f"Mean ROI:\n{roi_means[idx]:.2f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7),
        fontsize=10,
        va="top"
    )

    ax.set_ylim(roi_ymax, roi_ymin)
    ax.set_xticks([])
    ax.set_xlabel(f"{2*N+1} images")
    ax.set_ylabel("ROI Mean")

    plt.tight_layout()
    fig.canvas.draw()

    panel = np.array(fig.canvas.buffer_rgba())[:, :, :3]

    plt.close(fig)

    return panel


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def create_roi_panes(tif_folder, roi, N=10):

    tif_folder = Path(tif_folder)

    tiff_files = sorted(tif_folder.glob("*.tif"))

    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {tif_folder}")

    # compute ROI signal
    roi_means = compute_roi_means(tiff_files, roi)

    roi_ymin = float(roi_means.min())
    roi_ymax = float(roi_means.max())

    # output folder next to TIFF folder
    output_folder = tif_folder.parent / f"{tif_folder.name}_ROI_Panes"
    output_folder.mkdir(parents=True, exist_ok=True)

    pane_paths = []

    for idx, tif in enumerate(tiff_files):

        print(f"Rendering ROI pane {idx+1}/{len(tiff_files)}: {tif.name}")

        panel = make_roi_panel(
            idx,
            roi_means,
            roi_ymin,
            roi_ymax,
            N=N
        )

        out_path = output_folder / f"{tif.stem}.png"

        Image.fromarray(panel.astype(np.uint8)).save(out_path)

        pane_paths.append(out_path)

    return pane_paths