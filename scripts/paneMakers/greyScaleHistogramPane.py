import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


# ==========================================================
# GAUSSIAN MODEL
# ==========================================================

def gaussian(x, mu, sigma):
    sigma = max(sigma, 1e-8)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


# ==========================================================
# SIMPLE 2-GAUSSIAN SPLIT MODEL
# ==========================================================

def two_gaussian_fit(data):

    data = data.ravel()

    split = len(data) // 2
    a = data[:split]
    b = data[split:]

    mu1, s1 = np.mean(a), np.std(a) + 1e-8
    mu2, s2 = np.mean(b), np.std(b) + 1e-8

    return (mu1, s1), (mu2, s2)


# ==========================================================
# ROI HIST + FIT PANEL
# ==========================================================

def make_roi_panel(img, roi, bins, y_max):

    x1, x2, y1, y2 = roi
    roi_pixels = img[y1:y2, x1:x2].ravel()

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    # --------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------
    counts, edges, _ = ax.hist(
        roi_pixels,
        bins=bins,
        color="lightgray",
        edgecolor="black",
        alpha=0.85
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    # ==================================================
    # 1-GAUSSIAN FIT
    # ==================================================
    mu = np.mean(roi_pixels)
    sigma = np.std(roi_pixels) + 1e-8

    g1 = gaussian(centers, mu, sigma)

    g1_scaled = g1 * np.max(counts) / np.max(g1)

    pred1 = np.interp(centers, centers, g1_scaled)
    err1 = np.sqrt(np.mean((counts - pred1) ** 2))

    ax.plot(centers, g1_scaled, color="red", linewidth=2, label="1-Gaussian")

    # ==================================================
    # 2-GAUSSIAN FIT (split approximation)
    # ==================================================
    (mu1, s1), (mu2, s2) = two_gaussian_fit(roi_pixels)

    g2 = (
        gaussian(centers, mu1, s1) +
        gaussian(centers, mu2, s2)
    )

    g2_scaled = g2 * np.max(counts) / np.max(g2)

    # small vertical offset so it doesn't hide behind 1-Gaussian
    offset = 0.03 * np.max(counts)
    g2_scaled = g2_scaled + offset

    err2 = np.sqrt(np.mean((counts - g2_scaled) ** 2))

    ax.plot(centers, g2_scaled, color="blue", linewidth=2, label="2-Gaussian")

    # --------------------------------------------------
    # AXIS + CONSISTENCY
    # --------------------------------------------------
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")

    # --------------------------------------------------
    # ERROR TEXT
    # --------------------------------------------------
    ax.text(
        0.98, 0.98,
        f"1G RMSE: {err1:.2f}\n2G RMSE: {err2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7)
    )

    ax.legend(loc="upper left")

    plt.tight_layout()

    fig.canvas.draw()

    panel = np.array(fig.canvas.buffer_rgba())[:, :, :3]

    plt.close(fig)

    return panel


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def create_roi_panes(
    tif_folder,
    destination,
    roi,
    bins=50
):

    tif_folder = Path(tif_folder)
    destination = Path(destination)

    tiff_files = sorted(tif_folder.glob("*.tif"))

    if len(tiff_files) == 0:
        raise ValueError("No TIFF files found")

    output_folder = destination 
    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # GLOBAL Y SCALE (histogram consistency)
    # --------------------------------------------------
    all_counts = []

    for tif in tiff_files:

        img = np.array(Image.open(tif))
        x1, x2, y1, y2 = roi

        roi_pixels = img[y1:y2, x1:x2].ravel()

        counts, _ = np.histogram(roi_pixels, bins=bins)

        all_counts.append(counts)

    y_max = float(np.max(all_counts))

    # --------------------------------------------------
    # RENDER PANES
    # --------------------------------------------------
    pane_paths = []

    for idx, tif in enumerate(tiff_files):

        #print(f"Rendering {idx+1}/{len(tiff_files)}: {tif.name}")

        img = np.array(Image.open(tif))

        panel = make_roi_panel(
            img=img,
            roi=roi,
            bins=bins,
            y_max=y_max
        )

        out_path = output_folder / f"{tif.stem}.png"

        Image.fromarray(panel.astype(np.uint8)).save(out_path)

        pane_paths.append(out_path)
    print(pane_paths)
    return pane_paths