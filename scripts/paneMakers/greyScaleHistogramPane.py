import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from scipy.optimize import minimize

# ==========================================================
# GAUSSIAN MODEL
# ==========================================================

def gaussian(x, mu, sigma):
    sigma = max(float(sigma), 1e-8)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


# ==========================================================
# 2-GAUSSIAN MIXTURE MODEL
# ==========================================================

def two_gaussian_mixture(x, params):
    w1, mu1, s1, w2, mu2, s2 = params

    s1 = max(float(s1), 1e-8)
    s2 = max(float(s2), 1e-8)

    g1 = gaussian(x, mu1, s1)
    g2 = gaussian(x, mu2, s2)

    return w1 * g1 + w2 * g2


# ==========================================================
# SAFE SCALING (IMPORTANT FIX)
# ==========================================================

def safe_scale(y, pred):
    denom = np.dot(pred, pred)
    if not np.isfinite(denom) or denom < 1e-12:
        return pred
    scale = np.dot(y, pred) / denom
    return pred * scale


# ==========================================================
# ERROR
# ==========================================================

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ==========================================================
# INITIAL GUESSES
# ==========================================================

def init_1gauss(data):
    return np.mean(data), np.std(data) + 1e-8


def init_2gauss(data):
    data = np.sort(data)
    split = len(data) // 2

    a = data[:split]
    b = data[split:]

    mu1, s1 = np.mean(a), np.std(a) + 1e-8
    mu2, s2 = np.mean(b), np.std(b) + 1e-8

    return [0.5, mu1, s1, 0.5, mu2, s2]


# ==========================================================
# 1-GAUSSIAN FIT
# ==========================================================

def fit_1gauss(x, y, data):

    mu0, s0 = init_1gauss(data)

    def objective(params):
        mu, sigma = params
        pred = gaussian(x, mu, sigma)
        pred = safe_scale(y, pred)
        return rmse(y, pred)

    res = minimize(objective, [mu0, s0], method="Nelder-Mead")

    mu_opt, sigma_opt = res.x

    pred = gaussian(x, mu_opt, sigma_opt)
    pred = safe_scale(y, pred)

    err = rmse(y, pred)

    return (mu_opt, sigma_opt), pred, err


# ==========================================================
# 2-GAUSSIAN FIT
# ==========================================================

def fit_2gauss(x, y, data):

    p0 = init_2gauss(data)

    def objective(params):
        pred = two_gaussian_mixture(x, params)
        pred = safe_scale(y, pred)
        return rmse(y, pred)

    bounds = [
        (0, 2),        # w1
        (None, None),  # mu1
        (1e-6, None),  # s1
        (0, 2),        # w2
        (None, None),  # mu2
        (1e-6, None)   # s2
    ]

    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds)

    pred = two_gaussian_mixture(x, res.x)
    pred = safe_scale(y, pred)

    err = rmse(y, pred)

    return res.x, pred, err


# ==========================================================
# ROI PANEL
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
    # 1-GAUSSIAN
    # ==================================================
    (_, _), pred1, err1 = fit_1gauss(
        centers, counts, roi_pixels
    )

    ax.plot(
        centers, pred1,
        color="red",
        linewidth=2,
        label="1-Gaussian"
    )

    # ==================================================
    # 2-GAUSSIAN
    # ==================================================
    _, pred2, err2 = fit_2gauss(
        centers, counts, roi_pixels
    )

    pred2_plot = pred2 + 0.03 * np.max(counts)

    ax.plot(
        centers,
        pred2_plot,
        color="blue",
        linewidth=2,
        label="2-Gaussian"
    )

    # --------------------------------------------------
    # AXES
    # --------------------------------------------------
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")

    # --------------------------------------------------
    # ERROR DISPLAY
    # --------------------------------------------------
    ax.text(
        0.98, 0.98,
        f"1G RMSE: {err1:.3f}\n2G RMSE: {err2:.3f}",
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
# PIPELINE
# ==========================================================

def create_roi_panes(tif_folder, destination, roi, bins=50):

    tif_folder = Path(tif_folder)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(tif_folder.glob("*.tif"))

    if len(tiff_files) == 0:
        raise ValueError("No TIFF files found")

    # --------------------------------------------------
    # GLOBAL SCALE
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
    # RENDER
    # --------------------------------------------------
    pane_paths = []

    for tif in tiff_files:

        img = np.array(Image.open(tif))

        panel = make_roi_panel(
            img=img,
            roi=roi,
            bins=bins,
            y_max=y_max
        )

        out_path = destination / f"{tif.stem}.png"

        Image.fromarray(panel.astype(np.uint8)).save(out_path)

        pane_paths.append(out_path)

    #print(pane_paths)
    return pane_paths