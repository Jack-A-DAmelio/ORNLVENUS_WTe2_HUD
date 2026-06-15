import json, csv, re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Circle
from scipy.optimize import curve_fit


# ==========================================================
# SETTINGS
# ==========================================================

REMOVE_SPECIAL = True
BINS = 100

Y_MAX = None  # will auto-compute if None


# ==========================================================
# IO
# ==========================================================

def load_image(p):
    return tifffile.imread(p)


def extract_run_number(f):
    m = re.search(r"_(\d+_\d+)", f)
    return m.group(1) if m else f.stem


# ==========================================================
# ROI
# ==========================================================

def normalize_roi(roi):
    if isinstance(roi, dict):
        return roi
    x1, x2, y1, y2 = roi
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}


def select_roi(img):
    roi = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.set_title("Select ROI then close")

    def on_select(e1, e2):
        x1, y1 = int(e1.xdata), int(e1.ydata)
        x2, y2 = int(e2.xdata), int(e2.ydata)
        roi["x1"], roi["x2"] = min(x1, x2), max(x1, x2)
        roi["y1"], roi["y2"] = min(y1, y2), max(y1, y2)

    RectangleSelector(ax, on_select, useblit=True, button=[1], interactive=True)
    plt.show()

    if not roi:
        raise RuntimeError("No ROI selected")

    return roi


# ==========================================================
# IMAGE OPS
# ==========================================================

def crop_image(img, roi):
    roi = normalize_roi(roi)
    return img[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]


def normalize_image(img):
    img = img.astype(np.float32)
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx - mn == 0:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def build_histogram(img, bins):
    px = img.ravel()
    px = px[np.isfinite(px)]

    if REMOVE_SPECIAL:
        px = px[(px != 0.0) & (px != 1.0)]

    h, edges = np.histogram(px, bins=bins, range=(0, 1))
    g = (edges[:-1] + edges[1:]) / 2
    return g, h


# ==========================================================
# GAUSSIAN MODEL + WEIGHTED FIT
# ==========================================================

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


def weighted_residual(params, x, y):
    a, mu, sigma = params
    model = gaussian(x, a, mu, sigma)

    weight_scale = 0.01#bigger is more forceful
    w = np.exp(-(x - mu)**2 / (2 * weight_scale**2))

    return w * (model - y)


def fit_gaussian(g, h):

    if len(h) == 0 or np.sum(h) == 0:
        return None

    # ---- initial guess from histogram peak
    peak_idx = np.argmax(h)
    mu0 = g[peak_idx]
    a0 = h[peak_idx]
    sigma0 = 0.1

    p0 = [a0, mu0, sigma0]

    try:
        popt, _ = curve_fit(
            gaussian,
            g,
            h,
            p0=p0,
            maxfev=10000
        )
        return popt
    except RuntimeError:
        return None


# ==========================================================
# PLOT
# ==========================================================

def plot_histogram(g, h, fit_params, out, run, show=False):

    global Y_MAX
    if Y_MAX is None:
        Y_MAX = max(Y_MAX or 0, np.max(h))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(g, h, color="black", label="Histogram")

    # ---- fit overlay
    if fit_params is not None:
        a, mu, sigma = fit_params
        ax.plot(g, gaussian(g, *fit_params), color="red", label="Gaussian fit")

        ax.text(
            0.02, 0.95,
            f"μ = {mu:.4f}\nσ = {sigma:.4f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8)
        )
    else:
        ax.text(
            0.02, 0.95,
            "Fit failed",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="orange", edgecolor="black", alpha=0.8)
        )

    # ---- axis constraints
    ax.set_xlim(0, 1)
    ax.set_ylim(0, Y_MAX * 1.05)

    for v in [0.25, 0.5, 0.75]:
        ax.axvline(v, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Normalized intensity (0–1)")
    ax.set_ylabel("Counts (fixed scale)")
    ax.set_title(run)
    ax.legend()

    plt.tight_layout()

    outpath = Path(out) / f"Hist_{run}.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# ==========================================================
# PIPELINE
# ==========================================================

def create_roi_panes(folder, dest, roi, bins=100, show=False, remove_special_pixels=True):

    global REMOVE_SPECIAL
    REMOVE_SPECIAL = remove_special_pixels

    folder = Path(folder)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = sorted(folder.glob("*.tif"))
    if not files:
        raise ValueError("No TIFF files found")

    csv_path = dest / "parameters.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["tiff", "run", "roi", "mu", "sigma", "amp"])

    outputs = []

    for f in files:

        img = load_image(f)
        crop = crop_image(img, roi)
        crop = normalize_image(crop)

        g, h = build_histogram(crop, bins)
        fit = fit_gaussian(g, h)

        run = f.stem

        plot_histogram(g, h, fit, dest, run, show=show)

        # log
        if fit is None:
            row = [str(f), run, json.dumps(roi), np.nan, np.nan, np.nan]
        else:
            a, mu, sigma = fit
            row = [str(f), run, json.dumps(roi), mu, sigma, a]

        with open(csv_path, "a", newline="") as fcsv:
            csv.writer(fcsv).writerow(row)

        outputs.append(dest / f"Hist_{run}.png")

    return outputs