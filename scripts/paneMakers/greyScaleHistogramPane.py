import json, csv, re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Circle


# ==========================================================
# SETTINGS
# ==========================================================

REMOVE_SPECIAL = True   # remove exact 0 and 1 pixels if needed
BINS = 100              # histogram resolution


# ==========================================================
# IO
# ==========================================================

def load_image(p):
    """Load TIFF image as numpy array."""
    return tifffile.imread(p)


def extract_run_number(f):
    """Extract run identifier from filename (optional utility)."""
    m = re.search(r"_(\d+_\d+)", f)
    if m:
        return m.group(1)
    return f.stem


# ==========================================================
# ROI SELECTION
# ==========================================================

def normalize_roi(roi):
    """Ensure ROI is always dict format."""
    if isinstance(roi, dict):
        return roi
    x1, x2, y1, y2 = roi
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}


def select_roi(img):
    """
    Interactive ROI selection using matplotlib rectangle selector.
    Returns bounding box dictionary.
    """
    roi = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.set_title("Select ROI then close window")

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
# IMAGE PROCESSING
# ==========================================================

def crop_image(img, roi):
    """Crop image to ROI."""
    roi = normalize_roi(roi)
    return img[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]


def normalize_image(img):
    """
    Normalize image intensities to [0, 1] using min-max scaling.
    This ensures all histograms are comparable in range.
    """
    img = img.astype(np.float32)

    mn = np.nanmin(img)
    mx = np.nanmax(img)

    # avoid divide-by-zero if flat image
    if mx - mn == 0:
        return np.zeros_like(img, dtype=np.float32)

    return (img - mn) / (mx - mn)


def build_histogram(img, bins):
    """
    Build histogram of pixel intensities in [0, 1].
    """
    px = img.ravel()
    px = px[np.isfinite(px)]

    # optionally remove extreme artifacts
    if REMOVE_SPECIAL:
        px = px[(px != 0.0) & (px != 1.0)]

    h, edges = np.histogram(px, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, h


# ==========================================================
# CSV OUTPUT
# ==========================================================

def init_csv(out):
    """Create output CSV if it doesn't exist."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "parameters.csv"

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["tiff", "run", "roi"])

    return csv_path


def append_csv(csv_path, tiff, run, roi):
    """Append one processed entry."""
    row = [str(tiff), run, json.dumps(roi)]

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ==========================================================
# PLOTTING
# ==========================================================

def plot_histogram(x, h, out, run, show=False):
    """
    Plot normalized histogram.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # main histogram
    ax.plot(x, h, color="black", label="Histogram")

    # reference guides
    ax.set_xlim(0, 1)
    for v in [0.25, 0.5, 0.75]:
        ax.axvline(v, color="gray", linestyle="--", linewidth=1)

    # styling
    ax.set_xlabel("Normalized intensity (0 → 1)")
    ax.set_ylabel("Pixel count")
    ax.set_title(run)

    ax.legend()

    # small visual markers (kept from your original style)
    ax.add_patch(Circle((0.05, 0.05), 0.03,
                        transform=ax.transAxes,
                        facecolor="black"))

    ax.add_patch(Circle((0.95, 0.05), 0.03,
                        transform=ax.transAxes,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=1))

    plt.tight_layout()

    outpath = Path(out) / f"Hist_{run}.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def create_roi_panes(folder, dest, roi, bins=100, show=False, remove_special_pixels=True):
    """
    Full processing pipeline:
    - load TIFFs
    - crop ROI
    - normalize (0–1)
    - compute histogram
    - plot + save
    - log CSV
    """

    global REMOVE_SPECIAL
    REMOVE_SPECIAL = remove_special_pixels

    folder = Path(folder)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = sorted(folder.glob("*.tif"))
    if not files:
        raise ValueError("No TIFF files found")

    csv_path = init_csv(dest)
    outputs = []

    for i, f in enumerate(files):

        # ---- load image
        img = load_image(f)

        # ---- crop ROI
        crop = crop_image(img, roi)

        # ---- normalize to 0–1 (IMPORTANT STEP)
        crop = normalize_image(crop)

        # ---- histogram
        x, h = build_histogram(crop, bins)

        # ---- run label
        run = f.stem

        # ---- plot result
        plot_histogram(x, h, dest, run, show=show)

        # ---- save metadata
        append_csv(csv_path, f, run, roi)

        outputs.append(dest / f"Hist_{run}.png")

    return outputs