import os
import re
import json
import tifffile
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pathlib import Path
from PIL import Image

# ==========================================================
# IMAGE IO
# ==========================================================

def load_image(image_path):
    return tifffile.imread(image_path)


# ==========================================================
# RUN PARSING (UNCHANGED)
# ==========================================================

def extract_run_number(filename):
    match = re.search(r"_(\d+_\d+)", filename)

    if match:
        return match.group(1)

    raise ValueError("Could not extract run number from filename.")


# ==========================================================
# ROI SELECTION (UNCHANGED FUNCTIONALITY)
# ==========================================================

def select_roi(image):

    roi = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    ax.set_title("Select ROI and close window")

    def on_select(eclick, erelease):

        x1 = int(eclick.xdata)
        y1 = int(eclick.ydata)
        x2 = int(erelease.xdata)
        y2 = int(erelease.ydata)

        roi["x1"] = min(x1, x2)
        roi["x2"] = max(x1, x2)
        roi["y1"] = min(y1, y2)
        roi["y2"] = max(y1, y2)

        print("ROI selected:", roi)

    RectangleSelector(ax, on_select, useblit=True, button=[1], interactive=True)

    plt.show()

    if not roi:
        raise RuntimeError("No ROI selected")

    return roi


def save_roi(roi, output_folder):
    path = Path(output_folder) / "ROI_coordinates.json"

    with open(path, "w") as f:
        json.dump(roi, f, indent=4)

    print(f"ROI saved: {path}")


# ==========================================================
# CROPPING
# ==========================================================

def crop_image(image, roi):
    return image[
        roi["y1"]:roi["y2"],
        roi["x1"]:roi["x2"]
    ]


# ==========================================================
# HISTOGRAM
# ==========================================================

def calculate_histogram(image, bins=500):

    pixels = image.flatten()

    histogram, edges = np.histogram(
        pixels,
        bins=bins,
        range=(pixels.min(), pixels.max())
    )

    grey_values = (edges[:-1] + edges[1:]) / 2

    return grey_values, histogram


# ==========================================================
# GAUSSIANS (UNCHANGED MATH)
# ==========================================================

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(
        -(x - mean) ** 2 / (2 * sigma ** 2)
    )


def double_gaussian(x,
                    amp1, mean1, sigma1,
                    amp2, mean2, sigma2):

    return (
        gaussian(x, amp1, mean1, sigma1)
        + gaussian(x, amp2, mean2, sigma2)
    )


# ==========================================================
# FITTING (UNCHANGED LOGIC)
# ==========================================================

def fit_gaussians(grey_values, histogram):

    mean_guess = np.average(grey_values, weights=histogram)
    sigma_guess = np.sqrt(
        np.average((grey_values - mean_guess) ** 2, weights=histogram)
    )

    amp_guess = histogram.max()

    # -------------------------
    # 1 Gaussian
    # -------------------------
    popt1, _ = curve_fit(
        gaussian,
        grey_values,
        histogram,
        p0=[amp_guess, mean_guess, sigma_guess]
    )

    # -------------------------
    # 2 Gaussian
    # -------------------------
    midpoint = len(grey_values) // 2

    popt2, _ = curve_fit(
        double_gaussian,
        grey_values,
        histogram,
        p0=[
            amp_guess / 2,
            grey_values[midpoint // 2],
            sigma_guess / 2,

            amp_guess / 2,
            grey_values[midpoint + midpoint // 2],
            sigma_guess / 2
        ],
        maxfev=10000
    )

    fit1 = gaussian(grey_values, *popt1)
    fit2 = double_gaussian(grey_values, *popt2)

    rss1 = np.sum((histogram - fit1) ** 2)
    rss2 = np.sum((histogram - fit2) ** 2)

    return popt1, popt2, rss1, rss2


# ==========================================================
# PLOTTING (UNCHANGED OUTPUT)
# ==========================================================

def plot_gaussian_comparison(
        grey_values,
        histogram,
        popt1,
        popt2,
        rss1,
        rss2,
        output_folder,
        run_number,
        save_plot=True,
        show_plot=False):

    plt.figure(figsize=(8, 6))

    plt.plot(grey_values, histogram, label="Histogram")

    plt.plot(
        grey_values,
        gaussian(grey_values, *popt1),
        label=f"1 Gaussian RSS={rss1:.2e}"
    )

    plt.plot(
        grey_values,
        double_gaussian(grey_values, *popt2),
        label=f"2 Gaussian RSS={rss2:.2e}"
    )

    plt.xlabel("Grey Value")
    plt.ylabel("Pixel Count")
    plt.title(f"Gaussian Fit Comparison - {run_number}")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        path = Path(output_folder) / f"GaussianFitComparison_{run_number}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# ==========================================================
# FIT RESULTS SAVE (UNCHANGED)
# ==========================================================

def save_fit_results(popt1, popt2, rss1, rss2, output_folder, run_number):

    amp1, mean1, sigma1 = popt1

    # popt2 is a 2-Gaussian model (6 parameters)
    amp2_1, mean2_1, sigma2_1, amp2_2, mean2_2, sigma2_2 = popt2

    path = Path(output_folder) / f"GaussianFit_{run_number}.csv"

    data = np.array([[
        amp1, mean1, sigma1,
        amp2_1, mean2_1, sigma2_1,
        amp2_2, mean2_2, sigma2_2,
        rss1, rss2
    ]])

    header = (
        "Amp1,Mean1,Sigma1,"
        "Amp2_A,Mean2_A,Sigma2_A,"
        "Amp2_B,Mean2_B,Sigma2_B,"
        "RSS1,RSS2"
    )

    np.savetxt(path, data, delimiter=",", header=header, comments="")

    print(f"Saved: {path}")

# ==========================================================
# PIPELINE WRAPPER (NEW - YOUR WORKFLOW INTEGRATION)
# ==========================================================

def process_image(image_path, output_folder, bins=500, show=False):

    image = load_image(image_path)

    run_number = extract_run_number(Path(image_path).name)

    roi = select_roi(image)
    save_roi(roi, output_folder)

    roi_image = crop_image(image, roi)

    grey_values, histogram = calculate_histogram(roi_image, bins=bins)

    popt1, popt2, rss1, rss2 = fit_gaussians(grey_values, histogram)

    save_fit_results(
        popt1, popt2, rss1, rss2,
        output_folder,
        run_number
    )

    plot_gaussian_comparison(
        grey_values,
        histogram,
        popt1,
        popt2,
        rss1,
        rss2,
        output_folder,
        run_number,
        show_plot=show
    )

    return {
        "run": run_number,
        "roi": roi,
        "popt1": popt1,
        "popt2": popt2,
        "rss1": rss1,
        "rss2": rss2
    }


# ==========================================================
# PIPELINE: ROI PANES + FITTING
# ==========================================================

def create_roi_panes(
    tif_folder,
    destination,
    roi,
    bins=500,
    show=False
):

    tif_folder = Path(tif_folder)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(tif_folder.glob("*.tif"))

    if len(tiff_files) == 0:
        raise ValueError("No TIFF files found")

    print(f"[INFO] Found {len(tiff_files)} TIFF files")

    pane_paths = []

    # ==========================================================
    # STEP 1: PRECOMPUTE GLOBAL Y-SCALE (for consistent plots)
    # ==========================================================

    all_histograms = []

    for tif in tiff_files:

        img = np.array(Image.open(tif))

        x1, x2, y1, y2 = roi
        roi_pixels = img[y1:y2, x1:x2].ravel()

        counts, _ = np.histogram(
            roi_pixels,
            bins=bins
        )

        all_histograms.append(counts)

    y_max = float(np.max(all_histograms))

    # ==========================================================
    # STEP 2: PROCESS EACH IMAGE USING YOUR EXISTING PIPELINE
    # ==========================================================

    for i, tif in enumerate(tiff_files):

        print(f"[PROCESS] {i+1}/{len(tiff_files)}: {tif.name}")

        image = np.array(Image.open(tif))

        run_number = tif.stem

        # --------------------------------------------------
        # ROI crop (same logic as process_image, but reused here)
        # --------------------------------------------------
        x1, x2, y1, y2 = roi
        roi_image = image[y1:y2, x1:x2]

        # --------------------------------------------------
        # histogram
        # --------------------------------------------------
        grey_values, histogram = calculate_histogram(
            roi_image,
            bins=bins
        )

        # --------------------------------------------------
        # fit (UNCHANGED MODEL)
        # --------------------------------------------------
        popt1, popt2, rss1, rss2 = fit_gaussians(
            grey_values,
            histogram
        )

        # --------------------------------------------------
        # plot (same comparison function you already have)
        # --------------------------------------------------
        plot_gaussian_comparison(
            grey_values,
            histogram,
            popt1,
            popt2,
            rss1,
            rss2,
            output_folder=destination,
            run_number=run_number,
            save_plot=True,
            show_plot=show
        )

        # --------------------------------------------------
        # save parameters
        # --------------------------------------------------
        save_fit_results(
            popt1,
            popt2,
            rss1,
            rss2,
            output_folder=destination,
            run_number=run_number
        )

        # --------------------------------------------------
        # store pane path
        # --------------------------------------------------
        pane_paths.append(
            Path(destination) / f"GaussianFitComparison_{run_number}.png"
        )
    
    return pane_paths