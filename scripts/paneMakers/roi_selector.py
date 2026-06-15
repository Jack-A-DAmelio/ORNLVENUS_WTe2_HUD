from pathlib import Path
import re
import numpy as np
from PIL import Image
import populateHDFSpreadSheet
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector




from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import json

# ==========================================================
# APPLY ROI TO ALL TIFFS
# ==========================================================

from pathlib import Path
from PIL import Image, ImageDraw
import json

# ==========================================================
# APPLY ROI TO TIFF LIST
# ==========================================================
from pathlib import Path
from PIL import Image, ImageDraw
import json

# ==========================================================
# APPLY ROI TO TIFF LIST (OVERWRITE ORIGINALS)
# ==========================================================

def apply_roi_to_tiffs(tiff_paths, roi_box, roi_json_path=None):

    x_min, x_max, y_min, y_max = roi_box

    roi_meta = {
        "roi_box": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        },
        "files": []
    }

    for tif_path in tiff_paths:

        tif_path = Path(tif_path)

        img = Image.open(tif_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=(0, 255, 0),
            width=3
        )

        img.save(tif_path)

        roi_meta["files"].append(str(tif_path))

    if roi_json_path is not None:

        roi_json_path = Path(roi_json_path)

        with open(roi_json_path, "w") as f:
            json.dump(roi_meta, f, indent=4)

    return roi_meta

# ==========================================================
# ROI SELECTION
# ==========================================================

def select_roi(tif_folder):

    tiff_files = sorted(
        Path(tif_folder).glob("*.tif")
    )

    if len(tiff_files) == 0:
        raise ValueError(
            f"No TIFF files found in {tif_folder}"
        )

    img = np.array(
        Image.open(tiff_files[0])
    )

    def to_display(img):

        img = img.astype(np.float32)

        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99)

        return np.clip(
            (img - lo) / (hi - lo + 1e-8),
            0,
            1
        )

    roi = {"box": None}

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        to_display(img),
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

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=False,
        button=[1],
        interactive=True
    )

    plt.show()

    if roi["box"] is None:
        raise ValueError(
            "No ROI selected"
        )

    return roi["box"]