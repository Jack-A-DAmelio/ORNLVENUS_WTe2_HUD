from pathlib import Path
import re
import numpy as np
from PIL import Image


# ==========================================================
# NORMALIZE FILENAMES
# ==========================================================

def normalize_key(path):
    """
    Convert filenames like:

        GlobalRunWindow_23139_23139.tif
        GaussianFitComparison_GlobalRunWindow_23139_23139.png
        TemperaturePane_23139_23139.png

    into:

        23139_23139

    so files can be matched regardless of:
        - extension
        - folder
        - prefixes
    """

    name = Path(path).stem

    match = re.search(r"(\d+_\d+)$", name)

    if match:
        return match.group(1)

    return None


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(path):
    """
    Load image and convert to uint8 RGB.
    Handles TIFF, PNG, grayscale, RGB, RGBA.
    """

    img = Image.open(path)

    arr = np.array(img)

    arr = np.squeeze(arr)

    # grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # RGBA -> RGB
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    arr = arr.astype(np.float32)

    arr -= arr.min()

    if arr.max() > 0:
        arr /= arr.max()

    arr = (arr * 255).astype(np.uint8)

    return arr


# ==========================================================
# GLUE FUNCTION
# ==========================================================

def glue_multiple_pane_sets(
    pane_lists,
    output_folder
):
    """
    pane_lists should look like:

    [
        [temp1.png, temp2.png, temp3.png],
        [avg1.png, avg2.png, avg3.png],
        [hist1.png, hist2.png, hist3.png],
        [img1.tif, img2.tif, img3.tif]
    ]

    Matching is based ONLY on the run-number portion
    of the filename.

    Example:

        GlobalRunWindow_23139_23139.tif
        GaussianFitComparison_GlobalRunWindow_23139_23139.png

    both map to:

        23139_23139
    """

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # STEP 1: BUILD LOOKUPS
    # --------------------------------------------------

    maps = []

    for set_index, pane_list in enumerate(pane_lists):

        lookup = {}

        for p in pane_list:

            p = Path(p)

            key = normalize_key(p)

            if key is None:
                print(f"[SKIP] Could not parse: {p.name}")
                continue

            lookup[key] = p

        maps.append(lookup)

        print(
            f"[INFO] Pane set {set_index}: "
            f"{len(lookup)} files"
        )

        # helpful debugging
        if len(lookup):
            print(
                "       Example:",
                next(iter(lookup.values()))
            )

    # --------------------------------------------------
    # STEP 2: DEBUG PRINT KEYS
    # --------------------------------------------------

    print("\n===== KEYS BY DATASET =====")

    for i, m in enumerate(maps):

        print(f"\nSET {i}")

        for key in sorted(m.keys())[:5]:
            print("   ", key)

    print("\n===========================\n")

    # --------------------------------------------------
    # STEP 3: FIND COMMON KEYS
    # --------------------------------------------------

    if len(maps) == 0:
        print("[WARNING] No pane sets supplied")
        return []

    common_keys = set(maps[0].keys())

    for m in maps[1:]:
        common_keys &= set(m.keys())

    common_keys = sorted(common_keys)

    print(
        f"Found {len(common_keys)} matching groups"
    )

    if len(common_keys) == 0:

        print(
            "[WARNING] No overlapping filenames"
        )

        return []

    # --------------------------------------------------
    # STEP 4: STITCH PANELS
    # --------------------------------------------------

    outputs = []

    for key in common_keys:

        images = []

        for lookup in maps:

            img = load_image(
                lookup[key]
            )

            images.append(img)

        # --------------------------------------------------
        # MATCH HEIGHTS
        # --------------------------------------------------

        target_height = min(
            img.shape[0]
            for img in images
        )

        resized = []

        for img in images:

            pil = Image.fromarray(img)

            if pil.height != target_height:

                new_width = int(
                    pil.width *
                    target_height /
                    pil.height
                )

                pil = pil.resize(
                    (
                        new_width,
                        target_height
                    ),
                    Image.NEAREST
                )

            resized.append(
                np.array(pil)
            )

        # --------------------------------------------------
        # CONCATENATE
        # --------------------------------------------------

        combined = np.concatenate(
            resized,
            axis=1
        )

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------

        out_path = (
            output_folder /
            f"HUD_{key}.png"
        )

        Image.fromarray(
            combined.astype(np.uint8)
        ).save(out_path)

        outputs.append(out_path)

        print(
            f"[SAVED] {out_path.name}"
        )

    print(
        f"\nCreated {len(outputs)} HUD images"
    )

    return outputs