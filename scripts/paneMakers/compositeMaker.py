from pathlib import Path
import re
import numpy as np
from PIL import Image
import math


# ==========================================================
# NORMALIZE FILENAMES
# ==========================================================

def normalize_key(path):

    name = Path(path).stem
    match = re.search(r"(\d+_\d+)$", name)

    if match:
        return match.group(1)

    return None


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(path):

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

def glue_multiple_pane_sets(pane_lists, output_folder):

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # BUILD LOOKUPS
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

        print(f"[INFO] Pane set {set_index}: {len(lookup)} files")

        if len(lookup):
            print("       Example:", next(iter(lookup.values())))

    # --------------------------------------------------
    # COMMON KEYS
    # --------------------------------------------------

    if len(maps) == 0:
        print("[WARNING] No pane sets supplied")
        return []

    common_keys = set(maps[0].keys())

    for m in maps[1:]:
        common_keys &= set(m.keys())

    common_keys = sorted(common_keys)

    print(f"Found {len(common_keys)} matching groups")

    if len(common_keys) == 0:
        print("[WARNING] No overlapping filenames")
        return []

    # --------------------------------------------------
    # PROCESS EACH KEY
    # --------------------------------------------------

    outputs = []

    for key in common_keys:

        images = []

        for lookup in maps:
            images.append(load_image(lookup[key]))

        # --------------------------------------------------
        # GLOBAL HEIGHT NORMALIZATION
        # --------------------------------------------------

        target_height = min(img.shape[0] for img in images)

        resized = []

        for img in images:

            pil = Image.fromarray(img)

            if pil.height != target_height:

                new_width = int(pil.width * target_height / pil.height)
                pil = pil.resize((new_width, target_height), Image.NEAREST)

            resized.append(np.array(pil))

        # --------------------------------------------------
        # GRID SHAPE (ROWS/COLS)
        # --------------------------------------------------

        n = len(resized)

        if n <= 2:
            rows = 1
        elif n <= 4:
            rows = 2
        else:
            rows = 3

        cols = math.ceil(n / rows)

        # pad to full grid
        h, w = resized[0].shape[:2]
        blank = np.zeros((h, w, 3), dtype=np.uint8)

        while len(resized) < rows * cols:
            resized.append(blank)

        # --------------------------------------------------
        # BUILD GRID ROWS
        # --------------------------------------------------

        grid_rows = []
        row_widths = []

        for r in range(rows):

            row_imgs = resized[r * cols:(r + 1) * cols]

            row_h = min(img.shape[0] for img in row_imgs)

            fixed_row = []

            for img in row_imgs:

                pil = Image.fromarray(img)

                if pil.height != row_h:
                    new_w = int(pil.width * row_h / pil.height)
                    pil = pil.resize((new_w, row_h), Image.NEAREST)

                fixed_row.append(np.array(pil))

            row = np.concatenate(fixed_row, axis=1)
            grid_rows.append(row)
            row_widths.append(row.shape[1])

        # --------------------------------------------------
        # PAD ROWS TO SAME WIDTH (CRITICAL FIX)
        # --------------------------------------------------

        max_width = max(row_widths)
        h = grid_rows[0].shape[0]

        padded_rows = []

        for row in grid_rows:

            if row.shape[1] < max_width:

                pad = np.zeros(
                    (h, max_width - row.shape[1], 3),
                    dtype=np.uint8
                )

                row = np.concatenate([row, pad], axis=1)

            padded_rows.append(row)

        # FINAL COMBINE
        combined = np.concatenate(padded_rows, axis=0)

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------

        out_path = output_folder / f"HUD_{key}.png"

        Image.fromarray(combined.astype(np.uint8)).save(out_path)

        outputs.append(out_path)

        print(f"[SAVED] {out_path.name}")

    print(f"\nCreated {len(outputs)} HUD images")

    return outputs