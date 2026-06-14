from pathlib import Path
import numpy as np
from PIL import Image


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(path):
    img = Image.open(path)
    arr = np.array(img)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    arr = arr.astype(np.float32)
    arr -= arr.min()

    if arr.max() > 0:
        arr /= arr.max()

    return (arr * 255).astype(np.uint8)


# ==========================================================
# GLUE FUNCTION (PURE STRING MATCHING)
# ==========================================================

def glue_multiple_pane_sets(pane_lists, output_folder):

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # STEP 1: map basename (NO EXT) → full path
    # --------------------------------------------------

    maps = []

    for pane_list in pane_lists:

        lookup = {}

        for p in pane_list:

            p = Path(p)

            key = p.stem   # <-- THIS IS THE ONLY KEY

            lookup[key] = p

        maps.append(lookup)

        print(f"[INFO] loaded {len(lookup)} files")

    # --------------------------------------------------
    # STEP 2: find common filenames
    # --------------------------------------------------

    common_keys = set(maps[0].keys())

    for m in maps[1:]:
        common_keys &= set(m.keys())

    common_keys = sorted(common_keys)

    print(f"\nFound {len(common_keys)} matching groups")

    if not common_keys:
        print("[WARNING] No overlapping filenames")
        return []

    # --------------------------------------------------
    # STEP 3: stitch
    # --------------------------------------------------

    outputs = []

    for key in common_keys:

        images = []

        for m in maps:
            img = load_image(m[key])
            images.append(img)

        # normalize height
        target_height = min(img.shape[0] for img in images)

        resized = []

        for img in images:

            pil = Image.fromarray(img)

            if pil.height != target_height:
                new_width = int(pil.width * target_height / pil.height)
                pil = pil.resize((new_width, target_height), Image.NEAREST)

            resized.append(np.array(pil))

        combined = np.concatenate(resized, axis=1)

        out_path = output_folder / f"{key}.png"

        Image.fromarray(combined.astype(np.uint8)).save(out_path)

        outputs.append(out_path)

    return outputs