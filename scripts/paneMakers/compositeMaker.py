from pathlib import Path
import re
import numpy as np
from PIL import Image
import populateHDFSpreadSheet

def glue_multiple_pane_sets(
    pane_lists,
    output_folder
):
    """
    pane_lists:
        [
            [Path(...), Path(...), ...],
            [Path(...), Path(...), ...],
            [Path(...), Path(...), ...]
        ]

    output:
        one glued image for each matching run range
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # build lookup tables
    # --------------------------------------------------

    pane_maps = []

    for pane_list in pane_lists:

        lookup = {}

        for p in pane_list:

            p = Path(p)

            m = re.search(
                r'(\d+)_(\d+)',
                p.stem
            )

            if m is None:
                continue

            key = (
                int(m.group(1)),
                int(m.group(2))
            )

            lookup[key] = p

        pane_maps.append(lookup)

    # --------------------------------------------------
    # find common keys
    # --------------------------------------------------

    common_keys = set(
        pane_maps[0].keys()
    )

    for lookup in pane_maps[1:]:
        common_keys &= set(
            lookup.keys()
        )

    common_keys = sorted(
        common_keys
    )

    print(
        f"Found {len(common_keys)} matching pane groups"
    )

    # --------------------------------------------------
    # glue
    # --------------------------------------------------

    output_paths = []

    for key in common_keys:

        images = []

        for lookup in pane_maps:

            img = Image.open(lookup[key])

            arr = np.array(img)

            # ---------------------------
            # FIX SHAPE ISSUES
            # ---------------------------
            arr = np.squeeze(arr)

            # if weird higher-D arrays sneak in
            if arr.ndim > 3:
                arr = arr.reshape(arr.shape[-2], arr.shape[-1])

            # force RGB
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]

            # ---------------------------
            # FIX INTENSITY (prevents black TIFFs)
            # ---------------------------
            arr = arr.astype(np.float32)
            arr = arr - arr.min()

            if arr.max() > 0:
                arr = arr / arr.max()

            arr = (arr * 255).astype(np.uint8)

            images.append(arr)

        heights = [
            img.shape[0]
            for img in images
        ]

        target_height = min(
            heights
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

        combined = np.concatenate(
            resized,
            axis=1
        )

        start_run, end_run = key

        out_path = (
            output_folder /
            f"HUD_{start_run}_{end_run}.png"
        )

        Image.fromarray(
            combined.astype(np.uint8)
        ).save(out_path)

        output_paths.append(
            out_path
        )

    return output_paths
 