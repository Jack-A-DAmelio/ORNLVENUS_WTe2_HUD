import paneMakers.rolledImagePanes
import paneMakers.temperaturePane
import paneMakers.averageGreyscalePane
from pathlib import Path
import re
import numpy as np
from PIL import Image
import populateHDFSpreadSheet
MASTER_IMAGE_SOURCE = Path("/SNS/VENUS/IPTS-36967/shared/autoreduce/images/tpx1/raw/radiography/")
ENERGY_FRAME_MIN = 560
ENERGY_FRAME_MAX = 830


STARTING_RUN_NUMBER = 23417
ENDING_RUN_NUMBER = 23535
RUN_NUMBERS = list(range(STARTING_RUN_NUMBER, ENDING_RUN_NUMBER + 1))

EXPECTED_IMAGE_DURATION = 5 #How long one frame was collected for, not the total length of the rolled frames

MASTER_DESTINATION = Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/HUDtest/")
OB_PATH = Path(
	"/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/OBs/")

def prepare_image_panes(tif_folder, destination, csv):

    temperaturePane = True
    averageGreyscalePane = True

    
    temperaturePanels = []
    if temperaturePane:
        temperaturePanels = paneMakers.temperaturePane.prepare_temperaturePane(tif_folder, csv, destination / "temperaturePanes")
    averageGreyscalePanes = []
    if averageGreyscalePane:
        averageGreyscalePanes = paneMakers.averageGreyscalePane.run_roi_pipeline(tif_folder, destination / "averageGrayScalePanes")

    return [temperaturePanels, averageGreyscalePanes]



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
 

def batch_process_images_into_rolls(master_image_source = MASTER_IMAGE_SOURCE, run_number_range = RUN_NUMBERS, master_file_destination = MASTER_DESTINATION, scanLen_min = EXPECTED_IMAGE_DURATION, ob_path = OB_PATH, roll_length= 5): #will perform a series of data processing steps, and then caclulate plots to make composite image
    def grey_balance(arr, low=1, high=99):
        """
        Percentile-based contrast normalization.
        Works on grayscale or RGB uint16/float images.
        """

        arr = arr.astype(np.float32)

        # if RGB, operate per-channel
        if arr.ndim == 3:
            out = np.zeros_like(arr, dtype=np.float32)

            for c in range(arr.shape[2]):
                channel = arr[:, :, c]

                lo = np.percentile(channel, low)
                hi = np.percentile(channel, high)

                if hi - lo < 1e-6:
                    out[:, :, c] = 0
                else:
                    out[:, :, c] = (channel - lo) / (hi - lo)

            arr = out
        else:
            lo = np.percentile(arr, low)
            hi = np.percentile(arr, high)

            if hi - lo > 1e-6:
                arr = (arr - lo) / (hi - lo)
            else:
                arr = np.zeros_like(arr)

        
        return np.clip(arr, 0, 1)
    #Creation of simple division normalized images, some of which will be summed ("Rolled") for higher contrast
    #choose roll lengths
    roll_lengths = [roll_length]
    for rollLength in roll_lengths:

        window_size = rollLength / scanLen_min
        if window_size % 1 != 0:
            print ("Error: Cannot make ", rollLength, " min rolls with ", scanLen_min, " min scans")
        else:
            print("Rolling ", window_size, "scans together")

        window_size = int(window_size)
        #Determine what run numbers already exist in target location
        targetLocation_for_rolls = Path (master_file_destination / (str(rollLength) + "rolls/"))

        if targetLocation_for_rolls.exists() == False:
            print("Creating:", targetLocation_for_rolls)
            targetLocation_for_rolls.mkdir(parents=True, exist_ok=True)

        # ==========================================================
        # enumerate all .tif files in this folder
        # ==========================================================
        tif_files = sorted(targetLocation_for_rolls.glob("*.tif"))

        print(f"Found {len(tif_files)} .tif files in {targetLocation_for_rolls}")
        pre_existing_images = []
        # optional: print names for debug
        for f in tif_files:
            pre_existing_images.append(f.name)
        


        
        runs_which_have_not_been_processed_yet = []
        if len(tif_files) == 0:
            print("No prexisting data found")
            runs_which_have_not_been_processed_yet = run_number_range
        else: #find the most recent run number that has bin processed

            max_run = -float("inf")

            for f in tif_files:

                # extract all numbers in filename
                numbers = re.findall(r"\d+", f.stem)

                if not numbers:
                    continue

                # convert to ints
                numbers = list(map(int, numbers))

                # take max number in this file
                file_max = max(numbers)

                # update global max
                if file_max > max_run:
                    max_run = file_max

            print("Largest run number found:", max_run)

            #compute which preceding frames are need to complete the immediate frames after the max frame
            runs_which_have_not_been_processed_yet = list(range(max_run-window_size, run_number_range[-1]))



        #def process_runs_rolling_global(master_image_source, file_destination, run_numbers, open_beam, window_size, image_duration, duration_percentage_filter = 20):
        newImages = paneMakers.rolledImagePanes.process_runs_rolling_global(master_image_source, master_file_destination / (str(rollLength) + "rolls/"), runs_which_have_not_been_processed_yet, ob_path / ("June_OB_" + str(rollLength) + "min.tif" ),  window_size, scanLen_min*60, energyLabels = [560, 830] , duration_percentage_filter = 20)
        tifPaths = []
        for tif in pre_existing_images:
            print(tif)
            tifPaths.append(MASTER_DESTINATION / (str(rollLength) +"rolls") / tif)

        #print(ASTER_DESTINATION / (str(roll) +"rolls") / tif)
        allImages = tifPaths + newImages
        
        balancedImages = []

        for img_path in allImages:

            arr = np.array(Image.open(img_path))

            arr = np.squeeze(arr)

            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)

            arr = grey_balance(arr)

            arr = (arr * 255).astype(np.uint8)

            Image.fromarray(arr).save(img_path)

            balancedImages.append(img_path)
        
  
        return balancedImages



def main():
    roll = 5
    tifPaths = batch_process_images_into_rolls(roll_length = roll)
   
    csvPath = populateHDFSpreadSheet.update_HDF_sheet(MASTER_DESTINATION / "HDFSpreadsheet", MASTER_IMAGE_SOURCE)
    print("Making Temp Panes")
    panes = prepare_image_panes(MASTER_DESTINATION / (str(roll) +"rolls"), MASTER_DESTINATION / (str(roll) +"rolls"), csvPath)
    panes.append(tifPaths)
    glue_multiple_pane_sets(panes, MASTER_DESTINATION / "HUD")



    return 0

	
if __name__ == "__main__":
	main()