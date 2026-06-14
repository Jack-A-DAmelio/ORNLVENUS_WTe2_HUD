from pathlib import Path
import os
import h5py
import numpy as np
import tifffile
from astropy.io import fits
from PIL import Image

import re
# ==========================================================
# User Settings
# ==========================================================


# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

#OUTPUT_BASE = Path.cwd() / "OUTPUT_FOLDER"
#OUTPUT_BASE.mkdir(parents=True, exist_ok=True)




# ==========================================================
# IMAGE LOADER (TIFF + FITS via Astropy)
# ==========================================================

def load_image(path, badfilelog):

	path = Path(path)

	try:

		suffix = path.suffix.lower()

		# -------------------------
		# TIFF
		# -------------------------
		if suffix in [".tif", ".tiff"]:
			return tifffile.imread(path).astype(np.float32)

		# -------------------------
		# FITS (Astropy)
		# -------------------------
		if suffix in [".fits", ".fit"]:

			with fits.open(path, memmap=False) as hdul:
				data = hdul[0].data

				if data is None:
					raise ValueError("Empty FITS file")

				data = np.squeeze(data)
				return np.asarray(data, dtype=np.float32)

		raise ValueError(f"Unsupported file type: {suffix}")

	except Exception as e:

		with open( badfilelog, "a") as f:
			f.write(f"{path} :: {repr(e)}\n")

		print(f"[SKIP] {path} -> {e}")
		return None


# ==========================================================
# OPTIONAL: HDF Loader
# ==========================================================
def get_HDF_for_single_run(Run_num):
	nexusLocation= "/SNS/users/damelio2/data/SNS/VENUS/IPTS-36967/nexus/"

	hdf5_input = nexusLocation + "VENUS_" + str(Run_num) + ".nxs.h5"
	return hdf5_input


# ==========================================================
# OPTIONAL: HDF getDuration
# ==========================================================
def get_HDF_duration(hdf5_input):
	# Duration
	with h5py.File(hdf5_input, "r") as f:
		duration = float(np.array(f["entry/duration"]))
	return duration




# ==========================================================
# OPEN BEAM
# ==========================================================

def load_open_beam(path,  badfilelog):
	print("Loading open beam...")
	return load_image(path, badfilelog)


# ==========================================================
# FILE DISCOVERY (single BASE_DIR)
# ==========================================================

def find_run_folder(base_dir, run):
	for d in base_dir.iterdir():
		if d.is_dir() and f"Run_{run}_" in d.name:
			return d
	return None


def collect_images(run_folder, frame_min, frame_max):

	files = []

	for ext in ["*.tif", "*.tiff", "*.fit", "*.fits"]:
		for f in run_folder.glob(ext):

			try:
				frame_number = int(f.stem.split("_")[-1])

				if frame_min <= frame_number <= frame_max:
					files.append(f)

			except Exception:
				continue

	files.sort()
	return files


# ==========================================================
# STACK SUM
# ==========================================================

def sum_image_stack(files,  badfilelog):

	summed = None

	for f in files:

		img = load_image(f,  badfilelog)

		if img is None:
			continue

		# ONLY convert FITS files (not TIFF)
		suffix = f.suffix.lower()
		if suffix in [".fits", ".fit"]:
			convert_to_tiff(f, img)

		if summed is None:
			summed = np.zeros_like(img, dtype=np.float32)

		if img.shape != summed.shape:
			continue

		summed += img

	return summed


# ==========================================================
# NORMALIZATION + ROTATION
# ==========================================================

def normalize_by_open_beam(img, ob):

	if img.shape != ob.shape:
		return None

	with np.errstate(divide="ignore", invalid="ignore"):
		out = img / ob
		out[~np.isfinite(out)] = 0

	return out


def rotate(img):
	return np.rot90(img, k=-1)


def save(img, path):
	tifffile.imwrite(path, img.astype(np.float32))
	#print("Saved:", path)





# ==========================================================
# WORKFLOW 3: GLOBAL CROSS-CONTAINER ROLLING
# ==========================================================

def discover_all_runs(MASTER_SOURCE, run_min, run_max):

	run_map = []

	for root, dirs, files in os.walk(MASTER_SOURCE):

		root_path = Path(root)

		for d in dirs:

			if "Run_" not in d:
				continue

			try:
				run_id = int(d.split("Run_")[1].split("_")[0])
			except Exception:
				continue

			if run_min <= run_id <= run_max:
				run_map.append((run_id, root_path / d))

	run_map.sort(key=lambda x: x[0])

	return run_map


def process_runs_rolling_global(master_image_source, file_destination, run_numbers, open_beam, window_size, image_duration, energyLabels = [560, 830] ,duration_percentage_filter = 20):
	print("Beginning Rolling in range", run_numbers)
	out_dir = file_destination
	out_dir.mkdir(parents=True, exist_ok=True)
	badfilelog = out_dir / "bad_files.log"
	print(open_beam)
	open_beam = load_open_beam(open_beam,  badfilelog)
	energy_min = energyLabels[0]
	energy_max = energyLabels[1]

	run_min = min(run_numbers)
	run_max = max(run_numbers)

	run_map = discover_all_runs(master_image_source, run_min, run_max)
	if len(run_map) < window_size:
		print("Not enough global runs found")
		return False

	

	
	
	num_windows = int(len(run_map) - window_size + 1)
	#print(window_size)
	images_made = 0
	created_images = []
	for i in range(num_windows):
		#print(i)

		window_runs = run_map[i:i + window_size]
		#print(window_runs)
		window_sum = None

		start_run = window_runs[0][0]
		end_run = window_runs[-1][0]
		
		rangeContainsBadFrames = False #will control if we do compute a roll, based on if the duration is too short (aborted run)  or too long (beam down)
		for run_id, folder, in window_runs:
			hdf5_data = get_HDF_for_single_run(run_id)
			duration = get_HDF_duration(hdf5_data)
			if duration > image_duration* (1 + (0.01*duration_percentage_filter)) or duration < image_duration * (1 - (0.01*duration_percentage_filter)):
				rangeContainsBadFrames = True
				print("Bad Frame:", run_id, duration, "Expected:", image_duration)
		
		#print(window_runs, "aaa")
		if rangeContainsBadFrames:
			
			print("Skip")			
		else:
			#print("here")
			for run_id, folder in window_runs:

				files = collect_images(folder, energy_min, energy_max)
				
				summed = sum_image_stack(files, badfilelog)

				if summed is None:
					continue

				if window_sum is None:
					window_sum = np.zeros_like(summed, dtype=np.float32)

				window_sum += summed

			if window_sum is None:
				continue
			#print(type(open_beam))
			img = normalize_by_open_beam(window_sum, open_beam)
			img = rotate(img)

			out_path = out_dir / f"GlobalRunWindow_{start_run}_{end_run}.tif"
			#print(out_path)

			save(img, out_path)
			created_images.append(out_path)
			images_made += 1

	print("Created ", images_made, " images")

	return created_images


from pathlib import Path
import numpy as np
from PIL import Image


def auto_balance_images(allImages):

    if len(allImages) == 0:
        return []

    # Determine source folder
    source_folder = Path(allImages[0]).parent

    # Create sibling output folder
    output_folder = (
        source_folder /
        f"{source_folder.name}_autobalanced"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    balancedImages = []

    for img_path in allImages:

        img_path = Path(img_path)

        arr = np.array(
            Image.open(img_path)
        )

        arr = np.squeeze(arr)

        if arr.ndim == 2:
            arr = np.stack(
                [arr] * 3,
                axis=-1
            )

        arr = grey_balance(arr)

        arr = np.clip(
            arr,
            0,
            1
        )

        arr = (
            arr * 255
        ).astype(np.uint8)

        out_path = (
            output_folder /
            img_path.name
        )

        Image.fromarray(arr).save(
            out_path
        )

        balancedImages.append(
            out_path
        )

    return balancedImages
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
def batch_process_images_into_rolls(master_image_source, run_number_range, master_file_destination, scanLen_min, ob_path, roll_length): #will perform a series of data processing steps, and then caclulate plots to make composite image
    
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
        newImages = process_runs_rolling_global(master_image_source, master_file_destination / (str(rollLength) + "rolls/"), runs_which_have_not_been_processed_yet, ob_path / ( str(rollLength) + "min_ob.tif" ),  window_size, scanLen_min*60, energyLabels = [560, 830] , duration_percentage_filter = 20)
        tifPaths = []
        for tif in pre_existing_images:
            print(tif)
            tifPaths.append(master_file_destination / (str(rollLength) +"rolls") / tif)

        #print(ASTER_DESTINATION / (str(roll) +"rolls") / tif)
        allImages = tifPaths + newImages
        
        balancedImages = auto_balance_images(allImages)

      
        
  
        return allImages, balancedImages


# ==========================================================
# MAIN
# ==========================================================

def main():

	open_beam = load_open_beam(OB_PATH)


	# Workflow 3 (NEW)
	process_runs_rolling_global(MASTER_SOURCE, RUN_NUMBERS, open_beam, WINDOW_SIZE)

	print("\nDone.")


#if __name__ == "__main__":
	#main()
