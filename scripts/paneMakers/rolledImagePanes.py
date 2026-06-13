from pathlib import Path
import os
import h5py
import numpy as np
import tifffile
from astropy.io import fits


# ==========================================================
# User Settings
# ==========================================================




MASTER_SOURCE = Path("/SNS/VENUS/IPTS-36967/shared/autoreduce/images/tpx1/raw/radiography/")
FRAME_MIN = 560
FRAME_MAX = 830


A = 23417
B = 23533
RUN_NUMBERS = list(range(A, B + 1))

EXPECTED_IMAGE_DURATION_SECONDS = 5 * 60 #How long one frame was collected for, not the total length of the rolled frames
DURATION_CHECK_THRESHOLD_PERCENTAGE = 20 # Runs that deviate by more than this percentage 

NORMALIZE_BOOL = True
WINDOW_SIZE = 6


BAD_FILE_LOG = Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/Nucleation/quarterPercent/Crystal Images_rolling/badfile.log")
FINAL_IMAGE_LENGTH = int(WINDOW_SIZE*EXPECTED_IMAGE_DURATION_SECONDS/60)
MASTER_DESTINATION = Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/Nucleation/quarterPercent/Crystal Images_rolling/" + str(FINAL_IMAGE_LENGTH) + "_roll")
OB_PATH = Path(
	"/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/OBs/June_OB_" + str(FINAL_IMAGE_LENGTH) + "min.tif"
)
# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

#OUTPUT_BASE = Path.cwd() / "OUTPUT_FOLDER"
#OUTPUT_BASE.mkdir(parents=True, exist_ok=True)




# ==========================================================
# IMAGE LOADER (TIFF + FITS via Astropy)
# ==========================================================

def load_image(path):

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

		with open(BAD_FILE_LOG, "a") as f:
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

'''
    with h5py.File(hdf5_input, "r") as f:
        # Times (convert to Python strings)
        start_time = np.array(f["entry/start_time"], dtype=str).item()
        end_time = np.array(f["entry/end_time"], dtype=str).item()

        start_time = clean_iso(start_time)
        end_time = clean_iso(end_time)

        # Duration
        duration = float(np.array(f["entry/duration"]))

        # Temperatures
        start_T = float(np.array(f["entry/DASlogs/BL10:SE:ND2:CH1:PV/minimum_value"]))
        end_T = float(np.array(f["entry/DASlogs/BL10:SE:ND2:CH1:PV/maximum_value"]))
        avg_T = float(np.array(f["entry/DASlogs/BL10:SE:ND2:CH1:PV/average_value"]))

        return {
            "run_number": Run_num,
            "filename": hdf5_input,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "start_T": start_T,
            "end_T": end_T,
            "avg_T": avg_T
        }
'''
# ==========================================================
# OPTIONAL: FITS → TIFF CONVERSION
# ==========================================================

def convert_to_tiff(path, image_array):

	out_name = path.stem + ".tif"
	out_path = CONVERTED_DIR / out_name

	tifffile.imwrite(out_path, image_array.astype(np.float32))

	print(f"[CONVERTED] {path.name} -> {out_path.name}")


# ==========================================================
# OPEN BEAM
# ==========================================================

def load_open_beam(path):
	print("Loading open beam...")
	return load_image(path)


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

def sum_image_stack(files):

	summed = None

	for f in files:

		img = load_image(f)

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


def process_runs_rolling_global(MASTER_SOURCE, run_numbers, open_beam, window_size):

	print(f"\n--- GLOBAL Rolling across ALL containers (N={window_size}) ---")
	print("Final Image Length:", FINAL_IMAGE_LENGTH)

	run_min = min(run_numbers)
	run_max = max(run_numbers)

	run_map = discover_all_runs(MASTER_SOURCE, run_min, run_max)
	output_path = "postProcess_Run_" +  str(str(run_map[0][1]).split("/")[-1].split("Run_")[1])
	if len(run_map) < window_size:
		print("Not enough global runs found")
		return False

	out_dir = MASTER_DESTINATION  / output_path 
	out_dir.mkdir(parents=True, exist_ok=True)
	CONVERTED_DIR = MASTER_DESTINATION  / output_path / "converted_tiffs"
	CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

	BAD_FILE_LOG = MASTER_DESTINATION  / output_path / "bad_files.log"
	
	num_windows = len(run_map) - window_size + 1

	for i in range(num_windows):

		window_runs = run_map[i:i + window_size]

		window_sum = None

		start_run = window_runs[0][0]
		end_run = window_runs[-1][0]
		
		rangeContainsBadFrames = False #will control if we do compute a roll, based on if the duration is too short (aborted run)  or too long (beam down)
		for run_id, folder, in window_runs:
			hdf5_data = get_HDF_for_single_run(run_id)
			duration = get_HDF_duration(hdf5_data)
			if duration > EXPECTED_IMAGE_DURATION_SECONDS * (1 + (0.01*DURATION_CHECK_THRESHOLD_PERCENTAGE)) or duration < EXPECTED_IMAGE_DURATION_SECONDS * (1 - (0.01*DURATION_CHECK_THRESHOLD_PERCENTAGE)):
				rangeContainsBadFrames = True
				print("Bad Frame:", run_id, duration, "Expected:", EXPECTED_IMAGE_DURATION_SECONDS)

		#print(window_runs, "aaa")
		if rangeContainsBadFrames:
			
			print("Skip")			
		else:
			for run_id, folder in window_runs:

				files = collect_images(folder, FRAME_MIN, FRAME_MAX)
				
				summed = sum_image_stack(files)

				if summed is None:
					continue

				if window_sum is None:
					window_sum = np.zeros_like(summed, dtype=np.float32)

				window_sum += summed

			if window_sum is None:
				continue

			img = normalize_by_open_beam(window_sum, open_beam)
			img = rotate(img)

			out_path = out_dir / f"GlobalRunWindow_{start_run}_{end_run}.tif"

			save(img, out_path)

	return True


# ==========================================================
# MAIN
# ==========================================================

def main():

	open_beam = load_open_beam(OB_PATH)


	# Workflow 3 (NEW)
	process_runs_rolling_global(MASTER_SOURCE, RUN_NUMBERS, open_beam, WINDOW_SIZE)

	print("\nDone.")


if __name__ == "__main__":
	main()
