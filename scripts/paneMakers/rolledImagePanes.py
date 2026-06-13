from pathlib import Path
import os
import h5py
import numpy as np
import tifffile
from astropy.io import fits


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
