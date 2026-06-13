import paneMakers.rolledImagePanes
from pathlib import Path
import re

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

def batch_process_images_into_rolls(master_image_source = MASTER_IMAGE_SOURCE, run_number_range = RUN_NUMBERS, master_file_destination = MASTER_DESTINATION, scanLen_min = EXPECTED_IMAGE_DURATION, ob_path = OB_PATH, roll_length= 5): #will perform a series of data processing steps, and then caclulate plots to make composite image

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
        allImages = pre_existing_images + newImages

        return allImages





    DURATION_CHECK_THRESHOLD_PERCENTAGE = 20 # Runs that deviate by more than this percentage 

    NORMALIZE_BOOL = True
    
    BAD_FILE_LOG = Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/Nucleation/quarterPercent/Crystal Images_rolling/badfile.log")

    #FINAL_IMAGE_LENGTH = int(WINDOW_SIZE*EXPECTED_IMAGE_DURATION_SECONDS/60)

    #OB_PATH = Path(
      #  "/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/OBs/June_OB_" + str(FINAL_IMAGE_LENGTH) + "min.tif"
  #  )
    return 0
def main():
    batch_process_images_into_rolls(roll_length = 5)
    return 0

	
if __name__ == "__main__":
	main()