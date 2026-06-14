import sys
from pathlib import Path

SCRIPTS_ROOT="/SNS/VENUS/IPTS-36967/shared/Jack/fullHUDscripting/ORNLVENUS_HUD_WTe2/scripts"
sys.path.insert(0,SCRIPTS_ROOT)  # allow local module imports

import paneMakers.rolledImagePanes

import paneMakers.compositeMaker
import paneMakers.roi_selector

import controllerMethods
import re,numpy as np,matplotlib.pyplot as plt
from PIL import Image
import populateHDFSpreadSheet
from matplotlib.widgets import RectangleSelector

NEXUS="/SNS/users/damelio2/data/SNS/VENUS/IPTS-36967/nexus/"

MASTER_IMAGE_SOURCE=Path("/SNS/VENUS/IPTS-36967/shared/autoreduce/images/tpx1/raw/radiography/")
ENERGY_FRAME_MIN=560  # lower TOF energy window bound
ENERGY_FRAME_MAX=830  # upper TOF energy window bound

STARTING_RUN_NUMBER=23139
ENDING_RUN_NUMBER=23141
RUN_NUMBERS=list(range(STARTING_RUN_NUMBER,ENDING_RUN_NUMBER+1))  # run range list

EXPECTED_IMAGE_DURATION=5  # seconds per frame
ROLL_LENGTH_IN_MIN=[5]  # roll grouping size in minutes

MASTER_DESTINATION=Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/HUDtest_SampleD/")
OB_PATH=Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/OBs/")



def main():
    roi=False  # ROI is selected once and reused across rolls

    for roll in ROLL_LENGTH_IN_MIN:
        tifFolder=MASTER_DESTINATION/(str(roll)+"rolls") 
        #tifPaths,auto_balance_images=paneMakers.rolledImagePanes.batch_process_images_into_rolls(master_image_source=MASTER_IMAGE_SOURCE, run_number_range=RUN_NUMBERS, master_file_destination=MASTER_DESTINATION, scanLen_min=EXPECTED_IMAGE_DURATION, nexus=NEXUS, ob_path=OB_PATH, roll_length=roll)  # generate rolled TIFF datasets + auto-balanced images
        tifPaths,auto_balance = controllerMethods.load_existing_roll_outputs(tifFolder)
         # output folder for this roll group

        if roi==False:
            roi=paneMakers.roi_selector.select_roi(tifFolder)  # user selects ROI once per batch

        csvPath=populateHDFSpreadSheet.update_HDF_sheet(
            MASTER_DESTINATION/"HDFSpreadsheet",
            MASTER_IMAGE_SOURCE,
            NEXUS
        )  # regenerate/update metadata spreadsheet

        panes=controllerMethods.prepare_image_panes(tifFolder,tifFolder,csvPath,roi)  # build all analysis panes

       
        paneMakers.compositeMaker.glue_multiple_pane_sets(
            panes,
            tifFolder/"HUD"
        )  # combine all outputs into final HUD composite

    return 0

if __name__=="__main__":
    main()