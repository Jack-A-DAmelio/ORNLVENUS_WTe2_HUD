import paneMakers.rolledImagePanes
import paneMakers.temperaturePane
import paneMakers.averageGreyscalePane
import paneMakers.compositeMaker
import paneMakers.roi_selector
import paneMakers.greyScaleHistogramPane
from pathlib import Path
import re
import numpy as np
from PIL import Image
import populateHDFSpreadSheet
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
MASTER_IMAGE_SOURCE = Path("/SNS/VENUS/IPTS-36967/shared/autoreduce/images/tpx1/raw/radiography/") #The IPTS folder that contains all data
ENERGY_FRAME_MIN = 560# The specific images per run, correspond to ToF Energy
ENERGY_FRAME_MAX = 830



STARTING_RUN_NUMBER = 23417# THe run number to be analyzed
ENDING_RUN_NUMBER = 23535
RUN_NUMBERS = list(range(STARTING_RUN_NUMBER, ENDING_RUN_NUMBER + 1))

EXPECTED_IMAGE_DURATION = 5 #How long one frame was collected for, not the total length of the rolled frames
ROLL_LENGTH_IN_MIN = [5] # how long the total roll is, should be a multiple of EXPECTED_IMAGE_DURATION

MASTER_DESTINATION = Path("/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/HUDtest/") #Where are these files will go
OB_PATH = Path(
	"/SNS/VENUS/IPTS-36967/shared/Batch_analysis_6-12-25/June/OBs/")# the ob for the whole roll, make sure hte length is right

def prepare_image_panes(tif_folder, destination, csv, roi = False):

    temperaturePane = True
    averageGreyscalePane = True
    greyScaleHistogram = True

    
    temperaturePanels = []
    if temperaturePane:
        temperaturePanels = paneMakers.temperaturePane.prepare_temperaturePane(tif_folder, csv, destination / "temperaturePanes")
    averageGreyscalePanes = []
    if averageGreyscalePane:
        averageGreyscalePanes = paneMakers.averageGreyscalePane.run_roi_pipeline(tif_folder, destination / "averageGrayScalePanes", roi)
    greyScaleHistogramPanes = []
    if greyScaleHistogram:
        greyScaleHistogramPanes = paneMakers.greyScaleHistogramPane.create_roi_panes(tif_folder, destination / "histogramPanes", roi)

    return [temperaturePanels, averageGreyscalePanes,greyScaleHistogramPanes]





def main():
    roi = False
    for roll in ROLL_LENGTH_IN_MIN:

        #Make the tif files, avoiding making already made data
        tifPaths, auto_balance_images = paneMakers.rolledImagePanes.batch_process_images_into_rolls(master_image_source = MASTER_IMAGE_SOURCE, run_number_range = RUN_NUMBERS, master_file_destination = MASTER_DESTINATION, scanLen_min = EXPECTED_IMAGE_DURATION, ob_path = OB_PATH, roll_length = roll)
        tifFolder = MASTER_DESTINATION / (str(roll) +"rolls")
        if roi == False:#Only make the ROI once for whole batch
            roi = paneMakers.roi_selector.select_roi(tifFolder)
        #path to tif folder 
        
        #update HDF spreadshet
        csvPath = populateHDFSpreadSheet.update_HDF_sheet(MASTER_DESTINATION / "HDFSpreadsheet", MASTER_IMAGE_SOURCE)
        # createPanes of data analysis
        panes = prepare_image_panes(tifFolder,tifFolder, csvPath, roi)
        panes.append(auto_balance_images)
        #Make image composits
        paneMakers.compositeMaker.glue_multiple_pane_sets(panes, tifFolder / "HUD")



    return 0

	
if __name__ == "__main__":
	main()