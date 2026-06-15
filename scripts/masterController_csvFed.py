import sys
from pathlib import Path
import ast
import pandas as pd

SCRIPTS_ROOT="/SNS/VENUS/IPTS-36967/shared/Jack/fullHUDscripting/ORNLVENUS_HUD_WTe2/scripts"
sys.path.insert(0,SCRIPTS_ROOT)

import paneMakers.rolledImagePanes
import paneMakers.compositeMaker
import paneMakers.roi_selector

import controllerMethods
import populateHDFSpreadSheet


# ================= USER INPUTS =================

INPUT_CSV = Path("/SNS/users/lfigari/data/SNS/VENUS/IPTS-36967/shared/FeedTheRollOvernight.csv")

NEXUS="/SNS/users/lfigari/data/SNS/VENUS/IPTS-36967/nexus/"

MASTER_IMAGE_SOURCE=Path(
    "/SNS/VENUS/IPTS-36967/shared/autoreduce/images/tpx1/raw/radiography/"
)

ENERGY_FRAME_MIN=560
ENERGY_FRAME_MAX=830


# =================================================


def process_roll(
        run_numbers,
        expected_length,
        roll,
        master_destination,
        ob_path,
        roi
    ):

    print(f"Processing {roll} minute roll")
    
    # for num in run_numbers:
    #     num = int(num)

    paneMakers.rolledImagePanes.batch_process_images_into_rolls(
        master_image_source=MASTER_IMAGE_SOURCE,
        run_number_range=run_numbers,
        master_file_destination=master_destination,
        scanLen_min=expected_length,
        nexus=NEXUS,
        ob_path=ob_path,
        roll_length=roll
    )

    tifFolder = master_destination / (str(roll) + "rolls/")
    print(tifFolder)


    csvPath = populateHDFSpreadSheet.update_HDF_sheet(
        tifFolder/"HDFSpreadsheet",
        MASTER_IMAGE_SOURCE,
        NEXUS
    )


    panes = controllerMethods.prepare_image_panes(
        tifFolder,
        tifFolder,
        csvPath,
        roi
    )


    # reload auto balance output
    _, auto_balance = controllerMethods.load_existing_roll_outputs(tifFolder)

    panes.append(auto_balance)


    paneMakers.compositeMaker.glue_multiple_pane_sets(
        panes,
        tifFolder/"HUD"
    )



def main():

    dataframe = pd.read_csv(INPUT_CSV)

    roi = False


    for index, row in dataframe.iterrows():

        print("\n================================")
        print("Processing row:")
        print(row["DropOffAddress"])
        print("================================")


        OB_PATH = Path(row["OB_Address"])

        STARTING_RUN_NUMBER = int(row["StartRunNum"])
        ENDING_RUN_NUMBER = int(row["EndRunNum"])

        RUN_NUMBERS = list(
            range(
                STARTING_RUN_NUMBER,
                ENDING_RUN_NUMBER + 1
            )
        )


        EXPECTED_IMAGE_DURATION = int(row["ExpectedLength"])

        MASTER_DESTINATION = Path(row["DropOffAddress"])


        ROLL_LENGTHS = ast.literal_eval(row["WindowArray"])

        ROLL_LENGTHS = [
            roll for roll in ROLL_LENGTHS
            if roll >= EXPECTED_IMAGE_DURATION
        ]


        for roll in ROLL_LENGTHS:

            # generate the first roll so ROI can be selected
            paneMakers.rolledImagePanes.batch_process_images_into_rolls(
                master_image_source=MASTER_IMAGE_SOURCE,
                run_number_range=RUN_NUMBERS,
                master_file_destination=MASTER_DESTINATION,
                scanLen_min=EXPECTED_IMAGE_DURATION,
                nexus=NEXUS,
                ob_path=OB_PATH,
                roll_length=roll
            )


            tifFolder = MASTER_DESTINATION / (str(roll) + "rolls/")

            sdasdasad.asdasada(auto_balance, roi)
            if roi is False:
                roi = paneMakers.roi_selector.select_roi(tifFolder)


            csvPath = populateHDFSpreadSheet.update_HDF_sheet(
                tifFolder/"HDFSpreadsheet",
                MASTER_IMAGE_SOURCE,
                NEXUS
            )


            panes = controllerMethods.prepare_image_panes(
                tifFolder,
                tifFolder,
                csvPath,
                roi
            )


            _, auto_balance = controllerMethods.load_existing_roll_outputs(
                tifFolder
            )

            panes.append(auto_balance)


            paneMakers.compositeMaker.glue_multiple_pane_sets(
                panes,
                tifFolder/"HUD"
            )


    return 0



if __name__=="__main__":
    main()