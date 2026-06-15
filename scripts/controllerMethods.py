import paneMakers.temperaturePane
import paneMakers.averageGreyscalePane
import paneMakers.greyScaleHistogramPane


from pathlib import Path

def load_existing_roll_outputs(roll_folder):
    roll_folder=Path(roll_folder)  # folder like ".../5rolls"
    tif_paths=sorted(roll_folder.glob("GlobalRunWindow_*.tif"))  # raw rolled images only
    auto_folder=roll_folder/f"{roll_folder.name}_autobalanced"  # derived auto-balance folder

    if auto_folder.exists():
        auto_paths=sorted(auto_folder.glob("GlobalRunWindow_*.tif"))  # balanced images
    else:
        auto_paths=[]

    return tif_paths,auto_paths

def prepare_image_panes(tif_folder,destination,csv,roi=False):
    make_temperature_pane= True
    make_average_greyscale_pane=True
    make_histogram_pane= True

    panes=[]  # store all generated pane sets

    if make_temperature_pane:
        temperature_panes=paneMakers.temperaturePane.prepare_temperaturePane(
            tif_folder,csv,destination/"temperaturePanes"
        )  # generate temperature-based analysis panes

        print(f"[INFO] Temperature panes: {len(temperature_panes)}")
        panes.append(temperature_panes)

    if make_average_greyscale_pane:
        average_greyscale_panes=paneMakers.averageGreyscalePane.run_roi_pipeline(
            tif_folder,destination/"averageGrayScalePanes",roi
        )  # ROI-based average intensity panes

        print(f"[INFO] Average greyscale panes: {len(average_greyscale_panes)}")
        panes.append(average_greyscale_panes)

    if make_histogram_pane:
        histogram_panes=paneMakers.greyScaleHistogramPane.create_roi_panes(
            tif_folder,destination/"histogramPanes",roi
        )  # histogram + Gaussian fitting panes

        print(f"[INFO] Histogram panes: {len(histogram_panes)}")
        panes.append(histogram_panes)

    print("\n=== PREPARED PANE SETS ===")

    for i,pane_set in enumerate(panes):
        print(f"Set {i}: {len(pane_set)} files")  # number of outputs per pipeline
        if len(pane_set)>0:
            print(f"    Example: {pane_set[0]}")  # preview output file

    print("==========================\n")

    return panes