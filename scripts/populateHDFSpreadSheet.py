import os
import csv
import re
import h5py
import numpy as np
from pathlib import Path

# ==========================================================
# USER CONFIG
# ==========================================================

BASE_DIR = "/SNS/users/lfigari/data/SNS/VENUS/IPTS-36967/nexus/"

ROOT_DIR = Path(
    "/SNS/VENUS/IPTS-36967/shared/autoreduce/images/tpx1/raw/radiography/"
)

script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(script_dir, "runSummary_MayALL.csv")

# ==========================================================
# HDF FILE PATH
# ==========================================================

def get_HDF_for_single_run(run_num):
    return os.path.join(BASE_DIR, f"VENUS_{run_num}.nxs.h5")

# ==========================================================
# CLEAN ISO TIME
# ==========================================================

def clean_iso(ts):
    if "." not in ts:
        return ts

    main, rest = ts.split(".", 1)

    if "+" in rest:
        frac, tz = rest.split("+", 1)
        tz = "+" + tz
    elif "-" in rest:
        frac, tz = rest.split("-", 1)
        tz = "-" + tz
    else:
        frac, tz = rest, ""

    frac = frac[:6]
    return f"{main}.{frac}{tz}"

# ==========================================================
# FAST RUN SUMMARY (single HDF open)
# ==========================================================

def summarize_run(run_num, iso_thresh=0.2):

    hdf5_input = get_HDF_for_single_run(run_num)

    with h5py.File(hdf5_input, "r") as f:

        # -------------------------
        # Sample name
        # -------------------------
        sample_name = str(
            f["entry/DASlogs/BL10:Exp:IM:ImageFilePath/value"][()]
        ).split("/")[-1]

        # -------------------------
        # Temperature arrays
        # -------------------------
        T_L = f["entry/DASlogs/BL10:SE:ND2:CH1:PV/value"][()]
        T_R = f["entry/DASlogs/BL10:SE:ND2:CH2:PV/value"][()]

        startT = (float(T_L[0]) + float(T_R[0])) / 2
        endT   = (float(T_L[-1]) + float(T_R[-1])) / 2

        avgT_L = float(f["entry/DASlogs/BL10:SE:ND2:CH1:PV/average_value"][()])
        avgT_R = float(f["entry/DASlogs/BL10:SE:ND2:CH2:PV/average_value"][()])
        avgT = (avgT_L + avgT_R) / 2

        # -------------------------
        # Rate
        # -------------------------
        rate_L = float(f["entry/DASlogs/BL10:SE:ND2:Loop1:SP:RateSet/average_value"][()])
        rate_R = float(f["entry/DASlogs/BL10:SE:ND2:Loop2:SP:RateSet/average_value"][()])
        rate = (rate_L + rate_R) / 2

        # -------------------------
        # Duration
        # -------------------------
        duration = float(f["entry/duration"][()])

        # -------------------------
        # Times
        # -------------------------
        start_time = clean_iso(str(f["entry/start_time"][()].decode() if isinstance(f["entry/start_time"][()], bytes) else f["entry/start_time"][()]))
        end_time   = clean_iso(str(f["entry/end_time"][()].decode() if isinstance(f["entry/end_time"][()], bytes) else f["entry/end_time"][()]))

    # -------------------------
    # Direction
    # -------------------------
    deltaT = endT - startT

    if abs(deltaT) < iso_thresh:
        direction = "Holding"
    elif deltaT > 0:
        direction = "Heating"
    else:
        direction = "Cooling"

    return [
        run_num,
        sample_name,
        avgT,
        direction,
        rate,
        start_time,
        end_time,
        duration,
        startT,
        endT
    ]

# ==========================================================
# GET RUN NUMBERS (FAST NON-RECURSIVE SCAN)
# ==========================================================


def get_run_numbers():

    pattern = re.compile(r"Run_(\d+)")

    run_numbers = []

    for outer in os.scandir(ROOT_DIR):

        if not outer.is_dir():
            continue

        run_number = None

        # ONLY look at immediate children of this folder
        for child in os.scandir(outer.path):

            if not child.is_dir():
                continue

            match = pattern.search(child.name)

            if match:
                run_number = int(match.group(1))
                run_numbers.append(run_number)

        print(f"{outer.name}: {run_number}")

    run_numbers = sorted(set(run_numbers))

    print(f"\nTotal runs found: {len(run_numbers)}")

    return run_numbers

# ==========================================================
# MAIN
# ==========================================================

def main():

    Run_nums = get_run_numbers()

    header = [
        "Index",
        "RunNum",
        "SampleName",
        "AvgT",
        "Direction",
        "Rate",
        "StartTime",
        "EndTime",
        "Duration",
        "StartT",
        "EndT"
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(header)

        for idx, run_num in enumerate(Run_nums):

            try:
                row = summarize_run(run_num)
                writer.writerow([idx] + row)

                print(f"[{idx+1}/{len(Run_nums)}] Run {run_num}")

            except Exception as e:
                print(f"FAILED run {run_num}: {e}")

    print("\nWrote:", OUTPUT_CSV)

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()
