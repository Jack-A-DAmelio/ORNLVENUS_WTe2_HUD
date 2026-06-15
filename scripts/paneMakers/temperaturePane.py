import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ==========================================================
# RUN RANGE PARSING
# ==========================================================

def parse_run_range(filename):
    name = filename.replace(".tif", "")
    parts = name.split("_")

    try:
        start_run = int(parts[1])
        end_run = int(parts[2])
        return list(range(start_run, end_run + 1))
    except:
        return []

# ==========================================================
# BUILD RUN INDEX MAPPING (CORE FIX)
# ==========================================================

def build_run_index(df):
    run_list = sorted(df["RunNum"].unique())
    run_to_idx = {r: i for i, r in enumerate(run_list)}
    return run_list, run_to_idx

# ==========================================================
# TEMPERATURE CONTEXT
# ==========================================================

def get_temperature_context(run_range,
                            run_to_avgT,
                            run_to_idx):

    window_idx = []
    window_T = []

    for r in run_range:
        if r in run_to_avgT and r in run_to_idx:
            window_idx.append(run_to_idx[r])
            window_T.append(run_to_avgT[r])

    if len(window_idx) == 0:
        return None

    window_idx = np.array(window_idx)
    window_T = np.array(window_T)

    return {
        "window_idx": window_idx,
        "window_T": window_T
    }

# ==========================================================
# PANEL RENDERING
# ==========================================================

def make_temperature_panel(data,
                           full_T_series,
                           window_mask,
                           y_min=0,
                           y_max=1200):

    x = np.arange(len(full_T_series))
    y = np.array(full_T_series, dtype=float)

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

    # full temperature trace
    ax.plot(x, y, color="black", linewidth=1)

    # window highlight (orange bar)
    ax.fill_between(
        x,
        y_min,
        y_max,
        where=window_mask,
        color="orange",
        alpha=0.25
    )

    # overlay window points
    win_x = x[window_mask]
    win_y = y[window_mask]

    if len(win_x) > 0:
        ax.plot(win_x, win_y, color="orange", linewidth=2.5)

    avg_T = float(np.mean(win_y)) if len(win_y) else np.nan

    ax.text(
        0.02, 0.95,
        f"Avg T: {avg_T:.2f} °C",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7),
        fontsize=10,
        va="top"
    )

    # ======================================================
    # FIXED AXES (YOUR REQUIREMENT)
    # ======================================================

    ax.set_xlim(0, len(full_T_series))
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Run index (global sequence)")
    ax.set_ylabel("Temperature (°C)")

    fig.tight_layout()
    fig.canvas.draw()

    panel = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return panel

# ==========================================================
# MAIN PIPELINE
# ==========================================================
def prepare_temperaturePane(tif_folder, csv_file, output_folder, N=20):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # LOAD + CLEAN CSV
    # ==========================================================

    df = pd.read_csv(csv_file)

    df["RunNum"] = (
        df["RunNum"].astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .astype(int)
    )

    df = df.sort_values("RunNum").reset_index(drop=True)

    run_list = df["RunNum"].tolist()
    run_to_idx = {r: i for i, r in enumerate(run_list)}
    run_to_avgT = dict(zip(df["RunNum"], df["AvgT"]))

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

    def parse_run_range(filename):
        name = filename.replace(".tif", "")
        parts = name.split("_")
        try:
            start_run = int(parts[1])
            end_run = int(parts[2])
            return list(range(start_run, end_run + 1))
        except:
            return []

    def get_window_indices(run_range):
        idxs, temps = [], []
        for r in run_range:
            if r in run_to_idx and r in run_to_avgT:
                idxs.append(run_to_idx[r])
                temps.append(run_to_avgT[r])
        if not idxs:
            return None
        return np.array(idxs, dtype=int), np.array(temps, dtype=float)

    def make_panel(all_runs, full_T, window_idx_set, window_T):

        x = np.arange(len(all_runs))
        y = full_T

        valid = ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        mask = window_idx_set[valid]

        wx = x[mask]
        wy = y[mask]

        avg_T = np.mean(window_T) if len(window_T) else np.nan

        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

        ax.plot(x, y, color="gray", alpha=0.6, linewidth=1)
        ax.plot(wx, wy, color="orange", linewidth=2.5)

        if len(wx) > 0:
            ax.axvspan(wx.min(), wx.max(), color="orange", alpha=0.15)

        ax.text(
            0.02, 0.95,
            f"Avg T: {avg_T:.2f} °C",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
            fontsize=10,
            va="top"
        )

        ax.set_xlim(0, len(all_runs) - 1)
        ax.set_ylim(0, 1200)

        ax.set_xlabel("Run index (TIFF subset)")
        ax.set_ylabel("Temperature (°C)")

        fig.tight_layout()
        fig.canvas.draw()

        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        return img

    # ==========================================================
    # TIFF FILES → DEFINE GLOBAL X SPACE
    # ==========================================================

    tiff_files = sorted(Path(tif_folder).glob("*.tif"))
    print(f"[INFO] {len(tiff_files)} TIFF files found")

    all_runs = []
    parsed = []

    for tif in tiff_files:
        rrange = parse_run_range(tif.name)
        valid = [r for r in rrange if r in run_to_idx]
        if not valid:
            continue

        mid = valid[len(valid)//2]
        all_runs.append(mid)
        parsed.append((tif, valid, mid))

    all_runs = sorted(set(all_runs))
    run_to_local = {r:i for i,r in enumerate(all_runs)}

    full_T = np.array([run_to_avgT.get(r, np.nan) for r in all_runs])

    # ==========================================================
    # BUILD PANELS
    # ==========================================================

    panels = []
    records = []

    for i, (tif, run_range, mid_run) in enumerate(parsed):

        out = get_window_indices(run_range)
        if out is None:
            continue

        window_idx, window_T = out

        window_mask = np.zeros(len(all_runs), dtype=bool)

        for r in run_range:
            if r in run_to_local:
                window_mask[run_to_local[r]] = True

        panel = make_panel(
            all_runs,
            full_T,
            window_mask,
            window_T
        )

        out_path = output_folder / f"{tif.stem}.png"
        Image.fromarray(panel.astype(np.uint8)).save(out_path)

        panels.append(out_path)

        records.append({
            "index": i,
            "filename": tif.name,
            "run": mid_run,
            "temperature": run_to_avgT[mid_run]
        })

    # ==========================================================
    # SUMMARY
    # ==========================================================

    if records:
        temps = np.array([r["temperature"] for r in records], dtype=float)

        plt.figure()
        plt.plot(temps, color="black")
        plt.ylabel("Temperature (°C)")
        plt.xlabel("TIFF index")
        plt.tight_layout()
        plt.savefig(output_folder / "Temperature_FullRange.png", dpi=200)
        plt.close()

        pd.DataFrame(records).to_csv(
            output_folder / "TIFF_Temperature_Log.csv",
            index=False
        )

    return panels

# ==========================================================
# SUMMARY
# ==========================================================

def save_summary(records, output_folder):

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if len(records) == 0:
        print("[WARN] No records generated")
        return

    temps = np.array([r["temperature"] for r in records], dtype=float)

    plt.figure()
    plt.plot(temps, color="black")
    plt.ylabel("Temperature (°C)")
    plt.xlabel("Image index")
    plt.tight_layout()
    plt.savefig(output_folder / "Temperature_FullRange.png", dpi=200)
    plt.close()

    pd.DataFrame(records).to_csv(
        output_folder / "TIFF_Temperature_Log.csv",
        index=False
    )

# ==========================================================
# WRAPPER (PIPELINE SAFE)
# ==========================================================

def run_pipeline(tif_folder, csv_file, output_folder, N=10):
    return prepare_temperaturePane(
        tif_folder,
        csv_file,
        output_folder,
        N=N
    )