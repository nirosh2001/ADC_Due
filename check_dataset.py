"""
Quick sanity check for saved SCD dataset files.
Usage: python check_dataset.py <folder_path>
Example: python check_dataset.py D:/FYP/Dataset/DroneX/QuadCopter/5.00m/20260225_143000
"""

import sys
import os
import numpy as np


def check_dataset(folder):
    print(f"Checking dataset in: {folder}\n")

    if not os.path.isdir(folder):
        print(f"ERROR: Folder does not exist: {folder}")
        return

    # --- 1. Check scd_data.npz ---
    scd_path = os.path.join(folder, "scd_data.npz")
    if os.path.exists(scd_path):
        size_mb = os.path.getsize(scd_path) / (1024 * 1024)
        try:
            data = np.load(scd_path)
            scd = data["scd"]
            print(f"[OK] scd_data.npz ({size_mb:.2f} MB)")
            print(f"     Shape : {scd.shape}")
            print(f"     Dtype : {scd.dtype}")
            print(f"     Range : [{scd.min():.2f}, {scd.max():.2f}] dB")
            print(f"     NaN?  : {np.isnan(scd).any()}")
            print(f"     Inf?  : {np.isinf(scd).any()}")
        except Exception as e:
            print(f"[FAIL] scd_data.npz - Error loading: {e}")
    else:
        print(f"[MISSING] scd_data.npz")

    print()

    # --- 2. Check raw_iq_data.npz ---
    raw_path = os.path.join(folder, "raw_iq_data.npz")
    if os.path.exists(raw_path):
        size_mb = os.path.getsize(raw_path) / (1024 * 1024)
        try:
            data = np.load(raw_path)
            i_ch = data["i_channel"]
            q_ch = data["q_channel"]
            print(f"[OK] raw_iq_data.npz ({size_mb:.2f} MB)")
            print(f"     I-channel: shape={i_ch.shape}, dtype={i_ch.dtype}")
            print(f"       Range : [{i_ch.min():.6f}, {i_ch.max():.6f}] V")
            print(f"       Mean  : {i_ch.mean():.6f} V")
            print(f"       Std   : {i_ch.std():.6f} V")
            print(f"     Q-channel: shape={q_ch.shape}, dtype={q_ch.dtype}")
            print(f"       Range : [{q_ch.min():.6f}, {q_ch.max():.6f}] V")
            print(f"       Mean  : {q_ch.mean():.6f} V")
            print(f"       Std   : {q_ch.std():.6f} V")
            print(f"     Samples : {len(i_ch)}")
            print(f"     NaN?    : I={np.isnan(i_ch).any()}, Q={np.isnan(q_ch).any()}")
        except Exception as e:
            print(f"[FAIL] raw_iq_data.npz - Error loading: {e}")
    else:
        print(f"[MISSING] raw_iq_data.npz")

    print()

    # --- 3. Check scd_pattern.png ---
    img_path = os.path.join(folder, "scd_pattern.png")
    if os.path.exists(img_path):
        size_kb = os.path.getsize(img_path) / 1024
        print(f"[OK] scd_pattern.png ({size_kb:.1f} KB)")
    else:
        print(f"[MISSING] scd_pattern.png")

    # --- 4. Check settings.txt ---
    settings_path = os.path.join(folder, "settings.txt")
    if os.path.exists(settings_path):
        size_kb = os.path.getsize(settings_path) / 1024
        with open(settings_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[OK] settings.txt ({size_kb:.1f} KB, {len(lines)} lines)")
        # Print key settings
        for line in lines:
            line = line.strip()
            if any(k in line for k in ["Sample Rate", "FFT Size (NFFT)", "Segments", "Overlap", "Alpha Step"]):
                print(f"     {line}")
    else:
        print(f"[MISSING] settings.txt")

    print()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find the most recent dataset folder
        base = os.path.join("D:", os.sep, "FYP", "Dataset")
        if os.path.isdir(base):
            # Walk to find the most recent session folder
            latest = None
            for root, dirs, files in os.walk(base):
                if "settings.txt" in files or "scd_data.npz" in files:
                    if latest is None or root > latest:
                        latest = root
            if latest:
                print(f"Auto-detected latest dataset: {latest}\n")
                check_dataset(latest)
            else:
                print(f"No datasets found in {base}")
                print(f"Usage: python {sys.argv[0]} <folder_path>")
        else:
            print(f"Usage: python {sys.argv[0]} <folder_path>")
    else:
        check_dataset(sys.argv[1])
