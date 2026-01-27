# plot_ads8867_dual.py
# Real-time plot of two ADS8867 ADC streams from your Arduino Due binary frames.
#
# Stream format:
#   Header: b'ADC\n' + uint32 seq_end + uint16 count
#   Payload: count * (uint16 adc1, uint16 adc2) little-endian
#
# Run (no args):
#   python plot_ads8867_dual.py
#
# Install once:
#   pip install pyserial numpy matplotlib

import struct
import sys
from collections import deque

import numpy as np
import serial
from serial.tools import list_ports
import matplotlib.pyplot as plt


FORCE_PORT = None  # e.g. "COM10" (Windows) or "/dev/ttyACM0" (Linux). Leave None to auto-pick.

MAGIC = b"ADC\n"
HDR_LEN = 10  # 4 + 4 + 2

# Plot settings (edit here if you want, no terminal args needed)
DISPLAY_POINTS = 4000      # points shown on screen (per channel)
DECIMATE = 10              # plot every Nth sample (keeps matplotlib responsive)
UPDATE_EVERY_FRAMES = 2    # redraw after this many frames


def find_best_port():
    ports = list(list_ports.comports())
    if not ports:
        return None

    def score(p):
        txt = f"{p.device} {p.description} {p.manufacturer} {p.hwid}".lower()
        s = 0
        if "arduino" in txt: s += 50
        if "due" in txt: s += 40
        if "bossa" in txt: s += 30
        if "cdc" in txt or "usb serial" in txt: s += 10
        if "vid:pid=2341" in txt or "vid:pid=2a03" in txt: s += 25
        if "vid:pid=03eb" in txt: s += 10
        return s

    return max(ports, key=score).device


def parse_frames(buf: bytearray):
    """Return list of (seq_end, count, payload_bytes). Leaves remainder in buf."""
    frames = []
    while True:
        if len(buf) < HDR_LEN:
            break

        idx = buf.find(MAGIC)
        if idx == -1:
            if len(buf) > 3:
                del buf[:-3]
            break

        if idx > 0:
            del buf[:idx]

        if len(buf) < HDR_LEN:
            break

        seq_end = struct.unpack_from("<I", buf, 4)[0]
        count = struct.unpack_from("<H", buf, 8)[0]
        frame_len = HDR_LEN + count * 4

        if len(buf) < frame_len:
            break

        payload = bytes(buf[HDR_LEN:frame_len])
        frames.append((seq_end, count, payload))
        del buf[:frame_len]

    return frames


def main():
    port = FORCE_PORT or find_best_port()
    if not port:
        print("No serial ports found. Plug in the Due (Native USB) and try again.")
        sys.exit(1)

    print(f"Opening port: {port}")
    print("Plotting ADC1 and ADC2 (Ctrl+C to stop)\n")

    ser = serial.Serial(port=port, baudrate=115200, timeout=0.1)

    # Deques for rolling plot
    y1 = deque(maxlen=DISPLAY_POINTS)
    y2 = deque(maxlen=DISPLAY_POINTS)
    x  = deque(maxlen=DISPLAY_POINTS)
    sample_index = 0

    # Matplotlib setup
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.canvas.manager.set_window_title("ADS8867 Dual ADC - Live Plot")

    line1, = ax1.plot([], [], linewidth=1)
    ax1.set_title("ADC1")
    ax1.set_ylabel("Code")
    ax1.grid(True)

    line2, = ax2.plot([], [], linewidth=1)
    ax2.set_title("ADC2")
    ax2.set_ylabel("Code")
    ax2.set_xlabel("Sample (decimated)")
    ax2.grid(True)

    buf = bytearray()
    frames_since_draw = 0

    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)

                for _, count, payload in parse_frames(buf):
                    # Convert payload -> numpy array of shape (count,2)
                    arr = np.frombuffer(payload, dtype="<u2").reshape(-1, 2)

                    # Decimate for plotting speed
                    if DECIMATE > 1:
                        arr = arr[::DECIMATE]

                    if arr.size == 0:
                        continue

                    n = arr.shape[0]
                    xs = range(sample_index, sample_index + n)
                    sample_index += n

                    x.extend(xs)
                    y1.extend(arr[:, 0].tolist())
                    y2.extend(arr[:, 1].tolist())

                    frames_since_draw += 1

            # Redraw periodically
            if frames_since_draw >= UPDATE_EVERY_FRAMES and len(x) > 10:
                frames_since_draw = 0

                xx = np.fromiter(x, dtype=np.int64)
                yy1 = np.fromiter(y1, dtype=np.float64)
                yy2 = np.fromiter(y2, dtype=np.float64)

                line1.set_data(xx, yy1)
                line2.set_data(xx, yy2)

                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()

                plt.pause(0.1)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
