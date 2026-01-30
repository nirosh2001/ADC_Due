import argparse
import struct
import time
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

MAGIC = b"ADC\n"
HDR_LEN = 10  # 'A''D''C''\n' + uint32 seq + uint16 count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=5, help="COMx or /dev/ttyACM0")
    ap.add_argument("--vref", type=float, default=5, help="VREF in volts")
    ap.add_argument("--timeout", type=float, default=0.1)
    ap.add_argument("--decim", type=int, default=6, help="plot every Nth sample")
    ap.add_argument("--buffer_points", type=int, default=50000, help="max decimated points stored")
    ap.add_argument("--init_xwindow", type=int, default=4000, help="initial visible X window (points)")
    ap.add_argument("--min_xwindow", type=int, default=200, help="minimum visible X window")
    ap.add_argument("--init_yrange", type=float, default=1.0, help="initial Y range (+/- volts)")
    ap.add_argument("--send_start", action="store_true", help="send 'S' to start sampling (if Due supports it)")
    ap.add_argument("--sample_rate", type=float, default=16000, help="ADC sample rate in Hz (for FFT freq axis)")
    ap.add_argument("--fft_points", type=int, default=2048, help="FFT window size (power of 2 recommended)")
    ap.add_argument("--max_freq", type=float, default=100, help="Initial max frequency to display in FFT (Hz)")
    args = ap.parse_args()

    ser = serial.Serial(args.port, baudrate=115200, timeout=args.timeout)
    if args.send_start:
        time.sleep(0.2)
        ser.write(b"S")

    rx = bytearray()

    # ADS8867: signed int16 two's complement -> volts (diff)
    lsb = args.vref / 32768.0

    # store DECIMATED samples
    x = deque(maxlen=args.buffer_points)
    ch1v = deque(maxlen=args.buffer_points)
    ch2v = deque(maxlen=args.buffer_points)
    k = 0

    # Effective sample rate after decimation (for FFT frequency axis)
    fs_eff = args.sample_rate / max(1, args.decim)

    plt.ion()
    fig, ((ax1, ax_fft1), (ax2, ax_fft2)) = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(bottom=0.24, hspace=0.35, wspace=0.25)  # space for sliders

    # Time domain plots (left column)
    (line1,) = ax1.plot([], [])
    ax1.set_title("CH1 Time Domain")
    ax1.set_ylabel("Volts")
    ax1.set_xlabel("Sample index (decimated)")
    ax1.grid(True)

    (line2,) = ax2.plot([], [])
    ax2.set_title("CH2 Time Domain")
    ax2.set_xlabel("Sample index (decimated)")
    ax2.set_ylabel("Volts")
    ax2.grid(True)

    # FFT plots (right column)
    (line_fft1,) = ax_fft1.plot([], [])
    ax_fft1.set_title("CH1 FFT Magnitude")
    ax_fft1.set_ylabel("Magnitude (dB)")
    ax_fft1.set_xlabel("Frequency (Hz)")
    ax_fft1.grid(True)

    (line_fft2,) = ax_fft2.plot([], [])
    ax_fft2.set_title("CH2 FFT Magnitude")
    ax_fft2.set_xlabel("Frequency (Hz)")
    ax_fft2.set_ylabel("Magnitude (dB)")
    ax_fft2.grid(True)

    # ----- Sliders -----
    # X window slider
    ax_x = fig.add_axes([0.12, 0.17, 0.76, 0.03])
    init_x = max(args.min_xwindow, min(args.init_xwindow, args.buffer_points))
    sx = Slider(
        ax=ax_x,
        label="X Window",
        valmin=args.min_xwindow,
        valmax=args.buffer_points,
        valinit=init_x,
        valstep=max(1, args.min_xwindow // 10),
    )

    # Y range slider (symmetric +/- range)
    ax_y = fig.add_axes([0.12, 0.12, 0.76, 0.03])
    init_y = max(0.001, min(float(args.init_yrange), float(args.vref)))
    sy = Slider(
        ax=ax_y,
        label="Y Range (±V)",
        valmin=0.001,
        valmax=float(args.vref),
        valinit=init_y,
        valstep=float(args.vref) / 500.0,  # smooth enough
    )

    # Max frequency slider for FFT zoom
    ax_fmax = fig.add_axes([0.12, 0.07, 0.76, 0.03])
    max_nyquist = fs_eff / 2.0
    init_fmax = min(args.max_freq, max_nyquist)
    s_fmax = Slider(
        ax=ax_fmax,
        label="Max Freq (Hz)",
        valmin=1.0,
        valmax=max_nyquist,
        valinit=init_fmax,
        valstep=1.0,
    )

    # FFT bin size slider (power of 2 values)
    ax_fft_n = fig.add_axes([0.12, 0.02, 0.50, 0.03])
    fft_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    init_fft_idx = fft_sizes.index(args.fft_points) if args.fft_points in fft_sizes else 5
    s_fft_n = Slider(
        ax=ax_fft_n,
        label="FFT Size",
        valmin=0,
        valmax=len(fft_sizes) - 1,
        valinit=init_fft_idx,
        valstep=1,
    )

    # FFT info text (observation time and frequency resolution)
    init_fft_size = fft_sizes[init_fft_idx]
    init_obs_time = init_fft_size / fs_eff
    init_freq_res = fs_eff / init_fft_size
    fft_info_text = fig.text(
        0.65, 0.035,
        f"T_obs: {init_obs_time*1000:.1f} ms  |  Δf: {init_freq_res:.2f} Hz",
        fontsize=9, ha='left', va='center'
    )

    running = True
    def on_close(_evt):
        nonlocal running
        running = False
    fig.canvas.mpl_connect("close_event", on_close)

    def redraw():
        if len(x) < 2:
            return

        # window size
        win = int(sx.val)
        win = max(args.min_xwindow, min(win, len(x)))

        xs = list(x)[-win:]
        y1 = list(ch1v)[-win:]
        y2 = list(ch2v)[-win:]

        # Time domain plots
        line1.set_data(xs, y1)
        line2.set_data(xs, y2)

        # x limits to last window
        ax1.set_xlim(xs[0], xs[-1])
        ax2.set_xlim(xs[0], xs[-1])

        # y limits from slider
        yr = float(sy.val)
        ax1.set_ylim(-yr, yr)
        ax2.set_ylim(-yr, yr)

        # FFT computation (uses full buffer, independent of X window)
        fft_n = min(fft_sizes[int(s_fft_n.val)], len(ch1v))
        if fft_n >= 16:
            # Update FFT info text
            obs_time = fft_n / fs_eff
            freq_res = fs_eff / fft_n
            fft_info_text.set_text(f"T_obs: {obs_time*1000:.1f} ms  |  Δf: {freq_res:.2f} Hz")

            # Use last fft_n samples from full buffer for FFT
            y1_fft = np.array(list(ch1v)[-fft_n:])
            y2_fft = np.array(list(ch2v)[-fft_n:])

            # Apply Hanning window to reduce spectral leakage
            window = np.hanning(fft_n)
            y1_windowed = y1_fft * window
            y2_windowed = y2_fft * window

            # Compute FFT (only positive frequencies)
            fft1 = np.fft.rfft(y1_windowed)
            fft2 = np.fft.rfft(y2_windowed)

            # Magnitude in dB (with floor to avoid log(0))
            mag1 = np.abs(fft1)
            mag2 = np.abs(fft2)
            mag1_db = 20 * np.log10(np.maximum(mag1, 1e-10))
            mag2_db = 20 * np.log10(np.maximum(mag2, 1e-10))

            # Frequency axis
            freqs = np.fft.rfftfreq(fft_n, d=1.0 / fs_eff)

            # Update FFT plots
            line_fft1.set_data(freqs, mag1_db)
            line_fft2.set_data(freqs, mag2_db)

            # X limits from slider (zoom into low frequencies)
            fmax_display = float(s_fmax.val)
            ax_fft1.set_xlim(0, fmax_display)
            ax_fft2.set_xlim(0, fmax_display)
            
            # Y limits for FFT (auto with some padding, only for visible range)
            visible_mask = freqs <= fmax_display
            if np.any(visible_mask):
                vis_mag1 = mag1_db[visible_mask]
                vis_mag2 = mag2_db[visible_mask]
                if len(vis_mag1) > 0:
                    ax_fft1.set_ylim(max(-120, np.min(vis_mag1) - 10), np.max(vis_mag1) + 10)
                if len(vis_mag2) > 0:
                    ax_fft2.set_ylim(max(-120, np.min(vis_mag2) - 10), np.max(vis_mag2) + 10)

        fig.canvas.draw_idle()

    def on_slider(_val):
        redraw()

    sx.on_changed(on_slider)
    sy.on_changed(on_slider)
    s_fmax.on_changed(on_slider)
    s_fft_n.on_changed(on_slider)

    last_redraw = time.time()

    while running:
        plt.pause(0.001)

        # Read ALL available data to prevent OS serial buffer buildup
        waiting = ser.in_waiting
        if waiting > 0:
            chunk = ser.read(waiting)
            rx.extend(chunk)

        # Aggressive buffer trim to stay real-time (skip old data)
        if len(rx) > 30000:
            rx[:] = rx[-10000:]

        while True:
            m = rx.find(MAGIC)
            if m < 0:
                break

            if m > 0:
                del rx[:m]

            if len(rx) < HDR_LEN:
                break

            count = struct.unpack_from("<H", rx, 8)[0]
            payload_len = count * 4

            if len(rx) < HDR_LEN + payload_len:
                break

            payload = bytes(rx[HDR_LEN:HDR_LEN + payload_len])
            del rx[:HDR_LEN + payload_len]

            u16 = np.frombuffer(payload, dtype="<u2")
            i16 = u16.view("<i2")
            pairs = i16.reshape(-1, 2)

            a1 = pairs[:, 0].astype(np.float32) * lsb
            a2 = pairs[:, 1].astype(np.float32) * lsb

            d = max(1, args.decim)
            a1 = a1[::d]
            a2 = a2[::d]

            for v1, v2 in zip(a1, a2):
                x.append(k)
                ch1v.append(float(v1))
                ch2v.append(float(v2))
                k += 1

        now = time.time()
        if now - last_redraw > 0.05:
            redraw()
            last_redraw = now

    ser.close()

if __name__ == "__main__":
    main()
