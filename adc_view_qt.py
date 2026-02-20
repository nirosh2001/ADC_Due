import sys
import os
import time
import struct
import argparse
import warnings
from datetime import datetime
import numpy as np
import serial
from scipy.signal import butter, sosfilt
from gpu_compute import GPUCompute

# Suppress overflow warnings from pyqtgraph internals
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyqtgraph')

import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSlider, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QGridLayout, QTabWidget, QComboBox, QProgressBar, QCheckBox, QLineEdit,
    QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import pyqtgraph as pg

MAGIC = b"ADC\n"
HDR_LEN = 10


class SerialReaderThread(threading.Thread):
    """Dedicated thread that continuously reads and parses serial data.
    
    Writes decoded samples directly into the shared ring buffer so the
    main/GUI thread is never blocked waiting for serial I/O.  The ring
    buffer arrays (ch1v, ch2v) are written with simple index stores and
    numpy slice-assigns which are safe for a single-writer / single-reader
    pattern without locks.
    """

    def __init__(self, ser, viewer):
        super().__init__(daemon=True)
        self.ser = ser
        self.v = viewer          # reference to ADCViewer for shared state
        self.rx = bytearray()
        self._stop_event = threading.Event()
        # Lock only for ser.write from main thread while we read
        self.ser_lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def run(self):
        MAGIC_LOCAL = MAGIC
        HDR_LOCAL = HDR_LEN
        v = self.v
        ser = self.ser

        while not self._stop_event.is_set():
            try:
                # Block up to 5 ms for data — keeps CPU usage low while
                # still draining USB faster than the main timer could.
                with self.ser_lock:
                    waiting = ser.in_waiting
                    if waiting > 0:
                        chunk = ser.read(waiting)
                    else:
                        chunk = ser.read(1)  # blocks up to timeout
                if not chunk:
                    continue
                self.rx.extend(chunk)
            except Exception:
                if self._stop_event.is_set():
                    break
                continue

            # Trim buffer if too large
            if len(self.rx) > 30000:
                self.rx[:] = self.rx[-10000:]

            # Check for text messages (lines not starting with 'ADC\n')
            while b'\n' in self.rx:
                newline_pos = self.rx.index(b'\n')
                if newline_pos >= 3 and self.rx[newline_pos-3:newline_pos+1] == MAGIC_LOCAL:
                    break
                line_bytes = bytes(self.rx[:newline_pos])
                del self.rx[:newline_pos + 1]
                try:
                    line_text = line_bytes.decode('utf-8', errors='ignore').strip()
                    if line_text and not line_text.startswith('ADC'):
                        v._last_arduino_msg = line_text
                except Exception:
                    pass

            # Parse ADC packets
            while True:
                m = self.rx.find(MAGIC_LOCAL)
                if m < 0:
                    break
                if m > 0:
                    del self.rx[:m]
                if len(self.rx) < HDR_LOCAL:
                    break

                count = struct.unpack_from("<H", self.rx, 8)[0]
                payload_len = count * 4

                if len(self.rx) < HDR_LOCAL + payload_len:
                    break

                payload = bytes(self.rx[HDR_LOCAL:HDR_LOCAL + payload_len])
                del self.rx[:HDR_LOCAL + payload_len]

                u16 = np.frombuffer(payload, dtype="<u2")
                i16 = u16.view("<i2")
                pairs = i16.reshape(-1, 2)

                a1 = pairs[:, 0].astype(np.float32) * v.lsb
                a2 = pairs[:, 1].astype(np.float32) * v.lsb

                # Sample counting
                n_samples = len(pairs)
                v.sample_count_ch1 += n_samples
                v.sample_count_ch2 += n_samples
                v.total_samples_ch1 += n_samples
                v.total_samples_ch2 += n_samples

                # Decimate
                d = max(1, v.decim)
                a1 = a1[::d]
                a2 = a2[::d]

                # Write to ring buffer — bulk copy for speed
                n = len(a1)
                idx = v.write_idx
                bs = v.buffer_size
                end = idx + n
                if end <= bs:
                    v.ch1v[idx:end] = a1
                    v.ch2v[idx:end] = a2
                else:
                    first = bs - idx
                    v.ch1v[idx:bs] = a1[:first]
                    v.ch2v[idx:bs] = a2[:first]
                    rest = n - first
                    v.ch1v[0:rest] = a1[first:]
                    v.ch2v[0:rest] = a2[first:]
                v.write_idx = (idx + n) % bs
                v.data_count = min(v.data_count + n, bs)
                v.k += n

    def safe_write(self, data):
        """Thread-safe serial write (called from main thread)."""
        with self.ser_lock:
            self.ser.write(data)


class ADCViewer(QMainWindow):
    def __init__(self, port="COM5", sample_rate=30000, vref=5.0, decim=1, send_start=False):
        super().__init__()
        self.setWindowTitle("ADC Viewer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Serial setup
        self.ser = serial.Serial(port, baudrate=115200, timeout=0.005)
        self._last_arduino_msg = None   # set by reader thread
        
        if send_start:
            time.sleep(0.2)
            self.ser.write(b"S")
        
        # Parameters
        self.sample_rate = sample_rate
        self.vref = vref
        self.decim = decim
        self.lsb = vref / 32768.0
        self.fs_eff = sample_rate / max(1, decim)
        
        # Data buffers - use numpy ring buffer for performance
        self.buffer_size = 500000
        self.ch1v = np.zeros(self.buffer_size, dtype=np.float32)
        self.ch2v = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0  # Current write position
        self.data_count = 0  # Total samples in buffer (up to buffer_size)
        self.k = 0  # Total samples received
        
        # Display downsampling - max points to actually plot
        self.max_display_points = 2000
        
        # Sample counting - per ADC
        self.sample_count_ch1 = 0  # Samples this interval for Q channel (ADC1)
        self.sample_count_ch2 = 0  # Samples this interval for I channel (ADC2)
        self.total_samples_ch1 = 0  # Total samples for Q channel (ADC1)
        self.total_samples_ch2 = 0  # Total samples for I channel (ADC2)
        self.sample_count_start = time.time()
        self.measured_rate_ch1 = 0.0  # Samples per second for Q channel
        self.measured_rate_ch2 = 0.0  # Samples per second for I channel
        
        # FFT update throttle
        self.last_fft_update = 0
        self.fft_update_interval = 0.1  # Update FFT every 100ms
        
        # DC offset optimization state
        self.dc_optimize_running = False
        self.dc_optimize_iteration = 0
        self.dc_optimize_max_iterations = 20
        self.dc_optimize_target_mv = 0.5  # Target: mean < 0.5mV
        self.dc_optimize_timer = None
        
        # Brent optimization state
        self.brent_state = None  # Will hold optimization state machine
        self.brent_discard_count = 0
        self.brent_samples_to_discard = 2000
        self.brent_samples_for_mean = 4000
        
        # HD2 optimization state (Q-Channel)
        self.hd2_optimize_running = False
        self.hd2_optimize_timer = None
        self.hd2_state = None
        self.hd2_target_freq = 2000  # Target frequency in Hz
        self.hd2_freq_tolerance = 100  # Search window ±100 Hz
        self.hd2_samples_to_discard = 3000
        self.hd2_samples_for_fft = 8192
        
        # HD2 I-Channel optimization state
        self.hd2i_optimize_running = False
        self.hd2i_optimize_timer = None
        self.hd2i_state = None
        
        # IQ Calibration (Image Rejection) optimization state
        self.iq_cal_optimize_running = False
        self.iq_cal_optimize_timer = None
        self.iq_cal_state = None
        self.iq_cal_target_freq = -1000  # Target frequency in Hz (negative for image)
        self.iq_cal_freq_tolerance = 100  # Search window ±100 Hz
        self.iq_cal_samples_to_discard = 3000
        self.iq_cal_samples_for_fft = 8192
        
        # DSP Calibration state (software gain/phase adjustment)
        self.dsp_gain = 1.0  # Software gain multiplier for Q channel
        self.dsp_phase = 0.0  # Software phase shift in degrees for Q channel
        self.dsp_cal_optimize_running = False
        self.dsp_cal_optimize_timer = None
        self.dsp_cal_state = None
        self.dsp_cal_target_freq = -2500  # Target frequency in Hz
        self.dsp_cal_freq_tolerance = 100  # Search window ±100 Hz
        self.dsp_cal_samples_to_discard = 2000
        self.dsp_cal_samples_for_fft = 8192
        
        # DC filter state (high-pass filter 0-10Hz)
        self.dc_filter_enabled = False
        self.dc_filter_cutoff = 10.0  # Hz
        self.dc_filter_order = 2  # Butterworth order
        self.dc_filter_sos = None  # Filter coefficients (SOS format)
        self._design_dc_filter()
        
        # 2500Hz bandpass filter state
        self.bp_2500hz_filter_enabled = False
        self.bp_2500hz_center = 2500.0  # Hz center frequency
        self.bp_2500hz_bandwidth = 200.0  # Hz bandwidth (±100Hz)
        self.bp_2500hz_order = 4  # Butterworth order
        self.bp_2500hz_sos = None  # Filter coefficients (SOS format)
        self._design_bp_2500hz_filter()
        
        # Frequency shift to baseband state
        self.freq_shift_enabled = False
        self.freq_shift_hz = 1600.0  # Default shift frequency in Hz
        
        # Baseband DC-blocking filter state (applied after shift)
        self.bb_filter_enabled = False
        self.bb_filter_cutoff = 10.0  # Hz highpass cutoff (DC block)
        self.bb_filter_order = 4  # Butterworth order (4th order)
        self.bb_filter_sos = None  # Filter coefficients (SOS format)
        self._design_bb_filter()
        
        # Complex notch filter state (for removing interference at specific freq)
        self.notch_filter_enabled = False
        self.notch_filter_freq = -3200.0  # Hz center frequency for notch
        self.notch_filter_bw = 100.0  # Hz bandwidth of notch
        self.notch_filter_order = 4  # Butterworth order (4th order)
        self.notch_filter_sos = None  # Filter coefficients (SOS format)
        self._design_notch_filter()
        
        # SCD (Spectral Correlation Density) state
        self.scd_enabled = False
        self.scd_nfft = 1024  # FFT size for SCD
        self.scd_noverlap = 768  # Overlap for SCD segments (75%)
        self.scd_overlap_pct = 75  # Overlap percentage
        self.scd_num_segments = 128  # Number of segments to average
        self.scd_alpha_step = 1  # Alpha bin step: 1=every bin, 2=even only
        self.scd_last_update = 0
        self.scd_update_interval = 0.5  # Update SCD every 500ms (slower than FFT)
        self.scd_data = None  # Cached SCD result

        # GPU acceleration
        self.gpu = GPUCompute()
        self.use_gpu = self.gpu.available  # auto-enable if GPU found
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # ===== LEFT: Plots =====
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=False)  # Disable for performance
        pg.setConfigOptions(useOpenGL=True)   # Use GPU acceleration
        
        # Time domain plots
        self.plot_ch1 = pg.PlotWidget(title="Q Channel (DCOQ)")
        self.plot_ch1.setLabel('left', 'Volts')
        self.plot_ch1.showGrid(x=True, y=True)
        self.plot_ch1.disableAutoRange()  # Manual range for performance
        self.curve_ch1 = self.plot_ch1.plot(pen='y')
        self.mean_line_ch1 = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        self.plot_ch1.addItem(self.mean_line_ch1)
        self.mean_text_ch1 = pg.TextItem(anchor=(0, 1), color='r')
        self.plot_ch1.addItem(self.mean_text_ch1)
        
        self.plot_ch2 = pg.PlotWidget(title="I Channel (DCOI)")
        self.plot_ch2.setLabel('left', 'Volts')
        self.plot_ch2.showGrid(x=True, y=True)
        self.plot_ch2.disableAutoRange()
        self.curve_ch2 = self.plot_ch2.plot(pen='c')
        self.mean_line_ch2 = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        self.plot_ch2.addItem(self.mean_line_ch2)
        self.mean_text_ch2 = pg.TextItem(anchor=(0, 1), color='r')
        self.plot_ch2.addItem(self.mean_text_ch2)
        
        # DSP Gain/Phase Difference display (shown in Q channel plot)
        self.dsp_info_text = pg.TextItem(anchor=(1, 0), color='#00FF00')
        self.dsp_info_text.setPos(0, 0)
        self.plot_ch1.addItem(self.dsp_info_text)
        
        # FFT plots
        self.plot_fft1 = pg.PlotWidget(title="Q Channel FFT")
        self.plot_fft1.setLabel('left', 'Magnitude (dB)')
        self.plot_fft1.setLabel('bottom', 'Frequency (Hz)')
        self.plot_fft1.showGrid(x=True, y=True)
        self.plot_fft1.disableAutoRange()
        self.curve_fft1 = self.plot_fft1.plot(pen='y')
        # Peak markers for FFT1
        self.peaks_fft1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(255, 0, 0, 120), symbol='o')
        self.plot_fft1.addItem(self.peaks_fft1)
        self.peak_texts_fft1 = [pg.TextItem(color='r', anchor=(0.5, 1.2)) for _ in range(4)]
        for txt in self.peak_texts_fft1:
            self.plot_fft1.addItem(txt)
        
        self.plot_fft2 = pg.PlotWidget(title="I Channel FFT")
        self.plot_fft2.setLabel('left', 'Magnitude (dB)')
        self.plot_fft2.setLabel('bottom', 'Frequency (Hz)')
        self.plot_fft2.showGrid(x=True, y=True)
        self.plot_fft2.disableAutoRange()
        self.curve_fft2 = self.plot_fft2.plot(pen='c')
        # Peak markers for FFT2
        self.peaks_fft2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(255, 0, 0, 120), symbol='o')
        self.plot_fft2.addItem(self.peaks_fft2)
        self.peak_texts_fft2 = [pg.TextItem(color='r', anchor=(0.5, 1.2)) for _ in range(4)]
        for txt in self.peak_texts_fft2:
            self.plot_fft2.addItem(txt)
        
        # Complex FFT plot (I + jQ)
        self.plot_fft_complex = pg.PlotWidget(title="Complex FFT (I + jQ)")
        self.plot_fft_complex.setLabel('left', 'Magnitude (dB)')
        self.plot_fft_complex.setLabel('bottom', 'Frequency (Hz)')
        self.plot_fft_complex.showGrid(x=True, y=True)
        self.plot_fft_complex.disableAutoRange()
        self.curve_fft_complex = self.plot_fft_complex.plot(pen=pg.mkPen('m', width=1))
        # Add zero-frequency line
        self.zero_freq_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.plot_fft_complex.addItem(self.zero_freq_line)
        # Peak markers for complex FFT
        self.peaks_fft_complex = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(255, 0, 0, 120), symbol='o')
        self.plot_fft_complex.addItem(self.peaks_fft_complex)
        self.peak_texts_fft_complex = [pg.TextItem(color='r', anchor=(0.5, 1.2)) for _ in range(4)]
        for txt in self.peak_texts_fft_complex:
            self.plot_fft_complex.addItem(txt)
        
        # Create FFT tab widget
        self.fft_tabs = QTabWidget()
        
        # Tab 1: I/Q separate FFTs
        iq_fft_widget = QWidget()
        iq_fft_layout = QVBoxLayout(iq_fft_widget)
        iq_fft_layout.setContentsMargins(0, 0, 0, 0)
        iq_fft_layout.addWidget(self.plot_fft1)
        iq_fft_layout.addWidget(self.plot_fft2)
        self.fft_tabs.addTab(iq_fft_widget, "I/Q FFT")
        
        # Tab 2: Complex FFT
        complex_fft_widget = QWidget()
        complex_fft_layout = QVBoxLayout(complex_fft_widget)
        complex_fft_layout.setContentsMargins(0, 0, 0, 0)
        complex_fft_layout.addWidget(self.plot_fft_complex)
        self.fft_tabs.addTab(complex_fft_widget, "Complex FFT")
        
        # Tab 3: Spectral Correlation Density (SCD)
        scd_widget = QWidget()
        scd_layout = QVBoxLayout(scd_widget)
        scd_layout.setContentsMargins(0, 0, 0, 0)
        
        # SCD controls
        scd_controls = QHBoxLayout()
        
        self.chk_scd_enable = QCheckBox("Enable SCD")
        self.chk_scd_enable.setChecked(False)
        self.chk_scd_enable.stateChanged.connect(self.on_scd_enable_toggled)
        scd_controls.addWidget(self.chk_scd_enable)
        
        scd_controls.addWidget(QLabel("FFT:"))
        self.combo_scd_nfft = QComboBox()
        self.combo_scd_nfft.addItems(["64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768"])
        self.combo_scd_nfft.setCurrentText("1024")
        self.combo_scd_nfft.currentTextChanged.connect(self.on_scd_nfft_changed)
        scd_controls.addWidget(self.combo_scd_nfft)
        
        scd_controls.addWidget(QLabel("Segs:"))
        self.spin_scd_segments = QSpinBox()
        self.spin_scd_segments.setRange(8, 1024)
        self.spin_scd_segments.setValue(128)
        self.spin_scd_segments.valueChanged.connect(self.on_scd_segments_changed)
        scd_controls.addWidget(self.spin_scd_segments)

        scd_controls.addWidget(QLabel("Overlap:"))
        self.combo_scd_overlap = QComboBox()
        self.combo_scd_overlap.addItems(["50%", "75%", "87.5%"])
        self.combo_scd_overlap.setCurrentText("75%")
        self.combo_scd_overlap.setToolTip("Segment overlap — higher = more segments from same data, smoother result")
        self.combo_scd_overlap.currentTextChanged.connect(self.on_scd_overlap_changed)
        scd_controls.addWidget(self.combo_scd_overlap)

        scd_controls.addWidget(QLabel("α step:"))
        self.combo_scd_alpha_step = QComboBox()
        self.combo_scd_alpha_step.addItems(["1 (fine)", "2 (fast)"])
        self.combo_scd_alpha_step.setCurrentIndex(0)
        self.combo_scd_alpha_step.setToolTip("Alpha bin step — 1 = full resolution (all bins), 2 = every other bin (2× faster)")
        self.combo_scd_alpha_step.currentIndexChanged.connect(self.on_scd_alpha_step_changed)
        scd_controls.addWidget(self.combo_scd_alpha_step)
        
        self.btn_scd_update = QPushButton("Update Now")
        self.btn_scd_update.setStyleSheet("background-color: #673AB7; color: white;")
        self.btn_scd_update.clicked.connect(self.update_scd_plot)
        scd_controls.addWidget(self.btn_scd_update)
        
        scd_controls.addStretch()
        scd_layout.addLayout(scd_controls)
        
        # SCD frequency range controls (second row)
        scd_controls2 = QHBoxLayout()
        
        scd_controls2.addWidget(QLabel("Max Freq:"))
        self.spin_scd_maxfreq = QSpinBox()
        self.spin_scd_maxfreq.setRange(10, 15000)
        self.spin_scd_maxfreq.setValue(500)
        self.spin_scd_maxfreq.setSuffix(" Hz")
        self.spin_scd_maxfreq.setSingleStep(50)
        scd_controls2.addWidget(self.spin_scd_maxfreq)
        
        scd_controls2.addWidget(QLabel("Max Alpha:"))
        self.spin_scd_maxalpha = QSpinBox()
        self.spin_scd_maxalpha.setRange(10, 15000)
        self.spin_scd_maxalpha.setValue(200)
        self.spin_scd_maxalpha.setSuffix(" Hz")
        self.spin_scd_maxalpha.setSingleStep(50)
        scd_controls2.addWidget(self.spin_scd_maxalpha)
        
        scd_controls2.addWidget(QLabel("Min dB:"))
        self.spin_scd_min_db = QSpinBox()
        self.spin_scd_min_db.setRange(-200, 0)
        self.spin_scd_min_db.setValue(-60)
        self.spin_scd_min_db.setSuffix(" dB")
        self.spin_scd_min_db.setSingleStep(5)
        self.spin_scd_min_db.setToolTip("Noise floor clamp — values below this are clipped for better contrast")
        scd_controls2.addWidget(self.spin_scd_min_db)

        self.chk_scd_use_shifted = QCheckBox("Use Shifted Signal")
        self.chk_scd_use_shifted.setChecked(True)
        self.chk_scd_use_shifted.setToolTip("Apply freq shift and baseband filter to SCD")
        scd_controls2.addWidget(self.chk_scd_use_shifted)

        self.chk_scd_gpu = QCheckBox("GPU")
        self.chk_scd_gpu.setChecked(self.use_gpu)
        self.chk_scd_gpu.setEnabled(self.gpu.available)
        self.chk_scd_gpu.setToolTip(self.gpu.status_string())
        self.chk_scd_gpu.stateChanged.connect(lambda s: setattr(self, 'use_gpu', bool(s)))
        scd_controls2.addWidget(self.chk_scd_gpu)

        self.lbl_gpu_status = QLabel(self.gpu.status_string())
        self.lbl_gpu_status.setStyleSheet("font-size: 9px; color: " + ("#4CAF50;" if self.gpu.available else "gray;"))
        scd_controls2.addWidget(self.lbl_gpu_status)

        scd_controls2.addStretch()
        scd_layout.addLayout(scd_controls2)

        # SCD Dataset Save controls (third row)
        scd_dataset_controls = QHBoxLayout()

        scd_dataset_controls.addWidget(QLabel("Drone Name:"))
        self.edit_drone_name = QLineEdit()
        self.edit_drone_name.setPlaceholderText("e.g. DJI Mavic 3")
        self.edit_drone_name.setMinimumWidth(100)
        scd_dataset_controls.addWidget(self.edit_drone_name)

        scd_dataset_controls.addWidget(QLabel("Drone Type:"))
        self.edit_drone_type = QLineEdit()
        self.edit_drone_type.setPlaceholderText("e.g. Quadcopter")
        self.edit_drone_type.setMinimumWidth(100)
        scd_dataset_controls.addWidget(self.edit_drone_type)

        scd_dataset_controls.addWidget(QLabel("Distance (m):"))
        self.spin_distance = QDoubleSpinBox()
        self.spin_distance.setRange(0.0, 99999.0)
        self.spin_distance.setValue(10.0)
        self.spin_distance.setSingleStep(0.5)
        self.spin_distance.setDecimals(2)
        self.spin_distance.setSuffix(" m")
        scd_dataset_controls.addWidget(self.spin_distance)

        self.btn_save_scd = QPushButton("Save SCD Dataset")
        self.btn_save_scd.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 4px 12px;")
        self.btn_save_scd.clicked.connect(self.save_scd_dataset)
        scd_dataset_controls.addWidget(self.btn_save_scd)

        scd_dataset_controls.addStretch()
        scd_layout.addLayout(scd_dataset_controls)

        # SCD 2D plot using ImageView
        self.scd_plot = pg.PlotWidget(title="Spectral Correlation Density")
        self.scd_plot.setLabel('left', 'Cyclic Frequency α (Hz)')
        self.scd_plot.setLabel('bottom', 'Frequency f (Hz)')
        self.scd_plot.showGrid(x=True, y=True)
        
        # Create ImageItem for 2D SCD display
        self.scd_image = pg.ImageItem()
        self.scd_plot.addItem(self.scd_image)
        
        # Add colorbar
        self.scd_colorbar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get('turbo'))
        self.scd_colorbar.setImageItem(self.scd_image)
        
        scd_layout.addWidget(self.scd_plot)
        
        # SCD info label
        self.scd_info_label = QLabel("SCD: Disabled | Click 'Enable SCD' to compute")
        self.scd_info_label.setStyleSheet("font-size: 10px; color: gray; padding: 2px;")
        scd_layout.addWidget(self.scd_info_label)
        
        self.fft_tabs.addTab(scd_widget, "SCD")
        
        # Arrange plots: time domain on left, FFT tabs on right
        plot_grid = QGridLayout()
        plot_grid.addWidget(self.plot_ch1, 0, 0)
        plot_grid.addWidget(self.plot_ch2, 1, 0)
        plot_grid.addWidget(self.fft_tabs, 0, 1, 2, 1)
        plot_layout.addLayout(plot_grid)
        
        # Status bar
        self.status_label = QLabel("Samples: 0  |  Rate: -- sps")
        self.status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px;")
        plot_layout.addWidget(self.status_label)
        
        main_layout.addWidget(plot_widget, stretch=3)
        
        # ===== RIGHT: Controls =====
        control_panel = QWidget()
        control_panel_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(320)
        
        # Create tab widget for controls
        self.control_tabs = QTabWidget()
        control_panel_layout.addWidget(self.control_tabs)
        
        # ===== Main Tab =====
        main_tab = QWidget()
        control_layout = QVBoxLayout(main_tab)
        
        # --- ADC Control ---
        adc_group = QGroupBox("ADC Control")
        adc_layout = QVBoxLayout(adc_group)
        
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        self.btn_start.clicked.connect(lambda: self._reader.safe_write(b"S"))
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: lightcoral; font-weight: bold;")
        self.btn_stop.clicked.connect(lambda: self._reader.safe_write(b"P"))
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        adc_layout.addLayout(btn_layout)
        control_layout.addWidget(adc_group)
        
        # --- Display Settings ---
        display_group = QGroupBox("Display Settings")
        display_layout = QGridLayout(display_group)
        
        display_layout.addWidget(QLabel("X Window:"), 0, 0)
        self.spin_xwindow = QSpinBox()
        self.spin_xwindow.setRange(100, 50000)
        self.spin_xwindow.setValue(4000)
        self.spin_xwindow.setSingleStep(100)
        display_layout.addWidget(self.spin_xwindow, 0, 1)
        
        display_layout.addWidget(QLabel("Y Range (±V):"), 1, 0)
        self.spin_yrange = QDoubleSpinBox()
        self.spin_yrange.setRange(0.001, self.vref)
        self.spin_yrange.setValue(1.0)
        self.spin_yrange.setSingleStep(0.1)
        self.spin_yrange.setDecimals(3)
        display_layout.addWidget(self.spin_yrange, 1, 1)
        
        display_layout.addWidget(QLabel("Max Freq (Hz):"), 2, 0)
        self.spin_maxfreq = QSpinBox()
        self.spin_maxfreq.setRange(1, int(self.fs_eff / 2))
        self.spin_maxfreq.setValue(8000)
        display_layout.addWidget(self.spin_maxfreq, 2, 1)
        
        display_layout.addWidget(QLabel("FFT Size:"), 3, 0)
        self.combo_fft = QComboBox()
        self.combo_fft.addItems(["64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384"])
        self.combo_fft.setCurrentText("2048")
        display_layout.addWidget(self.combo_fft, 3, 1)
        
        # FFT info label
        self.fft_info_label = QLabel("T_obs: -- ms  |  Δf: -- Hz")
        self.fft_info_label.setStyleSheet("font-size: 10px; color: gray;")
        display_layout.addWidget(self.fft_info_label, 4, 0, 1, 2)
        
        # DC Filter checkbox
        self.chk_dc_filter = QCheckBox("DC Filter (0-10Hz HP)")
        self.chk_dc_filter.setChecked(False)
        self.chk_dc_filter.stateChanged.connect(self.on_dc_filter_toggled)
        display_layout.addWidget(self.chk_dc_filter, 5, 0, 1, 2)
        
        # 2500Hz Bandpass Filter checkbox
        self.chk_bp_2500hz = QCheckBox("2500Hz Tone Filter (BP)")
        self.chk_bp_2500hz.setChecked(False)
        self.chk_bp_2500hz.stateChanged.connect(self.on_bp_2500hz_toggled)
        display_layout.addWidget(self.chk_bp_2500hz, 6, 0, 1, 2)
        
        # Frequency shift controls
        self.chk_freq_shift = QCheckBox("Freq Shift to Baseband")
        self.chk_freq_shift.setChecked(False)
        self.chk_freq_shift.stateChanged.connect(self.on_freq_shift_toggled)
        display_layout.addWidget(self.chk_freq_shift, 7, 0)
        
        self.spin_freq_shift = QSpinBox()
        self.spin_freq_shift.setRange(-15000, 15000)
        self.spin_freq_shift.setValue(1600)
        self.spin_freq_shift.setSuffix(" Hz")
        self.spin_freq_shift.valueChanged.connect(self.on_freq_shift_value_changed)
        display_layout.addWidget(self.spin_freq_shift, 7, 1)
        
        # Baseband DC-blocking filter controls
        self.chk_bb_filter = QCheckBox("Baseband DC Block")
        self.chk_bb_filter.setChecked(False)
        self.chk_bb_filter.stateChanged.connect(self.on_bb_filter_toggled)
        display_layout.addWidget(self.chk_bb_filter, 8, 0)
        
        self.spin_bb_cutoff = QSpinBox()
        self.spin_bb_cutoff.setRange(1, 100)
        self.spin_bb_cutoff.setValue(10)
        self.spin_bb_cutoff.setSuffix(" Hz")
        self.spin_bb_cutoff.valueChanged.connect(self.on_bb_cutoff_changed)
        display_layout.addWidget(self.spin_bb_cutoff, 8, 1)
        
        # Baseband notch filter controls (applied on baseband after shift)
        self.chk_notch_filter = QCheckBox("BB Notch Filter")
        self.chk_notch_filter.setChecked(False)
        self.chk_notch_filter.stateChanged.connect(self.on_notch_filter_toggled)
        display_layout.addWidget(self.chk_notch_filter, 9, 0)
        
        self.spin_notch_freq = QSpinBox()
        self.spin_notch_freq.setRange(-15000, 15000)
        self.spin_notch_freq.setValue(-3200)
        self.spin_notch_freq.setSuffix(" Hz")
        self.spin_notch_freq.valueChanged.connect(self.on_notch_freq_changed)
        display_layout.addWidget(self.spin_notch_freq, 9, 1)
        
        display_layout.addWidget(QLabel("Notch BW:"), 10, 0)
        self.spin_notch_bw = QSpinBox()
        self.spin_notch_bw.setRange(10, 1000)
        self.spin_notch_bw.setValue(100)
        self.spin_notch_bw.setSuffix(" Hz")
        self.spin_notch_bw.valueChanged.connect(self.on_notch_bw_changed)
        display_layout.addWidget(self.spin_notch_bw, 10, 1)
        
        control_layout.addWidget(display_group)
        
        # --- DVGA Control ---
        dvga_group = QGroupBox("DVGA Control")
        dvga_layout = QGridLayout(dvga_group)
        
        dvga_layout.addWidget(QLabel("Gain (0-63):"), 0, 0)
        self.spin_dvga = QSpinBox()
        self.spin_dvga.setRange(0, 63)
        self.spin_dvga.setValue(32)
        dvga_layout.addWidget(self.spin_dvga, 0, 1)
        
        self.btn_dvga_send = QPushButton("Send")
        self.btn_dvga_send.setStyleSheet("background-color: lightyellow;")
        self.btn_dvga_send.clicked.connect(self.send_dvga)
        dvga_layout.addWidget(self.btn_dvga_send, 0, 2)
        
        self.btn_dvga_sweep = QPushButton("Sweep")
        self.btn_dvga_sweep.setStyleSheet("background-color: khaki;")
        self.btn_dvga_sweep.clicked.connect(lambda: self._reader.safe_write(b"G"))
        dvga_layout.addWidget(self.btn_dvga_sweep, 1, 0, 1, 3)
        
        control_layout.addWidget(dvga_group)
        
        # --- Attenuator Control ---
        att_group = QGroupBox("Attenuator Control")
        att_layout = QGridLayout(att_group)
        
        att_layout.addWidget(QLabel("ATT (0-127):"), 0, 0)
        self.spin_att = QSpinBox()
        self.spin_att.setRange(0, 127)
        self.spin_att.setValue(1)
        att_layout.addWidget(self.spin_att, 0, 1)
        
        self.btn_att_send = QPushButton("Send")
        self.btn_att_send.setStyleSheet("background-color: lightblue;")
        self.btn_att_send.clicked.connect(self.send_att)
        att_layout.addWidget(self.btn_att_send, 0, 2)
        
        control_layout.addWidget(att_group)
        
        # --- Arduino Messages ---
        msg_group = QGroupBox("Arduino Messages")
        msg_layout = QVBoxLayout(msg_group)
        self.msg_label = QLabel("(no messages)")
        self.msg_label.setWordWrap(True)
        self.msg_label.setStyleSheet("font-size: 10px; color: darkblue;")
        msg_layout.addWidget(self.msg_label)
        control_layout.addWidget(msg_group)
        
        control_layout.addStretch()  # Push everything up
        self.control_tabs.addTab(main_tab, "Main")
        
        # ===== Demodulator Tab =====
        demod_tab = QWidget()
        demod_tab_layout = QVBoxLayout(demod_tab)
        
        # Initialize demod_spins dictionary
        self.demod_spins = {}
        
        # --- DC Offset Adjustment Group ---
        dc_offset_group = QGroupBox("DC Offset Adjustment")
        dc_offset_layout = QGridLayout(dc_offset_group)
        
        # DCOI register
        dc_offset_layout.addWidget(QLabel("DCOI (I-Ch):"), 0, 0)
        self.spin_dcoi = QSpinBox()
        self.spin_dcoi.setRange(0, 255)
        self.spin_dcoi.setValue(128)
        self.spin_dcoi.setMinimumWidth(100)
        self.demod_spins["DCOI"] = self.spin_dcoi  # Add to dict immediately
        dc_offset_layout.addWidget(self.spin_dcoi, 0, 1)
        
        btn_dcoi_send = QPushButton("➤")
        btn_dcoi_send.setFixedWidth(28)
        btn_dcoi_send.setStyleSheet("background-color: #5a5a8a; color: white; font-weight: bold; font-size: 12px;")
        btn_dcoi_send.clicked.connect(lambda checked: self.send_demod_reg("DCOI", "0"))
        dc_offset_layout.addWidget(btn_dcoi_send, 0, 2)
        
        # DCOQ register
        dc_offset_layout.addWidget(QLabel("DCOQ (Q-Ch):"), 1, 0)
        self.spin_dcoq = QSpinBox()
        self.spin_dcoq.setRange(0, 255)
        self.spin_dcoq.setValue(128)
        self.spin_dcoq.setMinimumWidth(100)
        self.demod_spins["DCOQ"] = self.spin_dcoq  # Add to dict immediately
        dc_offset_layout.addWidget(self.spin_dcoq, 1, 1)
        
        btn_dcoq_send = QPushButton("➤")
        btn_dcoq_send.setFixedWidth(28)
        btn_dcoq_send.setStyleSheet("background-color: #5a5a8a; color: white; font-weight: bold; font-size: 12px;")
        btn_dcoq_send.clicked.connect(lambda checked: self.send_demod_reg("DCOQ", "1"))
        dc_offset_layout.addWidget(btn_dcoq_send, 1, 2)
        
        # DC offset info label
        self.dc_offset_info = QLabel("Range: ±75mV (~0.59mV/code)")
        self.dc_offset_info.setStyleSheet("font-size: 10px; color: gray;")
        dc_offset_layout.addWidget(self.dc_offset_info, 2, 0, 1, 3)
        
        # Optimize button
        self.btn_dc_optimize = QPushButton("⚡ Optimize DC Offset")
        self.btn_dc_optimize.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_dc_optimize.clicked.connect(self.toggle_dc_optimization)
        dc_offset_layout.addWidget(self.btn_dc_optimize, 3, 0, 1, 3)
        
        # Optimization status
        self.dc_optimize_status = QLabel("")
        self.dc_optimize_status.setStyleSheet("font-size: 10px; color: #333;")
        self.dc_optimize_status.setWordWrap(True)
        dc_offset_layout.addWidget(self.dc_optimize_status, 4, 0, 1, 3)
        
        demod_tab_layout.addWidget(dc_offset_group)
        
        # --- Demodulator Calibration Control ---
        demod_group = QGroupBox("Demodulator Calibration")
        demod_layout = QGridLayout(demod_group)
        
        # Define remaining demodulator registers (excluding DCOI and DCOQ)
        self.demod_regs = [
            ("HD2IX", "2", "HD2 I-Ch X", 255, 128),
            ("HD2IY", "3", "HD2 I-Ch Y", 255, 128),
            ("HD2QX", "4", "HD2 Q-Ch X", 255, 128),
            ("HD2QY", "5", "HD2 Q-Ch Y", 255, 128),
            ("HD3IX", "6", "HD3 I-Ch X", 255, 128),
            ("HD3IY", "7", "HD3 I-Ch Y", 255, 128),
            ("HD3QX", "8", "HD3 Q-Ch X", 255, 128),
            ("HD3QY", "9", "HD3 Q-Ch Y", 255, 128),
            ("GERR", "A", "IQ Gain Error", 63, 32),
            ("DEMOD_ATT", "B", "Step Attenuator", 31, 0),
            ("AMPG", "C", "IF Amp Gain (8-15dB)", 7, 6),
            ("PHA", "F", "IQ Phase Error (±2.5°)", 511, 256),  # 9-bit, default 0x100
        ]
        
        for row, (name, reg_id, desc, max_val, default_val) in enumerate(self.demod_regs):
            # Label
            demod_layout.addWidget(QLabel(f"{name}:"), row, 0)
            
            # Spinbox
            spin = QSpinBox()
            spin.setRange(0, max_val)
            spin.setValue(default_val)
            spin.setMinimumWidth(100)
            self.demod_spins[name] = spin
            demod_layout.addWidget(spin, row, 1)
            
            # Send button
            btn_send = QPushButton("➤")
            btn_send.setFixedWidth(28)
            btn_send.setStyleSheet("background-color: #5a5a8a; color: white; font-weight: bold; font-size: 12px;")
            btn_send.clicked.connect(lambda checked, n=name, r=reg_id: self.send_demod_reg(n, r))
            demod_layout.addWidget(btn_send, row, 2)
        
        demod_tab_layout.addWidget(demod_group)
        
        demod_tab_layout.addStretch()  # Push everything up
        self.control_tabs.addTab(demod_tab, "Demodulator")
        
        # ===== HD2 Calibration Tab =====
        hd2_cal_tab = QWidget()
        hd2_cal_tab_layout = QVBoxLayout(hd2_cal_tab)
        
        # --- Shared HD2 Settings ---
        hd2_settings_group = QGroupBox("HD2 Settings")
        hd2_settings_layout = QGridLayout(hd2_settings_group)
        
        # Target frequency (shared for both I and Q)
        hd2_settings_layout.addWidget(QLabel("Target Freq (Hz):"), 0, 0)
        self.spin_hd2_freq = QSpinBox()
        self.spin_hd2_freq.setRange(100, 8000)
        self.spin_hd2_freq.setValue(2000)
        self.spin_hd2_freq.setSingleStep(100)
        hd2_settings_layout.addWidget(self.spin_hd2_freq, 0, 1)
        
        # Frequency tolerance
        hd2_settings_layout.addWidget(QLabel("Search ± (Hz):"), 1, 0)
        self.spin_hd2_tol = QSpinBox()
        self.spin_hd2_tol.setRange(10, 500)
        self.spin_hd2_tol.setValue(100)
        hd2_settings_layout.addWidget(self.spin_hd2_tol, 1, 1)
        
        hd2_cal_tab_layout.addWidget(hd2_settings_group)
        
        # --- HD2 Q-Channel Optimization ---
        hd2q_group = QGroupBox("HD2 Q-Channel")
        hd2q_layout = QGridLayout(hd2q_group)
        
        # Current HD2QX/HD2QY values display
        hd2q_layout.addWidget(QLabel("HD2QX:"), 0, 0)
        self.label_hd2qx = QLabel("128")
        self.label_hd2qx.setStyleSheet("font-weight: bold;")
        hd2q_layout.addWidget(self.label_hd2qx, 0, 1)
        
        hd2q_layout.addWidget(QLabel("HD2QY:"), 1, 0)
        self.label_hd2qy = QLabel("128")
        self.label_hd2qy.setStyleSheet("font-weight: bold;")
        hd2q_layout.addWidget(self.label_hd2qy, 1, 1)
        
        # Optimize button
        self.btn_hd2_optimize = QPushButton("⚡ Optimize Q-Channel")
        self.btn_hd2_optimize.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px;")
        self.btn_hd2_optimize.clicked.connect(self.toggle_hd2_optimization)
        hd2q_layout.addWidget(self.btn_hd2_optimize, 2, 0, 1, 2)
        
        # Status label
        self.hd2_optimize_status = QLabel("")
        self.hd2_optimize_status.setStyleSheet("font-size: 10px; color: #333;")
        self.hd2_optimize_status.setWordWrap(True)
        hd2q_layout.addWidget(self.hd2_optimize_status, 3, 0, 1, 2)
        
        hd2_cal_tab_layout.addWidget(hd2q_group)
        
        # --- HD2 I-Channel Optimization ---
        hd2i_group = QGroupBox("HD2 I-Channel")
        hd2i_layout = QGridLayout(hd2i_group)
        
        # Current HD2IX/HD2IY values display
        hd2i_layout.addWidget(QLabel("HD2IX:"), 0, 0)
        self.label_hd2ix = QLabel("128")
        self.label_hd2ix.setStyleSheet("font-weight: bold;")
        hd2i_layout.addWidget(self.label_hd2ix, 0, 1)
        
        hd2i_layout.addWidget(QLabel("HD2IY:"), 1, 0)
        self.label_hd2iy = QLabel("128")
        self.label_hd2iy.setStyleSheet("font-weight: bold;")
        hd2i_layout.addWidget(self.label_hd2iy, 1, 1)
        
        # Optimize button
        self.btn_hd2i_optimize = QPushButton("⚡ Optimize I-Channel")
        self.btn_hd2i_optimize.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 6px;")
        self.btn_hd2i_optimize.clicked.connect(self.toggle_hd2i_optimization)
        hd2i_layout.addWidget(self.btn_hd2i_optimize, 2, 0, 1, 2)
        
        # Status label
        self.hd2i_optimize_status = QLabel("")
        self.hd2i_optimize_status.setStyleSheet("font-size: 10px; color: #333;")
        self.hd2i_optimize_status.setWordWrap(True)
        hd2i_layout.addWidget(self.hd2i_optimize_status, 3, 0, 1, 2)
        
        hd2_cal_tab_layout.addWidget(hd2i_group)
        
        hd2_cal_tab_layout.addStretch()
        self.control_tabs.addTab(hd2_cal_tab, "HD2 Cal")
        
        # ===== IQ Calibration Tab =====
        iq_cal_tab = QWidget()
        iq_cal_tab_layout = QVBoxLayout(iq_cal_tab)
        
        # --- Image Rejection Optimization ---
        iq_cal_group = QGroupBox("Image Rejection Optimization")
        iq_cal_layout = QGridLayout(iq_cal_group)
        
        # Target frequency (image frequency)
        iq_cal_layout.addWidget(QLabel("Image Freq (Hz):"), 0, 0)
        self.spin_iq_cal_freq = QSpinBox()
        self.spin_iq_cal_freq.setRange(-8000, 8000)
        self.spin_iq_cal_freq.setValue(-1000)
        self.spin_iq_cal_freq.setSingleStep(100)
        iq_cal_layout.addWidget(self.spin_iq_cal_freq, 0, 1)
        
        # Frequency tolerance
        iq_cal_layout.addWidget(QLabel("Search ± (Hz):"), 1, 0)
        self.spin_iq_cal_tol = QSpinBox()
        self.spin_iq_cal_tol.setRange(10, 500)
        self.spin_iq_cal_tol.setValue(100)
        iq_cal_layout.addWidget(self.spin_iq_cal_tol, 1, 1)
        
        # Info label
        iq_cal_info = QLabel("Optimizes PHA (phase) and GERR (gain)\nto minimize image at specified frequency")
        iq_cal_info.setStyleSheet("font-size: 10px; color: gray;")
        iq_cal_layout.addWidget(iq_cal_info, 2, 0, 1, 2)
        
        iq_cal_tab_layout.addWidget(iq_cal_group)
        
        # --- Current Values Group ---
        iq_values_group = QGroupBox("Current Values")
        iq_values_layout = QGridLayout(iq_values_group)
        
        # PHA value display and manual control
        iq_values_layout.addWidget(QLabel("PHA:"), 0, 0)
        self.spin_pha_manual = QSpinBox()
        self.spin_pha_manual.setRange(0, 511)
        self.spin_pha_manual.setValue(256)
        iq_values_layout.addWidget(self.spin_pha_manual, 0, 1)
        
        btn_pha_send = QPushButton("➤")
        btn_pha_send.setFixedWidth(28)
        btn_pha_send.setStyleSheet("background-color: #5a5a8a; color: white; font-weight: bold;")
        btn_pha_send.clicked.connect(lambda: self.send_iq_cal_reg("PHA", "F", self.spin_pha_manual.value()))
        iq_values_layout.addWidget(btn_pha_send, 0, 2)
        
        # GERR value display and manual control
        iq_values_layout.addWidget(QLabel("GERR:"), 1, 0)
        self.spin_gerr_manual = QSpinBox()
        self.spin_gerr_manual.setRange(0, 63)
        self.spin_gerr_manual.setValue(32)
        iq_values_layout.addWidget(self.spin_gerr_manual, 1, 1)
        
        btn_gerr_send = QPushButton("➤")
        btn_gerr_send.setFixedWidth(28)
        btn_gerr_send.setStyleSheet("background-color: #5a5a8a; color: white; font-weight: bold;")
        btn_gerr_send.clicked.connect(lambda: self.send_iq_cal_reg("GERR", "A", self.spin_gerr_manual.value()))
        iq_values_layout.addWidget(btn_gerr_send, 1, 2)
        
        # Phase info
        pha_info = QLabel("PHA: 0-511 (256=0°, ±2.5° range)")
        pha_info.setStyleSheet("font-size: 9px; color: gray;")
        iq_values_layout.addWidget(pha_info, 2, 0, 1, 3)
        
        # Gain info
        gerr_info = QLabel("GERR: 0-63 (32=0dB, ±0.5dB range)")
        gerr_info.setStyleSheet("font-size: 9px; color: gray;")
        iq_values_layout.addWidget(gerr_info, 3, 0, 1, 3)
        
        iq_cal_tab_layout.addWidget(iq_values_group)
        
        # --- Optimization Control ---
        iq_opt_group = QGroupBox("Optimization")
        iq_opt_layout = QVBoxLayout(iq_opt_group)
        
        # Optimize button
        self.btn_iq_cal_optimize = QPushButton("⚡ Optimize Image Rejection")
        self.btn_iq_cal_optimize.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;")
        self.btn_iq_cal_optimize.clicked.connect(self.toggle_iq_cal_optimization)
        iq_opt_layout.addWidget(self.btn_iq_cal_optimize)
        
        # Status label
        self.iq_cal_optimize_status = QLabel("")
        self.iq_cal_optimize_status.setStyleSheet("font-size: 10px; color: #333;")
        self.iq_cal_optimize_status.setWordWrap(True)
        iq_opt_layout.addWidget(self.iq_cal_optimize_status)
        
        # Current image level display
        self.iq_cal_level_label = QLabel("Image Level: -- dB")
        self.iq_cal_level_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #9C27B0;")
        iq_opt_layout.addWidget(self.iq_cal_level_label)
        
        iq_cal_tab_layout.addWidget(iq_opt_group)
        
        iq_cal_tab_layout.addStretch()
        self.control_tabs.addTab(iq_cal_tab, "IQ Cal")
        
        # ===== DSP Calibration Tab =====
        dsp_cal_tab = QWidget()
        dsp_cal_tab_layout = QVBoxLayout(dsp_cal_tab)
        
        # --- DSP Settings ---
        dsp_settings_group = QGroupBox("Target Frequency")
        dsp_settings_layout = QGridLayout(dsp_settings_group)
        
        dsp_settings_layout.addWidget(QLabel("Target Freq (Hz):"), 0, 0)
        self.spin_dsp_target_freq = QSpinBox()
        self.spin_dsp_target_freq.setRange(-8000, 8000)
        self.spin_dsp_target_freq.setValue(-2500)
        self.spin_dsp_target_freq.setSingleStep(100)
        dsp_settings_layout.addWidget(self.spin_dsp_target_freq, 0, 1)
        
        dsp_settings_layout.addWidget(QLabel("Search ± (Hz):"), 1, 0)
        self.spin_dsp_tol = QSpinBox()
        self.spin_dsp_tol.setRange(10, 500)
        self.spin_dsp_tol.setValue(100)
        dsp_settings_layout.addWidget(self.spin_dsp_tol, 1, 1)
        
        dsp_info = QLabel("Adjusts software DSP gain & phase\nto minimize target frequency component")
        dsp_info.setStyleSheet("font-size: 10px; color: gray;")
        dsp_settings_layout.addWidget(dsp_info, 2, 0, 1, 2)
        
        dsp_cal_tab_layout.addWidget(dsp_settings_group)
        
        # --- DSP Manual Controls ---
        dsp_manual_group = QGroupBox("DSP Parameters")
        dsp_manual_layout = QGridLayout(dsp_manual_group)
        
        # DSP Gain control
        dsp_manual_layout.addWidget(QLabel("DSP Gain:"), 0, 0)
        self.spin_dsp_gain = QDoubleSpinBox()
        self.spin_dsp_gain.setRange(0.5, 2.0)
        self.spin_dsp_gain.setValue(1.0)
        self.spin_dsp_gain.setSingleStep(0.01)
        self.spin_dsp_gain.setDecimals(4)
        self.spin_dsp_gain.valueChanged.connect(self.on_dsp_gain_changed)
        dsp_manual_layout.addWidget(self.spin_dsp_gain, 0, 1)
        
        dsp_gain_info = QLabel("Q-ch gain multiplier (1.0 = unity)")
        dsp_gain_info.setStyleSheet("font-size: 9px; color: gray;")
        dsp_manual_layout.addWidget(dsp_gain_info, 1, 0, 1, 2)
        
        # DSP Phase control
        dsp_manual_layout.addWidget(QLabel("DSP Phase (°):"), 2, 0)
        self.spin_dsp_phase = QDoubleSpinBox()
        self.spin_dsp_phase.setRange(-10.0, 10.0)
        self.spin_dsp_phase.setValue(0.0)
        self.spin_dsp_phase.setSingleStep(0.1)
        self.spin_dsp_phase.setDecimals(3)
        self.spin_dsp_phase.valueChanged.connect(self.on_dsp_phase_changed)
        dsp_manual_layout.addWidget(self.spin_dsp_phase, 2, 1)
        
        dsp_phase_info = QLabel("I/Q phase correction in degrees")
        dsp_phase_info.setStyleSheet("font-size: 9px; color: gray;")
        dsp_manual_layout.addWidget(dsp_phase_info, 3, 0, 1, 2)
        
        # Reset button
        self.btn_dsp_reset = QPushButton("Reset to Default")
        self.btn_dsp_reset.setStyleSheet("background-color: #607D8B; color: white;")
        self.btn_dsp_reset.clicked.connect(self.reset_dsp_params)
        dsp_manual_layout.addWidget(self.btn_dsp_reset, 4, 0, 1, 2)
        
        dsp_cal_tab_layout.addWidget(dsp_manual_group)
        
        # --- Measured IQ Parameters ---
        dsp_measured_group = QGroupBox("Measured I/Q Difference")
        dsp_measured_layout = QGridLayout(dsp_measured_group)
        
        dsp_measured_layout.addWidget(QLabel("Gain Ratio (I/Q):"), 0, 0)
        self.label_measured_gain = QLabel("--")
        self.label_measured_gain.setStyleSheet("font-weight: bold; color: #4CAF50;")
        dsp_measured_layout.addWidget(self.label_measured_gain, 0, 1)
        
        dsp_measured_layout.addWidget(QLabel("Phase Diff (°):"), 1, 0)
        self.label_measured_phase = QLabel("--")
        self.label_measured_phase.setStyleSheet("font-weight: bold; color: #4CAF50;")
        dsp_measured_layout.addWidget(self.label_measured_phase, 1, 1)
        
        measured_info = QLabel("Computed from I/Q signal correlation")
        measured_info.setStyleSheet("font-size: 9px; color: gray;")
        dsp_measured_layout.addWidget(measured_info, 2, 0, 1, 2)
        
        dsp_cal_tab_layout.addWidget(dsp_measured_group)
        
        # --- DSP Optimization ---
        dsp_opt_group = QGroupBox("Optimization")
        dsp_opt_layout = QVBoxLayout(dsp_opt_group)
        
        self.btn_dsp_cal_optimize = QPushButton("⚡ Optimize DSP for Target Freq")
        self.btn_dsp_cal_optimize.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.btn_dsp_cal_optimize.clicked.connect(self.toggle_dsp_cal_optimization)
        dsp_opt_layout.addWidget(self.btn_dsp_cal_optimize)
        
        self.dsp_cal_optimize_status = QLabel("")
        self.dsp_cal_optimize_status.setStyleSheet("font-size: 10px; color: #333;")
        self.dsp_cal_optimize_status.setWordWrap(True)
        dsp_opt_layout.addWidget(self.dsp_cal_optimize_status)
        
        self.dsp_cal_level_label = QLabel("Target Level: -- dB")
        self.dsp_cal_level_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #2196F3;")
        dsp_opt_layout.addWidget(self.dsp_cal_level_label)
        
        dsp_cal_tab_layout.addWidget(dsp_opt_group)
        
        dsp_cal_tab_layout.addStretch()
        self.control_tabs.addTab(dsp_cal_tab, "DSP Cal")
        
        main_layout.addWidget(control_panel, stretch=1)
    
    def send_dvga(self):
        val = self.spin_dvga.value()
        self._reader.safe_write(f"V{val}".encode())
        print(f"Sent DVGA: {val}")
    
    def send_att(self):
        val = self.spin_att.value()
        self._reader.safe_write(f"A{val}".encode())
        print(f"Sent ATT: {val}")
    
    def send_demod_reg(self, name, reg_id):
        val = self.demod_spins[name].value()
        cmd = f"D{reg_id}{val}"
        self._reader.safe_write(cmd.encode())
        print(f"Sent {name}: {val} (cmd={cmd})")
        self.msg_label.setText(f"Sent {name}={val}")
    
    def setup_timer(self):
        # Start serial reader thread — runs independently of GUI
        self._reader = SerialReaderThread(self.ser, self)
        self._reader.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)  # 50 Hz GUI update
    
    def update(self):
        # Pick up any text message from the reader thread
        msg = self._last_arduino_msg
        if msg is not None:
            self._last_arduino_msg = None
            print(f"[Arduino] {msg}")
            self.msg_label.setText(msg)

        self.update_plots()
        self.update_status()
    
    def update_plots(self):
        if self.data_count < 2:
            return
        
        # Snapshot ring buffer state (writer thread may update these at any time)
        widx = self.write_idx
        dcount = self.data_count
        
        win = min(self.spin_xwindow.value(), dcount)
        
        # Extract data from ring buffer (last 'win' samples)
        if win <= widx:
            y1 = self.ch1v[widx - win:widx].copy()
            y2 = self.ch2v[widx - win:widx].copy()
        else:
            # Wrap around
            part1_len = win - widx
            y1 = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:widx]])
            y2 = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:widx]])
        
        # Downsample for display if too many points
        if len(y1) > self.max_display_points:
            step = len(y1) // self.max_display_points
            y1_plot = y1[::step]
            y2_plot = y2[::step]
            xs = np.arange(len(y1_plot), dtype=np.float64) * step
        else:
            y1_plot = y1
            y2_plot = y2
            xs = np.arange(len(y1), dtype=np.float64)
        
        self.curve_ch1.setData(xs, y1_plot)
        self.curve_ch2.setData(xs, y2_plot)
        
        # Calculate and display mean (using up to 8000 samples)
        n_mean = min(8000, dcount)
        if n_mean > 0:
            if n_mean <= widx:
                mean_ch1 = np.mean(self.ch1v[widx - n_mean:widx])
                mean_ch2 = np.mean(self.ch2v[widx - n_mean:widx])
            else:
                part1_len = n_mean - widx
                mean_ch1 = np.mean(np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:widx]]))
                mean_ch2 = np.mean(np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:widx]]))
            
            # Update mean lines
            self.mean_line_ch1.setValue(mean_ch1)
            self.mean_line_ch2.setValue(mean_ch2)
            
            # Update mean text labels
            self.mean_text_ch1.setText(f"Mean: {mean_ch1*1000:.2f} mV")
            self.mean_text_ch1.setPos(10, mean_ch1)
            self.mean_text_ch2.setText(f"Mean: {mean_ch2*1000:.2f} mV")
            self.mean_text_ch2.setPos(10, mean_ch2)
        
        yr = self.spin_yrange.value()
        self.plot_ch1.setYRange(-yr, yr)
        self.plot_ch2.setYRange(-yr, yr)
        self.plot_ch1.setXRange(0, float(win))
        self.plot_ch2.setXRange(0, float(win))
        
        # FFT - throttled update
        now = time.time()
        if now - self.last_fft_update < self.fft_update_interval:
            return
        self.last_fft_update = now
        
        fft_n = min(int(self.combo_fft.currentText()), dcount)
        if fft_n >= 16:
            # Update FFT info
            obs_time = fft_n / self.fs_eff
            freq_res = self.fs_eff / fft_n
            self.fft_info_label.setText(f"T_obs: {obs_time*1000:.1f} ms  |  Δf: {freq_res:.2f} Hz")
            
            # Extract FFT data from ring buffer
            if fft_n <= widx:
                y1_fft = self.ch1v[widx - fft_n:widx].copy()
                y2_fft = self.ch2v[widx - fft_n:widx].copy()
            else:
                part1_len = fft_n - widx
                y1_fft = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:widx]])
                y2_fft = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:widx]])
            
            # Apply DC filter if enabled
            if self.dc_filter_enabled:
                y1_fft = self.apply_dc_filter(y1_fft)
                y2_fft = self.apply_dc_filter(y2_fft)
            
            # Apply 2500Hz bandpass filter if enabled
            if self.bp_2500hz_filter_enabled:
                y1_fft = self.apply_bp_2500hz_filter(y1_fft)
                y2_fft = self.apply_bp_2500hz_filter(y2_fft)
            
            window = np.hanning(fft_n).astype(np.float32)
            fft1 = np.fft.rfft(y1_fft * window)
            fft2 = np.fft.rfft(y2_fft * window)
            
            mag1_db = 20 * np.log10(np.maximum(np.abs(fft1), 1e-10))
            mag2_db = 20 * np.log10(np.maximum(np.abs(fft2), 1e-10))
            freqs = np.fft.rfftfreq(fft_n, d=1.0 / self.fs_eff)
            
            self.curve_fft1.setData(freqs, mag1_db)
            self.curve_fft2.setData(freqs, mag2_db)
            
            fmax = float(self.spin_maxfreq.value())
            self.plot_fft1.setXRange(0.0, fmax)
            self.plot_fft2.setXRange(0.0, fmax)
            
            # Auto Y range for FFT (visible range only) - fixed minimum at -80 dB
            visible_mask = freqs <= fmax
            if np.any(visible_mask):
                vis_mag1 = mag1_db[visible_mask]
                vis_mag2 = mag2_db[visible_mask]
                vis_freqs = freqs[visible_mask]
                if len(vis_mag1) > 0:
                    ymax1 = float(min(100.0, np.max(vis_mag1) + 10))
                    self.plot_fft1.setYRange(-80.0, ymax1)
                    # Find and mark top 4 peaks for FFT1
                    peak_indices1 = self.find_top_peaks(vis_mag1, n_peaks=4, min_distance=5)
                    peak_freqs1 = vis_freqs[peak_indices1]
                    peak_mags1 = vis_mag1[peak_indices1]
                    self.peaks_fft1.setData(peak_freqs1, peak_mags1)
                    for i, txt in enumerate(self.peak_texts_fft1):
                        if i < len(peak_freqs1):
                            txt.setText(f"{peak_freqs1[i]:.0f}Hz\n{peak_mags1[i]:.1f}dB")
                            txt.setPos(peak_freqs1[i], peak_mags1[i])
                            txt.setVisible(True)
                        else:
                            txt.setVisible(False)
                if len(vis_mag2) > 0:
                    ymax2 = float(min(100.0, np.max(vis_mag2) + 10))
                    self.plot_fft2.setYRange(-80.0, ymax2)
                    # Find and mark top 4 peaks for FFT2
                    peak_indices2 = self.find_top_peaks(vis_mag2, n_peaks=4, min_distance=5)
                    peak_freqs2 = vis_freqs[peak_indices2]
                    peak_mags2 = vis_mag2[peak_indices2]
                    self.peaks_fft2.setData(peak_freqs2, peak_mags2)
                    for i, txt in enumerate(self.peak_texts_fft2):
                        if i < len(peak_freqs2):
                            txt.setText(f"{peak_freqs2[i]:.0f}Hz\n{peak_mags2[i]:.1f}dB")
                            txt.setPos(peak_freqs2[i], peak_mags2[i])
                            txt.setVisible(True)
                        else:
                            txt.setVisible(False)
            
            # Complex FFT: I + jQ (ch2v is I, ch1v is Q)
            # Apply DSP correction before computing complex FFT
            i_for_fft = y2_fft.copy()
            q_for_fft = y1_fft.copy()
            
            # Apply DSP gain/phase correction if not unity
            if abs(self.dsp_gain - 1.0) > 1e-6 or abs(self.dsp_phase) > 1e-6:
                i_for_fft, q_for_fft = self.apply_dsp_correction(i_for_fft, q_for_fft)
            
            # Build complex signal
            complex_signal = i_for_fft + 1j * q_for_fft
            
            # Apply frequency shift to baseband if enabled
            if self.freq_shift_enabled:
                complex_signal = self.apply_freq_shift(complex_signal)
            
            # Apply baseband lowpass filter if enabled (after shift)
            if self.bb_filter_enabled and self.freq_shift_enabled:
                complex_signal = self.apply_bb_filter(complex_signal)
            
            # Apply complex notch filter if enabled
            if self.notch_filter_enabled:
                complex_signal = self.apply_notch_filter(complex_signal)
            
            # Full complex FFT shows both positive and negative frequencies
            complex_signal_windowed = complex_signal * window
            fft_complex = np.fft.fft(complex_signal_windowed)
            fft_complex_shifted = np.fft.fftshift(fft_complex)  # Shift zero-freq to center
            mag_complex_db = 20 * np.log10(np.maximum(np.abs(fft_complex_shifted), 1e-10))
            freqs_complex = np.fft.fftshift(np.fft.fftfreq(fft_n, d=1.0 / self.fs_eff))
            
            self.curve_fft_complex.setData(freqs_complex, mag_complex_db)
            
            # Compute and display I/Q gain ratio and phase difference
            gain_ratio, phase_diff = self.compute_iq_gain_phase_diff(y2_fft, y1_fft, fft_n)
            if gain_ratio is not None:
                self.label_measured_gain.setText(f"{gain_ratio:.4f}")
                self.label_measured_phase.setText(f"{phase_diff:.2f}°" if phase_diff is not None else "--")
                
                # Update the DSP info text on the time domain plot
                dsp_text = f"DSP: G={self.dsp_gain:.3f}, φ={self.dsp_phase:.2f}°\n"
                dsp_text += f"Meas: I/Q={gain_ratio:.3f}, Δφ={phase_diff:.1f}°" if phase_diff is not None else f"Meas: I/Q={gain_ratio:.3f}"
                self.dsp_info_text.setText(dsp_text)
                # Position in top-right corner of the plot
                yr = self.spin_yrange.value()
                self.dsp_info_text.setPos(float(win) - 10, yr * 0.9)
            
            # Set X range for complex FFT: -fmax to +fmax
            self.plot_fft_complex.setXRange(-fmax, fmax)
            
            # Auto Y range for complex FFT - fixed minimum at -80 dB
            visible_mask_complex = (freqs_complex >= -fmax) & (freqs_complex <= fmax)
            if np.any(visible_mask_complex):
                vis_mag_complex = mag_complex_db[visible_mask_complex]
                vis_freqs_complex = freqs_complex[visible_mask_complex]
                if len(vis_mag_complex) > 0:
                    ymax_c = float(min(100.0, np.max(vis_mag_complex) + 10))
                    self.plot_fft_complex.setYRange(-80.0, ymax_c)
                    # Find and mark top 4 peaks for complex FFT
                    peak_indices_c = self.find_top_peaks(vis_mag_complex, n_peaks=4, min_distance=5)
                    peak_freqs_c = vis_freqs_complex[peak_indices_c]
                    peak_mags_c = vis_mag_complex[peak_indices_c]
                    self.peaks_fft_complex.setData(peak_freqs_c, peak_mags_c)
                    for i, txt in enumerate(self.peak_texts_fft_complex):
                        if i < len(peak_freqs_c):
                            txt.setText(f"{peak_freqs_c[i]:.0f}Hz\n{peak_mags_c[i]:.1f}dB")
                            txt.setPos(peak_freqs_c[i], peak_mags_c[i])
                            txt.setVisible(True)
                        else:
                            txt.setVisible(False)
            
            # SCD update (if enabled and enough time has passed)
            if self.scd_enabled and (now - self.scd_last_update >= self.scd_update_interval):
                self.update_scd_plot()
    
    def find_top_peaks(self, mag_db, n_peaks=4, min_distance=5):
        """Find the top N peaks in the magnitude spectrum.
        
        Args:
            mag_db: Magnitude spectrum in dB
            n_peaks: Number of peaks to find
            min_distance: Minimum distance between peaks (in bins)
        
        Returns:
            Array of peak indices sorted by magnitude (highest first)
        """
        if len(mag_db) < 3:
            return np.array([], dtype=int)
        
        # Find local maxima: points higher than both neighbors
        peaks = []
        for i in range(1, len(mag_db) - 1):
            if mag_db[i] > mag_db[i-1] and mag_db[i] > mag_db[i+1]:
                peaks.append((i, mag_db[i]))
        
        if not peaks:
            # No local maxima found, return global max
            return np.array([np.argmax(mag_db)])
        
        # Sort by magnitude (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Select top peaks with minimum distance constraint
        selected = []
        for idx, mag in peaks:
            # Check if this peak is far enough from already selected peaks
            too_close = False
            for sel_idx in selected:
                if abs(idx - sel_idx) < min_distance:
                    too_close = True
                    break
            if not too_close:
                selected.append(idx)
                if len(selected) >= n_peaks:
                    break
        
        return np.array(selected, dtype=int)
    
    def toggle_dc_optimization(self):
        """Toggle DC offset optimization on/off."""
        if self.dc_optimize_running:
            self.stop_dc_optimization()
        else:
            self.start_dc_optimization()
    
    def start_dc_optimization(self):
        """Start the Brent's method coordinate descent DC offset optimization."""
        self.dc_optimize_running = True
        self.dc_optimize_iteration = 0
        
        # Update button appearance
        self.btn_dc_optimize.setText("⏹ Cancel Optimization")
        self.btn_dc_optimize.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        
        self.dc_optimize_status.setText("Starting Brent optimization...")
        
        # Initialize Brent optimization state machine
        # States: 'init', 'optimizing_dcoi', 'optimizing_dcoq', 'discarding', 'measuring', 'done'
        self.brent_state = {
            'phase': 'init',
            'current_channel': 'DCOI',  # Start with I-channel
            'iteration': 0,
            'max_coord_iters': 1,  # Single pass since I/Q channels are independent
            'current_coord_iter': 0,
            'samples_before_write': 0,
            'best_dcoi': self.spin_dcoi.value(),
            'best_dcoq': self.spin_dcoq.value(),
            'best_cost': float('inf'),
            # Brent's method state for each channel
            'brent_a': 0,
            'brent_b': 255,
            'brent_tol': 1,
            'brent_evaluations': [],
            'brent_step': 'bracket_init',
            'eval_queue': [],  # Queue of values to evaluate
            'eval_results': {},  # Results of evaluations
        }
        
        # Create timer for optimization loop
        self.dc_optimize_timer = QTimer()
        self.dc_optimize_timer.timeout.connect(self.brent_optimization_step)
        self.dc_optimize_timer.start(50)  # Run every 50ms
    
    def stop_dc_optimization(self):
        """Stop the DC offset optimization process."""
        self.dc_optimize_running = False
        
        if self.dc_optimize_timer:
            self.dc_optimize_timer.stop()
            self.dc_optimize_timer = None
        
        # Reset button appearance
        self.btn_dc_optimize.setText("⚡ Optimize DC Offset")
        self.btn_dc_optimize.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        
        if self.brent_state:
            self.dc_optimize_status.setText(
                f"Optimization stopped.\n"
                f"DCOI: {self.spin_dcoi.value()}, DCOQ: {self.spin_dcoq.value()}"
            )
        self.brent_state = None
    
    def get_channel_mean_mv(self, channel):
        """Get the mean value in mV for the specified channel.
        DCOI controls I channel (ch2v), DCOQ controls Q channel (ch1v).
        """
        n_samples = min(self.brent_samples_for_mean, self.data_count)
        if n_samples < 100:
            return None
        
        if n_samples <= self.write_idx:
            if channel == 'DCOI':
                data = self.ch2v[self.write_idx - n_samples:self.write_idx]  # I channel
            else:
                data = self.ch1v[self.write_idx - n_samples:self.write_idx]  # Q channel
        else:
            part1_len = n_samples - self.write_idx
            if channel == 'DCOI':
                data = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:self.write_idx]])  # I channel
            else:
                data = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:self.write_idx]])  # Q channel
        
        return np.mean(data) * 1000  # Return in mV
    
    def _finalize_channel(self, channel, best_val, best_cost):
        """Finalize optimization for a channel and set up transition to next."""
        state = self.brent_state
        if channel == 'DCOI':
            state['best_dcoi'] = best_val
            state['pending_next_channel'] = 'DCOQ'
            self.write_dco_register(channel, best_val)
            self.dc_optimize_status.setText(
                f"DCOI optimized to {best_val} (|mean|={best_cost:.2f}mV)\n"
                f"Settling before DCOQ optimization..."
            )
        else:
            state['best_dcoq'] = best_val
            self.spin_dcoq.setValue(best_val)
            self.send_demod_reg("DCOQ", "1")
            state['current_coord_iter'] += 1
            state['pending_check_convergence'] = True
            state['samples_before_write'] = self.k
            state['phase'] = 'discarding'
    
    def write_dco_register(self, channel, value):
        """Write a value to the DCO register and prepare for discard."""
        value = int(max(0, min(255, value)))
        if channel == 'DCOI':
            self.spin_dcoi.setValue(value)
            self.send_demod_reg("DCOI", "0")
        else:
            self.spin_dcoq.setValue(value)
            self.send_demod_reg("DCOQ", "1")
        
        # Record current sample count for discarding
        self.brent_state['samples_before_write'] = self.k
        self.brent_state['phase'] = 'discarding'
        print(f"[Optimize] write_dco_register: {channel}={value}, k={self.k}, phase=discarding, pending_next={self.brent_state.get('pending_next_channel')}")
    
    def brent_optimization_step(self):
        """State machine for Brent's method coordinate descent optimization."""
        if not self.dc_optimize_running or self.brent_state is None:
            return
        
        state = self.brent_state
        
        # Phase: discarding samples after register write
        if state['phase'] == 'discarding':
            samples_since_write = self.k - state['samples_before_write']
            if samples_since_write >= self.brent_samples_to_discard:
                # Check if we have a pending channel transition
                if state.get('pending_next_channel'):
                    next_channel = state.pop('pending_next_channel')
                    state['current_channel'] = next_channel
                    state['phase'] = 'start_channel'
                    print(f"[Optimize] Transitioning to {next_channel}")
                    self.dc_optimize_status.setText(f"Now optimizing {next_channel}...")
                    return
                
                # Check if we need to verify convergence after DCOQ
                if state.get('pending_check_convergence'):
                    state.pop('pending_check_convergence')
                    state['phase'] = 'check_convergence'
                    print("[Optimize] Entering check_convergence phase")
                    return
                
                state['phase'] = 'measuring'
                state['samples_before_write'] = self.k  # Reset for measurement
                print(f"[Optimize] Entering measuring phase for {state['current_channel']}")
            else:
                self.dc_optimize_status.setText(
                    f"Discarding samples: {samples_since_write}/{self.brent_samples_to_discard}\n"
                    f"Channel: {state['current_channel']}"
                )
            return
        
        # Phase: check convergence after both channels optimized
        if state['phase'] == 'check_convergence':
            # Wait for enough samples to measure
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.brent_samples_for_mean:
                self.dc_optimize_status.setText(
                    f"Measuring final offsets: {samples_since_discard}/{self.brent_samples_for_mean}"
                )
                return
            
            mean_i = self.get_channel_mean_mv('DCOI')  # I channel mean
            mean_q = self.get_channel_mean_mv('DCOQ')  # Q channel mean
            
            # Optimization complete - channels are independent, no need for multiple passes
            self.dc_optimize_status.setText(
                f"✓ Optimization complete!\n"
                f"I-Ch: {mean_i:.2f}mV, Q-Ch: {mean_q:.2f}mV\n"
                f"DCOI={state['best_dcoi']}, DCOQ={state['best_dcoq']}"
            )
            self.stop_dc_optimization()
            return
        
        # Phase: measuring mean after discard
        if state['phase'] == 'measuring':
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.brent_samples_for_mean:
                self.dc_optimize_status.setText(
                    f"Collecting samples: {samples_since_discard}/{self.brent_samples_for_mean}\n"
                    f"Channel: {state['current_channel']}"
                )
                return
            
            # We have enough samples, measure mean
            mean_mv = self.get_channel_mean_mv(state['current_channel'])
            if mean_mv is None:
                return
            
            # Store the result
            current_value = self.spin_dcoi.value() if state['current_channel'] == 'DCOI' else self.spin_dcoq.value()
            state['eval_results'][current_value] = abs(mean_mv)
            state['brent_evaluations'].append((current_value, abs(mean_mv)))
            
            # Continue with search
            state['phase'] = 'search_step'
        
        # Phase: initialization
        if state['phase'] == 'init':
            state['current_coord_iter'] = 0
            state['phase'] = 'start_channel'
        
        # Phase: start optimizing a channel
        if state['phase'] == 'start_channel':
            channel = state['current_channel']
            state['brent_evaluations'] = []
            state['eval_results'] = {}
            
            # Start with 5-point initial grid for coarse search
            state['initial_points'] = [0, 64, 128, 192, 255]
            state['initial_idx'] = 0
            state['search_phase'] = 'coarse'
            
            # Write first value
            first_val = state['initial_points'][0]
            self.write_dco_register(channel, first_val)
            self.dc_optimize_status.setText(
                f"Coord iter {state['current_coord_iter']+1}: Optimizing {channel}\n"
                f"Coarse search: {first_val}"
            )
            return
        
        # Phase: search stepping
        if state['phase'] == 'search_step':
            channel = state['current_channel']
            evals = state['brent_evaluations']
            
            if state['search_phase'] == 'coarse':
                # Still doing coarse search
                state['initial_idx'] += 1
                if state['initial_idx'] < len(state['initial_points']):
                    next_val = state['initial_points'][state['initial_idx']]
                    self.write_dco_register(channel, next_val)
                    self.dc_optimize_status.setText(
                        f"Coarse search {state['initial_idx']+1}/5: {channel}={next_val}"
                    )
                    return
                else:
                    # Coarse search done, find best and set up fine search
                    best_val, best_cost = min(evals, key=lambda x: x[1])
                    print(f"[Optimize] Coarse search done. Best: {best_val} with cost {best_cost:.2f}mV")
                    
                    # Check if already good enough
                    if best_cost < self.dc_optimize_target_mv:
                        print(f"[Optimize] {channel} already optimal at {best_val}")
                        self._finalize_channel(channel, best_val, best_cost)
                        return
                    
                    # Set up fine search around best value (±32 range, step ~8)
                    state['fine_center'] = best_val
                    state['fine_range'] = 32
                    state['search_phase'] = 'fine'
                    state['fine_step'] = 0
                    
                    # Evaluate center-16, center, center+16
                    left = max(0, best_val - 16)
                    right = min(255, best_val + 16)
                    state['fine_points'] = [left, best_val, right]
                    state['fine_points'] = sorted(set(state['fine_points']))  # Remove duplicates
                    
                    # Skip points we already evaluated
                    state['fine_points'] = [p for p in state['fine_points'] 
                                            if not any(abs(p - e[0]) < 2 for e in evals)]
                    
                    if state['fine_points']:
                        next_val = state['fine_points'].pop(0)
                        self.write_dco_register(channel, next_val)
                        self.dc_optimize_status.setText(f"Fine search: {channel}={next_val}")
                        return
                    else:
                        # No new points to evaluate, go to refine
                        state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'fine':
                # Continue fine search
                if state['fine_points']:
                    next_val = state['fine_points'].pop(0)
                    self.write_dco_register(channel, next_val)
                    self.dc_optimize_status.setText(f"Fine search: {channel}={next_val}")
                    return
                else:
                    state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'refine':
                # Ternary search refinement
                best_val, best_cost = min(evals, key=lambda x: x[1])
                
                # Check convergence
                if best_cost < self.dc_optimize_target_mv or len(evals) >= 20:
                    print(f"[Optimize] {channel} converged: val={best_val}, cost={best_cost:.2f}mV, evals={len(evals)}")
                    self._finalize_channel(channel, best_val, best_cost)
                    return
                
                # Find neighbors of best point
                sorted_evals = sorted(evals, key=lambda x: x[0])
                best_idx = next(i for i, e in enumerate(sorted_evals) if e[0] == best_val)
                
                # Determine search bounds
                left_bound = sorted_evals[best_idx - 1][0] if best_idx > 0 else max(0, best_val - 8)
                right_bound = sorted_evals[best_idx + 1][0] if best_idx < len(sorted_evals) - 1 else min(255, best_val + 8)
                
                # Try midpoints
                left_mid = (left_bound + best_val) // 2
                right_mid = (best_val + right_bound) // 2
                
                candidates = []
                if abs(left_mid - best_val) >= 2 and not any(abs(left_mid - e[0]) < 2 for e in evals):
                    candidates.append(left_mid)
                if abs(right_mid - best_val) >= 2 and not any(abs(right_mid - e[0]) < 2 for e in evals):
                    candidates.append(right_mid)
                
                if candidates:
                    next_val = candidates[0]
                    self.write_dco_register(channel, next_val)
                    self.dc_optimize_status.setText(
                        f"Refine: {channel}={next_val}\n"
                        f"Best: {best_val} ({best_cost:.2f}mV)"
                    )
                    return
                else:
                    # No more points to try, finalize
                    print(f"[Optimize] {channel} refined: val={best_val}, cost={best_cost:.2f}mV")
                    self._finalize_channel(channel, best_val, best_cost)
    
    # ========== HD2 Q-Channel Optimization ==========
    
    def toggle_hd2_optimization(self):
        """Toggle HD2 optimization on/off."""
        if self.hd2_optimize_running:
            self.stop_hd2_optimization()
        else:
            self.start_hd2_optimization()
    
    def start_hd2_optimization(self):
        """Start the HD2 Q-channel optimization using 2D coordinate descent."""
        self.hd2_optimize_running = True
        self.hd2_target_freq = self.spin_hd2_freq.value()
        self.hd2_freq_tolerance = self.spin_hd2_tol.value()
        
        # Update button appearance
        self.btn_hd2_optimize.setText("⏹ Cancel HD2 Optimization")
        self.btn_hd2_optimize.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        
        self.hd2_optimize_status.setText("Starting HD2 optimization...")
        
        # Initialize HD2 optimization state machine
        # Uses coordinate descent: optimize HD2QX, then HD2QY, repeat
        self.hd2_state = {
            'phase': 'init',
            'current_reg': 'HD2QX',  # Start with X
            'coord_iter': 0,
            'max_coord_iters': 3,  # Do 3 passes of X,Y optimization
            'samples_before_write': 0,
            'best_hd2qx': self.demod_spins['HD2QX'].value(),
            'best_hd2qy': self.demod_spins['HD2QY'].value(),
            'best_cost': float('inf'),
            'evaluations': [],  # List of ((x, y), cost) tuples
            'current_reg_evals': [],  # Evaluations for current register
            'search_phase': 'coarse',
            'initial_points': [],
            'initial_idx': 0,
            'fine_points': [],
        }
        
        # Create timer for optimization loop
        self.hd2_optimize_timer = QTimer()
        self.hd2_optimize_timer.timeout.connect(self.hd2_optimization_step)
        self.hd2_optimize_timer.start(50)  # Run every 50ms
    
    def stop_hd2_optimization(self):
        """Stop the HD2 optimization process."""
        self.hd2_optimize_running = False
        
        if self.hd2_optimize_timer:
            self.hd2_optimize_timer.stop()
            self.hd2_optimize_timer = None
        
        # Reset button appearance
        self.btn_hd2_optimize.setText("⚡ Optimize HD2 Q-Channel")
        self.btn_hd2_optimize.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 8px;")
        
        if self.hd2_state:
            self.hd2_optimize_status.setText(
                f"Optimization stopped.\n"
                f"HD2QX: {self.demod_spins['HD2QX'].value()}, HD2QY: {self.demod_spins['HD2QY'].value()}"
            )
        self.hd2_state = None
    
    def get_hd2_peak_magnitude(self):
        """Get the peak FFT magnitude in dB around the target frequency for Q channel."""
        fft_n = min(self.hd2_samples_for_fft, self.data_count)
        if fft_n < 256:
            return None
        
        # Extract FFT data from ring buffer (Q channel = ch1v)
        if fft_n <= self.write_idx:
            y_fft = self.ch1v[self.write_idx - fft_n:self.write_idx]
        else:
            part1_len = fft_n - self.write_idx
            y_fft = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:self.write_idx]])
        
        window = np.hanning(fft_n).astype(np.float32)
        fft_result = np.fft.rfft(y_fft * window)
        mag_db = 20 * np.log10(np.maximum(np.abs(fft_result), 1e-10))
        freqs = np.fft.rfftfreq(fft_n, d=1.0 / self.fs_eff)
        
        # Find peak in the target frequency range
        freq_mask = (freqs >= self.hd2_target_freq - self.hd2_freq_tolerance) & \
                    (freqs <= self.hd2_target_freq + self.hd2_freq_tolerance)
        
        if not np.any(freq_mask):
            return None
        
        peak_mag = np.max(mag_db[freq_mask])
        peak_freq = freqs[freq_mask][np.argmax(mag_db[freq_mask])]
        
        return peak_mag, peak_freq
    
    def write_hd2_register(self, reg_name, value):
        """Write a value to an HD2 register and prepare for discard."""
        value = int(max(0, min(255, value)))
        self.demod_spins[reg_name].setValue(value)
        
        # Send the register command
        reg_id = '4' if reg_name == 'HD2QX' else '5'  # HD2QX=4, HD2QY=5
        cmd = f"D{reg_id}{value}"
        self._reader.safe_write(cmd.encode())
        print(f"[HD2 Optimize] Sent {reg_name}={value}")
        
        # Update display labels
        if reg_name == 'HD2QX':
            self.label_hd2qx.setText(str(value))
        else:
            self.label_hd2qy.setText(str(value))
        
        # Record current sample count for discarding
        self.hd2_state['samples_before_write'] = self.k
        self.hd2_state['phase'] = 'discarding'
    
    def _hd2_finalize_register(self, reg_name, best_val, best_cost):
        """Finalize optimization for a register and transition to next."""
        state = self.hd2_state
        
        if reg_name == 'HD2QX':
            state['best_hd2qx'] = best_val
            self.write_hd2_register('HD2QX', best_val)
            state['pending_next_reg'] = 'HD2QY'
            self.hd2_optimize_status.setText(
                f"HD2QX optimized to {best_val} (peak={best_cost:.1f}dB)\n"
                f"Settling before HD2QY optimization..."
            )
        else:  # HD2QY
            state['best_hd2qy'] = best_val
            self.write_hd2_register('HD2QY', best_val)
            state['coord_iter'] += 1
            
            if state['coord_iter'] < state['max_coord_iters']:
                # Do another pass
                state['pending_next_reg'] = 'HD2QX'
                self.hd2_optimize_status.setText(
                    f"Pass {state['coord_iter']}/{state['max_coord_iters']} complete.\n"
                    f"Starting pass {state['coord_iter']+1}..."
                )
            else:
                # All passes done
                state['pending_check_convergence'] = True
                self.hd2_optimize_status.setText(
                    f"Measuring final HD2 level..."
                )
    
    def hd2_optimization_step(self):
        """State machine for HD2 coordinate descent optimization."""
        if not self.hd2_optimize_running or self.hd2_state is None:
            return
        
        state = self.hd2_state
        
        # Phase: discarding samples after register write
        if state['phase'] == 'discarding':
            samples_since_write = self.k - state['samples_before_write']
            if samples_since_write >= self.hd2_samples_to_discard:
                # Check if we have a pending register transition
                if state.get('pending_next_reg'):
                    next_reg = state.pop('pending_next_reg')
                    state['current_reg'] = next_reg
                    state['phase'] = 'start_register'
                    state['current_reg_evals'] = []
                    print(f"[HD2 Optimize] Transitioning to {next_reg}")
                    return
                
                # Check if we need to verify convergence
                if state.get('pending_check_convergence'):
                    state.pop('pending_check_convergence')
                    state['phase'] = 'check_convergence'
                    return
                
                state['phase'] = 'measuring'
                state['samples_before_write'] = self.k
            else:
                self.hd2_optimize_status.setText(
                    f"Discarding samples: {samples_since_write}/{self.hd2_samples_to_discard}\n"
                    f"Register: {state['current_reg']}"
                )
            return
        
        # Phase: check convergence after all passes
        if state['phase'] == 'check_convergence':
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.hd2_samples_for_fft:
                self.hd2_optimize_status.setText(
                    f"Measuring final: {samples_since_discard}/{self.hd2_samples_for_fft}"
                )
                return
            
            result = self.get_hd2_peak_magnitude()
            if result is None:
                return
            
            peak_mag, peak_freq = result
            
            self.hd2_optimize_status.setText(
                f"✓ HD2 Optimization complete!\n"
                f"Peak: {peak_mag:.1f} dB @ {peak_freq:.0f} Hz\n"
                f"HD2QX={state['best_hd2qx']}, HD2QY={state['best_hd2qy']}"
            )
            self.stop_hd2_optimization()
            return
        
        # Phase: measuring FFT after discard
        if state['phase'] == 'measuring':
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.hd2_samples_for_fft:
                self.hd2_optimize_status.setText(
                    f"Collecting samples: {samples_since_discard}/{self.hd2_samples_for_fft}\n"
                    f"Register: {state['current_reg']}"
                )
                return
            
            # Measure FFT peak
            result = self.get_hd2_peak_magnitude()
            if result is None:
                return
            
            peak_mag, peak_freq = result
            current_value = self.demod_spins[state['current_reg']].value()
            state['current_reg_evals'].append((current_value, peak_mag))
            state['evaluations'].append(((self.demod_spins['HD2QX'].value(), 
                                          self.demod_spins['HD2QY'].value()), peak_mag))
            
            print(f"[HD2 Optimize] {state['current_reg']}={current_value} -> {peak_mag:.1f}dB @ {peak_freq:.0f}Hz")
            
            # Continue with search
            state['phase'] = 'search_step'
        
        # Phase: initialization
        if state['phase'] == 'init':
            state['coord_iter'] = 0
            state['phase'] = 'start_register'
        
        # Phase: start optimizing a register
        if state['phase'] == 'start_register':
            reg = state['current_reg']
            state['current_reg_evals'] = []
            
            # Start with 5-point initial grid for coarse search
            state['initial_points'] = [0, 64, 128, 192, 255]
            state['initial_idx'] = 0
            state['search_phase'] = 'coarse'
            
            # Write first value
            first_val = state['initial_points'][0]
            self.write_hd2_register(reg, first_val)
            self.hd2_optimize_status.setText(
                f"Pass {state['coord_iter']+1}: Optimizing {reg}\n"
                f"Coarse search: {first_val}"
            )
            return
        
        # Phase: search stepping
        if state['phase'] == 'search_step':
            reg = state['current_reg']
            evals = state['current_reg_evals']
            
            if state['search_phase'] == 'coarse':
                # Still doing coarse search
                state['initial_idx'] += 1
                if state['initial_idx'] < len(state['initial_points']):
                    next_val = state['initial_points'][state['initial_idx']]
                    self.write_hd2_register(reg, next_val)
                    self.hd2_optimize_status.setText(
                        f"Coarse search {state['initial_idx']+1}/5: {reg}={next_val}"
                    )
                    return
                else:
                    # Coarse search done, find best (minimum peak)
                    best_val, best_cost = min(evals, key=lambda x: x[1])
                    print(f"[HD2 Optimize] Coarse done. Best: {reg}={best_val} -> {best_cost:.1f}dB")
                    
                    # Set up fine search around best value
                    state['search_phase'] = 'fine'
                    left = max(0, best_val - 32)
                    right = min(255, best_val + 32)
                    state['fine_points'] = [left, best_val - 16, best_val, best_val + 16, right]
                    state['fine_points'] = sorted(set(state['fine_points']))
                    
                    # Skip points we already evaluated
                    state['fine_points'] = [p for p in state['fine_points'] 
                                            if not any(abs(p - e[0]) < 4 for e in evals)]
                    
                    if state['fine_points']:
                        next_val = state['fine_points'].pop(0)
                        self.write_hd2_register(reg, next_val)
                        self.hd2_optimize_status.setText(f"Fine search: {reg}={next_val}")
                        return
                    else:
                        state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'fine':
                if state['fine_points']:
                    next_val = state['fine_points'].pop(0)
                    self.write_hd2_register(reg, next_val)
                    self.hd2_optimize_status.setText(f"Fine search: {reg}={next_val}")
                    return
                else:
                    state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'refine':
                best_val, best_cost = min(evals, key=lambda x: x[1])
                
                # Check if we've done enough evaluations
                if len(evals) >= 12:
                    print(f"[HD2 Optimize] {reg} done: val={best_val}, cost={best_cost:.1f}dB")
                    self._hd2_finalize_register(reg, best_val, best_cost)
                    return
                
                # Try midpoints around best
                sorted_evals = sorted(evals, key=lambda x: x[0])
                best_idx = next(i for i, e in enumerate(sorted_evals) if e[0] == best_val)
                
                left_bound = sorted_evals[best_idx - 1][0] if best_idx > 0 else max(0, best_val - 8)
                right_bound = sorted_evals[best_idx + 1][0] if best_idx < len(sorted_evals) - 1 else min(255, best_val + 8)
                
                left_mid = (left_bound + best_val) // 2
                right_mid = (best_val + right_bound) // 2
                
                candidates = []
                if abs(left_mid - best_val) >= 4 and not any(abs(left_mid - e[0]) < 4 for e in evals):
                    candidates.append(left_mid)
                if abs(right_mid - best_val) >= 4 and not any(abs(right_mid - e[0]) < 4 for e in evals):
                    candidates.append(right_mid)
                
                if candidates:
                    next_val = candidates[0]
                    self.write_hd2_register(reg, next_val)
                    self.hd2_optimize_status.setText(
                        f"Refine: {reg}={next_val}\n"
                        f"Best: {best_val} ({best_cost:.1f}dB)"
                    )
                    return
                else:
                    print(f"[HD2 Optimize] {reg} refined: val={best_val}, cost={best_cost:.1f}dB")
                    self._hd2_finalize_register(reg, best_val, best_cost)
    
    # ========== HD2 I-Channel Optimization ==========
    
    def toggle_hd2i_optimization(self):
        """Toggle HD2 I-channel optimization on/off."""
        if self.hd2i_optimize_running:
            self.stop_hd2i_optimization()
        else:
            self.start_hd2i_optimization()
    
    def start_hd2i_optimization(self):
        """Start the HD2 I-channel optimization using 2D coordinate descent."""
        self.hd2i_optimize_running = True
        self.hd2_target_freq = self.spin_hd2_freq.value()
        self.hd2_freq_tolerance = self.spin_hd2_tol.value()
        
        # Update button appearance
        self.btn_hd2i_optimize.setText("⏹ Cancel")
        self.btn_hd2i_optimize.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 6px;")
        
        self.hd2i_optimize_status.setText("Starting HD2 I-channel optimization...")
        
        # Initialize HD2 I-channel optimization state machine
        # Uses coordinate descent: optimize HD2IX, then HD2IY, repeat
        self.hd2i_state = {
            'phase': 'init',
            'current_reg': 'HD2IX',  # Start with X
            'coord_iter': 0,
            'max_coord_iters': 3,  # Do 3 passes of X,Y optimization
            'samples_before_write': 0,
            'best_hd2ix': self.demod_spins['HD2IX'].value(),
            'best_hd2iy': self.demod_spins['HD2IY'].value(),
            'best_cost': float('inf'),
            'evaluations': [],  # List of ((x, y), cost) tuples
            'current_reg_evals': [],  # Evaluations for current register
            'search_phase': 'coarse',
            'initial_points': [],
            'initial_idx': 0,
            'fine_points': [],
        }
        
        # Create timer for optimization loop
        self.hd2i_optimize_timer = QTimer()
        self.hd2i_optimize_timer.timeout.connect(self.hd2i_optimization_step)
        self.hd2i_optimize_timer.start(50)  # Run every 50ms
    
    def stop_hd2i_optimization(self):
        """Stop the HD2 I-channel optimization process."""
        self.hd2i_optimize_running = False
        
        if self.hd2i_optimize_timer:
            self.hd2i_optimize_timer.stop()
            self.hd2i_optimize_timer = None
        
        # Reset button appearance
        self.btn_hd2i_optimize.setText("⚡ Optimize I-Channel")
        self.btn_hd2i_optimize.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 6px;")
        
        if self.hd2i_state:
            self.hd2i_optimize_status.setText(
                f"Optimization stopped.\n"
                f"HD2IX: {self.demod_spins['HD2IX'].value()}, HD2IY: {self.demod_spins['HD2IY'].value()}"
            )
        self.hd2i_state = None
    
    def get_hd2i_peak_magnitude(self):
        """Get the peak FFT magnitude in dB around the target frequency for I channel."""
        fft_n = min(self.hd2_samples_for_fft, self.data_count)
        if fft_n < 256:
            return None
        
        # Extract FFT data from ring buffer (I channel = ch2v)
        if fft_n <= self.write_idx:
            y_fft = self.ch2v[self.write_idx - fft_n:self.write_idx]
        else:
            part1_len = fft_n - self.write_idx
            y_fft = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:self.write_idx]])
        
        window = np.hanning(fft_n).astype(np.float32)
        fft_result = np.fft.rfft(y_fft * window)
        mag_db = 20 * np.log10(np.maximum(np.abs(fft_result), 1e-10))
        freqs = np.fft.rfftfreq(fft_n, d=1.0 / self.fs_eff)
        
        # Find peak in the target frequency range
        freq_mask = (freqs >= self.hd2_target_freq - self.hd2_freq_tolerance) & \
                    (freqs <= self.hd2_target_freq + self.hd2_freq_tolerance)
        
        if not np.any(freq_mask):
            return None
        
        peak_mag = np.max(mag_db[freq_mask])
        peak_freq = freqs[freq_mask][np.argmax(mag_db[freq_mask])]
        
        return peak_mag, peak_freq
    
    def write_hd2i_register(self, reg_name, value):
        """Write a value to an HD2 I-channel register and prepare for discard."""
        value = int(max(0, min(255, value)))
        self.demod_spins[reg_name].setValue(value)
        
        # Send the register command
        reg_id = '2' if reg_name == 'HD2IX' else '3'  # HD2IX=2, HD2IY=3
        cmd = f"D{reg_id}{value}"
        self._reader.safe_write(cmd.encode())
        print(f"[HD2I Optimize] Sent {reg_name}={value}")
        
        # Update display labels
        if reg_name == 'HD2IX':
            self.label_hd2ix.setText(str(value))
        else:
            self.label_hd2iy.setText(str(value))
        
        # Record current sample count for discarding
        self.hd2i_state['samples_before_write'] = self.k
        self.hd2i_state['phase'] = 'discarding'
    
    def _hd2i_finalize_register(self, reg_name, best_val, best_cost):
        """Finalize optimization for a register and transition to next."""
        state = self.hd2i_state
        
        if reg_name == 'HD2IX':
            state['best_hd2ix'] = best_val
            self.write_hd2i_register('HD2IX', best_val)
            state['pending_next_reg'] = 'HD2IY'
            self.hd2i_optimize_status.setText(
                f"HD2IX optimized to {best_val} (peak={best_cost:.1f}dB)\n"
                f"Settling before HD2IY optimization..."
            )
        else:  # HD2IY
            state['best_hd2iy'] = best_val
            self.write_hd2i_register('HD2IY', best_val)
            state['coord_iter'] += 1
            
            if state['coord_iter'] < state['max_coord_iters']:
                # Do another pass
                state['pending_next_reg'] = 'HD2IX'
                self.hd2i_optimize_status.setText(
                    f"Pass {state['coord_iter']}/{state['max_coord_iters']} complete.\n"
                    f"Starting pass {state['coord_iter']+1}..."
                )
            else:
                # All passes done
                state['pending_check_convergence'] = True
                self.hd2i_optimize_status.setText(
                    f"Measuring final HD2 level..."
                )
    
    def hd2i_optimization_step(self):
        """State machine for HD2 I-channel coordinate descent optimization."""
        if not self.hd2i_optimize_running or self.hd2i_state is None:
            return
        
        state = self.hd2i_state
        
        # Phase: discarding samples after register write
        if state['phase'] == 'discarding':
            samples_since_write = self.k - state['samples_before_write']
            if samples_since_write >= self.hd2_samples_to_discard:
                # Check if we have a pending register transition
                if state.get('pending_next_reg'):
                    next_reg = state.pop('pending_next_reg')
                    state['current_reg'] = next_reg
                    state['phase'] = 'start_register'
                    state['current_reg_evals'] = []
                    print(f"[HD2I Optimize] Transitioning to {next_reg}")
                    return
                
                # Check if we need to verify convergence
                if state.get('pending_check_convergence'):
                    state.pop('pending_check_convergence')
                    state['phase'] = 'check_convergence'
                    return
                
                state['phase'] = 'measuring'
                state['samples_before_write'] = self.k
            else:
                self.hd2i_optimize_status.setText(
                    f"Discarding samples: {samples_since_write}/{self.hd2_samples_to_discard}\n"
                    f"Register: {state['current_reg']}"
                )
            return
        
        # Phase: check convergence after all passes
        if state['phase'] == 'check_convergence':
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.hd2_samples_for_fft:
                self.hd2i_optimize_status.setText(
                    f"Measuring final: {samples_since_discard}/{self.hd2_samples_for_fft}"
                )
                return
            
            result = self.get_hd2i_peak_magnitude()
            if result is None:
                return
            
            peak_mag, peak_freq = result
            
            self.hd2i_optimize_status.setText(
                f"✓ HD2 I-Ch Optimization complete!\n"
                f"Peak: {peak_mag:.1f} dB @ {peak_freq:.0f} Hz\n"
                f"HD2IX={state['best_hd2ix']}, HD2IY={state['best_hd2iy']}"
            )
            self.stop_hd2i_optimization()
            return
        
        # Phase: measuring FFT after discard
        if state['phase'] == 'measuring':
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.hd2_samples_for_fft:
                self.hd2i_optimize_status.setText(
                    f"Collecting samples: {samples_since_discard}/{self.hd2_samples_for_fft}\n"
                    f"Register: {state['current_reg']}"
                )
                return
            
            # Measure FFT peak
            result = self.get_hd2i_peak_magnitude()
            if result is None:
                return
            
            peak_mag, peak_freq = result
            current_value = self.demod_spins[state['current_reg']].value()
            state['current_reg_evals'].append((current_value, peak_mag))
            state['evaluations'].append(((self.demod_spins['HD2IX'].value(), 
                                          self.demod_spins['HD2IY'].value()), peak_mag))
            
            print(f"[HD2I Optimize] {state['current_reg']}={current_value} -> {peak_mag:.1f}dB @ {peak_freq:.0f}Hz")
            
            # Continue with search
            state['phase'] = 'search_step'
        
        # Phase: initialization
        if state['phase'] == 'init':
            state['coord_iter'] = 0
            state['phase'] = 'start_register'
        
        # Phase: start optimizing a register
        if state['phase'] == 'start_register':
            reg = state['current_reg']
            state['current_reg_evals'] = []
            
            # Start with 5-point initial grid for coarse search
            state['initial_points'] = [0, 64, 128, 192, 255]
            state['initial_idx'] = 0
            state['search_phase'] = 'coarse'
            
            # Write first value
            first_val = state['initial_points'][0]
            self.write_hd2i_register(reg, first_val)
            self.hd2i_optimize_status.setText(
                f"Pass {state['coord_iter']+1}: Optimizing {reg}\n"
                f"Coarse search: {first_val}"
            )
            return
        
        # Phase: search stepping
        if state['phase'] == 'search_step':
            reg = state['current_reg']
            evals = state['current_reg_evals']
            
            if state['search_phase'] == 'coarse':
                # Still doing coarse search
                state['initial_idx'] += 1
                if state['initial_idx'] < len(state['initial_points']):
                    next_val = state['initial_points'][state['initial_idx']]
                    self.write_hd2i_register(reg, next_val)
                    self.hd2i_optimize_status.setText(
                        f"Coarse search {state['initial_idx']+1}/5: {reg}={next_val}"
                    )
                    return
                else:
                    # Coarse search done, find best (minimum peak)
                    best_val, best_cost = min(evals, key=lambda x: x[1])
                    print(f"[HD2I Optimize] Coarse done. Best: {reg}={best_val} -> {best_cost:.1f}dB")
                    
                    # Set up fine search around best value
                    state['search_phase'] = 'fine'
                    left = max(0, best_val - 32)
                    right = min(255, best_val + 32)
                    state['fine_points'] = [left, best_val - 16, best_val, best_val + 16, right]
                    state['fine_points'] = sorted(set(state['fine_points']))
                    
                    # Skip points we already evaluated
                    state['fine_points'] = [p for p in state['fine_points'] 
                                            if not any(abs(p - e[0]) < 4 for e in evals)]
                    
                    if state['fine_points']:
                        next_val = state['fine_points'].pop(0)
                        self.write_hd2i_register(reg, next_val)
                        self.hd2i_optimize_status.setText(f"Fine search: {reg}={next_val}")
                        return
                    else:
                        state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'fine':
                if state['fine_points']:
                    next_val = state['fine_points'].pop(0)
                    self.write_hd2i_register(reg, next_val)
                    self.hd2i_optimize_status.setText(f"Fine search: {reg}={next_val}")
                    return
                else:
                    state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'refine':
                best_val, best_cost = min(evals, key=lambda x: x[1])
                
                # Check if we've done enough evaluations
                if len(evals) >= 12:
                    print(f"[HD2I Optimize] {reg} done: val={best_val}, cost={best_cost:.1f}dB")
                    self._hd2i_finalize_register(reg, best_val, best_cost)
                    return
                
                # Try midpoints around best
                sorted_evals = sorted(evals, key=lambda x: x[0])
                best_idx = next(i for i, e in enumerate(sorted_evals) if e[0] == best_val)
                
                left_bound = sorted_evals[best_idx - 1][0] if best_idx > 0 else max(0, best_val - 8)
                right_bound = sorted_evals[best_idx + 1][0] if best_idx < len(sorted_evals) - 1 else min(255, best_val + 8)
                
                left_mid = (left_bound + best_val) // 2
                right_mid = (best_val + right_bound) // 2
                
                candidates = []
                if abs(left_mid - best_val) >= 4 and not any(abs(left_mid - e[0]) < 4 for e in evals):
                    candidates.append(left_mid)
                if abs(right_mid - best_val) >= 4 and not any(abs(right_mid - e[0]) < 4 for e in evals):
                    candidates.append(right_mid)
                
                if candidates:
                    next_val = candidates[0]
                    self.write_hd2i_register(reg, next_val)
                    self.hd2i_optimize_status.setText(
                        f"Refine: {reg}={next_val}\n"
                        f"Best: {best_val} ({best_cost:.1f}dB)"
                    )
                    return
                else:
                    print(f"[HD2I Optimize] {reg} refined: val={best_val}, cost={best_cost:.1f}dB")
                    self._hd2i_finalize_register(reg, best_val, best_cost)
    
    # ========== IQ Calibration (Image Rejection) Optimization ==========
    
    def send_iq_cal_reg(self, name, reg_id, value):
        """Send a PHA or GERR register value."""
        cmd = f"D{reg_id}{value}"
        self._reader.safe_write(cmd.encode())
        print(f"[IQ Cal] Sent {name}={value}")
        self.msg_label.setText(f"Sent {name}={value}")
    
    def toggle_iq_cal_optimization(self):
        """Toggle IQ calibration optimization on/off."""
        if self.iq_cal_optimize_running:
            self.stop_iq_cal_optimization()
        else:
            self.start_iq_cal_optimization()
    
    def start_iq_cal_optimization(self):
        """Start the IQ calibration optimization using 2D coordinate descent on PHA and GERR."""
        self.iq_cal_optimize_running = True
        self.iq_cal_target_freq = self.spin_iq_cal_freq.value()
        self.iq_cal_freq_tolerance = self.spin_iq_cal_tol.value()
        
        # Update button appearance
        self.btn_iq_cal_optimize.setText("⏹ Cancel Optimization")
        self.btn_iq_cal_optimize.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        
        self.iq_cal_optimize_status.setText("Starting IQ calibration optimization...")
        
        # Initialize optimization state machine
        # Uses coordinate descent: optimize PHA, then GERR, repeat
        self.iq_cal_state = {
            'phase': 'init',
            'current_reg': 'PHA',  # Start with phase
            'coord_iter': 0,
            'max_coord_iters': 3,  # Do 3 passes of PHA,GERR optimization
            'samples_before_write': 0,
            'best_pha': self.spin_pha_manual.value(),
            'best_gerr': self.spin_gerr_manual.value(),
            'best_cost': float('inf'),
            'evaluations': [],  # List of ((pha, gerr), cost) tuples
            'current_reg_evals': [],  # Evaluations for current register
            'search_phase': 'coarse',
            'initial_points': [],
            'initial_idx': 0,
            'fine_points': [],
        }
        
        # Create timer for optimization loop
        self.iq_cal_optimize_timer = QTimer()
        self.iq_cal_optimize_timer.timeout.connect(self.iq_cal_optimization_step)
        self.iq_cal_optimize_timer.start(50)  # Run every 50ms
    
    def stop_iq_cal_optimization(self):
        """Stop the IQ calibration optimization process."""
        self.iq_cal_optimize_running = False
        
        if self.iq_cal_optimize_timer:
            self.iq_cal_optimize_timer.stop()
            self.iq_cal_optimize_timer = None
        
        # Reset button appearance
        self.btn_iq_cal_optimize.setText("⚡ Optimize Image Rejection")
        self.btn_iq_cal_optimize.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;")
        
        if self.iq_cal_state:
            self.iq_cal_optimize_status.setText(
                f"Optimization stopped.\n"
                f"PHA: {self.spin_pha_manual.value()}, GERR: {self.spin_gerr_manual.value()}"
            )
        self.iq_cal_state = None
    
    def get_complex_fft_peak_magnitude(self):
        """Get the peak FFT magnitude in dB around the target frequency in complex FFT."""
        fft_n = min(self.iq_cal_samples_for_fft, self.data_count)
        if fft_n < 256:
            return None
        
        # Extract FFT data from ring buffer (I = ch2v, Q = ch1v)
        if fft_n <= self.write_idx:
            i_fft = self.ch2v[self.write_idx - fft_n:self.write_idx]
            q_fft = self.ch1v[self.write_idx - fft_n:self.write_idx]
        else:
            part1_len = fft_n - self.write_idx
            i_fft = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:self.write_idx]])
            q_fft = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:self.write_idx]])
        
        # Complex FFT for I + jQ
        window = np.hanning(fft_n).astype(np.float32)
        complex_signal = (i_fft + 1j * q_fft) * window
        fft_result = np.fft.fft(complex_signal)
        fft_shifted = np.fft.fftshift(fft_result)
        mag_db = 20 * np.log10(np.maximum(np.abs(fft_shifted), 1e-10))
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_n, d=1.0 / self.fs_eff))
        
        # Find peak in the target frequency range (can be negative for image)
        target = self.iq_cal_target_freq
        tol = self.iq_cal_freq_tolerance
        freq_mask = (freqs >= target - tol) & (freqs <= target + tol)
        
        if not np.any(freq_mask):
            return None
        
        peak_mag = np.max(mag_db[freq_mask])
        peak_freq = freqs[freq_mask][np.argmax(mag_db[freq_mask])]
        
        # Update level display
        self.iq_cal_level_label.setText(f"Image Level: {peak_mag:.1f} dB @ {peak_freq:.0f} Hz")
        
        return peak_mag, peak_freq
    
    def write_iq_cal_register(self, reg_name, value):
        """Write a value to PHA or GERR register and prepare for discard."""
        if reg_name == 'PHA':
            value = int(max(0, min(511, value)))
            self.spin_pha_manual.setValue(value)
            reg_id = 'F'
        else:  # GERR
            value = int(max(0, min(63, value)))
            self.spin_gerr_manual.setValue(value)
            reg_id = 'A'
        
        cmd = f"D{reg_id}{value}"
        self._reader.safe_write(cmd.encode())
        print(f"[IQ Cal Optimize] Sent {reg_name}={value}")
        
        # Record current sample count for discarding
        self.iq_cal_state['samples_before_write'] = self.k
        self.iq_cal_state['phase'] = 'discarding'
    
    def _iq_cal_finalize_register(self, reg_name, best_val, best_cost):
        """Finalize optimization for a register and transition to next."""
        state = self.iq_cal_state
        
        if reg_name == 'PHA':
            state['best_pha'] = best_val
            self.write_iq_cal_register('PHA', best_val)
            state['pending_next_reg'] = 'GERR'
            self.iq_cal_optimize_status.setText(
                f"PHA optimized to {best_val} (image={best_cost:.1f}dB)\n"
                f"Settling before GERR optimization..."
            )
        else:  # GERR
            state['best_gerr'] = best_val
            self.write_iq_cal_register('GERR', best_val)
            state['coord_iter'] += 1
            
            if state['coord_iter'] < state['max_coord_iters']:
                # Do another pass
                state['pending_next_reg'] = 'PHA'
                self.iq_cal_optimize_status.setText(
                    f"Pass {state['coord_iter']}/{state['max_coord_iters']} complete.\n"
                    f"Starting pass {state['coord_iter']+1}..."
                )
            else:
                # All passes done
                state['pending_check_convergence'] = True
                self.iq_cal_optimize_status.setText("Measuring final image level...")
    
    def iq_cal_optimization_step(self):
        """State machine for IQ calibration coordinate descent optimization."""
        if not self.iq_cal_optimize_running or self.iq_cal_state is None:
            return
        
        state = self.iq_cal_state
        
        # Phase: discarding samples after register write
        if state['phase'] == 'discarding':
            samples_since_write = self.k - state['samples_before_write']
            if samples_since_write >= self.iq_cal_samples_to_discard:
                # Check if we have a pending register transition
                if state.get('pending_next_reg'):
                    next_reg = state.pop('pending_next_reg')
                    state['current_reg'] = next_reg
                    state['phase'] = 'start_register'
                    state['current_reg_evals'] = []
                    print(f"[IQ Cal Optimize] Transitioning to {next_reg}")
                    return
                
                # Check if we need to verify convergence
                if state.get('pending_check_convergence'):
                    state.pop('pending_check_convergence')
                    state['phase'] = 'check_convergence'
                    return
                
                state['phase'] = 'measuring'
                state['samples_before_write'] = self.k
            else:
                self.iq_cal_optimize_status.setText(
                    f"Discarding samples: {samples_since_write}/{self.iq_cal_samples_to_discard}\n"
                    f"Register: {state['current_reg']}"
                )
            return
        
        # Phase: check convergence after all passes
        if state['phase'] == 'check_convergence':
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.iq_cal_samples_for_fft:
                self.iq_cal_optimize_status.setText(
                    f"Measuring final: {samples_since_discard}/{self.iq_cal_samples_for_fft}"
                )
                return
            
            result = self.get_complex_fft_peak_magnitude()
            if result is None:
                return
            
            peak_mag, peak_freq = result
            
            self.iq_cal_optimize_status.setText(
                f"✓ IQ Calibration complete!\n"
                f"Image: {peak_mag:.1f} dB @ {peak_freq:.0f} Hz\n"
                f"PHA={state['best_pha']}, GERR={state['best_gerr']}"
            )
            self.stop_iq_cal_optimization()
            return
        
        # Phase: measuring FFT after discard
        if state['phase'] == 'measuring':
            samples_since_discard = self.k - state['samples_before_write']
            if samples_since_discard < self.iq_cal_samples_for_fft:
                self.iq_cal_optimize_status.setText(
                    f"Collecting samples: {samples_since_discard}/{self.iq_cal_samples_for_fft}\n"
                    f"Register: {state['current_reg']}"
                )
                return
            
            # Measure FFT peak
            result = self.get_complex_fft_peak_magnitude()
            if result is None:
                return
            
            peak_mag, peak_freq = result
            
            if state['current_reg'] == 'PHA':
                current_value = self.spin_pha_manual.value()
            else:
                current_value = self.spin_gerr_manual.value()
            
            state['current_reg_evals'].append((current_value, peak_mag))
            state['evaluations'].append(((self.spin_pha_manual.value(), 
                                          self.spin_gerr_manual.value()), peak_mag))
            
            print(f"[IQ Cal Optimize] {state['current_reg']}={current_value} -> {peak_mag:.1f}dB @ {peak_freq:.0f}Hz")
            
            # Continue with search
            state['phase'] = 'search_step'
        
        # Phase: initialization
        if state['phase'] == 'init':
            state['coord_iter'] = 0
            state['phase'] = 'start_register'
        
        # Phase: start optimizing a register
        if state['phase'] == 'start_register':
            reg = state['current_reg']
            state['current_reg_evals'] = []
            
            # Different coarse search points for PHA (0-511) vs GERR (0-63)
            if reg == 'PHA':
                state['initial_points'] = [0, 128, 256, 384, 511]
                max_val = 511
            else:  # GERR
                state['initial_points'] = [0, 16, 32, 48, 63]
                max_val = 63
            
            state['initial_idx'] = 0
            state['search_phase'] = 'coarse'
            state['max_val'] = max_val
            
            # Write first value
            first_val = state['initial_points'][0]
            self.write_iq_cal_register(reg, first_val)
            self.iq_cal_optimize_status.setText(
                f"Pass {state['coord_iter']+1}: Optimizing {reg}\n"
                f"Coarse search: {first_val}"
            )
            return
        
        # Phase: search stepping
        if state['phase'] == 'search_step':
            reg = state['current_reg']
            evals = state['current_reg_evals']
            max_val = state.get('max_val', 255)
            
            if state['search_phase'] == 'coarse':
                # Still doing coarse search
                state['initial_idx'] += 1
                if state['initial_idx'] < len(state['initial_points']):
                    next_val = state['initial_points'][state['initial_idx']]
                    self.write_iq_cal_register(reg, next_val)
                    self.iq_cal_optimize_status.setText(
                        f"Coarse search {state['initial_idx']+1}/5: {reg}={next_val}"
                    )
                    return
                else:
                    # Coarse search done, find best (minimum peak)
                    best_val, best_cost = min(evals, key=lambda x: x[1])
                    print(f"[IQ Cal Optimize] Coarse done. Best: {reg}={best_val} -> {best_cost:.1f}dB")
                    
                    # Set up fine search around best value
                    state['search_phase'] = 'fine'
                    step = 32 if reg == 'PHA' else 8
                    left = max(0, best_val - step)
                    right = min(max_val, best_val + step)
                    half_step = step // 2
                    state['fine_points'] = [left, best_val - half_step, best_val, best_val + half_step, right]
                    state['fine_points'] = [max(0, min(max_val, p)) for p in state['fine_points']]
                    state['fine_points'] = sorted(set(state['fine_points']))
                    
                    # Skip points we already evaluated
                    min_dist = 4 if reg == 'PHA' else 2
                    state['fine_points'] = [p for p in state['fine_points'] 
                                            if not any(abs(p - e[0]) < min_dist for e in evals)]
                    
                    if state['fine_points']:
                        next_val = state['fine_points'].pop(0)
                        self.write_iq_cal_register(reg, next_val)
                        self.iq_cal_optimize_status.setText(f"Fine search: {reg}={next_val}")
                        return
                    else:
                        state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'fine':
                if state['fine_points']:
                    next_val = state['fine_points'].pop(0)
                    self.write_iq_cal_register(reg, next_val)
                    self.iq_cal_optimize_status.setText(f"Fine search: {reg}={next_val}")
                    return
                else:
                    state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'refine':
                best_val, best_cost = min(evals, key=lambda x: x[1])
                
                # Check if we've done enough evaluations
                if len(evals) >= 10:
                    print(f"[IQ Cal Optimize] {reg} done: val={best_val}, cost={best_cost:.1f}dB")
                    self._iq_cal_finalize_register(reg, best_val, best_cost)
                    return
                
                # Try midpoints around best
                sorted_evals = sorted(evals, key=lambda x: x[0])
                best_idx = next(i for i, e in enumerate(sorted_evals) if e[0] == best_val)
                
                step = 8 if reg == 'PHA' else 2
                left_bound = sorted_evals[best_idx - 1][0] if best_idx > 0 else max(0, best_val - step)
                right_bound = sorted_evals[best_idx + 1][0] if best_idx < len(sorted_evals) - 1 else min(max_val, best_val + step)
                
                left_mid = (left_bound + best_val) // 2
                right_mid = (best_val + right_bound) // 2
                
                min_dist = 4 if reg == 'PHA' else 2
                candidates = []
                if abs(left_mid - best_val) >= min_dist and not any(abs(left_mid - e[0]) < min_dist for e in evals):
                    candidates.append(left_mid)
                if abs(right_mid - best_val) >= min_dist and not any(abs(right_mid - e[0]) < min_dist for e in evals):
                    candidates.append(right_mid)
                
                if candidates:
                    next_val = candidates[0]
                    self.write_iq_cal_register(reg, next_val)
                    self.iq_cal_optimize_status.setText(
                        f"Refine: {reg}={next_val}\n"
                        f"Best: {best_val} ({best_cost:.1f}dB)"
                    )
                    return
                else:
                    print(f"[IQ Cal Optimize] {reg} refined: val={best_val}, cost={best_cost:.1f}dB")
                    self._iq_cal_finalize_register(reg, best_val, best_cost)
    
    # ========== DSP Calibration (Software Gain/Phase) Optimization ==========
    
    def on_dsp_gain_changed(self, value):
        """Handle manual DSP gain change."""
        self.dsp_gain = value
    
    def on_dsp_phase_changed(self, value):
        """Handle manual DSP phase change."""
        self.dsp_phase = value
    
    def reset_dsp_params(self):
        """Reset DSP parameters to defaults."""
        self.spin_dsp_gain.setValue(1.0)
        self.spin_dsp_phase.setValue(0.0)
        self.dsp_gain = 1.0
        self.dsp_phase = 0.0
    
    def apply_dsp_correction(self, i_data, q_data):
        """Apply software DSP gain and phase correction to I/Q data.
        
        Returns corrected (I, Q) tuple.
        Phase rotation: I' + jQ' = (I + jQ) * e^(j*theta)
        where theta = dsp_phase in radians
        """
        # Apply gain correction to Q channel
        q_corrected = q_data * self.dsp_gain
        
        # Apply phase rotation to correct I/Q phase error
        theta = np.radians(self.dsp_phase)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Rotate: I' = I*cos - Q*sin, Q' = I*sin + Q*cos
        i_out = i_data * cos_theta - q_corrected * sin_theta
        q_out = i_data * sin_theta + q_corrected * cos_theta
        
        return i_out, q_out
    
    def _design_dc_filter(self):
        """Design a high-pass Butterworth filter to remove DC (0-10Hz)."""
        nyquist = self.fs_eff / 2.0
        if self.dc_filter_cutoff >= nyquist:
            # Cutoff too high, use a reasonable fraction
            self.dc_filter_cutoff = nyquist * 0.01
        normalized_cutoff = self.dc_filter_cutoff / nyquist
        # Ensure normalized cutoff is in valid range (0, 1)
        normalized_cutoff = max(0.001, min(0.999, normalized_cutoff))
        self.dc_filter_sos = butter(self.dc_filter_order, normalized_cutoff, btype='high', output='sos')
    
    def on_dc_filter_toggled(self, state):
        """Handle DC filter checkbox toggle."""
        self.dc_filter_enabled = (state == Qt.CheckState.Checked.value)
    
    def _design_bp_2500hz_filter(self):
        """Design a bandpass Butterworth filter to isolate 2500Hz tone."""
        nyquist = self.fs_eff / 2.0
        low_freq = self.bp_2500hz_center - self.bp_2500hz_bandwidth / 2.0
        high_freq = self.bp_2500hz_center + self.bp_2500hz_bandwidth / 2.0
        # Ensure frequencies are within valid range
        low_freq = max(1.0, low_freq)
        high_freq = min(nyquist - 1.0, high_freq)
        if low_freq >= high_freq:
            # Invalid filter, just don't design
            self.bp_2500hz_sos = None
            return
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist
        # Clamp to valid range
        low_normalized = max(0.001, min(0.999, low_normalized))
        high_normalized = max(0.001, min(0.999, high_normalized))
        self.bp_2500hz_sos = butter(self.bp_2500hz_order, [low_normalized, high_normalized], btype='band', output='sos')
    
    def on_bp_2500hz_toggled(self, state):
        """Handle 2500Hz bandpass filter checkbox toggle."""
        self.bp_2500hz_filter_enabled = (state == Qt.CheckState.Checked.value)
    
    def on_freq_shift_toggled(self, state):
        """Handle frequency shift checkbox toggle."""
        self.freq_shift_enabled = (state == Qt.CheckState.Checked.value)
    
    def on_freq_shift_value_changed(self, value):
        """Handle frequency shift value change."""
        self.freq_shift_hz = float(value)
    
    def on_bb_filter_toggled(self, state):
        """Handle baseband filter checkbox toggle."""
        self.bb_filter_enabled = (state == Qt.CheckState.Checked.value)
    
    def on_bb_cutoff_changed(self, value):
        """Handle baseband filter cutoff change."""
        self.bb_filter_cutoff = float(value)
        self._design_bb_filter()
    
    def _design_bb_filter(self):
        """Design a highpass Butterworth filter to block DC after baseband shift."""
        nyquist = self.fs_eff / 2.0
        if self.bb_filter_cutoff >= nyquist:
            # Cutoff too high, use a reasonable fraction
            self.bb_filter_cutoff = nyquist * 0.01
        normalized_cutoff = self.bb_filter_cutoff / nyquist
        # Ensure normalized cutoff is in valid range (0, 1)
        normalized_cutoff = max(0.001, min(0.999, normalized_cutoff))
        self.bb_filter_sos = butter(self.bb_filter_order, normalized_cutoff, btype='high', output='sos')
    
    def apply_freq_shift(self, complex_signal):
        """Shift complex signal by freq_shift_hz to baseband.
        
        Multiplies by exp(-j * 2 * pi * f_shift * t) to shift the signal
        so that f_shift moves to DC (0 Hz).
        """
        n_samples = len(complex_signal)
        t = np.arange(n_samples) / self.fs_eff
        shift_phasor = np.exp(-1j * 2 * np.pi * self.freq_shift_hz * t)
        return complex_signal * shift_phasor
    
    def apply_bb_filter(self, complex_signal):
        """Apply baseband DC-blocking highpass filter to complex signal.
        
        Filters I and Q components separately to remove DC.
        """
        if self.bb_filter_sos is None:
            return complex_signal
        i_filtered = sosfilt(self.bb_filter_sos, np.real(complex_signal)).astype(np.float32)
        q_filtered = sosfilt(self.bb_filter_sos, np.imag(complex_signal)).astype(np.float32)
        return i_filtered + 1j * q_filtered
    
    def on_notch_filter_toggled(self, state):
        """Handle notch filter checkbox toggle."""
        self.notch_filter_enabled = (state == Qt.CheckState.Checked.value)
    
    def on_notch_freq_changed(self, value):
        """Handle notch filter frequency change."""
        self.notch_filter_freq = float(value)
        self._design_notch_filter()
    
    def on_notch_bw_changed(self, value):
        """Handle notch filter bandwidth change."""
        self.notch_filter_bw = float(value)
        self._design_notch_filter()
    
    def _design_notch_filter(self):
        """Design a highpass Butterworth filter for the notch (to remove DC after shifting).
        
        When applied: we shift the signal so notch_freq is at DC, then
        apply this highpass to remove content at DC (within the bandwidth),
        then shift back. This creates a notch at the specified frequency.
        """
        nyquist = self.fs_eff / 2.0
        # Use half the bandwidth as the highpass cutoff
        cutoff = self.notch_filter_bw / 2.0
        if cutoff >= nyquist:
            cutoff = nyquist * 0.1
        normalized_cutoff = cutoff / nyquist
        normalized_cutoff = max(0.001, min(0.999, normalized_cutoff))
        self.notch_filter_sos = butter(self.notch_filter_order, normalized_cutoff, btype='high', output='sos')
    
    def apply_notch_filter(self, complex_signal):
        """Apply notch filter at specified frequency in complex domain.
        
        Shifts the signal so the notch frequency is at DC, applies a
        highpass filter to remove DC content, then shifts back.
        This effectively creates a notch at the specified baseband frequency.
        """
        if self.notch_filter_sos is None:
            return complex_signal
        
        n_samples = len(complex_signal)
        t = np.arange(n_samples) / self.fs_eff
        
        # Shift the notch frequency to DC
        shift_phasor = np.exp(-1j * 2 * np.pi * self.notch_filter_freq * t)
        shifted = complex_signal * shift_phasor
        
        # Apply highpass filter around DC (filters I and Q separately)
        i_filtered = sosfilt(self.notch_filter_sos, np.real(shifted)).astype(np.float32)
        q_filtered = sosfilt(self.notch_filter_sos, np.imag(shifted)).astype(np.float32)
        filtered = i_filtered + 1j * q_filtered
        
        # Shift back
        shift_phasor_inv = np.exp(1j * 2 * np.pi * self.notch_filter_freq * t)
        return filtered * shift_phasor_inv
    
    def apply_bp_2500hz_filter(self, data):
        """Apply 2500Hz bandpass filter to data array."""
        if self.bp_2500hz_sos is None:
            return data
        return sosfilt(self.bp_2500hz_sos, data).astype(np.float32)
    
    def apply_dc_filter(self, data):
        """Apply DC high-pass filter to data array.
        
        Uses a zero-phase filter (forward-backward filtering) equivalent 
        for real-time use via sosfilt.
        """
        if self.dc_filter_sos is None:
            return data
        # Apply the filter
        return sosfilt(self.dc_filter_sos, data).astype(np.float32)
    
    # ========== SCD Dataset Save ==========

    def save_scd_dataset(self):
        """Save the current SCD pattern as image + text data, plus a settings text file.
        
        Folder structure: D:/FYP/Dataset/<drone_name>/<drone_type>/<distance_m>/<session_timestamp>/
        """
        if self.scd_data is None:
            QMessageBox.warning(self, "No SCD Data", "No SCD data available. Enable SCD and click 'Update Now' first.")
            return

        drone_name = self.edit_drone_name.text().strip()
        drone_type = self.edit_drone_type.text().strip()
        distance = self.spin_distance.value()

        if not drone_name:
            QMessageBox.warning(self, "Missing Field", "Please enter a Drone Name.")
            return
        if not drone_type:
            QMessageBox.warning(self, "Missing Field", "Please enter a Drone Type.")
            return

        # Sanitize folder names (replace invalid chars)
        def sanitize(name):
            return "".join(c if c.isalnum() or c in (' ', '-', '_', '.') else '_' for c in name).strip()

        drone_name_safe = sanitize(drone_name)
        drone_type_safe = sanitize(drone_type)
        distance_str = f"{distance:.2f}m"
        now = datetime.now()
        session_stamp = now.strftime("%Y%m%d_%H%M%S")

        base_path = os.path.join("D:", os.sep, "FYP", "Dataset")
        folder = os.path.join(base_path, drone_name_safe, drone_type_safe, distance_str, session_stamp)
        os.makedirs(folder, exist_ok=True)

        # --- 1. Save SCD data as text file ---
        scd_txt_path = os.path.join(folder, "scd_data.txt")
        np.savetxt(scd_txt_path, self.scd_data, fmt="%.6f", delimiter="\t")

        # --- 2. Save SCD plot as image ---
        scd_img_path = os.path.join(folder, "scd_pattern.png")
        try:
            import pyqtgraph.exporters as exporters
            exporter = exporters.ImageExporter(self.scd_plot.plotItem)
            exporter.parameters()['width'] = 1920
            exporter.export(scd_img_path)
        except Exception as e:
            # Fallback: grab the widget as an image
            try:
                pixmap = self.scd_plot.grab()
                pixmap.save(scd_img_path, "PNG")
            except Exception as e2:
                QMessageBox.warning(self, "Image Save Error", f"Could not save image: {e2}")

        # --- 3. Save settings text file ---
        settings_path = os.path.join(folder, "settings.txt")
        with open(settings_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  SCD Dataset Capture Settings\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Date & Time       : {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Drone Name        : {drone_name}\n")
            f.write(f"Drone Type        : {drone_type}\n")
            f.write(f"Distance to Object: {distance:.2f} m\n\n")

            f.write("--- ADC Settings ---\n")
            f.write(f"Sample Rate       : {self.sample_rate} Hz\n")
            f.write(f"Effective Fs      : {self.fs_eff} Hz\n")
            f.write(f"VREF              : {self.vref} V\n")
            f.write(f"Decimation        : {self.decim}\n")
            f.write(f"Total Samples     : {self.data_count}\n\n")

            f.write("--- SCD Settings ---\n")
            f.write(f"SCD Enabled       : {self.scd_enabled}\n")
            f.write(f"FFT Size (NFFT)   : {self.scd_nfft}\n")
            f.write(f"Segments          : {self.scd_num_segments}\n")
            f.write(f"Overlap           : {self.scd_overlap_pct}%\n")
            f.write(f"Alpha Step        : {self.scd_alpha_step}\n")
            f.write(f"Max Freq Display  : {self.spin_scd_maxfreq.value()} Hz\n")
            f.write(f"Max Alpha Display : {self.spin_scd_maxalpha.value()} Hz\n")
            f.write(f"Min dB Floor      : {self.spin_scd_min_db.value()} dB\n")
            f.write(f"Use Shifted Signal: {self.chk_scd_use_shifted.isChecked()}\n")
            f.write(f"GPU Enabled       : {self.chk_scd_gpu.isChecked()}\n\n")

            f.write("--- Display / Filter Settings ---\n")
            f.write(f"X Window          : {self.spin_xwindow.value()}\n")
            f.write(f"Y Range           : ±{self.spin_yrange.value()} V\n")
            f.write(f"Max Freq (FFT)    : {self.spin_maxfreq.value()} Hz\n")
            f.write(f"FFT Size (display): {self.combo_fft.currentText()}\n")
            f.write(f"DC Filter Enabled : {self.dc_filter_enabled}\n")
            f.write(f"BP 2500Hz Filter  : {self.bp_2500hz_filter_enabled}\n")
            f.write(f"Freq Shift Enabled: {self.freq_shift_enabled}\n")
            f.write(f"Freq Shift Hz     : {self.freq_shift_hz}\n")
            f.write(f"BB DC Block       : {self.bb_filter_enabled}\n")
            f.write(f"BB Cutoff         : {self.bb_filter_cutoff} Hz\n")
            f.write(f"Notch Filter      : {self.notch_filter_enabled}\n")
            f.write(f"Notch Freq        : {self.notch_filter_freq} Hz\n")
            f.write(f"Notch BW          : {self.notch_filter_bw} Hz\n\n")

            f.write("--- DVGA / Attenuator ---\n")
            f.write(f"DVGA Gain         : {self.spin_dvga.value()}\n")
            f.write(f"Attenuator        : {self.spin_att.value()}\n\n")

            f.write("--- DSP Calibration ---\n")
            f.write(f"DSP Gain          : {self.dsp_gain}\n")
            f.write(f"DSP Phase         : {self.dsp_phase}°\n\n")

            f.write("--- Demodulator Registers ---\n")
            for name, spin in self.demod_spins.items():
                f.write(f"{name:18s}: {spin.value()}\n")
            f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("  End of Settings\n")
            f.write("=" * 60 + "\n")

        # Show success
        self.scd_info_label.setText(f"SCD: Saved to {folder}")
        QMessageBox.information(self, "Dataset Saved",
            f"SCD dataset saved successfully!\n\n"
            f"Folder: {folder}\n\n"
            f"Files:\n"
            f"  • scd_data.txt (numeric data)\n"
            f"  • scd_pattern.png (image)\n"
            f"  • settings.txt (all settings)")

    # ========== Spectral Correlation Density (SCD) ==========
    
    def on_scd_enable_toggled(self, state):
        """Handle SCD enable checkbox toggle."""
        self.scd_enabled = (state == Qt.CheckState.Checked.value)
        if self.scd_enabled:
            self.scd_info_label.setText("SCD: Enabled | Computing...")
        else:
            self.scd_info_label.setText("SCD: Disabled")
    
    def on_scd_nfft_changed(self, text):
        """Handle SCD FFT size change."""
        self.scd_nfft = int(text)
        self.scd_noverlap = int(self.scd_nfft * self.scd_overlap_pct / 100)
        self.scd_data = None  # Clear cached data
    
    def on_scd_segments_changed(self, value):
        """Handle SCD segments change."""
        self.scd_num_segments = value
        self.scd_data = None  # Clear cached data

    def on_scd_overlap_changed(self, text):
        """Handle SCD overlap change."""
        pct_map = {"50%": 50, "75%": 75, "87.5%": 87.5}
        self.scd_overlap_pct = pct_map.get(text, 75)
        self.scd_noverlap = int(self.scd_nfft * self.scd_overlap_pct / 100)
        self.scd_data = None

    def on_scd_alpha_step_changed(self, index):
        """Handle alpha step change: 0 → step 1 (fine), 1 → step 2 (fast)."""
        self.scd_alpha_step = 1 if index == 0 else 2
        self.scd_data = None
    
    def compute_scd(self, complex_signal):
        """Compute Spectral Correlation Density using FFT Accumulation Method.
        
        The SCD S_x^alpha(f) measures the correlation between frequency components
        at f+alpha/2 and f-alpha/2, averaged over time:
        
            S_x^alpha(f) = E{ X(f + alpha/2) * X*(f - alpha/2) }
        
        This implementation:
        - FFT-shifts first to work on symmetric [-Fs/2, Fs/2) grid
        - Uses only even alpha bin offsets so alpha/2 is integer bins
        - Avoids wrap-around by only correlating where both bins exist
        - Optionally offloads cross-correlation to GPU via OpenCL
        
        Args:
            complex_signal: Complex I/Q signal (I + jQ)
            
        Returns:
            scd: 2D array of SCD magnitude (alpha x f)
            freqs: Frequency axis (Hz)
            alphas: Cyclic frequency axis (Hz)
        """
        N = self.scd_nfft
        L = self.scd_noverlap
        num_segs = self.scd_num_segments

        total_samples = N + (num_segs - 1) * (N - L)
        if len(complex_signal) < N:
            return None, None, None

        signal = complex_signal[-min(len(complex_signal), total_samples):]
        window = np.hanning(N).astype(np.float32)

        max_possible = (len(signal) - N) // (N - L) + 1
        num_segs = min(num_segs, max_possible)
        if num_segs < 1:
            return None, None, None

        # Compute FFTs for each segment, apply fftshift immediately
        ffts = []
        for i in range(num_segs):
            start = i * (N - L)
            seg = signal[start:start + N]
            if len(seg) < N:
                break
            X = np.fft.fftshift(np.fft.fft(seg * window))
            ffts.append(X)
        
        if len(ffts) == 0:
            return None, None, None

        ffts = np.stack(ffts, axis=0).astype(np.complex64)  # [S, N]

        # Alpha bin offsets — step=1 gives full resolution, step=2 gives even-only
        max_m = N // 2
        step = self.scd_alpha_step
        if step == 1:
            # Full resolution: all integer offsets
            m_vals = np.arange(-max_m, max_m + 1, 1, dtype=np.int32)
        else:
            # Even only (alpha/2 stays integer → no interpolation)
            m_vals = np.arange(-max_m, max_m + 1, 2, dtype=np.int32)

        # ── GPU path ──
        if self.use_gpu and self.gpu.available:
            try:
                t0 = time.perf_counter()
                scd_mag = self.gpu.compute_scd_gpu(ffts, m_vals, N)
                self._scd_gpu_ms = (time.perf_counter() - t0) * 1000
                self._scd_backend = "GPU"
            except Exception as e:
                print(f"GPU SCD failed, falling back to CPU: {e}")
                scd_mag = self._compute_scd_cpu(ffts, m_vals, N)
        else:
            t0 = time.perf_counter()
            scd_mag = self._compute_scd_cpu(ffts, m_vals, N)
            self._scd_gpu_ms = (time.perf_counter() - t0) * 1000
            self._scd_backend = "CPU"

        # Frequency axis (already shifted)
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / self.fs_eff))
        # Alpha (cyclic frequency) axis
        alphas = m_vals * (self.fs_eff / N)
        
        return scd_mag, freqs, alphas

    def _compute_scd_cpu(self, ffts, m_vals, N):
        """CPU fallback for SCD cross-correlation.
        
        Supports both even m (integer half-offset) and odd m (linear
        interpolation between adjacent bins).
        """
        S = ffts.shape[0]
        scd = np.zeros((len(m_vals), N), dtype=np.complex64)

        for ai, m in enumerate(m_vals):
            m_int = int(m)
            if m_int % 2 == 0:
                # Even: alpha/2 is integer bins → exact
                mh = m_int // 2
                abs_mh = abs(mh)
                k0 = abs_mh
                k1 = N - abs_mh
                if k1 <= k0:
                    continue
                Xp = ffts[:, k0 + mh:k1 + mh]
                Xm = ffts[:, k0 - mh:k1 - mh]
                scd[ai, k0:k1] = np.mean(Xp * np.conj(Xm), axis=0)
            else:
                # Odd: alpha/2 = (m±1)/2 + 0.5 → interpolate
                mh_lo = m_int // 2       # floor
                mh_hi = mh_lo + 1 if m_int > 0 else mh_lo - 1  # ceil away from 0
                # Actually simpler: for odd m, half = m/2.0
                # Interpolate: X(k + m/2) ≈ 0.5*(X(k+floor) + X(k+ceil))
                mh_floor = m_int // 2
                mh_ceil = mh_floor + (1 if m_int > 0 else -1)
                abs_max = max(abs(mh_floor), abs(mh_ceil))
                k0 = abs_max
                k1 = N - abs_max
                if k1 <= k0:
                    continue
                Xp = 0.5 * (ffts[:, k0 + mh_floor:k1 + mh_floor] +
                            ffts[:, k0 + mh_ceil:k1 + mh_ceil])
                Xm = 0.5 * (ffts[:, k0 - mh_floor:k1 - mh_floor] +
                            ffts[:, k0 - mh_ceil:k1 - mh_ceil])
                scd[ai, k0:k1] = np.mean(Xp * np.conj(Xm), axis=0)

        return np.abs(scd).astype(np.float32)
    
    def update_scd_plot(self):
        """Compute and update the SCD plot."""
        # Snapshot ring buffer state (writer thread may update at any time)
        widx = self.write_idx
        dcount = self.data_count
        
        if dcount < self.scd_nfft * 2:
            self.scd_info_label.setText("SCD: Not enough samples")
            return
        
        # Get samples needed for SCD
        total_samples = self.scd_nfft + (self.scd_num_segments - 1) * (self.scd_nfft - self.scd_noverlap)
        total_samples = min(total_samples, dcount)
        
        # Extract data from ring buffer using snapshot
        if total_samples <= widx:
            i_data = self.ch2v[widx - total_samples:widx].copy()
            q_data = self.ch1v[widx - total_samples:widx].copy()
        else:
            part1_len = total_samples - widx
            i_data = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:widx]])
            q_data = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:widx]])
        
        # Apply DC filter if enabled
        if self.dc_filter_enabled:
            i_data = self.apply_dc_filter(i_data)
            q_data = self.apply_dc_filter(q_data)
        
        # Apply 2500Hz bandpass filter if enabled
        if self.bp_2500hz_filter_enabled:
            i_data = self.apply_bp_2500hz_filter(i_data)
            q_data = self.apply_bp_2500hz_filter(q_data)
        
        # Apply DSP correction if not unity
        if abs(self.dsp_gain - 1.0) > 1e-6 or abs(self.dsp_phase) > 1e-6:
            i_data, q_data = self.apply_dsp_correction(i_data, q_data)
        
        # Create complex signal
        complex_signal = i_data + 1j * q_data
        
        # Apply frequency shift and baseband filter if enabled and checkbox is checked
        if self.chk_scd_use_shifted.isChecked():
            if self.freq_shift_enabled:
                complex_signal = self.apply_freq_shift(complex_signal)
            if self.bb_filter_enabled and self.freq_shift_enabled:
                complex_signal = self.apply_bb_filter(complex_signal)
        
        # Compute SCD
        scd, freqs, alphas = self.compute_scd(complex_signal)
        
        if scd is None:
            self.scd_info_label.setText("SCD: Computation failed")
            return
        
        # Convert to dB
        scd_db = 20 * np.log10(np.maximum(scd, 1e-10))
        
        # Clamp to user-specified noise floor
        floor_db = float(self.spin_scd_min_db.value())
        scd_db = np.clip(scd_db, floor_db, None)
        scd_min = floor_db
        scd_max = np.max(scd_db)
        
        # Update image
        # Set up the transform to map array indices to frequency values
        tr = pg.QtGui.QTransform()
        tr.translate(freqs[0], alphas[0])
        tr.scale((freqs[-1] - freqs[0]) / len(freqs), (alphas[-1] - alphas[0]) / len(alphas))
        self.scd_image.setTransform(tr)
        self.scd_image.setImage(scd_db.T)  # Transpose for correct orientation
        
        # Update colorbar
        self.scd_colorbar.setLevels((scd_min, scd_max))
        
        # Update axes - use SCD-specific max freq controls for low-freq zoom
        fmax_scd = float(self.spin_scd_maxfreq.value())
        alpha_max_scd = float(self.spin_scd_maxalpha.value())
        self.scd_plot.setXRange(-fmax_scd, fmax_scd)
        self.scd_plot.setYRange(-alpha_max_scd, alpha_max_scd)
        
        # Build info text
        shifted_str = "(shifted)" if self.chk_scd_use_shifted.isChecked() and self.freq_shift_enabled else ""
        backend_str = getattr(self, '_scd_backend', 'CPU')
        ms_str = f"{getattr(self, '_scd_gpu_ms', 0):.1f}ms"
        delta_f = self.fs_eff / self.scd_nfft
        delta_alpha = delta_f * self.scd_alpha_step
        
        # Update info label
        self.scd_info_label.setText(
            f"SCD: {self.scd_nfft}-pt FFT, {self.scd_num_segments} segs, "
            f"{self.scd_overlap_pct:.0f}% overlap {shifted_str} | "
            f"Δf={delta_f:.2f} Hz, Δα={delta_alpha:.2f} Hz | "
            f"f: ±{fmax_scd:.0f} Hz | "
            f"α: ±{alpha_max_scd:.0f} Hz | "
            f"[{scd_min:.1f}, {scd_max:.1f}] dB | "
            f"{backend_str} {ms_str}"
        )
        
        self.scd_last_update = time.time()
        self.scd_data = scd_db
    
    def compute_iq_gain_phase_diff(self, i_data, q_data, fft_n=4096):
        """Compute gain ratio and phase difference between I and Q channels.
        
        Uses cross-correlation at the dominant frequency.
        """
        if len(i_data) < fft_n:
            return None, None
        
        # Use a sliding window correlation approach
        n = min(len(i_data), fft_n)
        i_seg = i_data[-n:]
        q_seg = q_data[-n:]
        
        # Remove DC
        i_seg = i_seg - np.mean(i_seg)
        q_seg = q_seg - np.mean(q_seg)
        
        # Compute RMS for gain ratio
        rms_i = np.sqrt(np.mean(i_seg**2))
        rms_q = np.sqrt(np.mean(q_seg**2))
        
        if rms_q > 1e-10:
            gain_ratio = rms_i / rms_q
        else:
            gain_ratio = 1.0
        
        # Compute phase difference using FFT at dominant frequency
        window = np.hanning(n).astype(np.float32)
        fft_i = np.fft.rfft(i_seg * window)
        fft_q = np.fft.rfft(q_seg * window)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs_eff)
        
        # Find dominant frequency (excluding DC)
        mag_i = np.abs(fft_i)
        mag_i[0] = 0  # Exclude DC
        # Find peak in range 100Hz to Nyquist
        freq_mask = (freqs >= 100) & (freqs <= self.fs_eff / 2)
        if np.any(freq_mask):
            masked_mag = np.zeros_like(mag_i)
            masked_mag[freq_mask] = mag_i[freq_mask]
            peak_idx = np.argmax(masked_mag)
            
            if mag_i[peak_idx] > 1e-10 and np.abs(fft_q[peak_idx]) > 1e-10:
                # Phase difference = angle(Q) - angle(I)
                phase_i = np.angle(fft_i[peak_idx])
                phase_q = np.angle(fft_q[peak_idx])
                phase_diff = np.degrees(phase_q - phase_i)
                # Normalize to -180 to 180
                while phase_diff > 180:
                    phase_diff -= 360
                while phase_diff < -180:
                    phase_diff += 360
                # Ideal IQ should have 90° difference; deviation from 90° is the error
                phase_error = phase_diff - 90.0 if phase_diff > 0 else phase_diff + 90.0
                return gain_ratio, phase_error
        
        return gain_ratio, 0.0
    
    def get_dsp_fft_peak_magnitude(self, apply_correction=True):
        """Get the peak FFT magnitude in dB around the DSP target frequency in complex FFT.
        
        Optionally applies DSP gain/phase correction.
        """
        fft_n = min(self.dsp_cal_samples_for_fft, self.data_count)
        if fft_n < 256:
            return None
        
        # Extract FFT data from ring buffer (I = ch2v, Q = ch1v)
        if fft_n <= self.write_idx:
            i_fft = self.ch2v[self.write_idx - fft_n:self.write_idx].copy()
            q_fft = self.ch1v[self.write_idx - fft_n:self.write_idx].copy()
        else:
            part1_len = fft_n - self.write_idx
            i_fft = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:self.write_idx]])
            q_fft = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:self.write_idx]])
        
        # Apply DSP correction if requested
        if apply_correction:
            i_fft, q_fft = self.apply_dsp_correction(i_fft, q_fft)
        
        # Complex FFT for I + jQ
        window = np.hanning(fft_n).astype(np.float32)
        complex_signal = (i_fft + 1j * q_fft) * window
        fft_result = np.fft.fft(complex_signal)
        fft_shifted = np.fft.fftshift(fft_result)
        mag_db = 20 * np.log10(np.maximum(np.abs(fft_shifted), 1e-10))
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_n, d=1.0 / self.fs_eff))
        
        # Find peak in the target frequency range
        target = self.dsp_cal_target_freq
        tol = self.dsp_cal_freq_tolerance
        freq_mask = (freqs >= target - tol) & (freqs <= target + tol)
        
        if not np.any(freq_mask):
            return None
        
        peak_mag = np.max(mag_db[freq_mask])
        peak_freq = freqs[freq_mask][np.argmax(mag_db[freq_mask])]
        
        # Update level display
        self.dsp_cal_level_label.setText(f"Target Level: {peak_mag:.1f} dB @ {peak_freq:.0f} Hz")
        
        return peak_mag, peak_freq
    
    def toggle_dsp_cal_optimization(self):
        """Toggle DSP calibration optimization on/off."""
        if self.dsp_cal_optimize_running:
            self.stop_dsp_cal_optimization()
        else:
            self.start_dsp_cal_optimization()
    
    def start_dsp_cal_optimization(self):
        """Start DSP calibration optimization using coordinate descent on gain and phase."""
        self.dsp_cal_optimize_running = True
        self.dsp_cal_target_freq = self.spin_dsp_target_freq.value()
        self.dsp_cal_freq_tolerance = self.spin_dsp_tol.value()
        
        # Update button appearance
        self.btn_dsp_cal_optimize.setText("⏹ Cancel Optimization")
        self.btn_dsp_cal_optimize.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        
        self.dsp_cal_optimize_status.setText("Starting DSP optimization...")
        
        # Initialize optimization state for coordinate descent on gain and phase
        self.dsp_cal_state = {
            'phase': 'init',
            'current_param': 'gain',  # Start with gain
            'coord_iter': 0,
            'max_coord_iters': 3,  # 3 passes of gain/phase optimization
            'samples_before_change': 0,
            'best_gain': self.dsp_gain,
            'best_phase': self.dsp_phase,
            'best_cost': float('inf'),
            'evaluations': [],
            'current_param_evals': [],
            'search_phase': 'coarse',
            'initial_points': [],
            'initial_idx': 0,
            'fine_points': [],
        }
        
        # Create timer for optimization loop
        self.dsp_cal_optimize_timer = QTimer()
        self.dsp_cal_optimize_timer.timeout.connect(self.dsp_cal_optimization_step)
        self.dsp_cal_optimize_timer.start(50)
    
    def stop_dsp_cal_optimization(self):
        """Stop the DSP calibration optimization."""
        self.dsp_cal_optimize_running = False
        
        if self.dsp_cal_optimize_timer:
            self.dsp_cal_optimize_timer.stop()
            self.dsp_cal_optimize_timer = None
        
        # Reset button appearance
        self.btn_dsp_cal_optimize.setText("⚡ Optimize DSP for Target Freq")
        self.btn_dsp_cal_optimize.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        
        if self.dsp_cal_state:
            self.dsp_cal_optimize_status.setText(
                f"Optimization stopped.\n"
                f"Gain: {self.dsp_gain:.4f}, Phase: {self.dsp_phase:.3f}°"
            )
        self.dsp_cal_state = None
    
    def set_dsp_param(self, param_name, value):
        """Set a DSP parameter and prepare for measurement."""
        if param_name == 'gain':
            # Clamp gain to valid range
            value = max(0.5, min(2.0, value))
            self.dsp_gain = value
            self.spin_dsp_gain.setValue(value)
        else:  # phase
            # Clamp phase to valid range
            value = max(-10.0, min(10.0, value))
            self.dsp_phase = value
            self.spin_dsp_phase.setValue(value)
        
        # Record current sample count for discarding
        self.dsp_cal_state['samples_before_change'] = self.k
        self.dsp_cal_state['phase'] = 'discarding'
        print(f"[DSP Cal] Set {param_name}={value:.4f}")
    
    def _dsp_cal_finalize_param(self, param_name, best_val, best_cost):
        """Finalize optimization for a parameter and transition to next."""
        state = self.dsp_cal_state
        
        if param_name == 'gain':
            state['best_gain'] = best_val
            self.set_dsp_param('gain', best_val)
            state['pending_next_param'] = 'phase'
            self.dsp_cal_optimize_status.setText(
                f"Gain optimized to {best_val:.4f} (level={best_cost:.1f}dB)\n"
                f"Now optimizing phase..."
            )
        else:  # phase
            state['best_phase'] = best_val
            self.set_dsp_param('phase', best_val)
            state['coord_iter'] += 1
            
            if state['coord_iter'] < state['max_coord_iters']:
                state['pending_next_param'] = 'gain'
                self.dsp_cal_optimize_status.setText(
                    f"Pass {state['coord_iter']}/{state['max_coord_iters']} complete.\n"
                    f"Starting pass {state['coord_iter']+1}..."
                )
            else:
                state['pending_check_convergence'] = True
                self.dsp_cal_optimize_status.setText("Measuring final level...")
    
    def dsp_cal_optimization_step(self):
        """State machine for DSP calibration coordinate descent optimization."""
        if not self.dsp_cal_optimize_running or self.dsp_cal_state is None:
            return
        
        state = self.dsp_cal_state
        
        # Phase: discarding samples after parameter change
        if state['phase'] == 'discarding':
            samples_since_change = self.k - state['samples_before_change']
            if samples_since_change >= self.dsp_cal_samples_to_discard:
                if state.get('pending_next_param'):
                    next_param = state.pop('pending_next_param')
                    state['current_param'] = next_param
                    state['phase'] = 'start_param'
                    state['current_param_evals'] = []
                    return
                
                if state.get('pending_check_convergence'):
                    state.pop('pending_check_convergence')
                    state['phase'] = 'check_convergence'
                    return
                
                state['phase'] = 'measuring'
                state['samples_before_change'] = self.k
            else:
                self.dsp_cal_optimize_status.setText(
                    f"Discarding: {samples_since_change}/{self.dsp_cal_samples_to_discard}\n"
                    f"Param: {state['current_param']}"
                )
            return
        
        # Phase: check convergence after all passes
        if state['phase'] == 'check_convergence':
            samples_since = self.k - state['samples_before_change']
            if samples_since < self.dsp_cal_samples_for_fft:
                self.dsp_cal_optimize_status.setText(
                    f"Measuring final: {samples_since}/{self.dsp_cal_samples_for_fft}"
                )
                return
            
            result = self.get_dsp_fft_peak_magnitude(apply_correction=True)
            if result is None:
                return
            
            peak_mag, peak_freq = result
            
            self.dsp_cal_optimize_status.setText(
                f"✓ DSP Optimization complete!\n"
                f"Level: {peak_mag:.1f} dB @ {peak_freq:.0f} Hz\n"
                f"Gain={state['best_gain']:.4f}, Phase={state['best_phase']:.3f}°"
            )
            self.stop_dsp_cal_optimization()
            return
        
        # Phase: measuring FFT after discard
        if state['phase'] == 'measuring':
            samples_since = self.k - state['samples_before_change']
            if samples_since < self.dsp_cal_samples_for_fft:
                self.dsp_cal_optimize_status.setText(
                    f"Collecting: {samples_since}/{self.dsp_cal_samples_for_fft}\n"
                    f"Param: {state['current_param']}"
                )
                return
            
            result = self.get_dsp_fft_peak_magnitude(apply_correction=True)
            if result is None:
                return
            
            peak_mag, peak_freq = result
            
            if state['current_param'] == 'gain':
                current_value = self.dsp_gain
            else:
                current_value = self.dsp_phase
            
            state['current_param_evals'].append((current_value, peak_mag))
            state['evaluations'].append(((self.dsp_gain, self.dsp_phase), peak_mag))
            
            print(f"[DSP Cal] {state['current_param']}={current_value:.4f} -> {peak_mag:.1f}dB @ {peak_freq:.0f}Hz")
            
            state['phase'] = 'search_step'
        
        # Phase: initialization
        if state['phase'] == 'init':
            state['coord_iter'] = 0
            state['phase'] = 'start_param'
        
        # Phase: start optimizing a parameter
        if state['phase'] == 'start_param':
            param = state['current_param']
            state['current_param_evals'] = []
            
            # Different search ranges for gain vs phase
            if param == 'gain':
                # Gain: search from 0.8 to 1.2 in 5 points
                state['initial_points'] = [0.8, 0.9, 1.0, 1.1, 1.2]
                state['param_min'] = 0.5
                state['param_max'] = 2.0
                state['step_size'] = 0.05
            else:  # phase
                # Phase: search from -5 to +5 degrees in 5 points
                state['initial_points'] = [-5.0, -2.5, 0.0, 2.5, 5.0]
                state['param_min'] = -10.0
                state['param_max'] = 10.0
                state['step_size'] = 0.5
            
            state['initial_idx'] = 0
            state['search_phase'] = 'coarse'
            
            first_val = state['initial_points'][0]
            self.set_dsp_param(param, first_val)
            self.dsp_cal_optimize_status.setText(
                f"Pass {state['coord_iter']+1}: Optimizing {param}\n"
                f"Coarse search: {first_val:.4f}"
            )
            return
        
        # Phase: search stepping
        if state['phase'] == 'search_step':
            param = state['current_param']
            evals = state['current_param_evals']
            
            if state['search_phase'] == 'coarse':
                state['initial_idx'] += 1
                if state['initial_idx'] < len(state['initial_points']):
                    next_val = state['initial_points'][state['initial_idx']]
                    self.set_dsp_param(param, next_val)
                    self.dsp_cal_optimize_status.setText(
                        f"Coarse {state['initial_idx']+1}/5: {param}={next_val:.4f}"
                    )
                    return
                else:
                    # Coarse done, find best
                    best_val, best_cost = min(evals, key=lambda x: x[1])
                    print(f"[DSP Cal] Coarse done. Best: {param}={best_val:.4f} -> {best_cost:.1f}dB")
                    
                    # Set up fine search
                    state['search_phase'] = 'fine'
                    step = state['step_size']
                    left = max(state['param_min'], best_val - step * 2)
                    right = min(state['param_max'], best_val + step * 2)
                    state['fine_points'] = [left, best_val - step, best_val, best_val + step, right]
                    state['fine_points'] = sorted(set(state['fine_points']))
                    
                    min_dist = step / 2
                    state['fine_points'] = [p for p in state['fine_points']
                                            if not any(abs(p - e[0]) < min_dist for e in evals)]
                    
                    if state['fine_points']:
                        next_val = state['fine_points'].pop(0)
                        self.set_dsp_param(param, next_val)
                        self.dsp_cal_optimize_status.setText(f"Fine: {param}={next_val:.4f}")
                        return
                    else:
                        state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'fine':
                if state['fine_points']:
                    next_val = state['fine_points'].pop(0)
                    self.set_dsp_param(param, next_val)
                    self.dsp_cal_optimize_status.setText(f"Fine: {param}={next_val:.4f}")
                    return
                else:
                    state['search_phase'] = 'refine'
            
            if state['search_phase'] == 'refine':
                best_val, best_cost = min(evals, key=lambda x: x[1])
                
                if len(evals) >= 10:
                    print(f"[DSP Cal] {param} done: val={best_val:.4f}, cost={best_cost:.1f}dB")
                    self._dsp_cal_finalize_param(param, best_val, best_cost)
                    return
                
                # Try midpoints
                sorted_evals = sorted(evals, key=lambda x: x[0])
                best_idx = next(i for i, e in enumerate(sorted_evals) if e[0] == best_val)
                
                step = state['step_size'] / 2
                left_bound = sorted_evals[best_idx - 1][0] if best_idx > 0 else max(state['param_min'], best_val - step)
                right_bound = sorted_evals[best_idx + 1][0] if best_idx < len(sorted_evals) - 1 else min(state['param_max'], best_val + step)
                
                left_mid = (left_bound + best_val) / 2
                right_mid = (best_val + right_bound) / 2
                
                min_dist = step / 2
                candidates = []
                if abs(left_mid - best_val) >= min_dist and not any(abs(left_mid - e[0]) < min_dist for e in evals):
                    candidates.append(left_mid)
                if abs(right_mid - best_val) >= min_dist and not any(abs(right_mid - e[0]) < min_dist for e in evals):
                    candidates.append(right_mid)
                
                if candidates:
                    next_val = candidates[0]
                    self.set_dsp_param(param, next_val)
                    self.dsp_cal_optimize_status.setText(
                        f"Refine: {param}={next_val:.4f}\n"
                        f"Best: {best_val:.4f} ({best_cost:.1f}dB)"
                    )
                    return
                else:
                    print(f"[DSP Cal] {param} refined: val={best_val:.4f}, cost={best_cost:.1f}dB")
                    self._dsp_cal_finalize_param(param, best_val, best_cost)
    
    def update_status(self):
        now = time.time()
        elapsed = now - self.sample_count_start
        if elapsed >= 1.0:
            self.measured_rate_ch1 = self.sample_count_ch1 / elapsed
            self.measured_rate_ch2 = self.sample_count_ch2 / elapsed
            self.sample_count_ch1 = 0
            self.sample_count_ch2 = 0
            self.sample_count_start = now
        
        rate_str_ch1 = f"{self.measured_rate_ch1:.0f}" if self.measured_rate_ch1 > 0 else "--"
        rate_str_ch2 = f"{self.measured_rate_ch2:.0f}" if self.measured_rate_ch2 > 0 else "--"
        
        self.status_label.setText(
            f"Q(ADC1): {self.total_samples_ch1:,} @ {rate_str_ch1} sps  |  "
            f"I(ADC2): {self.total_samples_ch2:,} @ {rate_str_ch2} sps"
        )
    
    def closeEvent(self, event):
        if self.dc_optimize_timer:
            self.dc_optimize_timer.stop()
        if self.hd2_optimize_timer:
            self.hd2_optimize_timer.stop()
        if self.hd2i_optimize_timer:
            self.hd2i_optimize_timer.stop()
        if self.iq_cal_optimize_timer:
            self.iq_cal_optimize_timer.stop()
        if self.dsp_cal_optimize_timer:
            self.dsp_cal_optimize_timer.stop()
        # Stop reader thread before closing serial port
        if hasattr(self, '_reader'):
            self._reader.stop()
            self._reader.join(timeout=1.0)
        self.ser.close()
        event.accept()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5", help="COMx or /dev/ttyACM0")
    ap.add_argument("--vref", type=float, default=5, help="VREF in volts")
    ap.add_argument("--decim", type=int, default=1, help="plot every Nth sample")
    ap.add_argument("--sample_rate", type=float, default=30000, help="ADC sample rate in Hz")
    ap.add_argument("--send_start", action="store_true", help="send 'S' to start sampling")
    args = ap.parse_args()
    
    app = QApplication(sys.argv)
    viewer = ADCViewer(
        port=args.port,
        sample_rate=args.sample_rate,
        vref=args.vref,
        decim=args.decim,
        send_start=args.send_start
    )
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
