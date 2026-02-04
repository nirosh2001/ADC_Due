import sys
import time
import struct
import argparse
import warnings
import numpy as np
import serial

# Suppress overflow warnings from pyqtgraph internals
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyqtgraph')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSlider, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QGridLayout, QTabWidget, QComboBox
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

MAGIC = b"ADC\n"
HDR_LEN = 10


class ADCViewer(QMainWindow):
    def __init__(self, port="COM5", sample_rate=16000, vref=5.0, decim=1, send_start=False):
        super().__init__()
        self.setWindowTitle("ADC Viewer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Serial setup
        self.ser = serial.Serial(port, baudrate=115200, timeout=0.01)
        self.rx = bytearray()
        
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
        self.buffer_size = 50000
        self.ch1v = np.zeros(self.buffer_size, dtype=np.float32)
        self.ch2v = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0  # Current write position
        self.data_count = 0  # Total samples in buffer (up to buffer_size)
        self.k = 0  # Total samples received
        
        # Display downsampling - max points to actually plot
        self.max_display_points = 2000
        
        # Sample counting
        self.sample_count = 0
        self.sample_count_start = time.time()
        self.measured_rate = 0.0
        
        # FFT update throttle
        self.last_fft_update = 0
        self.fft_update_interval = 0.1  # Update FFT every 100ms
        
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
        self.plot_ch1 = pg.PlotWidget(title="CH1 Time Domain")
        self.plot_ch1.setLabel('left', 'Volts')
        self.plot_ch1.showGrid(x=True, y=True)
        self.plot_ch1.disableAutoRange()  # Manual range for performance
        self.curve_ch1 = self.plot_ch1.plot(pen='y')
        
        self.plot_ch2 = pg.PlotWidget(title="CH2 Time Domain")
        self.plot_ch2.setLabel('left', 'Volts')
        self.plot_ch2.showGrid(x=True, y=True)
        self.plot_ch2.disableAutoRange()
        self.curve_ch2 = self.plot_ch2.plot(pen='c')
        
        # FFT plots
        self.plot_fft1 = pg.PlotWidget(title="CH1 FFT")
        self.plot_fft1.setLabel('left', 'Magnitude (dB)')
        self.plot_fft1.setLabel('bottom', 'Frequency (Hz)')
        self.plot_fft1.showGrid(x=True, y=True)
        self.plot_fft1.disableAutoRange()
        self.curve_fft1 = self.plot_fft1.plot(pen='y')
        
        self.plot_fft2 = pg.PlotWidget(title="CH2 FFT")
        self.plot_fft2.setLabel('left', 'Magnitude (dB)')
        self.plot_fft2.setLabel('bottom', 'Frequency (Hz)')
        self.plot_fft2.showGrid(x=True, y=True)
        self.plot_fft2.disableAutoRange()
        self.curve_fft2 = self.plot_fft2.plot(pen='c')
        
        # Arrange plots in 2x2 grid
        plot_grid = QGridLayout()
        plot_grid.addWidget(self.plot_ch1, 0, 0)
        plot_grid.addWidget(self.plot_fft1, 0, 1)
        plot_grid.addWidget(self.plot_ch2, 1, 0)
        plot_grid.addWidget(self.plot_fft2, 1, 1)
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
        self.btn_start.clicked.connect(lambda: self.ser.write(b"S"))
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: lightcoral; font-weight: bold;")
        self.btn_stop.clicked.connect(lambda: self.ser.write(b"P"))
        
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
        self.spin_maxfreq.setValue(100)
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
        self.btn_dvga_sweep.clicked.connect(lambda: self.ser.write(b"G"))
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
        
        # --- Demodulator Calibration Control ---
        demod_group = QGroupBox("Demodulator Calibration")
        demod_layout = QGridLayout(demod_group)
        
        # Define all demodulator registers: (name, reg_id, description, max_val, default_val)
        self.demod_regs = [
            ("DCOI", "0", "I-Ch DC Offset", 255, 128),
            ("DCOQ", "1", "Q-Ch DC Offset", 255, 128),
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
        ]
        
        self.demod_spins = {}
        
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
        
        main_layout.addWidget(control_panel, stretch=1)
    
    def send_dvga(self):
        val = self.spin_dvga.value()
        self.ser.write(f"V{val}".encode())
        print(f"Sent DVGA: {val}")
    
    def send_att(self):
        val = self.spin_att.value()
        self.ser.write(f"A{val}".encode())
        print(f"Sent ATT: {val}")
    
    def send_demod_reg(self, name, reg_id):
        val = self.demod_spins[name].value()
        cmd = f"D{reg_id}{val}"
        self.ser.write(cmd.encode())
        print(f"Sent {name}: {val}")
    
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)  # 50 Hz update
    
    def update(self):
        self.read_serial()
        self.update_plots()
        self.update_status()
    
    def read_serial(self):
        waiting = self.ser.in_waiting
        if waiting > 0:
            self.rx.extend(self.ser.read(waiting))
        
        # Trim buffer if too large
        if len(self.rx) > 30000:
            self.rx[:] = self.rx[-10000:]
        
        # Check for text messages (lines not starting with 'ADC\n')
        while b'\n' in self.rx:
            newline_pos = self.rx.index(b'\n')
            # Check if this is NOT an ADC packet header
            if newline_pos >= 3 and self.rx[newline_pos-3:newline_pos+1] == MAGIC:
                break
            # Extract the line as text
            line_bytes = bytes(self.rx[:newline_pos])
            del self.rx[:newline_pos + 1]
            try:
                line_text = line_bytes.decode('utf-8', errors='ignore').strip()
                if line_text and not line_text.startswith('ADC'):
                    print(f"[Arduino] {line_text}")
                    self.msg_label.setText(line_text)
            except:
                pass
        
        # Parse ADC packets
        while True:
            m = self.rx.find(MAGIC)
            if m < 0:
                break
            if m > 0:
                del self.rx[:m]
            if len(self.rx) < HDR_LEN:
                break
            
            count = struct.unpack_from("<H", self.rx, 8)[0]
            payload_len = count * 4
            
            if len(self.rx) < HDR_LEN + payload_len:
                break
            
            payload = bytes(self.rx[HDR_LEN:HDR_LEN + payload_len])
            del self.rx[:HDR_LEN + payload_len]
            
            u16 = np.frombuffer(payload, dtype="<u2")
            i16 = u16.view("<i2")
            pairs = i16.reshape(-1, 2)
            
            a1 = pairs[:, 0].astype(np.float32) * self.lsb
            a2 = pairs[:, 1].astype(np.float32) * self.lsb
            
            self.sample_count += len(pairs)
            
            # Decimate
            d = max(1, self.decim)
            a1 = a1[::d]
            a2 = a2[::d]
            
            # Write to ring buffer (numpy array)
            n = len(a1)
            for i in range(n):
                self.ch1v[self.write_idx] = a1[i]
                self.ch2v[self.write_idx] = a2[i]
                self.write_idx = (self.write_idx + 1) % self.buffer_size
                self.data_count = min(self.data_count + 1, self.buffer_size)
                self.k += 1
    
    def update_plots(self):
        if self.data_count < 2:
            return
        
        win = min(self.spin_xwindow.value(), self.data_count)
        
        # Extract data from ring buffer (last 'win' samples)
        if win <= self.write_idx:
            y1 = self.ch1v[self.write_idx - win:self.write_idx].copy()
            y2 = self.ch2v[self.write_idx - win:self.write_idx].copy()
        else:
            # Wrap around
            part1_len = win - self.write_idx
            y1 = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:self.write_idx]])
            y2 = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:self.write_idx]])
        
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
        
        fft_n = min(int(self.combo_fft.currentText()), self.data_count)
        if fft_n >= 16:
            # Update FFT info
            obs_time = fft_n / self.fs_eff
            freq_res = self.fs_eff / fft_n
            self.fft_info_label.setText(f"T_obs: {obs_time*1000:.1f} ms  |  Δf: {freq_res:.2f} Hz")
            
            # Extract FFT data from ring buffer
            if fft_n <= self.write_idx:
                y1_fft = self.ch1v[self.write_idx - fft_n:self.write_idx]
                y2_fft = self.ch2v[self.write_idx - fft_n:self.write_idx]
            else:
                part1_len = fft_n - self.write_idx
                y1_fft = np.concatenate([self.ch1v[self.buffer_size - part1_len:], self.ch1v[:self.write_idx]])
                y2_fft = np.concatenate([self.ch2v[self.buffer_size - part1_len:], self.ch2v[:self.write_idx]])
            
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
            
            # Auto Y range for FFT (visible range only) - use explicit floats to avoid overflow
            visible_mask = freqs <= fmax
            if np.any(visible_mask):
                vis_mag1 = mag1_db[visible_mask]
                vis_mag2 = mag2_db[visible_mask]
                if len(vis_mag1) > 0:
                    ymin1 = float(max(-120.0, np.min(vis_mag1) - 10))
                    ymax1 = float(min(100.0, np.max(vis_mag1) + 10))
                    self.plot_fft1.setYRange(ymin1, ymax1)
                if len(vis_mag2) > 0:
                    ymin2 = float(max(-120.0, np.min(vis_mag2) - 10))
                    ymax2 = float(min(100.0, np.max(vis_mag2) + 10))
                    self.plot_fft2.setYRange(ymin2, ymax2)
    
    def update_status(self):
        now = time.time()
        elapsed = now - self.sample_count_start
        if elapsed >= 1.0:
            self.measured_rate = self.sample_count / elapsed
            self.sample_count = 0
            self.sample_count_start = now
        
        rate_str = f"{self.measured_rate:.0f}" if self.measured_rate > 0 else "--"
        self.status_label.setText(f"Samples: {self.k * self.decim:,}  |  Rate: {rate_str} sps")
    
    def closeEvent(self, event):
        self.ser.close()
        event.accept()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5", help="COMx or /dev/ttyACM0")
    ap.add_argument("--vref", type=float, default=5, help="VREF in volts")
    ap.add_argument("--decim", type=int, default=1, help="plot every Nth sample")
    ap.add_argument("--sample_rate", type=float, default=16000, help="ADC sample rate in Hz")
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
