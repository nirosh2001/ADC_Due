# print_ads8867_dual.py
# Reads the Arduino Due binary stream and prints ADC1/ADC2 values.
#
# Stream format (as sent by the Due code):
#   Header: b'ADC\n' + uint32 seq_end + uint16 count
#   Payload: count * (uint16 adc1, uint16 adc2) little-endian
#
# Run (no args):
#   python print_ads8867_dual.py
#
# Install once:
#   pip install pyserial

import struct
import sys
from serial.tools import list_ports
import serial

FORCE_PORT = None  # e.g. "COM7" (Windows) or "/dev/ttyACM0" (Linux). Leave None to auto-pick.

MAGIC = b"ADC\n"
HDR_LEN = 10  # 4 + 4 + 2


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
            # keep last 3 bytes in case MAGIC splits across reads
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
    print("Printing ADC1, ADC2 values (Ctrl+C to stop)\n")

    ser = serial.Serial(port=port, baudrate=115200, timeout=0.1)

    buf = bytearray()

    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)

                for seq_end, count, payload in parse_frames(buf):
                    # payload: count pairs of uint16
                    # Unpack efficiently
                    # Each pair: adc1, adc2
                    for i in range(count):
                        off = i * 4
                        adc1 = payload[off] | (payload[off + 1] << 8)
                        adc2 = payload[off + 2] | (payload[off + 3] << 8)
                        print(f"{adc1}\t{adc2}")

            # If no data arrives, loop again (timeout keeps it responsive)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
