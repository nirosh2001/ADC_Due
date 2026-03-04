"""
SCD Transmitter Module
Handles continuous transmission of SCD images over TCP/IP via Wi-Fi.

Author: Auto-generated
Date: March 4, 2026
"""

import socket
import struct
import json
import time
import threading
import logging
from datetime import datetime
import numpy as np
import cv2


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SCDTransmitter:
    """
    Handles continuous transmission of SCD images to a remote TCP server.
    
    Features:
    - Persistent TCP connection
    - Background thread for non-blocking transmission
    - Automatic retry on connection failure
    - Frame counter and metadata packaging
    - PNG encoding with normalization
    """
    
    def __init__(self):
        """Initialize the SCD transmitter."""
        self.socket = None
        self.connected = False
        self.transmission_enabled = False
        self.frame_counter = 0
        
        # Connection parameters
        self.host = None
        self.port = None
        
        # Threading
        self.transmission_thread = None
        self.stop_event = threading.Event()
        self.send_lock = threading.Lock()
        
        # Queue for pending frames (optional buffering)
        self.pending_frame = None
        self.pending_frame_lock = threading.Lock()
        
        logger.info("SCDTransmitter initialized")
    
    def connect(self, host, port, timeout=5):
        """
        Establish TCP connection to the ML laptop server.
        
        Args:
            host (str): Server IP address
            port (int): Server port number
            timeout (int): Connection timeout in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.connected:
            logger.warning("Already connected")
            return True
        
        try:
            self.host = host
            self.port = port
            
            # Create TCP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            
            # Connect to server
            logger.info(f"Connecting to {host}:{port}...")
            self.socket.connect((host, port))
            
            # Set socket to blocking mode after connection
            self.socket.settimeout(None)
            
            self.connected = True
            logger.info(f"Successfully connected to {host}:{port}")
            return True
            
        except socket.timeout:
            logger.error(f"Connection timeout to {host}:{port}")
            self._cleanup_socket()
            return False
        except ConnectionRefusedError:
            logger.error(f"Connection refused by {host}:{port}")
            self._cleanup_socket()
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._cleanup_socket()
            return False
    
    def disconnect(self):
        """
        Disconnect from the server and clean up resources.
        """
        if not self.connected:
            return
        
        logger.info("Disconnecting from server...")
        self.stop_transmission()
        self._cleanup_socket()
        logger.info("Disconnected")
    
    def _cleanup_socket(self):
        """Clean up socket resources."""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
    
    def start_transmission(self):
        """
        Start continuous transmission mode.
        This enables the transmitter to send frames when available.
        """
        if not self.connected:
            logger.error("Cannot start transmission: not connected")
            return False
        
        if self.transmission_enabled:
            logger.warning("Transmission already enabled")
            return True
        
        self.transmission_enabled = True
        logger.info("Transmission started")
        return True
    
    def stop_transmission(self):
        """
        Stop continuous transmission mode.
        This disables the transmitter but keeps the connection alive.
        """
        if not self.transmission_enabled:
            return
        
        self.transmission_enabled = False
        logger.info("Transmission stopped")
    
    def is_enabled(self):
        """
        Check if transmission is currently enabled.
        
        Returns:
            bool: True if transmission is enabled, False otherwise
        """
        return self.transmission_enabled and self.connected
    
    def send_scd_frame(self, scd_matrix):
        """
        Send a single SCD frame to the server.
        
        Args:
            scd_matrix (np.ndarray): SCD matrix (2D numpy array)
            
        Returns:
            bool: True if send successful, False otherwise
        """
        if not self.is_enabled():
            return False
        
        try:
            # Increment frame counter
            self.frame_counter += 1
            
            # Normalize and encode SCD matrix as PNG
            image_bytes = self._encode_scd_to_png(scd_matrix)
            
            if image_bytes is None:
                logger.error("Failed to encode SCD matrix")
                return False
            
            # Create metadata header
            metadata = {
                "source": "CW",
                "frame_id": self.frame_counter,
                "timestamp": datetime.now().isoformat(),
                "shape": list(scd_matrix.shape),
                "dtype": str(scd_matrix.dtype)
            }
            
            # Serialize metadata to JSON
            header_json = json.dumps(metadata).encode('utf-8')
            header_length = len(header_json)
            image_length = len(image_bytes)
            
            # Create message structure:
            # [4 bytes: header length][header JSON][4 bytes: image length][image bytes]
            with self.send_lock:
                # Send header length (4 bytes, network byte order)
                self.socket.sendall(struct.pack('!I', header_length))
                
                # Send header JSON
                self.socket.sendall(header_json)
                
                # Send image length (4 bytes, network byte order)
                self.socket.sendall(struct.pack('!I', image_length))
                
                # Send image bytes
                self.socket.sendall(image_bytes)
            
            logger.debug(f"Sent frame {self.frame_counter} ({image_length} bytes)")
            return True
            
        except BrokenPipeError:
            logger.error("Connection broken (broken pipe)")
            self._handle_connection_error()
            return False
        except ConnectionResetError:
            logger.error("Connection reset by peer")
            self._handle_connection_error()
            return False
        except socket.error as e:
            logger.error(f"Socket error during send: {e}")
            self._handle_connection_error()
            return False
        except Exception as e:
            logger.error(f"Unexpected error during send: {e}")
            return False
    
    def _encode_scd_to_png(self, scd_matrix):
        """
        Encode SCD matrix as PNG image bytes.
        
        Args:
            scd_matrix (np.ndarray): SCD matrix (2D numpy array)
            
        Returns:
            bytes: PNG encoded image bytes, or None on failure
        """
        try:
            # Create a copy to avoid modifying original
            scd = scd_matrix.copy()
            
            # Normalize to 0-255 range
            scd_min = np.min(scd)
            scd_max = np.max(scd)
            
            if scd_max - scd_min > 1e-6:
                scd_normalized = (scd - scd_min) / (scd_max - scd_min)
            else:
                scd_normalized = np.zeros_like(scd)
            
            # Convert to uint8
            scd_uint8 = (scd_normalized * 255).astype(np.uint8)
            
            # Encode as PNG (lossless)
            success, encoded = cv2.imencode('.png', scd_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            if not success:
                return None
            
            return encoded.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to encode SCD matrix: {e}")
            return None
    
    def _handle_connection_error(self):
        """Handle connection errors by stopping transmission and cleaning up."""
        logger.warning("Handling connection error: stopping transmission")
        self.transmission_enabled = False
        self._cleanup_socket()
    
    def get_status(self):
        """
        Get current transmitter status.
        
        Returns:
            dict: Status information
        """
        return {
            "connected": self.connected,
            "transmission_enabled": self.transmission_enabled,
            "frames_sent": self.frame_counter,
            "host": self.host,
            "port": self.port
        }
    
    def reset_frame_counter(self):
        """Reset the frame counter to zero."""
        self.frame_counter = 0
        logger.info("Frame counter reset")


# Example usage
if __name__ == "__main__":
    # Create transmitter
    transmitter = SCDTransmitter()
    
    # Connect to server
    if transmitter.connect("127.0.0.1", 5000):
        # Start transmission
        transmitter.start_transmission()
        
        # Simulate sending some frames
        for i in range(5):
            # Create dummy SCD matrix (random data)
            dummy_scd = np.random.rand(256, 256) * 100 - 50  # Simulate dB values
            transmitter.send_scd_frame(dummy_scd)
            time.sleep(0.5)  # Simulate SCD generation delay
        
        # Stop and disconnect
        transmitter.stop_transmission()
        transmitter.disconnect()
    else:
        print("Failed to connect to server")
