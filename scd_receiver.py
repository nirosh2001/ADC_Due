"""
SCD Receiver Module
TCP server that receives SCD images from SCDTransmitter over Wi-Fi.

Author: Auto-generated
Date: March 4, 2026
"""

import socket
import struct
import json
import os
import threading
import logging
from datetime import datetime
import numpy as np
import cv2


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SCDReceiver:
    """
    TCP server that receives SCD images from remote clients.
    
    Features:
    - Multi-threaded server accepting multiple clients
    - Robust packet parsing with proper length handling
    - Automatic PNG decoding and saving
    - Graceful error handling
    """
    
    def __init__(self):
        """Initialize the SCD receiver."""
        self.server_socket = None
        self.running = False
        self.server_thread = None
        self.client_threads = []
        self.output_folder = "received_frames"
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        logger.info("SCDReceiver initialized")
    
    def start_server(self, host="0.0.0.0", port=5000):
        """
        Start the TCP server.
        
        Args:
            host (str): Host address to bind to (0.0.0.0 = all interfaces)
            port (int): Port number to listen on
        """
        if self.running:
            logger.warning("Server is already running")
            return
        
        try:
            # Create TCP socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to address and port
            self.server_socket.bind((host, port))
            
            # Start listening (backlog of 5 connections)
            self.server_socket.listen(5)
            
            self.running = True
            
            logger.info(f"Server started on {host}:{port}")
            logger.info(f"Waiting for connections...")
            
            # Start accept loop in separate thread
            self.server_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.server_thread.start()
            
        except OSError as e:
            logger.error(f"Failed to start server: {e}")
            self.running = False
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
    
    def _accept_loop(self):
        """Main accept loop that handles incoming client connections."""
        while self.running:
            try:
                # Accept incoming connection
                conn, addr = self.server_socket.accept()
                logger.info(f"New connection from {addr[0]}:{addr[1]}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr),
                    daemon=True
                )
                client_thread.start()
                self.client_threads.append(client_thread)
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
                break
    
    def stop_server(self):
        """Stop the server and close all connections."""
        if not self.running:
            return
        
        logger.info("Stopping server...")
        self.running = False
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None
        
        logger.info("Server stopped")
    
    def receive_exact_bytes(self, conn, n):
        """
        Receive exactly n bytes from the connection.
        
        Handles partial receives by looping until all bytes are received.
        
        Args:
            conn: Socket connection
            n (int): Number of bytes to receive
            
        Returns:
            bytes: Exactly n bytes, or None if connection closed
        """
        data = bytearray()
        while len(data) < n:
            try:
                chunk = conn.recv(n - len(data))
                if not chunk:
                    # Connection closed
                    return None
                data.extend(chunk)
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                return None
        return bytes(data)
    
    def handle_client(self, conn, addr):
        """
        Handle a connected client and receive SCD frames.
        
        Args:
            conn: Socket connection
            addr: Client address tuple (host, port)
        """
        client_id = f"{addr[0]}:{addr[1]}"
        logger.info(f"Client handler started for {client_id}")
        
        try:
            # Continuously receive frames from this client
            while self.running:
                # 1. Receive header length (4 bytes)
                header_length_bytes = self.receive_exact_bytes(conn, 4)
                if header_length_bytes is None:
                    logger.info(f"Client {client_id} disconnected")
                    break
                
                header_length = struct.unpack('!I', header_length_bytes)[0]
                
                if header_length > 10000:  # Sanity check (10KB max for header)
                    logger.error(f"Invalid header length: {header_length}")
                    break
                
                # 2. Receive header JSON
                header_json_bytes = self.receive_exact_bytes(conn, header_length)
                if header_json_bytes is None:
                    logger.warning(f"Failed to receive header from {client_id}")
                    break
                
                # Parse header JSON
                try:
                    header = json.loads(header_json_bytes.decode('utf-8'))
                except Exception as e:
                    logger.error(f"Failed to parse header JSON: {e}")
                    continue
                
                source = header.get("source", "UNKNOWN")
                frame_id = header.get("frame_id", 0)
                timestamp = header.get("timestamp", "")
                shape = header.get("shape", [0, 0])
                
                # 3. Receive image length (4 bytes)
                image_length_bytes = self.receive_exact_bytes(conn, 4)
                if image_length_bytes is None:
                    logger.warning(f"Failed to receive image length from {client_id}")
                    break
                
                image_length = struct.unpack('!I', image_length_bytes)[0]
                
                if image_length > 50000000:  # Sanity check (50MB max)
                    logger.error(f"Invalid image length: {image_length}")
                    break
                
                # 4. Receive PNG image bytes
                image_bytes = self.receive_exact_bytes(conn, image_length)
                if image_bytes is None:
                    logger.warning(f"Failed to receive image data from {client_id}")
                    break
                
                # Decode PNG image
                try:
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    
                    # Decode PNG
                    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        logger.error(f"Failed to decode PNG image (frame {frame_id})")
                        continue
                    
                    # Save image to disk
                    filename = f"{source}_frame_{frame_id}.png"
                    filepath = os.path.join(self.output_folder, filename)
                    
                    cv2.imwrite(filepath, img)
                    
                    # Log success
                    logger.info(f"Received frame {frame_id} from {source}, saved successfully.")
                    
                    # Print to console (as required)
                    print(f"Received frame {frame_id} from {source}, saved successfully.")
                    
                except Exception as e:
                    logger.error(f"Error processing image (frame {frame_id}): {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        
        finally:
            # Close connection
            try:
                conn.close()
            except Exception:
                pass
            logger.info(f"Client {client_id} handler terminated")


def main():
    """Main function to start the SCD receiver server."""
    # Create receiver
    receiver = SCDReceiver()
    
    # Start server on all interfaces, port 5000
    receiver.start_server(host="0.0.0.0", port=5000)
    
    # Keep running until Ctrl+C
    try:
        logger.info("Server is running. Press Ctrl+C to stop.")
        # Keep main thread alive
        while receiver.running:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        logger.info("\nReceived Ctrl+C, shutting down...")
        receiver.stop_server()


if __name__ == "__main__":
    main()
