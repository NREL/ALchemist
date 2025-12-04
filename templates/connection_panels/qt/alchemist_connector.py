"""
ALchemist Connection Panel - Qt/PySide6 Template

A reusable Qt widget for connecting to ALchemist sessions from external applications.
Copy this file into your Qt application and integrate it into your UI.

Requirements:
    pip install PySide6 requests

Usage:
    from alchemist_connector import AlchemistConnector
    
    # Create and add to your layout
    connector = AlchemistConnector(api_url="http://localhost:8000/api/v1")
    layout.addWidget(connector)
    
    # Connect to signals
    connector.connected.connect(on_connected)
    connector.disconnected.connect(on_disconnected)
    connector.error_occurred.connect(on_error)

Author: ALchemist Team
Date: December 2025
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QGroupBox, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
import requests
from typing import Optional, Dict, Any
from datetime import datetime


class AlchemistConnector(QWidget):
    """
    Reusable Qt widget for connecting to ALchemist optimization sessions.
    
    Signals:
        connected(str, dict): Emitted when successfully connected (session_id, session_info)
        disconnected(): Emitted when disconnected from session
        error_occurred(str): Emitted when an error occurs (error_message)
        state_updated(dict): Emitted when session state is updated (state_data)
    """
    
    # Qt Signals
    connected = Signal(str, dict)  # session_id, session_info
    disconnected = Signal()
    error_occurred = Signal(str)  # error_message
    state_updated = Signal(dict)  # state_data
    
    def __init__(self, api_url: str = "http://localhost:8000/api/v1", 
                 client_name: str = "QtController",
                 auto_lock: bool = True,
                 parent=None):
        """
        Initialize the ALchemist connector widget.
        
        Args:
            api_url: Base URL for ALchemist API (default: http://localhost:8000/api/v1)
            client_name: Name to identify this client when locking sessions
            auto_lock: Automatically lock session on connect and unlock on disconnect
            parent: Parent Qt widget
        """
        super().__init__(parent)
        
        self.api_url = api_url.rstrip('/')
        self.client_name = client_name
        self.auto_lock = auto_lock
        self.session_id: Optional[str] = None
        self.is_connected: bool = False
        self.session_info: Optional[Dict[str, Any]] = None
        self.lock_token: Optional[str] = None
        
        # Polling timer for state updates
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_session_state)
        self.poll_interval_ms = 5000  # 5 seconds
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Group box for connection controls
        group = QGroupBox("ALchemist Connection")
        group_layout = QVBoxLayout()
        
        # API URL display
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("API URL:"))
        self.url_label = QLabel(self.api_url)
        self.url_label.setStyleSheet("color: gray; font-family: monospace;")
        url_layout.addWidget(self.url_label)
        url_layout.addStretch()
        group_layout.addLayout(url_layout)
        
        # Session ID input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Session ID:"))
        self.session_input = QLineEdit()
        self.session_input.setPlaceholderText("Paste session ID from ALchemist web UI...")
        self.session_input.setFont(QFont("monospace"))
        input_layout.addWidget(self.session_input)
        group_layout.addLayout(input_layout)
        
        # Connect/Disconnect buttons
        button_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        self.disconnect_btn.setEnabled(False)
        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.disconnect_btn)
        button_layout.addStretch()
        group_layout.addLayout(button_layout)
        
        # Status display
        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        group_layout.addWidget(self.status_label)
        
        # Session info display
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setPlaceholderText("Session information will appear here...")
        group_layout.addWidget(self.info_text)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
    def _on_connect_clicked(self):
        """Handle connect button click."""
        session_id = self.session_input.text().strip()
        if not session_id:
            self._set_error("Please enter a session ID")
            return
            
        self.connect_to_session(session_id)
        
    def _on_disconnect_clicked(self):
        """Handle disconnect button click."""
        self.disconnect()
        
    def connect_to_session(self, session_id: str) -> bool:
        """
        Connect to an existing ALchemist session.
        
        Args:
            session_id: The session ID to connect to
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Validate session exists
            response = requests.get(f"{self.api_url}/sessions/{session_id}", timeout=5)
            response.raise_for_status()
            
            self.session_info = response.json()
            self.session_id = session_id
            
            # Lock the session if auto_lock is enabled
            if self.auto_lock:
                try:
                    lock_response = requests.post(
                        f"{self.api_url}/sessions/{session_id}/lock",
                        json={
                            "locked_by": self.client_name,
                            "client_id": f"qt-{id(self)}"
                        },
                        timeout=5
                    )
                    lock_response.raise_for_status()
                    lock_data = lock_response.json()
                    self.lock_token = lock_data.get("lock_token")
                    print(f"✓ Session locked by {self.client_name}")
                except Exception as e:
                    print(f"⚠ Warning: Could not lock session: {e}")
                    # Continue anyway - locking is optional
            
            self.is_connected = True
            
            # Update UI
            self.session_input.setEnabled(False)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self._set_status(f"Connected to session: {session_id[:8]}...", "green")
            self._update_info_display()
            
            # Start polling for state updates
            self.poll_timer.start(self.poll_interval_ms)
            
            # Emit signal
            self.connected.emit(session_id, self.session_info)
            
            return True
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect: {str(e)}"
            self._set_error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
            
    def disconnect(self):
        """Disconnect from current session."""
        if not self.is_connected:
            return
        
        # Unlock the session if we have a token
        if self.lock_token and self.session_id:
            try:
                unlock_response = requests.delete(
                    f"{self.api_url}/sessions/{self.session_id}/lock",
                    params={"lock_token": self.lock_token},
                    timeout=5
                )
                unlock_response.raise_for_status()
                print(f"✓ Session unlocked")
            except Exception as e:
                print(f"⚠ Warning: Could not unlock session: {e}")
                # Continue anyway - disconnect should always work
            
        # Stop polling
        self.poll_timer.stop()
        
        # Reset state
        session_id_copy = self.session_id  # Keep for logging
        self.session_id = None
        self.session_info = None
        self.lock_token = None
        self.is_connected = False
        
        # Update UI
        self.session_input.setEnabled(True)
        self.session_input.clear()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self._set_status("Disconnected", "gray")
        self.info_text.clear()
        
        # Emit signal
        self.disconnected.emit()
        
    def _poll_session_state(self):
        """Poll session state for updates (called by timer)."""
        if not self.is_connected or not self.session_id:
            return
            
        try:
            response = requests.get(
                f"{self.api_url}/sessions/{self.session_id}/state",
                timeout=5
            )
            response.raise_for_status()
            state = response.json()
            
            # Update info display
            self._update_info_display(state)
            
            # Emit signal for external handlers
            self.state_updated.emit(state)
            
        except requests.exceptions.RequestException as e:
            # Don't disconnect on polling errors, just log
            print(f"Warning: Failed to poll session state: {e}")
            
    def _update_info_display(self, state: Optional[Dict] = None):
        """Update the session info text display."""
        if not self.session_info:
            return
            
        # Extract values from nested structure
        n_variables = self.session_info.get('search_space', {}).get('n_variables', 0)
        n_experiments = self.session_info.get('data', {}).get('n_experiments', 0)
        
        lines = [
            f"Session ID: {self.session_id}",
            f"Created: {self.session_info.get('created_at', 'Unknown')}",
            f"Variables: {n_variables}",
            f"Experiments: {n_experiments}",
        ]
        
        if state:
            lines.extend([
                "",
                "Current State:",
                f"  Model Trained: {'Yes' if state.get('model_trained') else 'No'}",
                f"  Last Updated: {datetime.now().strftime('%H:%M:%S')}"
            ])
            
        self.info_text.setPlainText("\n".join(lines))
        
    def _set_status(self, message: str, color: str = "black"):
        """Update status label with color."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
    def _set_error(self, message: str):
        """Display error message."""
        self._set_status(f"Error: {message}", "red")
        
    # Public API methods
    
    def get_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.session_id
        
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get the current session information."""
        return self.session_info
        
    def is_session_connected(self) -> bool:
        """Check if currently connected to a session."""
        return self.is_connected
        
    def set_poll_interval(self, milliseconds: int):
        """
        Set the polling interval for state updates.
        
        Args:
            milliseconds: Polling interval in milliseconds (default: 5000)
        """
        self.poll_interval_ms = milliseconds
        if self.poll_timer.isActive():
            self.poll_timer.setInterval(milliseconds)


# Example usage
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create connector widget
    connector = AlchemistConnector()
    
    # Connect to signals
    def on_connected(session_id, info):
        print(f"✓ Connected to session: {session_id}")
        n_vars = info.get('search_space', {}).get('n_variables', 0)
        n_exps = info.get('data', {}).get('n_experiments', 0)
        print(f"  Variables: {n_vars}")
        print(f"  Experiments: {n_exps}")
        
    def on_disconnected():
        print("✗ Disconnected from session")
        
    def on_error(error_msg):
        print(f"✗ Error: {error_msg}")
        
    def on_state_updated(state):
        print(f"↻ State update: {state['n_experiments']} experiments, "
              f"model {'trained' if state['model_trained'] else 'not trained'}")
    
    connector.connected.connect(on_connected)
    connector.disconnected.connect(on_disconnected)
    connector.error_occurred.connect(on_error)
    connector.state_updated.connect(on_state_updated)
    
    connector.show()
    
    sys.exit(app.exec())
