# ALchemist Qt Connection Panel

A reusable Qt/PySide6 widget for connecting external applications to ALchemist optimization sessions.

## Overview

This template provides a plug-and-play connection panel that handles:
- Session ID input and validation
- Connection/disconnection to ALchemist API
- Automatic state polling and updates
- Signal-based event handling
- Clean UI with status indicators

## Requirements

```bash
pip install PySide6 requests
```

## Quick Start

### 1. Copy the Template

Copy `alchemist_connector.py` into your Qt application directory.

### 2. Basic Integration

```python
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from alchemist_connector import AlchemistConnector

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Add ALchemist connector
        self.alchemist = AlchemistConnector()
        layout.addWidget(self.alchemist)
        
        # Connect to signals
        self.alchemist.connected.connect(self.on_alchemist_connected)
        self.alchemist.state_updated.connect(self.on_state_updated)
        
        # Add your application-specific widgets here
        # ...
        
        self.setCentralWidget(central_widget)
        
    def on_alchemist_connected(self, session_id, info):
        print(f"Connected to ALchemist session: {session_id}")
        # Enable your optimization controls here
        
    def on_state_updated(self, state):
        # Update your UI with latest optimization state
        n_experiments = state['n_experiments']
        model_trained = state['model_trained']
        # ...

if __name__ == "__main__":
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec()
```

### 3. Using the API in Your Application

Once connected, use the session ID to interact with ALchemist:

```python
import requests

# Get session ID from connector
session_id = self.alchemist.get_session_id()
api_url = "http://localhost:8000/api/v1"

# Add experiment data
data = {
    "inputs": {"temperature": 350.0, "flow_rate": 5.0},
    "output": 0.85
}
response = requests.post(
    f"{api_url}/sessions/{session_id}/experiments",
    params={"auto_train": True},
    json=data
)

# Get next suggestion
suggestion_request = {
    "strategy": "qEI",
    "goal": "maximize",
    "n_suggestions": 1
}
response = requests.post(
    f"{api_url}/sessions/{session_id}/acquisition/suggest",
    json=suggestion_request
)
next_point = response.json()["suggestions"][0]
```

## API Reference

### AlchemistConnector

**Constructor:**
```python
AlchemistConnector(api_url="http://localhost:8000/api/v1", parent=None)
```

**Signals:**
- `connected(str, dict)` - Emitted when successfully connected
  - Args: session_id, session_info dict
- `disconnected()` - Emitted when disconnected
- `error_occurred(str)` - Emitted on error
  - Args: error_message
- `state_updated(dict)` - Emitted when session state updates (every 5s by default)
  - Args: state dict with keys: `n_variables`, `n_experiments`, `model_trained`, `last_suggestion`

**Methods:**
- `connect_to_session(session_id: str) -> bool` - Connect to session
- `disconnect()` - Disconnect from current session
- `get_session_id() -> Optional[str]` - Get current session ID
- `get_session_info() -> Optional[dict]` - Get session information
- `is_session_connected() -> bool` - Check connection status
- `set_poll_interval(milliseconds: int)` - Set state polling interval (default: 5000ms)

## Workflow Example

### User Workflow

1. **Setup in ALchemist Web UI**
   - User opens http://localhost:8000
   - Creates new session
   - Defines variables (temperature, flow_rate, etc.)
   - Copies session ID: `abc-123-def-456`

2. **Connect from External Application**
   - User pastes session ID into your Qt app
   - Clicks "Connect"
   - Your app validates and connects
   - Web UI automatically enters monitoring mode

3. **Autonomous Operation**
   - Your app controls the optimization loop
   - Reads hardware sensors
   - Preprocesses data
   - Adds experiments via API
   - Gets suggestions via API
   - Web dashboard displays progress in real-time

### Code Example: Complete Optimization Loop

```python
class ReactorController(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup ALchemist connector
        self.alchemist = AlchemistConnector()
        self.alchemist.connected.connect(self.start_optimization)
        
        # Your hardware interface
        self.reactor = ReactorHardware()  # Your custom class
        
    def start_optimization(self, session_id, info):
        """Called when connected to ALchemist."""
        print(f"Starting optimization with {info['variable_count']} variables")
        
        # Start optimization loop
        self.run_optimization_loop()
        
    def run_optimization_loop(self):
        """Main optimization loop."""
        session_id = self.alchemist.get_session_id()
        api_url = self.alchemist.api_url
        
        while not self.stop_requested:
            # Get next experiment suggestion
            response = requests.post(
                f"{api_url}/sessions/{session_id}/acquisition/suggest",
                json={"strategy": "qEI", "goal": "maximize", "n_suggestions": 1}
            )
            next_point = response.json()["suggestions"][0]
            
            # Execute experiment on hardware
            self.reactor.set_conditions(
                temperature=next_point["temperature"],
                flow_rate=next_point["flow_rate"]
            )
            self.reactor.wait_for_steady_state()
            result = self.reactor.measure_output()
            
            # Send result back to ALchemist
            requests.post(
                f"{api_url}/sessions/{session_id}/experiments",
                params={"auto_train": True},
                json={"inputs": next_point, "output": result}
            )
            
            # State update signal will fire automatically
```

## Customization

### Styling

The connector uses basic Qt widgets. Customize appearance by:

```python
# Change colors
connector.status_label.setStyleSheet("color: blue; font-size: 14px;")

# Resize components
connector.info_text.setMaximumHeight(200)

# Use your app's theme
connector.setStyleSheet(your_stylesheet)
```

### Extended Functionality

Add custom features by subclassing:

```python
class CustomConnector(AlchemistConnector):
    def __init__(self, api_url, parent=None):
        super().__init__(api_url, parent)
        
        # Add emergency stop button
        self.stop_btn = QPushButton("Emergency Stop")
        self.stop_btn.clicked.connect(self.emergency_stop)
        self.layout().addWidget(self.stop_btn)
        
    def emergency_stop(self):
        # Your emergency stop logic
        self.disconnect()
        # Signal hardware to stop...
```

## Troubleshooting

### Connection Fails

1. Verify ALchemist API is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check session ID is correct (copy from web UI)

3. Ensure API URL matches your deployment

### State Not Updating

- Default poll interval is 5 seconds
- Check console for polling errors
- Increase poll interval if needed:
  ```python
  connector.set_poll_interval(10000)  # 10 seconds
  ```

### Session Expired

- Sessions expire after 24 hours by default
- Create new session in web UI
- Consider implementing auto-reconnect

## Examples

See the complete example applications:
- `examples/middle_man_apps/qt_simple_optimizer/` - Basic optimization loop
- `examples/middle_man_apps/qt_reactor_controller/` - Reactor control example

## Support

For questions or issues:
- GitHub Issues: https://github.com/NREL/ALchemist/issues
- Email: ccoatney@nrel.gov

## License

Same as ALchemist main project - see LICENSE file.
