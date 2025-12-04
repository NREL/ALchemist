# Quick Start: External Application Control

**Goal:** Control ALchemist optimization from your own application while monitoring progress in the web dashboard.

---

## 5-Minute Setup

### 1. Setup Optimization in Web UI (2 minutes)

```bash
# Start ALchemist
alchemist-web
```

Open browser → http://localhost:8000

1. Click **"New Session"**
2. Click **"Edit Info"** to add metadata (optional)
3. Go to **Variables** tab → Add your variables
4. Go to **Experiments** tab → Add initial data (optional)
5. Click the **📋 copy button** next to session ID in header
   - Session ID now in clipboard: `abc-123-def-456`

### 2. Connect from Your Qt Application (3 minutes)

**Copy the template into your project:**
```bash
cp templates/connection_panels/qt/alchemist_connector.py your_project/
```

**Integrate into your app:**
```python
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from alchemist_connector import AlchemistConnector

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Add ALchemist connector to your UI
        self.alchemist = AlchemistConnector()
        layout = QVBoxLayout()
        layout.addWidget(self.alchemist)
        
        # Connect signals
        self.alchemist.connected.connect(self.on_connected)
        
        # ... rest of your app UI ...
        
    def on_connected(self, session_id, info):
        print(f"✅ Connected! {info['experiment_count']} experiments")
        # Your optimization code starts here
        self.run_optimization_loop(session_id)

if __name__ == "__main__":
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec()
```

**Run your app and connect:**
1. Paste session ID into connector
2. Click "Connect"
3. Your app now controls the optimization!

### 3. Run Autonomous Optimization

```python
import requests

def run_optimization_loop(self, session_id):
    """Your optimization loop."""
    api_url = "http://localhost:8000/api/v1"
    
    while not converged:
        # Get next experiment suggestion
        response = requests.post(
            f"{api_url}/sessions/{session_id}/acquisition/suggest",
            json={"strategy": "qEI", "goal": "maximize", "n_suggestions": 1}
        )
        next_point = response.json()["suggestions"][0]
        
        # Run your experiment (hardware, simulation, etc.)
        result = your_experiment_function(next_point)
        
        # Send result back to ALchemist
        requests.post(
            f"{api_url}/sessions/{session_id}/experiments",
            params={"auto_train": True},
            json={"inputs": next_point, "output": result}
        )
        
        # Web dashboard automatically updates!
```

---

## What You Get

✅ **Web UI shows real-time progress** - Experiments, model, plots update automatically  
✅ **Your app controls everything** - When to run, what to measure, how to preprocess  
✅ **Clean separation** - ALchemist handles optimization, you handle domain logic  
✅ **No conflicts** - One controller at a time, clear ownership  

---

## Complete Example

See `templates/connection_panels/qt/simple_optimizer_example.py` for a full working app:

```bash
cd templates/connection_panels/qt
python simple_optimizer_example.py
```

This shows:
- Full Qt application with connector integrated
- Autonomous optimization loop in worker thread
- Real-time logging and status updates
- Proper connection handling and cleanup

---

## Troubleshooting

**"Failed to connect"**
- Check ALchemist is running: `curl http://localhost:8000/health`
- Verify session ID is correct (copy fresh from web UI)

**"No suggestions returned"**
- Add more experiments first (model needs data to train)
- Check session has variables defined

**Connection drops**
- Sessions expire after 24 hours
- Create new session in web UI

---

## Next: Customize for Your Hardware

Replace the simulation function with your real hardware interface:

```python
def run_reactor_experiment(self, inputs):
    """Replace this with your hardware control."""
    # Set reactor conditions
    self.reactor.set_temperature(inputs["temperature"])
    self.reactor.set_flow_rate(inputs["flow_rate"])
    
    # Wait for steady state
    self.reactor.wait_for_steady_state()
    
    # Measure output
    spectrum = self.spectrometer.read()
    feature = self.extract_feature(spectrum)
    
    return feature
```

---

## Support

- Full Qt template docs: `templates/connection_panels/qt/README.md`
- API reference: `api/API_ENDPOINTS.md`
- Issues: https://github.com/NREL/ALchemist/issues
