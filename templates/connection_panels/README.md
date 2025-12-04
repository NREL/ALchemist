# ALchemist Connection Panel Templates

Reusable UI components for connecting external applications to ALchemist optimization sessions.

## Overview

These templates provide ready-to-use connection panels that you can drop into your own applications. They handle the "handshake" between your application and ALchemist, allowing you to:

1. Connect to existing ALchemist sessions via session ID
2. Control optimization programmatically
3. Monitor session state in real-time
4. Use ALchemist web UI as a monitoring dashboard

## Available Templates

### Qt/PySide6 (`qt/`)
Full-featured Qt widget for desktop applications.
- **Best for:** Hardware control applications, desktop tools
- **Status:** ✅ Complete
- **Requirements:** PySide6, requests

### CustomTkinter (`customtkinter/`)
Modern Tkinter widget matching ALchemist's desktop UI.
- **Best for:** Python desktop apps, rapid prototyping
- **Status:** 🚧 Coming soon
- **Requirements:** customtkinter, requests

### React/TypeScript (`react/`)
Web-based connection component and hooks.
- **Best for:** Web applications, remote control dashboards
- **Status:** 🚧 Coming soon
- **Requirements:** React 18+, TypeScript

### Streamlit (`streamlit/`)
Simple connection widget for Streamlit dashboards.
- **Best for:** Quick prototypes, data science workflows
- **Status:** 🚧 Coming soon
- **Requirements:** streamlit, requests

### Python API (`python/`)
Pure Python connector class (no UI).
- **Best for:** Scripts, Jupyter notebooks, headless applications
- **Status:** 🚧 Coming soon
- **Requirements:** requests

## Quick Start

Each template directory contains:
- Ready-to-use component code
- Integration guide (`README.md`)
- Example usage
- API reference

### Basic Workflow

1. **Setup in ALchemist Web UI**
   - Create session
   - Define variables
   - Copy session ID

2. **Integrate Template**
   - Copy template into your application
   - Add to your UI
   - Connect signal handlers

3. **Connect and Control**
   - Paste session ID
   - Click "Connect"
   - Your app takes control
   - Web UI becomes monitoring dashboard

## Example: Qt Integration

```python
from templates.connection_panels.qt import AlchemistConnector

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Add ALchemist connector
        self.alchemist = AlchemistConnector()
        layout.addWidget(self.alchemist)
        
        # Connect signals
        self.alchemist.connected.connect(self.on_connected)
        self.alchemist.state_updated.connect(self.on_state_update)
        
    def on_connected(self, session_id, info):
        # Start your optimization loop
        self.run_optimization(session_id)
```

## Architecture

### The Handshake Process

```
User → Web UI (setup session) 
  ↓
User copies session_id
  ↓
User → External App (paste session_id)
  ↓
External App → API (validate session)
  ↓
External App takes control
  ↓
Web UI → Monitor mode (automatic)
```

### API Communication

All templates communicate with ALchemist via REST API:

```
http://localhost:8000/api/v1/sessions/{session_id}
```

Key endpoints:
- `GET /sessions/{id}` - Validate and get session info
- `GET /sessions/{id}/state` - Poll current state
- `POST /sessions/{id}/experiments` - Add data
- `POST /sessions/{id}/acquisition/suggest` - Get next suggestion

## Design Philosophy

These templates are designed to be:

1. **Copy-paste ready** - Drop into your project and use
2. **Minimal dependencies** - Only essential packages
3. **Framework-agnostic** - Work with your existing code
4. **Customizable** - Easy to modify and extend
5. **Well-documented** - Clear examples and API docs

## Contributing

Want to contribute a template for another framework?

1. Create directory: `templates/connection_panels/{framework}/`
2. Include:
   - Main component file
   - README.md with usage guide
   - Example code
   - Requirements list
3. Submit pull request

### Template Requirements

All templates should:
- Handle session ID input and validation
- Provide connect/disconnect functionality
- Poll session state (configurable interval)
- Emit events/signals for state changes
- Display connection status
- Include error handling

## License

Same as ALchemist main project.

## Support

- Documentation: https://nrel.github.io/ALchemist/
- Issues: https://github.com/NREL/ALchemist/issues
- Email: ccoatney@nrel.gov
