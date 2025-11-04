# Session Save/Load Feature

## Overview

Added complete session save/load functionality to ALchemist web UI. Sessions can now be exported as `.pkl` files and reimported later, preserving all variables, experiments, and model state.

## Features

### Backend (FastAPI)

**New API Endpoints:**
- `GET /api/v1/sessions/{session_id}/export` - Download session as .pkl file
- `POST /api/v1/sessions/import` - Upload and restore a session from .pkl file

**Session Store Enhancements:**
- Added `export_session()` - Serialize session to bytes for download
- Added `import_session()` - Deserialize and restore session from bytes
- Sessions automatically persist to disk in `cache/sessions/` directory
- Survives server restarts during development

### Frontend (React)

**New Hooks:**
- `useExportSession()` - Export current session with automatic download
- `useImportSession()` - Import session from file upload

**UI Updates:**
- **"Save Session"** button in active session header
- **"Load Session"** button on welcome screen (no session)
- File picker accepts `.pkl` files only
- Toast notifications for success/error feedback

## Usage

### Saving a Session

1. With an active session, click **"Save Session"** button
2. Browser downloads `alchemist_session_XXXXXXXX.pkl` file
3. Session includes:
   - All defined variables (with units/descriptions)
   - All experiment data
   - Trained model state (if any)
   - Session metadata

### Loading a Session

1. From welcome screen, click **"Load Session"**
2. Select a previously saved `.pkl` file
3. Session is restored with a new session ID
4. All data, variables, and experiments are preserved

## Technical Details

### Session Persistence

Sessions are automatically saved to disk:
```
cache/sessions/
├── 8af6350e-5429-4f00-9c1d-e1375b08990d.pkl
├── 5dd100dc-4062-4398-bcba-c030fac003ec.pkl
└── ...
```

- **Survives server restarts** - Development hot-reload won't lose sessions
- **TTL still enforced** - Expired sessions cleaned up automatically
- **Pickle format** - Python serialization for complete state preservation

### File Format

The `.pkl` files contain:
```python
{
    "session": OptimizationSession,  # Complete session object
    "created_at": datetime,
    "last_accessed": datetime,
    "expires_at": datetime
}
```

## Benefits

✅ **No more lost work** - Server restarts don't clear sessions  
✅ **Share sessions** - Send .pkl files to collaborators  
✅ **Backup/restore** - Save checkpoints during long experiments  
✅ **Version control** - Keep session files in version control (carefully!)  
✅ **Easy migration** - Move sessions between dev/prod environments

## Future Enhancements

Potential improvements:
- Export to human-readable JSON format
- Session versioning/compatibility checks
- Compression for large session files
- Session templates/presets
- Session history/snapshots

## Testing

**To test:**
1. Create a session
2. Add variables (e.g., Temperature, Pressure)
3. Load some experiment data
4. Click "Save Session" - file downloads
5. Clear session
6. Click "Load Session" - select downloaded file
7. Verify all data is restored correctly

**Expected behavior:**
- Variables appear in table with units/descriptions
- Experiments appear in data table
- Session info shows correct counts
- New session ID assigned on import
