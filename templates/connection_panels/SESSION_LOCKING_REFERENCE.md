# Session Locking Quick Reference

## For External App Developers

### Qt/PySide6

```python
from alchemist_connector import AlchemistConnector

# Create connector with auto-locking
connector = AlchemistConnector(
    client_name="MyOptimizer",  # Shown in web UI
    auto_lock=True               # Auto lock/unlock
)

# Connect (automatically locks session)
connector.connect_to_session(session_id)

# Run your optimization loop
for iteration in range(100):
    # Get suggestions
    suggestions = connector.get_suggestions()
    
    # Run experiments
    results = run_experiments(suggestions)
    
    # Add results
    connector.add_results(results)

# Disconnect (automatically unlocks)
connector.disconnect()
```

### Python API Client

```python
import requests

API_URL = "http://localhost:8000/api/v1"

# Lock session
response = requests.post(
    f"{API_URL}/sessions/{session_id}/lock",
    json={"locked_by": "MyScript", "client_id": "script-001"}
)
lock_data = response.json()
lock_token = lock_data["lock_token"]  # Save for unlock!

try:
    # Run optimization
    while not converged:
        # ... your code ...
        pass
finally:
    # Always unlock, even on error
    requests.delete(
        f"{API_URL}/sessions/{session_id}/lock",
        params={"lock_token": lock_token}
    )
```

### MATLAB

```matlab
% Lock session
url = sprintf('%s/sessions/%s/lock', API_URL, session_id);
body = struct('locked_by', 'MATLAB', 'client_id', 'matlab-001');
options = weboptions('RequestMethod', 'post', 'MediaType', 'application/json');
lock_data = webwrite(url, body, options);
lock_token = lock_data.lock_token;

try
    % Run optimization
    % ... your code ...
catch
    % Error handling
end

% Unlock
url = sprintf('%s/sessions/%s/lock?lock_token=%s', API_URL, session_id, lock_token);
options = weboptions('RequestMethod', 'delete');
webwrite(url, '', options);
```

## API Reference

### Lock Session
```http
POST /sessions/{session_id}/lock
Content-Type: application/json

{
  "locked_by": "ClientName",
  "client_id": "optional-unique-id"
}

Response 200:
{
  "locked": true,
  "locked_by": "ClientName",
  "locked_at": "2024-12-04T16:23:38.849288",
  "lock_token": "503bee63-55b3-4d7c-9cec-31e68d83f465"
}
```

**IMPORTANT:** Save the `lock_token` - you'll need it to unlock!

### Unlock Session
```http
DELETE /sessions/{session_id}/lock?lock_token={token}

Response 200:
{
  "locked": false,
  "locked_by": null,
  "locked_at": null,
  "lock_token": null
}
```

Force unlock (no token - use with caution):
```http
DELETE /sessions/{session_id}/lock
```

### Check Lock Status
```http
GET /sessions/{session_id}/lock

Response 200:
{
  "locked": true,
  "locked_by": "ClientName",
  "locked_at": "2024-12-04T16:23:38.849288",
  "lock_token": null  // Never exposed in status checks
}
```

## Web UI Behavior

When a session is locked:

1. **Blue lock badge** appears next to session ID: `🔒 ClientName`
2. **Monitor mode** activates automatically
3. **All controls** become read-only
4. **Toast notification**: "External controller connected: ClientName"
5. **Real-time updates** continue (charts, tables, etc.)

When unlocked:
1. Lock badge disappears
2. Interactive mode restores
3. Toast notification: "External controller disconnected - resuming interactive mode"

## Error Handling

### Session Not Found
```json
{
  "detail": "Session {session_id} not found or expired"
}
```
HTTP 404

### Invalid Lock Token
```json
{
  "detail": "Invalid lock token"
}
```
HTTP 403

### Best Practices

1. **Always unlock in finally block** to prevent stuck locks
2. **Use descriptive client names** for web UI clarity
3. **Test locking locally** before deploying
4. **Save lock tokens** immediately after locking
5. **Handle network errors** gracefully

## Troubleshooting

**Web UI stuck in monitor mode?**
- Check lock status: `GET /sessions/{id}/lock`
- Force unlock: `DELETE /sessions/{id}/lock` (no token)
- Or restart session

**"Invalid lock token" error?**
- Token may have expired (session restart)
- Wrong token provided
- Session was force-unlocked

**Lock not working?**
- Verify API is running: `http://localhost:8000/api/docs`
- Check session exists: `GET /sessions/{id}`
- Verify network connectivity

## Testing

Use the included test script:

```bash
cd templates/connection_panels/qt
python test_locking.py
```

This will:
1. Create a test session
2. Auto-lock it
3. Show lock status
4. Demonstrate web UI behavior
5. Test unlock on disconnect

## Example: Reactor Control Integration

```python
from alchemist_connector import AlchemistConnector
import reactor_control  # Your hardware interface

connector = AlchemistConnector(client_name="ReactorController", auto_lock=True)
connector.connect_to_session(session_id)

for cycle in range(50):
    # Get next experiment to run
    suggestions = connector.get_suggestions(n=1)
    if not suggestions:
        break
    
    params = suggestions[0]
    
    # Run on physical hardware
    print(f"Running: {params}")
    reactor_control.set_temperature(params['temperature'])
    reactor_control.set_pressure(params['pressure'])
    reactor_control.start_reaction()
    
    # Wait for completion and measure
    reactor_control.wait_for_completion()
    yield_value = reactor_control.measure_yield()
    
    # Report back to ALchemist
    connector.add_experiment(params, yield_value)
    print(f"Yield: {yield_value:.2f}%")

connector.disconnect()
print("Optimization complete!")
```

The web UI will show real-time progress while your hardware runs!
