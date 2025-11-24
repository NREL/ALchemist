Session Components Design

Goal: Mirror desktop UX for session management, audit locking, pending suggestions, and Add Point flow.

Components to implement:

- SessionMetadata.tsx
  - Modal to view/edit session name, description, tags, author.
  - Calls PATCH `/sessions/{id}/metadata`.
  - Props: `sessionId`, `metadata`, `onSaved` callback.

- AuditLockDialog.tsx
  - Modal for lock confirmation and notes input.
  - Used for Data/Model/Acquisition locks.
  - Props: `sessionId`, `lockType` ('data'|'model'|'acquisition'), `payload` (for acquisition: strategy, parameters, suggestions), `onSuccess`.
  - Calls POST `/sessions/{id}/audit/lock` with validated payload.

- PendingSuggestionsPanel.tsx
  - Shows staged suggestions (mirrors desktop `pending_suggestions`).
  - Allows edit, "Add Point" to open AddPoint form pre-filled, and remove.
  - Props: `sessionId`, `pending`, `onAddPoint`, `onRemove`.
  - Add Point calls a local UI dialog to enter Output/Noise and then calls POST `/sessions/{id}/experiments` or appropriate endpoint (note: API route for adding single experiment may exist; otherwise use client-side session API adapter).

- SessionMenu.tsx (toolbar/menu)
  - New/Open/Save/Save As/Export Audit
  - New: POST `/sessions` to create session and adopt returned `session_id`.
  - Open: upload JSON via `/sessions/import` or `/sessions/upload` endpoint; update local app state with new session id and metadata.
  - Save/Save As: GET `/sessions/{id}/download` to download JSON file; suggests filename from `metadata.name`.
  - Export Audit: GET `/sessions/{id}/audit/export` (markdown)

API Contract notes:
- Create session: POST `/sessions` -> { session_id, created_at, expires_at }
- Update metadata: PATCH `/sessions/{id}/metadata` with { name?, description?, tags? }
- Download session JSON: GET `/sessions/{id}/download` (Content-Disposition filename uses metadata.name)
- Upload session JSON: POST `/sessions/import` (file upload) or POST `/sessions/upload`
- Lock decision: POST `/sessions/{id}/audit/lock` with body { lock_type: 'data'|'model'|'acquisition', notes?, strategy?, parameters?, suggestions? }
- Get audit entries: GET `/sessions/{id}/audit?entry_type=...`
- Export audit markdown: GET `/sessions/{id}/audit/export` (returns text/markdown)

UX invariants:
- Lock-in must be explicit; show confirmation modal with notes.
- Suggestions are staged, not auto-added.
- Use metadata.name for filenames.

Implementation plan: create component skeletons with minimal styling, include fetch wrappers, and wire to top-level app state via callbacks.

Example API helper (client-side):

```ts
async function createSession() {
  const res = await fetch('/api/v1/sessions', { method: 'POST' })
  return res.json() // { session_id, created_at, expires_at }
}

async function downloadSession(sessionId: string) {
  const res = await fetch(`/api/v1/sessions/${sessionId}/download`)
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `session_${sessionId}.json`
  a.click()
  URL.revokeObjectURL(url)
}

async function lockDecision(sessionId: string, body: any) {
  const res = await fetch(`/api/v1/sessions/${sessionId}/audit/lock`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) })
  return res.json()
}
```

Integration notes:
- The React app should keep `sessionId` in top-level state and pass it to these components.
- When acquisition returns suggestions, call the PendingSuggestionsPanel with the server-provided suggestions and display the AuditLockDialog for user confirmation prior to calling `lockDecision`.

Add Point flow (frontend):

- When the user clicks `Add Point` on a pending suggestion, open `AddPointDialog` pre-filled with suggested variable values and the recorded `Iteration` and `_reason`.
- The user supplies observed `Output` and optional `Noise` and `Reason` values and confirms.
- The front-end constructs the experiment payload (flat dict with variable columns, `Output`, optional `Noise`, `Iteration`, `Reason`) and POSTs it to `POST /sessions/{session_id}/experiments`.
- The API returns the updated experiment count and optionally triggers retraining if `auto_train=true` is passed as a query parameter. The UI should check the response and refresh session state (e.g., call `/sessions/{id}/state` and `/sessions/{id}/experiments/summary`).

Example payload:

```json
{
  "temperature": 350,
  "pressure": 5.8,
  "Output": 0.852,
  "Noise": 0.02,
  "Iteration": 3,
  "Reason": "Expected Improvement"
}
```

After success:
- Remove the suggestion from `pending_suggestions` (the desktop UI does this), update the UI experiment table, and optionally show a success toast.
- If multiple pending suggestions remain, show the next suggestion in the Add Point dialog (desktop uses `current_suggestion_index`).
