# Session & Audit Migration Plan: Desktop → REST API / React UI

Date: 2025-11-21
Author: GitHub Copilot (pair-programmer)

## Objective

Document the session management and audit-logging implementation present in the desktop UI and `alchemist_core`, evaluate the changes made to the session API, and provide a prioritized, minimal-change migration plan to implement the same session management and audit logging behavior in the REST API and React UI. The migration should prioritize reusing `alchemist_core` APIs, keep new code minimal and modular, and ensure cross-compatibility between `.json` desktop sessions and the REST API session store (currently using pickles).

## Clarifications & constraints (user)

- **REST API as a thin wrapper:** The REST API is intended to be a functional wrapper around the `alchemist_core` session API. The API should expose the same core functionality and stay aligned with the core implementation rather than reimplementing logic.
- **Core-first migration:** Confirm whether audit logging is fully present in the core session API; the migration workflow is to ensure audit logging/session management are complete and stable in `alchemist_core` first, and then expose that logic through the REST API.
- **React UI parity:** The React UI should mimic the desktop app's user workflow and UX for session metadata, lock confirmations, and audit log viewing. Frontend behavior should mirror desktop interactions.
- **Backwards compatibility de-prioritized:** Because this has not been widely released, backwards compatibility with legacy pickled session IDs is a low priority; prefer simplicity and a JSON-first approach unless you explicitly request maintained legacy support.

## Findings (what's implemented today)

- Core implementation lives in `alchemist_core`:
  - `alchemist_core/audit_log.py` provides `SessionMetadata`, `AuditEntry`, and `AuditLog` with: `lock_data`, `lock_model`, `lock_acquisition`, `to_dict`, `from_dict`, and `to_markdown` export.
  - `alchemist_core/session.py` exposes `OptimizationSession` with programmatic APIs: `add_variable`, `load_data`, `add_experiment`, `generate_initial_design`, `train_model`, `suggest_next`, `predict`, `lock_data`, `lock_model`, `lock_acquisition`, `save_session`, `load_session`, and `update_metadata`.
  - `OptimizationSession.save_session()` writes a JSON session file that includes metadata, audit log, search space, experiments, config, and an optional `model_config` section.
  - `OptimizationSession.load_session()` can reconstruct a session from the JSON format and will attempt to retrain the model from the saved `model_config`.
  - `AuditLog.to_markdown()` builds a publication-ready markdown audit trail including search space, experimental data table, and iteration-by-iteration model/acquisition summaries.

- Desktop application uses `.json` session files and allows user-controlled session metadata (name, description, tags), custom save location, and Cmd+S / File menu integration.

- REST API / React UI currently uses a pickle-based session mechanism (pickled session objects), where the API's session store serves multiple sessions concurrently by referencing their auto-generated hash identifiers and preserving pickled model instances.

- API endpoints already present to support sessions and audit log operations (in `api/routers/sessions.py`) include:
  - POST `/sessions` - create session
  - GET `/sessions/{id}` - session info
  - GET `/sessions/{id}/state` - lightweight state
  - POST `/sessions/{id}/audit/lock` - lock decision (data/model/acquisition)
  - GET `/sessions/{id}/audit` - get audit entries
  - GET `/sessions/{id}/audit/export` - markdown export
  - GET `/sessions/{id}/download` and POST `/sessions/upload` - JSON download/upload (these endpoints interact with `OptimizationSession.save_session` / `load_session`)
  - POST `/sessions/import` - import session file (different route name exists)

 The REST API `session_store` persists sessions as JSON under `cache/sessions/<session_id>.json` and reconstructs sessions with `OptimizationSession.load_session()`. I implemented small, focused changes to the store to:
  - add per-session `threading.Lock` objects to serialize mutating operations,
  - ensure the in-memory store key (`session_id`) is synchronized to `session.metadata.session_id` on create/import,
  - allow `create(name=...)` to pre-populate session metadata,
  - call `OptimizationSession.load_session(..., retrain_on_load=False)` during import and startup to avoid unintended expensive retraining.
 The main remaining items were addressed in small patches I applied:
  - Per-session locking added to `session_store` to avoid race conditions when mutating sessions (e.g., `lock_acquisition` increments iteration).
  - `OptimizationSession.load_session()` now supports `retrain_on_load=False` and `session_store` uses that during startup/import to avoid automatic retraining.
  - `session_store.create()` now accepts an optional `name`, `description`, and `tags` to populate metadata at creation.
  - The store now synchronizes the stored session's `metadata.session_id` with the store key.
1. Session file format
   - Desktop: JSON format (human-readable, git-friendly), `session.metadata.name` used for filenames, includes `session_id` in metadata.
   - REST API (current): Pickled sessions with auto-generated hash identifiers and pickled model instances; not human-readable and less cross-compatible.
   - Migration goal: Make REST API session store use JSON session format or at least store JSON alongside pickle, and prefer JSON for import/export to align with desktop.

2. Model persistence
   - Pickle: Stores live model objects (fast restore) but is insecure/unportable.
   - JSON: Session stores model *config* and hyperparameters (no live pickled model object) and optionally retrains on load using saved `model_config`.
   - Migration: Prefer storing `model_config` in session JSON and avoid pickling the model; optionally keep an internal cache for fast responses but mark it ephemeral and not part of exported session files.

3. Session identifiers
   - Current API uses auto hashes (from pickle) for session routing.
   - Desktop uses `metadata.session_id` (UUID) inside the JSON session file and user-controlled filename for export.
   - Migration: Use UUID `session.metadata.session_id` as canonical session ID in API and map it to session-store internal keys; keep an ephemeral numeric/short-hash map if necessary for multi-session handling but persist UUID in metadata. Ensure new sessions created by API also fill `SessionMetadata` via `SessionMetadata.create()`.

4. Audit logging behavior
   - Desktop and core implementation already implement append-only locking semantics; API endpoints call `session.lock_*` and `session.audit_log.*` methods which aligns behavior. The important part is to ensure REST/UI interaction patterns trigger locks only when user confirms (React UI must provide lock buttons and confirmation dialogs like the desktop) and to prevent automatic logging for exploration actions.

5. Upload/Download endpoints
   - REST router already provides `download` and `upload` endpoints that use `session.save_session()` and `OptimizationSession.load_session()`; confirm these paths are used by React UI.

6. React UI
   - Needs UI components: session metadata editor, lock buttons in Variables/GPR/Acquisition panels, audit log viewer, export and upload flows. These will call existing REST endpoints and must replicate desktop user confirmations and visual feedback.

7. Backwards compatibility
   - Existing pickled session store must still be supported for old clients. Plan should include a migration path: when a pickled session is loaded, export and store a JSON representation (via `save_session()`), and write a conversion utility to convert pickles to JSON (and vice-versa if necessary).

## Migration Principles / Constraints

- Reuse `alchemist_core` APIs: `OptimizationSession` and `AuditLog` already provide the programmatic behavior we want — prefer calling these functions from the API layer rather than reimplementing logic.
- Minimize new code: implement adapters in the API layer to convert existing pickled sessions into JSON-backed sessions, and implement small UI components in React that call the existing endpoints.
- Keep model retrainable: exported JSON must contain `model_config` (not pickled model). If API wants to cache trained model objects for speed, keep them ephemeral and not included in exported session files.
- Maintain append-only audit semantics and iteration numbering consistency: ensure session-store access is serialized to avoid race conditions around incrementing `_current_iteration` and writing entries.

## Prioritized TODOs (short-term → long-term)

### Phase 0 — Immediate research tasks (low-effort, high-value)
- [ ] Inspect `api/services/session_store.py` to confirm current pickled-session store behavior, TTL, disk persistence, and locking/concurrency model.
- [ ] Run unit tests locally (you said pytests are passing) to confirm test state and capture any failing tests after code changes later.

### Phase 1 — API storage migration & compatibility (core work)
- [ ] Add JSON-backed session persistence to `session_store` while preserving compatibility with existing pickled sessions. Implementation approach:
  - Extend `session_store` to persist both a) a JSON session export (via `OptimizationSession.save_session`) and b) an internal ephemeral pickled cache for fast restore if desired.
  - When creating a new session (POST `/sessions`), create an `OptimizationSession` with `SessionMetadata.create()`; return the `metadata.session_id` UUID to clients instead of a pickle hash.
  - Implement lookup by both legacy pickled-hash and `metadata.session_id` to support older clients; when a legacy pickled session is first accessed, convert and save a JSON snapshot.
- [ ] Avoid pickling models during export: modify `session_store` export routines to call `OptimizationSession.save_session()` and ensure `model_config` is included in JSON for later retraining; do not include pickled model serialized blobs in exported JSON.
- [ ] Ensure `session_store.create()` accepts an optional `name`/`metadata` payload to populate session metadata from REST clients.
- [ ] Implement per-session locking (threading.Lock or asyncio.Lock depending on server model) to serialize API access that mutates session state (notably `lock_acquisition` which increments iteration). This will prevent race conditions.

### Phase 2 — REST API endpoints & small adaptors
- [ ] Review and adjust `api/routers/sessions.py` to use `metadata.session_id` as canonical session identifier where feasible; add compatibility mapping if necessary.
- [ ] Ensure `POST /sessions` returns the `session.metadata.session_id` and that `GET /sessions/{id}` accepts both metadata UUID and legacy hash.
- [ ] Add an endpoint to list sessions with metadata (for UI session selection) and include creation time, last_modified, name, tags, and a 'format' marker (json vs legacy-pickle).
- [ ] Implement conversion utilities: `convert_pickle_to_json(pickle_path) -> json_str` and `convert_json_to_session(json_str) -> new session`, used by the import/upload flow and for migration.

### Phase 3 — React UI changes (frontend)
- [ ] Add `SessionMetadata` modal component in `alchemist-web` to edit name/description/tags and call `PATCH /sessions/{id}/metadata`.
- [ ] Add Lock buttons in three panels (Variables/Data panel, GPR panel, Acquisition panel) that call the API `POST /sessions/{id}/audit/lock` with the correct payloads and show confirmation dialogs.
- [ ] Add an `AuditLogViewer` component that requests `/sessions/{id}/audit` and `/sessions/{id}/audit/export` and displays a timeline and markdown export link.
- [ ] Update session open/save flows to use `/sessions/{id}/download` and `/sessions/import` (already present). For creating a new session, call `POST /sessions` and store returned `session_id` (UUID). Ensure filename uses `metadata.name` when downloading.
- [ ] Add UI visual locked-state indicators (orange → green) matching desktop logic for locked data/model/acquisition states.

### Phase 4 — Testing, docs, and rollout
- [ ] Add integration tests that create sessions via API, perform lock flows, download JSON, re-upload and verify audit entries preserved and `session.metadata.session_id` flows through correctly.
- [ ] Update docs: API docs, README, and memory notes to declare JSON as canonical session format and how to migrate legacy pickled sessions.
- [ ] Plan/execute migration for existing persisted pickled sessions: write a migration script to convert persisted pickles to JSON files (backup first).

## Implementation Notes & Code Reuse Strategy

- Reuse `OptimizationSession.save_session()` / `.load_session()` for export/import endpoints and as the canonical on-disk JSON format — they already include `model_config` and `audit_log`.
- Keep pickled models only as an optimization cache inside `session_store` (if needed) but never expose pickled blobs in exported files. Mark the cache ephemeral and local-only.
- Where possible, implement API-facing adapters that call into `alchemist_core` rather than duplicating logic. Example adapters:
  - `session_store.create_session_from_json(json_str)` → calls `OptimizationSession.load_session()` and registers the resulting object in the store.
  - `session_store.export_session_to_json(session_id)` → calls `session.save_session(temp_path)` and returns file contents.
- Add per-session locks (in-memory) to serialize modifications like `add_experiment`, `lock_acquisition`, and `save_session`. Consider using `asyncio.Lock` if the API is async and hosted in an async framework (FastAPI + uvicorn) — ensure locking implementation works in your server threading model.

## Risks & Mitigations

- Race conditions if multiple clients mutate same session concurrently — mitigate with per-session locks.
- Large model retrain time when `load_session()` retrains models on import — mitigate by making retrain optional (config flag) and by caching pickled models as an internal optimization (not part of exported JSON).
- Backwards compatibility with existing clients using pickle session IDs — mitigate by supporting old IDs in `session_store` lookups and migrating them to JSON / UUIDs when touched.

## Deliverables for next step

I created this file to capture findings and the migration plan: `SESSION_AUDIT_REST_MIGRATION.md` in the repository root. If you review and approve, I will:

1. Inspect `api/services/session_store.py` and propose minimal changes to support JSON-first sessions and per-session locking.
2. Implement backend changes in small, reviewable patches that reuse `alchemist_core` APIs.
3. Scaffold the React UI components and wire them to the existing API endpoints.

Please review this plan and let me know any preferences or constraints (e.g., must keep pickle support for X months, retrain on load must be disabled, or prefer synchronous locks). Once you approve, I will begin with step 1 (inspect `session_store`) and produce a short implementation plan for the code changes.

## Core audit logging status — confirmation & recommended small core changes

Summary: I confirm the core audit logging functionality is implemented in `alchemist_core` and exposed through `OptimizationSession`. The key features are present: append-only `AuditLog`, `AuditEntry` creation with deterministic hashing, `lock_data`/`lock_model`/`lock_acquisition` in both the `AuditLog` and `OptimizationSession`, JSON session save/load, and markdown export.

Recommended small changes to stabilize the core API before migrating to the REST API (small, localized, low-risk):

1. Add optional `retrain_on_load: bool = False` parameter to `OptimizationSession.load_session()` (and a `load_session_from_json(json_str, retrain=False)` helper) so the REST import flow can avoid expensive or unintended automatic retraining by default.
2. Add an `export_session_json()` convenience method on `OptimizationSession` that returns the JSON string for the current session (calls `save_session` under the hood but avoids filesystem IO), simplifying API adapter code.
3. Make `lock_data` optionally accept an `iteration` parameter (or document clearly why data locks intentionally omit iteration). This improves iteration grouping consistency in `AuditLog.to_markdown`.
4. Add clearer docstrings and small unit tests covering iteration semantics (model+acquisition iteration alignment) and the markdown export formatting to avoid regressions.
5. Document that `alchemist_core` session objects are not thread-safe and that the REST API should provide per-session synchronization (locks) when exposing mutating operations.

These changes keep logic in `alchemist_core`, follow your principle of minimal new code, and make the REST adapter work simpler and safer. If you approve, I'll implement these tiny core changes first (one patch at a time), then proceed to `session_store` migration.