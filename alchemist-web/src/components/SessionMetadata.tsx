// SessionMetadata.tsx - lightweight skeleton
import { useState } from 'react'

type Metadata = {
  session_id: string
  name: string
  description?: string
  tags?: string[]
  created_at?: string
  last_modified?: string
}

export default function SessionMetadata({ sessionId, metadata, onSaved }:
  { sessionId: string, metadata: Metadata, onSaved?: (m: Metadata) => void }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState(metadata?.name || '')
  const [description, setDescription] = useState(metadata?.description || '')
  const [tags, setTags] = useState((metadata?.tags || []).join(', '))

  async function save() {
    const body = { name, description, tags: tags.split(',').map(t=>t.trim()).filter(Boolean) }
    const res = await fetch(`/api/v1/sessions/${sessionId}/metadata`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    })
    if (res.ok) {
      const json = await res.json()
      onSaved && onSaved(json)
      setOpen(false)
    } else {
      console.error('Failed to update metadata')
    }
  }

  return (
    <div>
      <button onClick={() => setOpen(true)}>Edit Session Metadata</button>
      {open && (
        <div className="modal">
          <h3>Session Metadata</h3>
          <label>Name</label>
          <input value={name} onChange={e=>setName(e.target.value)} />
          <label>Description</label>
          <textarea value={description} onChange={e=>setDescription(e.target.value)} />
          <label>Tags (comma-separated)</label>
          <input value={tags} onChange={e=>setTags(e.target.value)} />
          <div className="actions">
            <button onClick={() => setOpen(false)}>Cancel</button>
            <button onClick={save}>Save</button>
          </div>
        </div>
      )}
    </div>
  )
}
