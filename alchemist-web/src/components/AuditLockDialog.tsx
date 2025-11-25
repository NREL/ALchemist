// AuditLockDialog.tsx - modal for lock confirmation + notes
import { useState } from 'react'

export default function AuditLockDialog({ sessionId, lockType, payload, onClose }:
  { sessionId: string, lockType: 'data'|'model'|'acquisition', payload?: any, onClose?: (success: boolean, entry?: any) => void }) {
  const [open, setOpen] = useState(true)
  const [notes, setNotes] = useState('')
  const [loading, setLoading] = useState(false)

  async function confirm() {
    setLoading(true)
    const body: any = { lock_type: lockType, notes }
    if (lockType === 'acquisition' && payload) {
      body.strategy = payload.strategy
      body.parameters = payload.parameters
      body.suggestions = payload.suggestions
    }

    try {
      const res = await fetch(`/api/v1/sessions/${sessionId}/audit/lock`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      })
      if (!res.ok) throw new Error('Lock failed')
      const json = await res.json()
      onClose && onClose(true, json.entry)
      setOpen(false)
    } catch (e) {
      console.error(e)
      onClose && onClose(false)
    } finally {
      setLoading(false)
    }
  }

  if (!open) return null
  return (
    <div className="modal">
      <h3>Confirm Lock: {lockType}</h3>
      <p>Optional notes to include in the audit trail:</p>
      <textarea value={notes} onChange={e=>setNotes(e.target.value)} />
      <div className="actions">
        <button onClick={() => { setOpen(false); onClose && onClose(false) }}>Cancel</button>
        <button onClick={confirm} disabled={loading}>{loading ? 'Saving...' : 'Lock'}</button>
      </div>
    </div>
  )
}
