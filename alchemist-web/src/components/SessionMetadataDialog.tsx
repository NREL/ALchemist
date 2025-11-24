/**
 * SessionMetadataDialog - Edit session metadata (name, description, tags, author)
 * Mimics desktop SessionMetadataDialog
 */
import { useState } from 'react';
import { X } from 'lucide-react';

interface SessionMetadataDialogProps {
  sessionId: string;
  metadata: {
    name?: string;
    description?: string;
    tags?: string[];
    author?: string;
    session_id?: string;
  };
  onSave: (metadata: {
    name?: string;
    description?: string;
    tags?: string[];
    author?: string;
  }) => Promise<void>;
  onCancel: () => void;
}

export function SessionMetadataDialog({
  sessionId,
  metadata,
  onSave,
  onCancel,
}: SessionMetadataDialogProps) {
  const [name, setName] = useState(metadata.name || '');
  const [description, setDescription] = useState(metadata.description || '');
  const [tagsStr, setTagsStr] = useState(metadata.tags?.join(', ') || '');
  const [author, setAuthor] = useState(metadata.author || '');
  const [isSaving, setIsSaving] = useState(false);

  async function handleSave() {
    setIsSaving(true);
    try {
      const tags = tagsStr.split(',').map(t => t.trim()).filter(t => t);
      await onSave({
        name: name.trim() || undefined,
        description: description.trim() || undefined,
        tags: tags.length > 0 ? tags : undefined,
        author: author.trim() || undefined,
      });
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onCancel}>
      <div className="bg-card border border-border rounded-lg shadow-lg w-full max-w-xl" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="border-b border-border p-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Session Metadata</h3>
          <button onClick={onCancel} className="p-1.5 rounded hover:bg-accent">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <div className="p-6 space-y-4">
          {/* Session Name */}
          <div className="space-y-1">
            <label className="block text-sm font-medium">Session Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Optimization Session"
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          {/* Description */}
          <div className="space-y-1">
            <label className="block text-sm font-medium">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter a brief description of this optimization session..."
              rows={4}
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 resize-y"
            />
          </div>

          {/* Tags */}
          <div className="space-y-1">
            <label className="block text-sm font-medium">Tags (comma-separated)</label>
            <input
              type="text"
              value={tagsStr}
              onChange={(e) => setTagsStr(e.target.value)}
              placeholder="catalyst, optimization, batch-1"
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          {/* Author */}
          <div className="space-y-1">
            <label className="block text-sm font-medium">Author</label>
            <input
              type="text"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              placeholder="Your name"
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          {/* Session ID (read-only) */}
          <div className="space-y-1">
            <label className="block text-sm font-medium text-muted-foreground">Session ID</label>
            <div className="px-3 py-2 text-sm rounded-md border border-border bg-muted text-foreground font-mono">
              {sessionId.substring(0, 16)}...
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-border p-4 flex justify-end gap-3">
          <button
            onClick={onCancel}
            disabled={isSaving}
            className="px-4 py-2 text-sm rounded-md border border-border hover:bg-accent disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {isSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
