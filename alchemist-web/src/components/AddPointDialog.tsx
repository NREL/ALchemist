/**
 * AddPointDialog - Modal dialog for adding experimental results
 * Mimics desktop UI add_point dialog with proper React styling
 */
import { useState } from 'react';
import { X } from 'lucide-react';

type Props = {
  suggestion: any;
  index?: number;
  total?: number;
  iteration?: number; // Current iteration number (read-only display)
  onCancel: () => void;
  onConfirm: (payload: any, options: { saveToFile: boolean; retrain: boolean }) => void;
  onPrev?: () => void;
  onNext?: () => void;
}

export default function AddPointDialog({ 
  suggestion, 
  index = 0, 
  total = 1, 
  iteration,
  onCancel, 
  onConfirm, 
  onPrev, 
  onNext 
}: Props) {
  // Build inputs from suggestion (exclude internal keys and Output/Noise)
  const baseInputs: Record<string, any> = {};
  Object.keys(suggestion || {}).forEach(k => {
    if (!k.startsWith('_') && k !== 'Output' && k !== 'Noise' && k !== 'Iteration' && k !== 'Reason') {
      baseInputs[k] = suggestion[k];
    }
  });

  const [inputs, setInputs] = useState<Record<string, any>>(baseInputs);
  const [output, setOutput] = useState<string>(suggestion?.Output?.toString() ?? '');
  const [noise, setNoise] = useState<string>(suggestion?.Noise?.toString() ?? '');
  
  // Auto-fill reason from acquisition strategy (_reason field from desktop workflow)
  const defaultReason = suggestion?._reason || suggestion?.Reason || 'Acquisition';
  const [reason, setReason] = useState<string>(defaultReason);
  
  const [saveToFile, setSaveToFile] = useState(true);
  const [retrain, setRetrain] = useState(true);
  
  // Display iteration (from suggestion or passed prop)
  const displayIteration = suggestion?.Iteration ?? iteration ?? 'N/A';

  function changeField(field: string, val: string) {
    setInputs((prev) => ({ ...prev, [field]: val }));
  }

  function confirm() {
    const payload: any = { inputs: { ...inputs } };
    if (output !== '') payload.output = Number(output);
    if (noise !== '') payload.noise = Number(noise);
    if (reason) payload.reason = reason;
    onConfirm(payload, { saveToFile, retrain });
  }

  return (
    <div className="bg-card border border-border rounded-lg shadow-lg w-full max-w-2xl max-h-[85vh] overflow-auto">
      {/* Header */}
      <div className="border-b border-border p-4 flex items-center justify-between">
        <div className="flex-1">
          <h3 className="text-lg font-semibold">
            {total > 1 ? `Pending Suggestion ${index + 1} of ${total}` : 'Add Experimental Result'}
          </h3>
          {total > 1 && (
            <p className="text-sm text-green-600 dark:text-green-500 mt-1">
              {defaultReason}
            </p>
          )}
        </div>
        
        {/* Navigation buttons */}
        {total > 1 && (
          <div className="flex gap-2 ml-4">
            <button
              onClick={onPrev}
              disabled={!onPrev}
              className="px-3 py-1.5 text-sm rounded border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ← Previous
            </button>
            <button
              onClick={onNext}
              disabled={!onNext}
              className="px-3 py-1.5 text-sm rounded border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next →
            </button>
          </div>
        )}
        
        <button
          onClick={onCancel}
          className="ml-2 p-1.5 rounded hover:bg-accent"
          title="Close"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Form content */}
      <div className="p-6 space-y-4">
        {/* Variable inputs in 2-column grid */}
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(inputs).map(([k, v]) => (
            <div key={k} className="space-y-1">
              <label className="block text-sm font-medium text-muted-foreground">{k}</label>
              <input
                type="text"
                value={String(v ?? '')}
                onChange={(e) => changeField(k, e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
          ))}
          
          {/* Output field */}
          <div className="space-y-1">
            <label className="block text-sm font-medium text-muted-foreground">Output</label>
            <input
              type="number"
              step="any"
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              autoFocus
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          
          {/* Noise field */}
          <div className="space-y-1">
            <label className="block text-sm font-medium text-muted-foreground">Noise (optional)</label>
            <input
              type="number"
              step="any"
              value={noise}
              onChange={(e) => setNoise(e.target.value)}
              placeholder="1e-6"
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        {/* Iteration (read-only) and Reason */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
          <div className="space-y-1">
            <label className="block text-sm font-medium text-muted-foreground">Iteration</label>
            <div className="px-3 py-2 text-sm rounded-md border border-border bg-muted text-foreground">
              {displayIteration}
            </div>
          </div>
          
          <div className="space-y-1">
            <label className="block text-sm font-medium text-muted-foreground">Reason</label>
            <input
              type="text"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        {/* Options checkboxes */}
        <div className="flex items-center gap-6 pt-4 border-t border-border">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={saveToFile}
              onChange={(e) => setSaveToFile(e.target.checked)}
              className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
            />
            <span>Save to file</span>
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={retrain}
              onChange={(e) => setRetrain(e.target.checked)}
              className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
            />
            <span>Retrain model</span>
          </label>
        </div>
      </div>

      {/* Footer with action buttons */}
      <div className="border-t border-border p-4 flex justify-end gap-3">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-sm rounded-md border border-border hover:bg-accent"
        >
          Cancel
        </button>
        <button
          onClick={confirm}
          className="px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
        >
          Save & Close
        </button>
      </div>
    </div>
  );
}
