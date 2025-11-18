/**
 * Experiments Panel - Manage experimental data
 * Mimics desktop UI experiment management
 */
import { useRef } from 'react';
import { useExperiments, useExperimentsSummary, useUploadExperiments } from '../../hooks/api/useExperiments';

interface ExperimentsPanelProps {
  sessionId: string;
}

export function ExperimentsPanel({ sessionId }: ExperimentsPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { data: experimentsData, isLoading: isLoadingExperiments } = useExperiments(sessionId);
  const { data: summaryData } = useExperimentsSummary(sessionId);
  const uploadExperiments = useUploadExperiments(sessionId);

  const handleLoadFromFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await uploadExperiments.mutateAsync(file);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const experiments = experimentsData?.experiments || [];
  const hasExperiments = experiments.length > 0;

  // Get column headers from first experiment
  const columns = hasExperiments ? Object.keys(experiments[0]) : [];

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground border-b pb-2">
        Experiment Data
      </h3>
      
      {/* Controls - Compact */}
      <button
        onClick={handleLoadFromFile}
        disabled={uploadExperiments.isPending}
        className="w-full bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
      >
        {uploadExperiments.isPending ? 'Loading...' : 'Load CSV'}
      </button>
      
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={handleFileSelected}
        className="hidden"
      />

      {/* Summary Stats - Compact */}
      {summaryData && summaryData.has_data && (
        <div className="p-3 bg-muted/30 rounded text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Experiments:</span>
            <span className="font-medium tabular-nums">{summaryData.n_experiments}</span>
          </div>
          {summaryData.target_stats && (
            <>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Range:</span>
                <span className="font-medium tabular-nums">
                  {summaryData.target_stats.min?.toFixed(2)} - {summaryData.target_stats.max?.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mean ± Std:</span>
                <span className="font-medium tabular-nums">
                  {summaryData.target_stats.mean?.toFixed(2)} ± {summaryData.target_stats.std?.toFixed(2)}
                </span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Experiments Table - Compact */}
      {isLoadingExperiments ? (
        <div className="text-xs text-muted-foreground">Loading...</div>
      ) : hasExperiments ? (
        <div className="border rounded overflow-hidden">
          <div className="overflow-x-auto max-h-64">
            <table className="w-full text-xs">
              <thead className="bg-muted/50 border-b sticky top-0">
                <tr>
                  {columns.map((col) => (
                    <th key={col} className="px-2 py-1.5 text-left font-medium text-xs">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y">
                {experiments.map((exp, idx) => (
                  <tr key={idx} className="hover:bg-accent/50">
                    {columns.map((col) => (
                      <td key={col} className="px-2 py-1.5 tabular-nums">
                        {typeof exp[col] === 'number' 
                          ? exp[col].toFixed(3) 
                          : exp[col] ?? '-'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="bg-muted/30 border-t px-2 py-1 text-xs text-muted-foreground">
            {experiments.length} row{experiments.length !== 1 ? 's' : ''}
          </div>
        </div>
      ) : (
        <div className="border border-dashed border-muted-foreground/20 rounded p-6 text-center">
          <p className="text-xs text-muted-foreground">No data loaded</p>
        </div>
      )}
    </div>
  );
}
