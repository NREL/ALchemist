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
    <div className="rounded-lg border bg-card p-6">
      <h2 className="text-2xl font-bold mb-6">Experiment Data</h2>
      
      {/* Controls */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={handleLoadFromFile}
          disabled={uploadExperiments.isPending}
          className="bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploadExperiments.isPending ? 'Loading...' : 'Load from CSV'}
        </button>
        
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileSelected}
          className="hidden"
        />
      </div>

      {/* Summary Stats */}
      {summaryData && summaryData.has_data && (
        <div className="mb-4 p-4 bg-muted/50 rounded-lg">
          <h3 className="font-semibold mb-2">Summary Statistics</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-muted-foreground">Experiments:</span>{' '}
              <span className="font-medium">{summaryData.n_experiments}</span>
            </div>
            {summaryData.target_stats && (
              <>
                <div>
                  <span className="text-muted-foreground">Mean Output:</span>{' '}
                  <span className="font-medium">{summaryData.target_stats.mean?.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Std Dev:</span>{' '}
                  <span className="font-medium">{summaryData.target_stats.std?.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Range:</span>{' '}
                  <span className="font-medium">
                    {summaryData.target_stats.min?.toFixed(3)} - {summaryData.target_stats.max?.toFixed(3)}
                  </span>
                </div>
              </>
            )}
            {summaryData.has_noise && (
              <div className="col-span-2">
                <span className="text-muted-foreground">✓ Noise data included</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Experiments Table */}
      {isLoadingExperiments ? (
        <div className="text-muted-foreground">Loading experiments...</div>
      ) : hasExperiments ? (
        <div className="border rounded-lg overflow-hidden">
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 border-b sticky top-0">
                <tr>
                  {columns.map((col) => (
                    <th key={col} className="px-3 py-2 text-left font-medium">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y">
                {experiments.map((exp, idx) => (
                  <tr key={idx} className="hover:bg-accent/50">
                    {columns.map((col) => (
                      <td key={col} className="px-3 py-2">
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
          
          {/* Footer */}
          <div className="bg-muted/30 border-t px-4 py-2 text-sm text-muted-foreground">
            {experiments.length} experiment{experiments.length !== 1 ? 's' : ''} loaded
          </div>
        </div>
      ) : (
        <div className="border border-dashed border-muted-foreground/25 rounded-lg p-12 text-center">
          <p className="text-muted-foreground mb-2">No experiments loaded yet</p>
          <p className="text-sm text-muted-foreground">
            Click "Load from CSV" to import experimental data
          </p>
        </div>
      )}
    </div>
  );
}
