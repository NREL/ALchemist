/**
 * Initial Design Panel - Generate DoE (Design of Experiments) points
 * For autonomous optimization workflows
 */
import { useState } from 'react';
import { useGenerateInitialDesign } from '../../hooks/api/useExperiments';
import { useVariables } from '../../hooks/api/useVariables';
import type { DoEMethod, LHSCriterion } from '../../api/types';
import { Download } from 'lucide-react';

interface InitialDesignPanelProps {
  sessionId: string;
}

export function InitialDesignPanel({ sessionId }: InitialDesignPanelProps) {
  const [method, setMethod] = useState<DoEMethod>('lhs');
  const [nPoints, setNPoints] = useState<number>(10);
  const [randomSeed, setRandomSeed] = useState<string>('');
  const [lhsCriterion, setLhsCriterion] = useState<LHSCriterion>('maximin');
  const [generatedPoints, setGeneratedPoints] = useState<Array<Record<string, any>> | null>(null);

  const { data: variablesData } = useVariables(sessionId);
  const generateDesign = useGenerateInitialDesign(sessionId);

  const hasVariables = variablesData && variablesData.variables.length > 0;

  const handleGenerate = async () => {
    const request = {
      method,
      n_points: nPoints,
      random_seed: randomSeed ? parseInt(randomSeed) : null,
      lhs_criterion: method === 'lhs' ? lhsCriterion : undefined,
    };

    const result = await generateDesign.mutateAsync(request);
    setGeneratedPoints(result.points);
  };

  const handleDownloadCSV = () => {
    if (!generatedPoints || generatedPoints.length === 0) return;

    // Get column headers from first point
    const headers = Object.keys(generatedPoints[0]);
    
    // Build CSV
    const csvRows = [
      headers.join(','),
      ...generatedPoints.map(point => 
        headers.map(h => point[h]).join(',')
      )
    ];
    
    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `initial_design_${method}_${nPoints}pts.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground border-b pb-2">
        Initial Design (DoE)
      </h3>

      {!hasVariables ? (
        <div className="border border-dashed border-muted-foreground/20 rounded p-4 text-center">
          <p className="text-xs text-muted-foreground">Define variables first</p>
        </div>
      ) : (
        <>
          {/* Compact Config Form */}
          <div className="space-y-2">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Method</label>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value as DoEMethod)}
                className="w-full px-2 py-1.5 text-sm border rounded bg-background"
              >
                <option value="lhs">LHS</option>
                <option value="sobol">Sobol</option>
                <option value="halton">Halton</option>
                <option value="hammersly">Hammersly</option>
                <option value="random">Random</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Points</label>
                <input
                  type="number"
                  value={nPoints}
                  onChange={(e) => setNPoints(parseInt(e.target.value) || 10)}
                  min={5}
                  max={100}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                />
              </div>
              
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Seed (opt)</label>
                <input
                  type="number"
                  value={randomSeed}
                  onChange={(e) => setRandomSeed(e.target.value)}
                  placeholder="Auto"
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                />
              </div>
            </div>

            {method === 'lhs' && (
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">LHS Criterion</label>
                <select
                  value={lhsCriterion}
                  onChange={(e) => setLhsCriterion(e.target.value as LHSCriterion)}
                  className="w-full px-2 py-1.5 text-sm border rounded bg-background"
                >
                  <option value="maximin">Maximin</option>
                  <option value="correlation">Correlation</option>
                  <option value="ratio">Ratio</option>
                </select>
              </div>
            )}
          </div>

          <button
            onClick={handleGenerate}
            disabled={generateDesign.isPending || !hasVariables}
            className="w-full bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {generateDesign.isPending ? 'Generating...' : 'Generate Design'}
          </button>

          {/* Results - Compact */}
          {generatedPoints && generatedPoints.length > 0 && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">
                  {generatedPoints.length} points generated
                </span>
                <button
                  onClick={handleDownloadCSV}
                  className="flex items-center gap-1 text-xs border px-2 py-1 rounded hover:bg-accent"
                >
                  <Download className="h-3 w-3" />
                  CSV
                </button>
              </div>
              
              <div className="border rounded overflow-hidden">
                <div className="overflow-x-auto max-h-48">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/50 border-b sticky top-0">
                      <tr>
                        {Object.keys(generatedPoints[0]).map((key) => (
                          <th key={key} className="px-2 py-1 text-left font-medium">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {generatedPoints.map((point, idx) => (
                        <tr key={idx} className="hover:bg-accent/50">
                          {Object.values(point).map((val, i) => (
                            <td key={i} className="px-2 py-1 tabular-nums">
                              {typeof val === 'number' ? val.toFixed(3) : val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
