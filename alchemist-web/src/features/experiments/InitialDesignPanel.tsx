/**
 * Initial Design Panel - Generate DoE (Design of Experiments) points
 * For autonomous optimization workflows
 */
import { useState } from 'react';
import { useGenerateInitialDesign } from '../../hooks/api/useExperiments';
import { useVariables } from '../../hooks/api/useVariables';
import type { DoEMethod, LHSCriterion } from '../../api/types';
import { Download, Sparkles } from 'lucide-react';

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
  const variables = variablesData?.variables || [];

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
    <div className="rounded-lg border bg-card p-6">
      <div className="flex items-center gap-2 mb-6">
        <Sparkles className="h-6 w-6 text-primary" />
        <h2 className="text-2xl font-bold">Initial Experimental Design</h2>
      </div>

      {!hasVariables ? (
        <div className="border border-dashed border-muted-foreground/25 rounded-lg p-8 text-center">
          <p className="text-muted-foreground">
            Define variables in the search space before generating initial design points.
          </p>
        </div>
      ) : (
        <>
          {/* Configuration Form */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            {/* Method Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">Sampling Method</label>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value as DoEMethod)}
                className="w-full px-3 py-2 border rounded-md bg-background"
              >
                <option value="lhs">Latin Hypercube Sampling (LHS)</option>
                <option value="sobol">Sobol Sequence</option>
                <option value="halton">Halton Sequence</option>
                <option value="hammersly">Hammersly Sequence</option>
                <option value="random">Random Sampling</option>
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                {method === 'lhs' && 'Space-filling design with maximin criterion'}
                {method === 'sobol' && 'Quasi-random low-discrepancy sequence'}
                {method === 'halton' && 'Deterministic quasi-random sequence'}
                {method === 'hammersly' && 'Hybrid quasi-random sequence'}
                {method === 'random' && 'Uniform random sampling'}
              </p>
            </div>

            {/* Number of Points */}
            <div>
              <label className="block text-sm font-medium mb-2">Number of Points</label>
              <input
                type="number"
                min={1}
                max={1000}
                value={nPoints}
                onChange={(e) => setNPoints(parseInt(e.target.value))}
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Recommended: 5-20 × number of variables ({variables.length} vars)
              </p>
            </div>

            {/* LHS Criterion (only for LHS) */}
            {method === 'lhs' && (
              <div>
                <label className="block text-sm font-medium mb-2">LHS Criterion</label>
                <select
                  value={lhsCriterion}
                  onChange={(e) => setLhsCriterion(e.target.value as LHSCriterion)}
                  className="w-full px-3 py-2 border rounded-md bg-background"
                >
                  <option value="maximin">Maximin</option>
                  <option value="correlation">Correlation</option>
                  <option value="ratio">Ratio</option>
                </select>
              </div>
            )}

            {/* Random Seed */}
            <div>
              <label className="block text-sm font-medium mb-2">Random Seed (Optional)</label>
              <input
                type="text"
                value={randomSeed}
                onChange={(e) => setRandomSeed(e.target.value)}
                placeholder="e.g., 42"
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
              <p className="text-xs text-muted-foreground mt-1">
                For reproducible designs
              </p>
            </div>
          </div>

          {/* Generate Button */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={handleGenerate}
              disabled={generateDesign.isPending}
              className="bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {generateDesign.isPending ? 'Generating...' : 'Generate Design'}
            </button>

            {generatedPoints && generatedPoints.length > 0 && (
              <button
                onClick={handleDownloadCSV}
                className="bg-secondary text-secondary-foreground px-4 py-2 rounded-md hover:bg-secondary/90 flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Download CSV
              </button>
            )}
          </div>

          {/* Generated Points Table */}
          {generatedPoints && generatedPoints.length > 0 && (
            <div className="border rounded-lg overflow-hidden">
              <div className="bg-muted/50 px-4 py-2 border-b">
                <h3 className="font-semibold">
                  Generated {generatedPoints.length} Design Points
                </h3>
              </div>
              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 border-b sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">#</th>
                      {Object.keys(generatedPoints[0]).map((col) => (
                        <th key={col} className="px-3 py-2 text-left font-medium">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {generatedPoints.map((point, idx) => (
                      <tr key={idx} className="hover:bg-accent/50">
                        <td className="px-3 py-2 text-muted-foreground">{idx + 1}</td>
                        {Object.entries(point).map(([key, value]) => (
                          <td key={key} className="px-3 py-2">
                            {typeof value === 'number' 
                              ? value.toFixed(3) 
                              : value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="bg-muted/30 border-t px-4 py-2 text-sm text-muted-foreground">
                💡 Download this as CSV and evaluate these experiments, then upload the results
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
