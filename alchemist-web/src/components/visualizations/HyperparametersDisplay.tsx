/**
 * Hyperparameters Display - Shows learned model hyperparameters
 * Mirrors desktop UI hyperparameters display from visualizations.py
 */
import { useHyperparameters } from '../../hooks/api/useVisualizations';
import { Loader2 } from 'lucide-react';

interface HyperparametersDisplayProps {
  sessionId: string;
}

export function HyperparametersDisplay({ sessionId }: HyperparametersDisplayProps) {
  const { data, isLoading, error } = useHyperparameters(sessionId);

  if (isLoading) {
    return (
      <div className="p-4 flex items-center justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !data) {
    return null; // Silent failure for hyperparameters display
  }

  return (
    <div className="p-4">
      <h3 className="text-lg font-semibold mb-2">Learned Hyperparameters</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 text-sm">
        {Object.entries(data.hyperparameters).map(([key, value]) => (
          <div key={key} className="flex flex-col">
            <span className="text-muted-foreground">{key}:</span>
            <span className="font-mono">
              {typeof value === 'number' ? value.toFixed(6) : String(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
