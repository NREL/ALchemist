/**
 * Contour Plot - 2D surface plot of model predictions
 * Placeholder implementation - TODO: Complete contour visualization
 */
import { Loader2 } from 'lucide-react';

interface ContourPlotProps {
  sessionId: string;
}

export function ContourPlot({ sessionId }: ContourPlotProps) {
  return (
    <div className="flex items-center justify-center h-96">
      <div className="text-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Contour plot implementation in progress...</p>
        <p className="text-xs text-muted-foreground mt-2">Session: {sessionId}</p>
      </div>
    </div>
  );
}
