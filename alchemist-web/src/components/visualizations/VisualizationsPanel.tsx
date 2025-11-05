/**
 * Visualizations Panel - Main container for model visualizations
 * Mimics desktop UI visualizations.py window structure
 */
import { useState } from 'react';
import { X } from 'lucide-react';
import { ParityPlot } from './ParityPlot';
import { MetricsPlot } from './MetricsPlot';
import { QQPlot } from './QQPlot';
import { CalibrationCurve } from './CalibrationCurve';
import { ContourPlot } from './ContourPlot';
import { HyperparametersDisplay } from './HyperparametersDisplay';

interface VisualizationsPanelProps {
  sessionId: string;
  isOpen: boolean;
  onClose: () => void;
  modelBackend?: string;
}

type PlotType = 'parity' | 'metrics' | 'qq' | 'calibration' | 'contour';
type MetricType = 'RMSE' | 'MAE' | 'MAPE' | 'R2';

export function VisualizationsPanel({ 
  sessionId, 
  isOpen, 
  onClose 
}: VisualizationsPanelProps) {
  const [activePlot, setActivePlot] = useState<PlotType>('parity');
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('RMSE');
  const [sigmaMultiplier, setSigmaMultiplier] = useState<string>('1.96');
  const [useCalibrated, setUseCalibrated] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-card rounded-lg shadow-xl w-full max-w-6xl h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-2xl font-bold">Model Visualizations</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-accent rounded-md transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Controls Row 1 - Plot Selection */}
        <div className="p-4 border-b border-border bg-muted/30">
          <div className="flex flex-wrap gap-2">
            {/* Plot Type Buttons */}
            <button
              onClick={() => setActivePlot('parity')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activePlot === 'parity'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Parity
            </button>
            <button
              onClick={() => setActivePlot('metrics')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activePlot === 'metrics'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Metrics
            </button>
            <button
              onClick={() => setActivePlot('qq')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activePlot === 'qq'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Q-Q
            </button>
            <button
              onClick={() => setActivePlot('calibration')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activePlot === 'calibration'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Calibration
            </button>
            <button
              onClick={() => setActivePlot('contour')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activePlot === 'contour'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Contour
            </button>

            {/* Metric Selector (for metrics plot) */}
            {activePlot === 'metrics' && (
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value as MetricType)}
                className="px-3 py-2 bg-background border border-input rounded-md text-sm"
              >
                <option value="RMSE">RMSE</option>
                <option value="MAE">MAE</option>
                <option value="MAPE">MAPE</option>
                <option value="R2">R²</option>
              </select>
            )}

            {/* Sigma Multiplier (for parity plot) */}
            {activePlot === 'parity' && (
              <>
                <span className="flex items-center text-sm text-muted-foreground ml-4">
                  Error bars:
                </span>
                <select
                  value={sigmaMultiplier}
                  onChange={(e) => setSigmaMultiplier(e.target.value)}
                  className="px-3 py-2 bg-background border border-input rounded-md text-sm"
                >
                  <option value="None">None</option>
                  <option value="1.0">1.0σ (68%)</option>
                  <option value="1.96">1.96σ (95%)</option>
                  <option value="2.0">2.0σ (95.4%)</option>
                  <option value="2.58">2.58σ (99%)</option>
                  <option value="3.0">3.0σ (99.7%)</option>
                </select>
              </>
            )}
          </div>
        </div>

        {/* Controls Row 2 - Calibration Toggle */}
        {(activePlot === 'parity' || activePlot === 'qq' || activePlot === 'calibration') && (
          <div className="px-4 py-2 border-b border-border bg-muted/20">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={useCalibrated}
                onChange={(e) => setUseCalibrated(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300"
              />
              <span>Use Calibrated Results</span>
            </label>
          </div>
        )}

        {/* Main Content Area - Plot Display */}
        <div className="flex-1 overflow-auto p-6">
          {activePlot === 'parity' && (
            <ParityPlot
              sessionId={sessionId}
              useCalibrated={useCalibrated}
              sigmaMultiplier={sigmaMultiplier}
            />
          )}
          {activePlot === 'metrics' && (
            <MetricsPlot
              sessionId={sessionId}
              selectedMetric={selectedMetric}
              cvSplits={5}
            />
          )}
          {activePlot === 'qq' && (
            <QQPlot
              sessionId={sessionId}
              useCalibrated={useCalibrated}
            />
          )}
          {activePlot === 'calibration' && (
            <CalibrationCurve
              sessionId={sessionId}
              useCalibrated={useCalibrated}
            />
          )}
          {/* ContourPlot always mounted but hidden to preserve state */}
          <div className={activePlot === 'contour' ? 'block' : 'hidden'}>
            <ContourPlot sessionId={sessionId} />
          </div>
        </div>

        {/* Footer - Hyperparameters */}
        <div className="border-t border-border">
          <HyperparametersDisplay sessionId={sessionId} />
        </div>
      </div>
    </div>
  );
}
