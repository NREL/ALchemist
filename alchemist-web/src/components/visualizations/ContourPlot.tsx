/**
 * Contour Plot - 2D surface plot of model predictions
 * Mirrors desktop UI contour plot from visualizations.py with side controls
 */
import { useState, useMemo, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { useContourData } from '../../hooks/api/useVisualizations';
import { useVariables } from '../../hooks/api/useVariables';
import type { VariableDetail } from '../../api/types';

interface ContourPlotProps {
  sessionId: string;
}

interface FixedValue {
  value: number | string;
  min?: number;
  max?: number;
  categories?: string[];
  type: 'Real' | 'Integer' | 'Categorical';
}

export function ContourPlot({ sessionId }: ContourPlotProps) {
  const { data: variables } = useVariables(sessionId);

  // Get continuous (Real) variables only
  const realVariables = useMemo(() => {
    if (!variables) return [];
    return variables.variables.filter((v: VariableDetail) => v.type === 'real');
  }, [variables]);

  const [xAxis, setXAxis] = useState<string>('');
  const [yAxis, setYAxis] = useState<string>('');
  const [fixedValues, setFixedValues] = useState<Record<string, FixedValue>>({});
  const [gridResolution, setGridResolution] = useState(50);
  const [showExperiments, setShowExperiments] = useState(false);
  const [showNextPoint, setShowNextPoint] = useState(false);

  // Initialize axes when variables load
  useEffect(() => {
    if (realVariables.length >= 2 && !xAxis) {
      setXAxis(realVariables[0].name);
      setYAxis(realVariables[1].name);
    }
  }, [realVariables, xAxis]);

  // Update fixed values when axes change
  useEffect(() => {
    if (!variables) return;

    const newFixed: Record<string, FixedValue> = {};
    variables.variables.forEach((variable: VariableDetail) => {
      if (variable.name === xAxis || variable.name === yAxis) return;

      if (variable.type === 'real' && variable.bounds) {
        const min = variable.bounds[0];
        const max = variable.bounds[1];
        newFixed[variable.name] = {
          value: (min + max) / 2,
          min,
          max,
          type: 'Real',
        };
      } else if (variable.type === 'integer' && variable.bounds) {
        const min = variable.bounds[0];
        const max = variable.bounds[1];
        newFixed[variable.name] = {
          value: Math.floor((min + max) / 2),
          min,
          max,
          type: 'Integer',
        };
      } else if (variable.type === 'categorical' && variable.categories) {
        newFixed[variable.name] = {
          value: variable.categories[0],
          categories: variable.categories,
          type: 'Categorical',
        };
      }
    });

    setFixedValues(newFixed);
  }, [xAxis, yAxis, variables]);

  // Fetch contour data
  const contourRequest = useMemo(() => {
    if (!xAxis || !yAxis) return null;

    const fixed: Record<string, number | string> = {};
    Object.entries(fixedValues).forEach(([key, val]) => {
      fixed[key] = val.value;
    });

    return {
      x_var: xAxis,
      y_var: yAxis,
      fixed_values: fixed,
      grid_resolution: gridResolution,
    };
  }, [xAxis, yAxis, fixedValues, gridResolution]);

  const {
    data: contourData,
    isLoading,
    error,
  } = useContourData(sessionId, contourRequest!);

  if (realVariables.length < 2) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">
          Need at least two Real (continuous) variables for contour plotting
        </p>
      </div>
    );
  }

  const handleFixedValueChange = (varName: string, value: number | string) => {
    setFixedValues((prev) => ({
      ...prev,
      [varName]: { ...prev[varName], value },
    }));
  };

  return (
    <div className="flex gap-6 h-full">
      {/* Main plot area */}
      <div className="flex-1 flex flex-col">
        {isLoading && (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        )}

        {error && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <p className="text-destructive font-medium">Error loading contour plot</p>
              <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
            </div>
          </div>
        )}

        {contourData && (
          <div className="flex-1">
            <ContourCanvas
              data={contourData}
              xAxis={xAxis}
              yAxis={yAxis}
              showExperiments={showExperiments}
              showNextPoint={showNextPoint}
            />
          </div>
        )}
      </div>

      {/* Right sidebar - Controls (mirrors desktop UI) */}
      <div className="w-64 bg-muted/30 rounded-lg p-4 space-y-4 overflow-auto">
        <h3 className="font-semibold text-lg">Contour Plot Options</h3>

        {/* X-Axis selector */}
        <div>
          <label className="text-sm font-medium block mb-1">X-Axis Variable:</label>
          <select
            value={xAxis}
            onChange={(e) => setXAxis(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm"
          >
            {realVariables.map((v: any) => (
              <option key={v.name} value={v.name}>
                {v.name}
              </option>
            ))}
          </select>
        </div>

        {/* Y-Axis selector */}
        <div>
          <label className="text-sm font-medium block mb-1">Y-Axis Variable:</label>
          <select
            value={yAxis}
            onChange={(e) => setYAxis(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm"
          >
            {realVariables.map((v: any) => (
              <option key={v.name} value={v.name}>
                {v.name}
              </option>
            ))}
          </select>
        </div>

        <div className="border-t border-border pt-4">
          <h4 className="text-sm font-semibold mb-3">Fixed Values</h4>
          <div className="space-y-3">
            {Object.entries(fixedValues).map(([varName, varInfo]) => (
              <div key={varName}>
                <label className="text-xs font-medium block mb-1">{varName}:</label>
                {varInfo.type === 'Real' && (
                  <div>
                    <input
                      type="range"
                      min={varInfo.min}
                      max={varInfo.max}
                      step={(varInfo.max! - varInfo.min!) / 100}
                      value={varInfo.value as number}
                      onChange={(e) => handleFixedValueChange(varName, parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <div className="text-xs text-muted-foreground mt-1">
                      {(varInfo.value as number).toFixed(3)}
                    </div>
                  </div>
                )}
                {varInfo.type === 'Integer' && (
                  <input
                    type="number"
                    min={varInfo.min}
                    max={varInfo.max}
                    step={1}
                    value={varInfo.value as number}
                    onChange={(e) => handleFixedValueChange(varName, parseInt(e.target.value))}
                    className="w-full px-2 py-1 bg-background border border-input rounded text-sm"
                  />
                )}
                {varInfo.type === 'Categorical' && (
                  <select
                    value={varInfo.value as string}
                    onChange={(e) => handleFixedValueChange(varName, e.target.value)}
                    className="w-full px-2 py-1 bg-background border border-input rounded text-sm"
                  >
                    {varInfo.categories?.map((cat) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-border pt-4">
          <h4 className="text-sm font-semibold mb-3">Display Options</h4>
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={showExperiments}
                onChange={(e) => setShowExperiments(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300"
              />
              <span>Show Experimental Points</span>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={showNextPoint}
                onChange={(e) => setShowNextPoint(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300"
              />
              <span>Show Next Point</span>
            </label>
          </div>
        </div>

        <div className="border-t border-border pt-4">
          <label className="text-sm font-medium block mb-1">Grid Resolution:</label>
          <input
            type="range"
            min={30}
            max={200}
            step={10}
            value={gridResolution}
            onChange={(e) => setGridResolution(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-muted-foreground mt-1">{gridResolution} × {gridResolution}</div>
        </div>
      </div>
    </div>
  );
}

/**
 * Canvas-based contour plot renderer
 * Uses HTML5 canvas for performance with large grids
 */
interface ContourCanvasProps {
  data: {
    x: number[];
    y: number[];
    z: number[][];
    experiments?: Array<{ x: number; y: number }>;
    next_point?: { x: number; y: number };
  };
  xAxis: string;
  yAxis: string;
  showExperiments: boolean;
  showNextPoint: boolean;
}

function ContourCanvas({ data, xAxis, yAxis, showExperiments, showNextPoint }: ContourCanvasProps) {
  const canvasRef = useState<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef[0];
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Find min/max z values for color scaling
    const allZ = data.z.flat();
    const minZ = Math.min(...allZ);
    const maxZ = Math.max(...allZ);

    // Draw contour using colored rectangles
    const cellWidth = width / data.x.length;
    const cellHeight = height / data.y.length;

    for (let i = 0; i < data.y.length; i++) {
      for (let j = 0; j < data.x.length; j++) {
        const zVal = data.z[i][j];
        const normalized = (zVal - minZ) / (maxZ - minZ);

        // Simple viridis-like colormap
        const hue = 240 - normalized * 240; // Blue to yellow
        const saturation = 70 + normalized * 30;
        const lightness = 30 + normalized * 50;

        ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        ctx.fillRect(j * cellWidth, (data.y.length - 1 - i) * cellHeight, cellWidth, cellHeight);
      }
    }

    // Draw experimental points
    if (showExperiments && data.experiments) {
      const xMin = Math.min(...data.x);
      const xMax = Math.max(...data.x);
      const yMin = Math.min(...data.y);
      const yMax = Math.max(...data.y);

      data.experiments.forEach((exp) => {
        const canvasX = ((exp.x - xMin) / (xMax - xMin)) * width;
        const canvasY = height - ((exp.y - yMin) / (yMax - yMin)) * height;

        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      });
    }

    // Draw next point
    if (showNextPoint && data.next_point) {
      const xMin = Math.min(...data.x);
      const xMax = Math.max(...data.x);
      const yMin = Math.min(...data.y);
      const yMax = Math.max(...data.y);

      const canvasX = ((data.next_point.x - xMin) / (xMax - xMin)) * width;
      const canvasY = height - ((data.next_point.y - yMin) / (yMax - yMin)) * height;

      ctx.fillStyle = 'red';
      ctx.strokeStyle = 'darkred';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(canvasX, canvasY - 8);
      ctx.lineTo(canvasX - 6, canvasY + 4);
      ctx.lineTo(canvasX + 6, canvasY + 4);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }, [data, showExperiments, showNextPoint, canvasRef]);

  return (
    <div className="w-full h-full flex flex-col">
      <div className="text-center mb-4">
        <h3 className="text-lg font-semibold">Contour Plot of Model Predictions</h3>
      </div>
      <div className="flex-1 flex items-center justify-center bg-muted/20 rounded-lg p-4">
        <canvas
          ref={(el) => canvasRef[1](el)}
          width={600}
          height={500}
          className="border border-border rounded"
        />
      </div>
      <div className="flex justify-between text-sm text-muted-foreground mt-2 px-4">
        <span>{xAxis}</span>
        <span>{yAxis}</span>
      </div>
    </div>
  );
}
