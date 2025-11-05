/**
 * Contour Plot - 2D surface plot of model predictions
 * Mirrors desktop UI contour plot from visualizations.py with side controls
 * Uses Plotly.js for professional contour visualization
 */
import { useState, useMemo, useEffect, useCallback } from 'react';
import { Loader2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Config } from 'plotly.js';
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
  const [committedFixedValues, setCommittedFixedValues] = useState<Record<string, FixedValue>>({});
  const [gridResolution, setGridResolution] = useState(50);
  const [committedGridResolution, setCommittedGridResolution] = useState(50);
  const [showExperiments, setShowExperiments] = useState(false);
  const [showNextPoint, setShowNextPoint] = useState(false);
  const [colormap, setColormap] = useState<string>('Viridis');

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
    setCommittedFixedValues(newFixed); // Also commit on initialization
  }, [xAxis, yAxis, variables]);

  // Fetch contour data (use committedFixedValues - only updates when slider is released)
  const contourRequest = useMemo(() => {
    if (!xAxis || !yAxis) return null;
    
    // Don't make request if we don't have fixed values initialized yet
    if (Object.keys(committedFixedValues).length === 0) {
      return null;
    }

    // CRITICAL: Ensure fixed values don't include x or y axis variables
    // This can happen during initialization race conditions
    const fixed: Record<string, number | string> = {};
    Object.entries(committedFixedValues).forEach(([key, val]) => {
      if (key !== xAxis && key !== yAxis) {
        fixed[key] = val.value;
      }
    });

    const request = {
      x_var: xAxis,
      y_var: yAxis,
      fixed_values: fixed,
      grid_resolution: committedGridResolution,
      include_experiments: showExperiments,
      include_suggestions: showNextPoint,
    };

    return request;
  }, [xAxis, yAxis, committedFixedValues, committedGridResolution, showExperiments, showNextPoint]);

  const {
    data: contourApiData,
    isLoading,
    error,
  } = useContourData(sessionId, contourRequest!, !!contourRequest);

  // Handler for updating fixed values (updates local state while dragging)
  const handleFixedValueChange = useCallback((varName: string, value: number | string) => {
    setFixedValues((prev) => ({
      ...prev,
      [varName]: { ...prev[varName], value },
    }));
  }, []);

  // Handler for committing fixed values (triggers API request when slider is released)
  const handleFixedValueCommit = useCallback((varName: string, value: number | string) => {
    setCommittedFixedValues((prev) => ({
      ...prev,
      [varName]: { ...prev[varName], value },
    }));
  }, []);

  if (realVariables.length < 2) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">
          Need at least two Real (continuous) variables for contour plotting
        </p>
      </div>
    );
  }

  return (
    <div className="flex gap-6 h-full">
      {/* Main plot area */}
      <div className="flex-1 flex flex-col min-h-[600px]">
        {isLoading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">Generating contour plot...</p>
            </div>
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

        {!isLoading && !error && contourApiData && (
          <div className="flex-1">
            <ContourPlotly
              data={contourApiData}
              xAxis={xAxis}
              yAxis={yAxis}
              showExperiments={showExperiments}
              showNextPoint={showNextPoint}
              colormap={colormap}
            />
          </div>
        )}
      </div>

      {/* Right sidebar - Controls (mirrors desktop UI) */}
      <div className="w-64 bg-muted/30 rounded-lg p-4 space-y-4 overflow-auto max-h-[800px]">
        <h3 className="font-semibold text-lg">Contour Plot Options</h3>

        {/* X-Axis selector */}
        <div>
          <label className="text-sm font-medium block mb-1">X-Axis Variable:</label>
          <select
            value={xAxis}
            onChange={(e) => setXAxis(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm"
          >
            {realVariables.map((v: VariableDetail) => (
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
            {realVariables.map((v: VariableDetail) => (
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
                      onMouseUp={(e) => handleFixedValueCommit(varName, parseFloat((e.target as HTMLInputElement).value))}
                      onTouchEnd={(e) => handleFixedValueCommit(varName, parseFloat((e.target as HTMLInputElement).value))}
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
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      handleFixedValueChange(varName, val);
                      handleFixedValueCommit(varName, val); // Commit immediately for number inputs
                    }}
                    className="w-full px-2 py-1 bg-background border border-input rounded text-sm"
                  />
                )}
                {varInfo.type === 'Categorical' && (
                  <select
                    value={varInfo.value as string}
                    onChange={(e) => {
                      handleFixedValueChange(varName, e.target.value);
                      handleFixedValueCommit(varName, e.target.value); // Commit immediately for selects
                    }}
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
          <label className="text-sm font-medium block mb-1">Colormap:</label>
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value)}
            className="w-full px-2 py-1 bg-background border border-input rounded text-sm"
          >
            <option value="Viridis">Viridis</option>
            <option value="Plasma">Plasma</option>
            <option value="Inferno">Inferno</option>
            <option value="Magma">Magma</option>
            <option value="Cividis">Cividis</option>
            <option value="Jet">Jet</option>
            <option value="Hot">Hot</option>
            <option value="Cool">Cool</option>
            <option value="RdBu">Red-Blue</option>
            <option value="YlOrRd">Yellow-Orange-Red</option>
          </select>
        </div>

        <div className="border-t border-border pt-4">
          <label className="text-sm font-medium block mb-1">Grid Resolution:</label>
          <input
            type="range"
            min={30}
            max={150}
            step={10}
            value={gridResolution}
            onChange={(e) => setGridResolution(parseInt(e.target.value))}
            onMouseUp={(e) => setCommittedGridResolution(parseInt((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedGridResolution(parseInt((e.target as HTMLInputElement).value))}
            className="w-full"
          />
          <div className="text-xs text-muted-foreground mt-1">{gridResolution} × {gridResolution}</div>
        </div>
      </div>
    </div>
  );
}

/**
 * Plotly-based contour plot renderer
 * Provides professional, interactive contour visualization matching desktop UI
 */
interface ContourPlotlyProps {
  data: {
    x_var: string;
    y_var: string;
    x_grid: number[][];
    y_grid: number[][];
    predictions: number[][];
    uncertainties: number[][];
    experiments?: {
      x: number[];
      y: number[];
      output: number[];
    } | null;
    suggestions?: {
      x: number[];
      y: number[];
    } | null;
    x_bounds: number[];
    y_bounds: number[];
    colorbar_bounds: number[];
  };
  xAxis: string;
  yAxis: string;
  showExperiments: boolean;
  showNextPoint: boolean;
  colormap: string;
}

function ContourPlotly({ 
  data, 
  xAxis, 
  yAxis, 
  showExperiments, 
  showNextPoint,
  colormap 
}: ContourPlotlyProps) {
  // Extract 1D arrays from 2D meshgrids for Plotly
  // x_grid is constant along columns, y_grid is constant along rows
  const xValues = data.x_grid[0]; // First row contains all unique x values
  const yValues = data.y_grid.map(row => row[0]); // First column contains all unique y values

  // Build traces array
  const traces: Data[] = [];

  // Main contour trace
  const contourTrace: Data = {
    type: 'contour',
    x: xValues,
    y: yValues,
    z: data.predictions,
    colorscale: colormap,
    colorbar: {
      title: {
        text: 'Prediction',
        side: 'right'
      },
      thickness: 20,
      len: 0.7,
    },
    contours: {
      coloring: 'heatmap',
    },
    hovertemplate: 
      `${xAxis}: %{x:.3f}<br>` +
      `${yAxis}: %{y:.3f}<br>` +
      'Prediction: %{z:.3f}<br>' +
      '<extra></extra>',
  };
  traces.push(contourTrace);

  // Add experimental points if requested and available
  if (showExperiments && data.experiments && data.experiments.x.length > 0) {
    const experimentTrace: Data = {
      type: 'scatter',
      mode: 'markers',
      x: data.experiments.x,
      y: data.experiments.y,
      marker: {
        color: 'white',
        size: 8,
        line: {
          color: 'black',
          width: 2
        },
        symbol: 'circle'
      },
      name: 'Experiments',
      hovertemplate: 
        `${xAxis}: %{x:.3f}<br>` +
        `${yAxis}: %{y:.3f}<br>` +
        'Output: %{text}<br>' +
        '<extra></extra>',
      text: data.experiments.output.map(v => v.toFixed(3)),
    };
    traces.push(experimentTrace);
  }

  // Add next suggested point if requested and available
  if (showNextPoint && data.suggestions && data.suggestions.x.length > 0) {
    const suggestionTrace: Data = {
      type: 'scatter',
      mode: 'markers',
      x: data.suggestions.x,
      y: data.suggestions.y,
      marker: {
        color: 'red',
        size: 12,
        line: {
          color: 'darkred',
          width: 2
        },
        symbol: 'diamond'
      },
      name: 'Next Point',
      hovertemplate: 
        `${xAxis}: %{x:.3f}<br>` +
        `${yAxis}: %{y:.3f}<br>` +
        '<extra></extra>',
    };
    traces.push(suggestionTrace);
  }

  // Layout configuration matching desktop UI
  const layout: Partial<Layout> = {
    title: {
      text: 'Contour Plot of Model Predictions',
      font: {
        size: 16,
        family: 'Arial, sans-serif'
      }
    },
    xaxis: {
      title: {
        text: xAxis,
        font: { size: 14 }
      },
      range: data.x_bounds,
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: {
        text: yAxis,
        font: { size: 14 }
      },
      range: data.y_bounds,
      showgrid: true,
      zeroline: false,
    },
    autosize: true,
    margin: {
      l: 80,
      r: 120,
      t: 80,
      b: 80
    },
    hovermode: 'closest',
    showlegend: Boolean((showExperiments && data.experiments) || (showNextPoint && data.suggestions)),
    legend: {
      x: 1.05,
      y: 1,
      xanchor: 'left',
      yanchor: 'top',
      bgcolor: 'rgba(255, 255, 255, 0.8)',
      bordercolor: '#ccc',
      borderwidth: 1
    },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: 'white',
  };

  // Plotly configuration
  const config: Partial<Config> = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: `contour_${xAxis}_${yAxis}`,
      height: 600,
      width: 800,
      scale: 2
    }
  };

  return (
    <div className="w-full h-full">
      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
}
