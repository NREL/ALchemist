/**
 * Q-Q Plot - Standardized Residuals vs Normal Distribution
 * Mirrors desktop UI Q-Q plot from visualizations.py
 */
import { useMemo } from 'react';
import {
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Line,
  ComposedChart,
  Area,
} from 'recharts';
import { useQQPlotData } from '../../hooks/api/useVisualizations';
import { Loader2 } from 'lucide-react';

interface QQPlotProps {
  sessionId: string;
  useCalibrated: boolean;
}

export function QQPlot({ sessionId, useCalibrated }: QQPlotProps) {
  const { data, isLoading, error } = useQQPlotData(sessionId, useCalibrated);

  const chartData = useMemo(() => {
    if (!data) return [];

    return data.theoretical_quantiles.map((theoretical, idx) => ({
      theoretical,
      sample: data.sample_quantiles[idx],
    }));
  }, [data]);

  const parityLineData = useMemo(() => {
    if (!data || data.theoretical_quantiles.length === 0) return [];

    const allValues = [...data.theoretical_quantiles, ...data.sample_quantiles];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    return [
      { theoretical: minVal, parity: minVal },
      { theoretical: maxVal, parity: maxVal },
    ];
  }, [data]);

  // Confidence band for small samples (±1.96/sqrt(n) around perfect line)
  const confidenceBandData = useMemo(() => {
    if (!data || data.n_samples >= 100) return [];

    const se = 1.96 / Math.sqrt(data.n_samples);
    
    // Create points along the perfect calibration line with upper/lower bounds
    return data.theoretical_quantiles.map(theoretical => ({
      theoretical,
      bounds: [theoretical - se, theoretical + se], // Store as tuple for Area component
    }));
  }, [data]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <p className="text-destructive font-medium">Error loading Q-Q plot</p>
          <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
        </div>
      </div>
    );
  }

  if (!data || data.theoretical_quantiles.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">No data available for Q-Q plot</p>
      </div>
    );
  }

  const resultsType = data.results_type === 'calibrated' ? ' (Calibrated)' : ' (Uncalibrated)';
  const title = `Q-Q Plot: Standardized Residuals vs. Normal Distribution${resultsType}\nMean(z) = ${data.z_mean.toFixed(3)}, Std(z) = ${data.z_std.toFixed(3)}, N = ${data.n_samples}`;

  // Determine calibration quality
  const isWellCalibrated = Math.abs(data.z_mean) < 0.1 && Math.abs(data.z_std - 1.0) < 0.2;
  const isBiased = Math.abs(data.z_mean) > 0.2;
  const isUnderConfident = data.z_std < 0.7;
  const isOverConfident = data.z_std > 1.3;

  let calibrationMessage = '';
  if (isWellCalibrated) {
    calibrationMessage = '✓ Model appears well-calibrated (unbiased, good uncertainty)';
  } else if (isBiased) {
    calibrationMessage = '⚠ Model may be biased (mean(z) far from 0)';
  } else if (isUnderConfident) {
    calibrationMessage = '⚠ Model may be under-confident (std(z) < 1)';
  } else if (isOverConfident) {
    calibrationMessage = '⚠ Model may be over-confident (std(z) > 1)';
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* Title */}
      <div className="text-center mb-4">
        <h3 className="text-lg font-semibold whitespace-pre-line">{title}</h3>
        {calibrationMessage && (
          <p className={`text-sm mt-2 ${isWellCalibrated ? 'text-green-600' : 'text-yellow-600'}`}>
            {calibrationMessage}
          </p>
        )}
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-[500px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="theoretical"
              type="number"
              domain={['auto', 'auto']}
              label={{
                value: 'Theoretical Quantiles (Standard Normal)',
                position: 'bottom',
                offset: 10,
              }}
            />
            <YAxis
              type="number"
              domain={['auto', 'auto']}
              label={{
                value: 'Observed Quantiles (Standardized Residuals)',
                angle: -90,
                position: 'left',
                offset: 10,
              }}
            />
            <Tooltip
              formatter={(value: number) => value.toFixed(3)}
              labelFormatter={(value: number) => `Theoretical: ${value.toFixed(3)}`}
            />
            <Legend verticalAlign="top" height={36} />

            {/* Confidence band (if small sample) - rendered as shaded area */}
            {confidenceBandData.length > 0 && (
              <Area
                data={confidenceBandData}
                dataKey="bounds"
                stroke="none"
                fill="#fecaca"
                fillOpacity={0.5}
                name="Approximate 95% CI"
                isAnimationActive={false}
              />
            )}

            {/* Perfect calibration line */}
            <Line
              data={parityLineData}
              dataKey="parity"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Perfect calibration"
              isAnimationActive={false}
            />

            {/* Scatter points - rendered last so they're on top */}
            <Scatter
              data={chartData}
              dataKey="sample"
              fill="#3b82f6"
              fillOpacity={0.7}
              stroke="#1e40af"
              strokeWidth={0.5}
              name="Observations"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
