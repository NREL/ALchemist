/**
 * FormField Components
 * Consistent styling for form inputs matching desktop UI
 */
import { type ReactNode } from 'react';

interface FormFieldProps {
  label: string;
  children: ReactNode;
  disabled?: boolean;
  help?: string;
  className?: string;
}

export function FormField({ label, children, disabled = false, help, className = '' }: FormFieldProps) {
  return (
    <div className={className}>
      <label className={`block text-sm font-medium mb-1.5 ${disabled ? 'text-muted-foreground' : ''}`}>
        {label}
      </label>
      {children}
      {help && (
        <p className="mt-1 text-xs text-muted-foreground italic">{help}</p>
      )}
    </div>
  );
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  options: { value: string; label: string }[];
}

export function Select({ options, className = '', ...props }: SelectProps) {
  return (
    <select
      {...props}
      className={`
        w-full px-3 py-2 text-sm rounded-md border bg-background
        disabled:opacity-50 disabled:cursor-not-allowed
        focus:outline-none focus:ring-2 focus:ring-primary/50
        ${className}
      `}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  showValue?: boolean;
  formatValue?: (val: number) => string;
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  disabled = false,
  showValue = true,
  formatValue = (v) => v.toFixed(4),
}: SliderProps) {
  return (
    <div className="space-y-1">
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed slider"
      />
      {showValue && (
        <div className="text-right text-sm font-medium tabular-nums">
          {formatValue(value)}
        </div>
      )}
      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: hsl(var(--primary));
          border-radius: 50%;
          cursor: pointer;
        }
        .slider::-moz-range-thumb {
          width: 16px;
          height: 16px;
          background: hsl(var(--primary));
          border-radius: 50%;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
}
