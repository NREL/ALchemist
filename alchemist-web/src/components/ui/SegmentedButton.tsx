/**
 * SegmentedButton Component
 * Mimics CustomTkinter's CTkSegmentedButton for mutually exclusive options
 */
import type { ReactNode } from 'react';

interface SegmentedButtonProps<T extends string> {
  options: readonly T[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  labels?: Record<T, ReactNode>;
  className?: string;
}

export function SegmentedButton<T extends string>({
  options,
  value,
  onChange,
  disabled = false,
  labels,
  className = '',
}: SegmentedButtonProps<T>) {
  return (
    <div className={`inline-flex rounded-md bg-muted p-1 ${className}`}>
      {options.map((option) => (
        <button
          key={option}
          onClick={() => onChange(option)}
          disabled={disabled}
          className={`
            px-4 py-2 text-sm font-medium rounded transition-all
            disabled:opacity-50 disabled:cursor-not-allowed
            ${
              value === option
                ? 'bg-primary text-primary-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }
          `}
        >
          {labels?.[option] || option}
        </button>
      ))}
    </div>
  );
}
