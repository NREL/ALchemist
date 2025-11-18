/**
 * CollapsibleSection Component
 * For advanced options that can be toggled on/off like desktop UI
 */
import { useState } from 'react';
import type { ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface CollapsibleSectionProps {
  title: string;
  children: ReactNode;
  defaultExpanded?: boolean;
  disabled?: boolean;
}

export function CollapsibleSection({
  title,
  children,
  defaultExpanded = false,
  disabled = false,
}: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={() => !disabled && setIsExpanded(!isExpanded)}
        disabled={disabled}
        className={`
          w-full flex items-center justify-between px-4 py-3 text-sm font-medium
          bg-muted/30 hover:bg-muted/50 transition-colors
          disabled:opacity-50 disabled:cursor-not-allowed
        `}
      >
        <span>{title}</span>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4" />
        ) : (
          <ChevronRight className="w-4 h-4" />
        )}
      </button>
      {isExpanded && (
        <div className="p-4 space-y-3 bg-card">
          {children}
        </div>
      )}
    </div>
  );
}
