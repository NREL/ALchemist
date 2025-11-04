/**
 * Variables Panel - Main interface for defining search space
 * Mimics desktop SpaceSetupWindow layout
 */
import { useState, useRef } from 'react';
import { useVariables } from '../../hooks/api/useVariables';
import { VariableList } from './VariableList';
import { VariableForm } from './VariableForm';
import { useLoadVariablesFromFile, useExportVariablesToFile } from '../../hooks/api/useFileOperations';
import type { VariableDetail } from '../../api/types';

interface VariablesPanelProps {
  sessionId: string;
}

export function VariablesPanel({ sessionId }: VariablesPanelProps) {
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingVariable, setEditingVariable] = useState<VariableDetail | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { data: variablesData, isLoading } = useVariables(sessionId);
  const loadFromFile = useLoadVariablesFromFile(sessionId);
  const exportToFile = useExportVariablesToFile(sessionId);

  const handleAddVariable = () => {
    setEditingVariable(null);
    setIsFormOpen(true);
  };

  const handleEditVariable = (variable: VariableDetail) => {
    setEditingVariable(variable);
    setIsFormOpen(true);
  };

  const handleLoadFromFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await loadFromFile.mutateAsync(file);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleExportToFile = async () => {
    await exportToFile.mutateAsync();
  };

  return (
    <div className="rounded-lg border bg-card p-6">
      <h2 className="text-2xl font-bold mb-6">Search Space Setup</h2>
      
      {/* Main layout: left panel (variables) + right panel (controls) */}
      <div className="flex gap-6">
        {/* Left Panel - Variable Display */}
        <div className="flex-1 space-y-4">
          {isLoading ? (
            <div className="text-muted-foreground">Loading variables...</div>
          ) : variablesData && variablesData.n_variables > 0 ? (
            <VariableList 
              variables={variablesData.variables} 
              sessionId={sessionId}
              onEdit={handleEditVariable}
            />
          ) : (
            <div className="border border-dashed border-muted-foreground/25 rounded-lg p-12 text-center">
              <p className="text-muted-foreground mb-2">No variables defined yet</p>
              <p className="text-sm text-muted-foreground">
                Click "Add Variable" to define your search space
              </p>
            </div>
          )}
        </div>

        {/* Right Panel - Control Buttons (matches desktop UI) */}
        <div className="w-40 space-y-2 flex-shrink-0">
          <button
            onClick={handleAddVariable}
            className="w-full bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90 text-sm"
          >
            Add Variable
          </button>
          
          <div className="pt-4 border-t border-border space-y-2">
            <p className="text-xs text-muted-foreground px-1 mb-2">File Operations</p>
            <button
              onClick={handleLoadFromFile}
              disabled={loadFromFile.isPending}
              className="w-full border border-input px-4 py-2 rounded-md hover:bg-accent text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loadFromFile.isPending ? 'Loading...' : 'Load from File'}
            </button>
            
            <button
              onClick={handleExportToFile}
              disabled={exportToFile.isPending || !variablesData || variablesData.n_variables === 0}
              className="w-full border border-input px-4 py-2 rounded-md hover:bg-accent text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {exportToFile.isPending ? 'Exporting...' : 'Save to File'}
            </button>
          </div>
          
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleFileSelected}
            className="hidden"
          />
        </div>
      </div>

      {/* Variable Form Modal/Dialog */}
      {isFormOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background rounded-lg p-6 max-w-lg w-full mx-4 max-h-[90vh] overflow-y-auto shadow-xl border">
            <VariableForm 
              sessionId={sessionId}
              onClose={() => setIsFormOpen(false)}
              editingVariable={editingVariable}
            />
          </div>
        </div>
      )}
    </div>
  );
}
