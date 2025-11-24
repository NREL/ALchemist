import { useEffect, useState, useRef } from 'react';
import { Toaster, toast } from 'sonner';
import { QueryProvider } from './providers/QueryProvider';
import { VisualizationProvider, useVisualization } from './providers/VisualizationProvider';
import { 
  getStoredSessionId, 
  clearStoredSessionId, 
  useCreateSession, 
  useSession,
  useExportSession,
  useImportSession,
  useUpdateSessionMetadata
} from './hooks/api/useSessions';
import { VariablesPanel } from './features/variables/VariablesPanel';
import { ExperimentsPanel } from './features/experiments/ExperimentsPanel';
import { InitialDesignPanel } from './features/experiments/InitialDesignPanel';
import { GPRPanel } from './features/models/GPRPanel';
import { AcquisitionPanel } from './features/acquisition/AcquisitionPanel';
import { MonitoringDashboard } from './features/monitoring/MonitoringDashboard';
import { VisualizationsPanel } from './components/visualizations';
import { TabView } from './components/ui';
import { SessionMetadataDialog } from './components/SessionMetadataDialog';
import { useTheme } from './hooks/useTheme';
import { Sun, Moon, X } from 'lucide-react';
import './index.css';

function AppContent() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isMonitoringMode, setIsMonitoringMode] = useState<boolean>(false);
  const [showMetadataDialog, setShowMetadataDialog] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const createSession = useCreateSession();
  const exportSession = useExportSession();
  const importSession = useImportSession();
  const updateMetadata = useUpdateSessionMetadata(sessionId);
  const { data: session, error: sessionError } = useSession(sessionId);
  const { theme, toggleTheme } = useTheme();
  const { isVisualizationOpen, closeVisualization, sessionId: vizSessionId } = useVisualization();
  // Global staged suggestions to mirror desktop main_app.pending_suggestions
  const [pendingSuggestions, setPendingSuggestions] = useState<any[]>([]);

  // Restore pending suggestions from audit log on session load (desktop workflow)
  useEffect(() => {
    if (!sessionId) return;
    
    async function restorePendingSuggestions() {
      try {
        const response = await fetch(`/api/v1/sessions/${sessionId}/audit?entry_type=acquisition_locked`);
        if (!response.ok) return;
        
        const data = await response.json();
        if (data.entries && data.entries.length > 0) {
          // Get latest acquisition entry
          const latestAcq = data.entries[data.entries.length - 1];
          const suggestions = latestAcq.parameters?.suggestions || [];
          
          if (suggestions.length > 0) {
            // Tag suggestions with strategy for reason auto-fill
            const strategy = latestAcq.parameters?.strategy || 'Acquisition';
            const taggedSuggestions = suggestions.map((s: any) => ({
              ...s,
              _reason: strategy,
              Iteration: latestAcq.parameters?.iteration
            }));
            setPendingSuggestions(taggedSuggestions);
            console.log(`✓ Restored ${suggestions.length} pending suggestions from audit log`);
          }
        }
      } catch (e) {
        console.error('Failed to restore pending suggestions:', e);
      }
    }
    
    restorePendingSuggestions();
  }, [sessionId]);

  // Check for monitoring mode URL parameter
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const monitorParam = urlParams.get('mode');
    if (monitorParam === 'monitor') {
      setIsMonitoringMode(true);
    }
  }, []);

  // Load session ID from localStorage on mount
  useEffect(() => {
    const storedId = getStoredSessionId();
    if (storedId) {
      setSessionId(storedId);
    }
  }, []);

  // Auto-clear invalid session
  useEffect(() => {
    if (sessionError && sessionId) {
      toast.error('Session expired or not found. Please create a new session.');
      handleClearSession();
    }
  }, [sessionError, sessionId]);

  // Prompt to save on page close/reload (desktop _quit behavior)
  useEffect(() => {
    if (!sessionId) return;
    
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      // Only prompt if session has data
      if (session && (session.variable_count > 0 || session.experiment_count > 0)) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
        return e.returnValue;
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [sessionId, session]);

  // Prompt to save on page close/reload (desktop _quit behavior)
  useEffect(() => {
    if (!sessionId) return;
    
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      // Only prompt if session has data
      if (session && (session.variable_count > 0 || session.experiment_count > 0)) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
        return e.returnValue;
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [sessionId, session]);

  // Handle session creation
  const handleCreateSession = async () => {
    try {
      const newSession = await createSession.mutateAsync({ ttl_hours: 24 });
      setSessionId(newSession.session_id);
      toast.success('Session created successfully!');
      // Show metadata dialog for new session
      setShowMetadataDialog(true);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to create session');
      console.error('Error creating session:', error);
    }
  };

  // Handle clearing/resetting session
  const handleClearSession = () => {
    clearStoredSessionId();
    setSessionId(null);
    toast.info('Session cleared');
  };

  // Handle session export
  const handleExportSession = async () => {
    if (!sessionId) return;
    try {
      await exportSession.mutateAsync(sessionId);
      toast.success('Session exported successfully!');
    } catch (error: any) {
      toast.error('Failed to export session');
      console.error('Error exporting session:', error);
    }
  };

  // Handle session import
  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        const newSession = await importSession.mutateAsync(file);
        setSessionId(newSession.session_id);
        toast.success('Session imported successfully!');
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } catch (error: any) {
        toast.error(error.response?.data?.detail || 'Failed to import session');
        console.error('Error importing session:', error);
      }
    }
  };

  // Handle audit log export (matches desktop File → Export Audit Log)
  const handleExportAuditLog = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`/api/v1/sessions/${sessionId}/audit/export`);
      if (!response.ok) throw new Error('Failed to export audit log');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit_log_${sessionId.substring(0, 8)}.md`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success('Audit log exported successfully!');
    } catch (error: any) {
      toast.error('Failed to export audit log');
      console.error('Error exporting audit log:', error);
    }
  };

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      {/* Monitoring Mode - Show dedicated dashboard */}
      {isMonitoringMode && sessionId ? (
        <MonitoringDashboard sessionId={sessionId} pollingInterval={90000} />
      ) : (
        <>
          {/* Header - Always visible */}
          <header className="border-b bg-card px-6 py-1 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex flex-col gap-0.5">
                  <img 
                    src={theme === 'dark' ? '/NEW_LOGO_DARK.png' : '/NEW_LOGO_LIGHT.png'} 
                    alt="ALchemist" 
                    className="h-auto"
                    style={{ width: '250px' }}
                  />
                  <p className="text-xs text-muted-foreground">
                    Active Learning Toolkit for Chemical and Materials Research
                  </p>
                </div>
                
                {/* Theme Toggle */}
                <button
                  onClick={toggleTheme}
                  className="p-2 rounded-md hover:bg-accent transition-colors"
                  title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                  {theme === 'dark' ? (
                    <Sun className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                  ) : (
                    <Moon className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                  )}
                </button>
              </div>
              
              {/* Session Controls */}
              {sessionId ? (
                <div className="flex items-center gap-3">
                  <div className="text-sm text-muted-foreground">
                    <code className="bg-muted px-2 py-1 rounded text-xs">
                      {sessionId.substring(0, 8)}
                    </code>
                    {session && (
                      <span className="ml-2">
                        {session.variable_count}V · {session.experiment_count}E
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => setShowMetadataDialog(true)}
                    className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-accent"
                    title="Edit session metadata"
                  >
                    Edit Info
                  </button>
                  <button
                    onClick={handleExportAuditLog}
                    className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-accent"
                    title="Export audit log as markdown"
                  >
                    Export Log
                  </button>
                  <button
                    onClick={handleExportSession}
                    disabled={exportSession.isPending}
                    className="text-xs bg-primary text-primary-foreground px-3 py-1.5 rounded hover:bg-primary/90 transition-colors disabled:opacity-50"
                  >
                    Save
                  </button>
                  <button
                    onClick={handleClearSession}
                    className="text-xs text-destructive hover:text-destructive/80 px-3 py-1.5 border border-destructive/30 rounded hover:bg-destructive/10 transition-colors"
                  >
                    Clear
                  </button>
                </div>
              ) : (
                <div className="flex gap-2">
                  <button 
                    onClick={handleCreateSession}
                    disabled={createSession.isPending}
                    className="text-sm bg-primary text-primary-foreground px-4 py-2 rounded hover:bg-primary/90 disabled:opacity-50"
                  >
                    {createSession.isPending ? 'Creating...' : 'New Session'}
                  </button>
                  <button 
                    onClick={handleImportClick}
                    disabled={importSession.isPending}
                    className="text-sm bg-secondary text-secondary-foreground px-4 py-2 rounded hover:bg-secondary/90 disabled:opacity-50"
                  >
                    {importSession.isPending ? 'Loading...' : 'Load Session'}
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json"
                    onChange={handleFileSelected}
                    className="hidden"
                  />
                </div>
              )}
            </div>
          </header>

          {/* Main Content Area - 3 Column Desktop Layout */}
          {sessionId ? (
            <div className="flex-1 flex overflow-hidden">
              {/* LEFT SIDEBAR - Variables & Experiments (fixed width, increased for better readability) */}
              <div className="w-[580px] flex-shrink-0 overflow-y-auto border-r bg-card p-4 space-y-4">
                <VariablesPanel sessionId={sessionId} />
                <ExperimentsPanel 
                  sessionId={sessionId} 
                  pendingSuggestions={pendingSuggestions}
                  onStageSuggestions={(s:any[])=>setPendingSuggestions(s)}
                />
                <InitialDesignPanel sessionId={sessionId} />
              </div>

              {/* CENTER - Visualization Area (expandable) */}
              <div className="flex-1 flex flex-col bg-background">
                {isVisualizationOpen && vizSessionId ? (
                  <>
                    {/* Visualization Header */}
                    <div className="border-b bg-card px-4 py-3 flex items-center justify-between">
                      <h3 className="font-semibold">Model Visualizations</h3>
                      <button
                        onClick={closeVisualization}
                        className="p-1 rounded hover:bg-accent transition-colors"
                        title="Close visualizations"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                    
                    {/* Embedded Visualizations */}
                    <div className="flex-1 overflow-auto">
                      <VisualizationsPanel 
                        sessionId={vizSessionId} 
                        embedded={true}
                      />
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center p-6">
                    <div className="text-center text-muted-foreground">
                      <div className="text-6xl mb-4">📊</div>
                      <p className="text-lg font-medium mb-2">Visualization Panel</p>
                      <p className="text-sm">
                        Train a model to see visualizations here
                      </p>
                      <p className="text-xs mt-2 text-muted-foreground/60">
                        Plots will be embedded in this panel
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT PANEL - Model & Acquisition Tabs (fixed width) */}
              <div className="w-[320px] flex-shrink-0 border-l bg-card">
                <TabView
                  tabs={[
                    {
                      id: 'model',
                      label: 'Model',
                      content: <GPRPanel sessionId={sessionId} />,
                    },
                    {
                      id: 'acquisition',
                      label: 'Acquisition',
                      content: (
                          <AcquisitionPanel 
                            sessionId={sessionId} 
                            modelBackend={session?.model_trained ? (session as any).model_backend : null} 
                            pendingSuggestions={pendingSuggestions}
                            onStageSuggestions={(s:any[])=>setPendingSuggestions(s)}
                          />
                        ),
                    },
                  ]}
                  defaultTab="model"
                />
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-background">
              <div className="text-center max-w-md">
                <div className="text-6xl mb-4">🧪</div>
                <h2 className="text-2xl font-bold mb-4">Welcome to ALchemist</h2>
                <p className="text-muted-foreground mb-6">
                  Create a new session or load a previously saved session to begin your optimization workflow.
                </p>
              </div>
            </div>
          )}
        </>
      )}
      
      {/* Toast notifications */}
      <Toaster position="top-right" richColors />
      
      {/* Session Metadata Dialog */}
      {showMetadataDialog && sessionId && session && (
        <SessionMetadataDialog
          sessionId={sessionId}
          metadata={session.metadata || {}}
          onSave={async (metadata) => {
            try {
              await updateMetadata.mutateAsync(metadata);
              toast.success('Session metadata updated');
              setShowMetadataDialog(false);
            } catch (e: any) {
              toast.error('Failed to update metadata: ' + (e?.message || String(e)));
            }
          }}
          onCancel={() => setShowMetadataDialog(false)}
        />
      )}
    </div>
  );
}

function App() {
  return (
    <QueryProvider>
      <VisualizationProvider>
        <AppContent />
      </VisualizationProvider>
    </QueryProvider>
  );
}

export default App;
