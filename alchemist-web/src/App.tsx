import { useEffect, useState, useRef } from 'react';
import { Toaster, toast } from 'sonner';
import { QueryProvider } from './providers/QueryProvider';
import { 
  getStoredSessionId, 
  clearStoredSessionId, 
  useCreateSession, 
  useSession,
  useExportSession,
  useImportSession
} from './hooks/api/useSessions';
import { VariablesPanel } from './features/variables/VariablesPanel';
import { ExperimentsPanel } from './features/experiments/ExperimentsPanel';
import { InitialDesignPanel } from './features/experiments/InitialDesignPanel';
import { GPRPanel } from './features/models/GPRPanel';
import { AcquisitionPanel } from './features/acquisition/AcquisitionPanel';
import { MonitoringDashboard } from './features/monitoring/MonitoringDashboard';
import './index.css';

function AppContent() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isMonitoringMode, setIsMonitoringMode] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const createSession = useCreateSession();
  const exportSession = useExportSession();
  const importSession = useImportSession();
  const { data: session, isLoading: isLoadingSession, error: sessionError } = useSession(sessionId);

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

  // Handle session creation
  const handleCreateSession = async () => {
    try {
      const newSession = await createSession.mutateAsync({ ttl_hours: 24 });
      setSessionId(newSession.session_id);
      toast.success('Session created successfully!');
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

  return (
    <div className="min-h-screen bg-background">
      {/* Monitoring Mode - Show dedicated dashboard */}
      {isMonitoringMode && sessionId ? (
        <MonitoringDashboard sessionId={sessionId} pollingInterval={90000} />
      ) : (
        // Normal Mode - Show full interface
        <div className="container mx-auto p-8">
          <header className="mb-8">
            <h1 className="text-4xl font-bold text-foreground mb-2">
              ALchemist
            </h1>
            <p className="text-muted-foreground">
              Active Learning Toolkit for Chemical and Materials Research
            </p>
          </header>

          <div className="space-y-4">
            {sessionId ? (
              <>
                {/* Session Info Card */}
                <div className="rounded-lg border bg-card p-6">
                  <div className="flex justify-between items-start mb-4">
                    <h2 className="text-2xl font-semibold">Session Active</h2>
                    <div className="flex gap-2">
                      <button
                        onClick={handleExportSession}
                        disabled={exportSession.isPending}
                        className="text-sm bg-primary text-primary-foreground px-3 py-1 rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {exportSession.isPending ? 'Exporting...' : 'Save Session'}
                      </button>
                      <button
                        onClick={handleClearSession}
                        className="text-sm text-destructive hover:text-destructive/80 px-3 py-1 border border-destructive/30 rounded-md hover:bg-destructive/10 transition-colors"
                      >
                        Clear Session
                      </button>
                    </div>
                  </div>
                  {isLoadingSession ? (
                    <p className="text-muted-foreground">Loading session info...</p>
                  ) : session ? (
                    <div className="space-y-2">
                      <p className="text-muted-foreground">
                        Session ID: <code className="bg-muted px-2 py-1 rounded">{sessionId}</code>
                      </p>
                      <p className="text-muted-foreground text-sm">
                        Created: {new Date(session.created_at).toLocaleString()}
                      </p>
                      <p className="text-muted-foreground text-sm">
                        Expires: {new Date(session.expires_at).toLocaleString()}
                      </p>
                      <p className="text-muted-foreground text-sm">
                        Variables: {session.variable_count} | Experiments: {session.experiment_count}
                      </p>
                    </div>
                  ) : (
                    <p className="text-muted-foreground">Session not found</p>
                  )}
                </div>

                {/* Variables Panel */}
                <VariablesPanel sessionId={sessionId} />

                {/* Experiments Panel */}
                <ExperimentsPanel sessionId={sessionId} />

                {/* Initial Design Panel */}
                <InitialDesignPanel sessionId={sessionId} />

                {/* GPR Model Panel */}
                <GPRPanel sessionId={sessionId} />

                {/* Acquisition Panel */}
                <AcquisitionPanel sessionId={sessionId} modelBackend={session?.model_trained ? (session as any).model_backend : null} />
              </>
            ) : (
              <div className="rounded-lg border bg-card p-6">
                <h2 className="text-2xl font-semibold mb-4">Welcome to ALchemist</h2>
                <p className="text-muted-foreground mb-4">
                  Create a new session or load a previously saved session.
                </p>
                <div className="flex gap-3">
                  <button 
                    onClick={handleCreateSession}
                    disabled={createSession.isPending}
                    className="bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {createSession.isPending ? 'Creating...' : 'Create New Session'}
                  </button>
                  <button 
                    onClick={handleImportClick}
                    disabled={importSession.isPending}
                    className="bg-secondary text-secondary-foreground px-4 py-2 rounded-md hover:bg-secondary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {importSession.isPending ? 'Loading...' : 'Load Session'}
                  </button>
                  {/* Hidden file input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pkl"
                    onChange={handleFileSelected}
                    className="hidden"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Toast notifications */}
      <Toaster position="top-right" richColors />
    </div>
  );
}

function App() {
  return (
    <QueryProvider>
      <AppContent />
    </QueryProvider>
  );
}

export default App;
