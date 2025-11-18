/**
 * Monitoring Dashboard - Read-only view for autonomous optimization
 * Polls session state and displays progress without allowing user intervention
 */
import { useEffect } from 'react';
import { useSessionState } from '../../hooks/api/useSessions';
import { useExperimentsSummary } from '../../hooks/api/useExperiments';
import { Activity, Bot, Database, Zap } from 'lucide-react';

interface MonitoringDashboardProps {
  sessionId: string;
  pollingInterval?: number; // milliseconds, default 60000 (1 minute)
}

export function MonitoringDashboard({ 
  sessionId, 
  pollingInterval = 60000 
}: MonitoringDashboardProps) {
  const { data: state, isLoading, dataUpdatedAt } = useSessionState(
    sessionId, 
    pollingInterval
  );
  const { data: summary } = useExperimentsSummary(sessionId);

  // Auto-scroll to top on mount
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const lastUpdateTime = new Date(dataUpdatedAt).toLocaleTimeString();

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        {/* Header with Bot Indicator */}
        <div className="mb-8 rounded-lg border-2 border-primary bg-primary/5 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Bot className="h-8 w-8 text-primary animate-pulse" />
            <div>
              <h1 className="text-3xl font-bold text-foreground">
                Autonomous Optimization Monitor
              </h1>
              <p className="text-muted-foreground mt-1">
                Read-only monitoring mode • Refreshes every {pollingInterval / 1000}s
              </p>
            </div>
          </div>
          <div className="mt-4 flex items-center gap-2 text-sm">
            <Activity className="h-4 w-4 text-primary" />
            <span className="text-muted-foreground">
              Last updated: {lastUpdateTime}
            </span>
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            <p className="mt-4 text-muted-foreground">Loading session state...</p>
          </div>
        ) : state ? (
          <div className="space-y-6">
            {/* Status Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Session Info Card */}
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="h-5 w-5 text-blue-500" />
                  <h3 className="font-semibold">Session</h3>
                </div>
                <p className="text-2xl font-bold">{state.session_id.slice(0, 8)}</p>
                <p className="text-xs text-muted-foreground mt-1">ID (truncated)</p>
              </div>

              {/* Variables Card */}
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  <h3 className="font-semibold">Variables</h3>
                </div>
                <p className="text-2xl font-bold">{state.n_variables}</p>
                <p className="text-xs text-muted-foreground mt-1">Search dimensions</p>
              </div>

              {/* Experiments Card */}
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-5 w-5 text-green-500" />
                  <h3 className="font-semibold">Experiments</h3>
                </div>
                <p className="text-2xl font-bold">{state.n_experiments}</p>
                <p className="text-xs text-muted-foreground mt-1">Data points collected</p>
              </div>

              {/* Model Status Card */}
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Bot className="h-5 w-5 text-purple-500" />
                  <h3 className="font-semibold">Model Status</h3>
                </div>
                <p className="text-2xl font-bold">
                  {state.model_trained ? '✓ Trained' : '○ Not Trained'}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {state.model_trained ? 'Ready for suggestions' : 'Awaiting data'}
                </p>
              </div>
            </div>

            {/* Summary Statistics */}
            {summary && summary.has_data && (
              <div className="rounded-lg border bg-card p-6">
                <h2 className="text-xl font-bold mb-4">Experiment Summary</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Experiments</p>
                    <p className="text-2xl font-bold">{summary.n_experiments}</p>
                  </div>
                  {summary.target_stats && (
                    <>
                      <div>
                        <p className="text-sm text-muted-foreground">Mean Output</p>
                        <p className="text-2xl font-bold">
                          {summary.target_stats.mean?.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Std Dev</p>
                        <p className="text-2xl font-bold">
                          {summary.target_stats.std?.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Range</p>
                        <p className="text-lg font-bold">
                          {summary.target_stats.min?.toFixed(2)} - {summary.target_stats.max?.toFixed(2)}
                        </p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Last Suggestion */}
            {state.last_suggestion && (
              <div className="rounded-lg border bg-card p-6">
                <h2 className="text-xl font-bold mb-4">Last Suggested Experiment</h2>
                <div className="bg-muted/50 rounded-lg p-4">
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {Object.entries(state.last_suggestion).map(([key, value]) => (
                      <div key={key}>
                        <p className="text-sm text-muted-foreground">{key}</p>
                        <p className="font-semibold">
                          {typeof value === 'number' ? value.toFixed(3) : value}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Information Notice */}
            <div className="rounded-lg border-2 border-blue-500/30 bg-blue-500/5 p-6">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <Bot className="h-5 w-5 text-blue-500" />
                Monitoring Mode Active
              </h3>
              <p className="text-sm text-muted-foreground">
                This dashboard is in read-only mode. An autonomous controller is managing 
                the optimization workflow. The view refreshes automatically every{' '}
                {pollingInterval / 1000} seconds to display the latest progress.
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                To stop autonomous optimization, use the hardware control interface (HMI) 
                or autonomous controller directly.
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-muted-foreground">Failed to load session state</p>
          </div>
        )}
      </div>
    </div>
  );
}
