import { useState, useEffect, useCallback } from 'react';
import { toast } from 'sonner';

interface LockStatus {
  locked: boolean;
  locked_by: string | null;
  locked_at: string | null;
}

interface UseLockStatusReturn {
  lockStatus: LockStatus | null;
  isLoading: boolean;
  error: Error | null;
  checkLockStatus: () => Promise<void>;
}

/**
 * Hook to poll session lock status and detect external controllers
 * 
 * When a session is locked by an external controller (e.g., Qt app),
 * this hook detects the lock and can trigger automatic monitor mode.
 * 
 * @param sessionId - The session ID to monitor
 * @param pollingInterval - How often to poll in milliseconds (default: 5000ms)
 * @param onLockStateChange - Callback when lock state changes
 */
export function useLockStatus(
  sessionId: string | null,
  pollingInterval: number = 5000,
  onLockStateChange?: (locked: boolean, lockedBy: string | null) => void
): UseLockStatusReturn {
  const [lockStatus, setLockStatus] = useState<LockStatus | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [previousLockState, setPreviousLockState] = useState<boolean | null>(null);

  const checkLockStatus = useCallback(async () => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/sessions/${sessionId}/lock`
      );

      if (!response.ok) {
        throw new Error(`Failed to check lock status: ${response.statusText}`);
      }

      const data: LockStatus = await response.json();
      setLockStatus(data);

      // Detect lock state changes
      if (previousLockState !== null && previousLockState !== data.locked) {
        if (data.locked) {
          toast.info(`External controller connected: ${data.locked_by}`, {
            duration: 5000,
          });
        } else {
          toast.info('External controller disconnected - resuming interactive mode', {
            duration: 3000,
          });
        }

        // Call the callback if provided
        if (onLockStateChange) {
          onLockStateChange(data.locked, data.locked_by);
        }
      }

      setPreviousLockState(data.locked);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      console.error('Error checking lock status:', error);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, previousLockState, onLockStateChange]);

  // Poll lock status at regular intervals
  useEffect(() => {
    if (!sessionId) return;

    // Check immediately
    checkLockStatus();

    // Set up polling
    const interval = setInterval(checkLockStatus, pollingInterval);

    return () => clearInterval(interval);
  }, [sessionId, pollingInterval, checkLockStatus]);

  return {
    lockStatus,
    isLoading,
    error,
    checkLockStatus,
  };
}
