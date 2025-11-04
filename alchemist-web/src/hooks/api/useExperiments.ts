/**
 * React Query hooks for experiments
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as experimentsAPI from '../../api/endpoints/experiments';
import type { Experiment } from '../../api/types';
import { toast } from 'sonner';

/**
 * Hook to get all experiments
 */
export function useExperiments(sessionId: string | null) {
  return useQuery({
    queryKey: ['experiments', sessionId],
    queryFn: () => experimentsAPI.getExperiments(sessionId!),
    enabled: !!sessionId,
  });
}

/**
 * Hook to get experiments summary
 */
export function useExperimentsSummary(sessionId: string | null) {
  return useQuery({
    queryKey: ['experiments-summary', sessionId],
    queryFn: () => experimentsAPI.getExperimentSummary(sessionId!),
    enabled: !!sessionId,
  });
}

/**
 * Hook to add a single experiment
 */
export function useCreateExperiment(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (experiment: Experiment) => 
      experimentsAPI.createExperiment(sessionId, experiment),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['experiments-summary', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success('Experiment added successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to add experiment');
    },
  });
}

/**
 * Hook to upload experiments from CSV file
 */
export function useUploadExperiments(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (file: File) => 
      experimentsAPI.uploadExperimentsCSV(sessionId, file),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['experiments', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['experiments-summary', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success(`Loaded ${data.n_experiments || 0} experiments successfully!`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to upload experiments');
    },
  });
}
