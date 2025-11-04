/**
 * API hooks for model training and management
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { toast } from 'sonner';
import type { 
  TrainModelRequest, 
  TrainModelResponse, 
  ModelInfo 
} from '../../api/types';

/**
 * Hook to get current model info
 */
export function useModelInfo(sessionId: string) {
  return useQuery({
    queryKey: ['model-info', sessionId],
    queryFn: async () => {
      const response = await apiClient.get<ModelInfo>(`/sessions/${sessionId}/model`);
      return response.data;
    },
    enabled: !!sessionId,
  });
}

/**
 * Hook to train a model
 */
export function useTrainModel(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (request: TrainModelRequest) => {
      const response = await apiClient.post<TrainModelResponse>(
        `/sessions/${sessionId}/model/train`,
        request
      );
      return response.data;
    },
    onSuccess: (data) => {
      // Invalidate model info to refetch
      queryClient.invalidateQueries({ queryKey: ['model-info', sessionId] });
      // Also invalidate session info since model status changed
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      // Show success toast
      toast.success(`Model trained successfully! R² = ${data.metrics.r2.toFixed(4)}`);
    },
    onError: (error: any) => {
      // Show error toast
      toast.error(error.response?.data?.detail || 'Failed to train model');
    },
  });
}
