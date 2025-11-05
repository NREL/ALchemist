/**
 * Models API endpoints
 */
import { apiClient } from '../client';
import type { 
  TrainModelRequest, 
  ModelInfo, 
  PredictionRequest,
  PredictionResponse
} from '../types';

/**
 * Train a surrogate model
 */
export const trainModel = async (
  sessionId: string,
  config: TrainModelRequest
): Promise<ModelInfo> => {
  const response = await apiClient.post<ModelInfo>(
    `/sessions/${sessionId}/model/train`,
    config
  );
  return response.data;
};

/**
 * Get model information
 */
export const getModelInfo = async (sessionId: string): Promise<ModelInfo> => {
  const response = await apiClient.get<ModelInfo>(`/sessions/${sessionId}/model`);
  return response.data;
};

/**
 * Make predictions with trained model
 */
export const predictWithModel = async (
  sessionId: string,
  inputs: PredictionRequest
): Promise<PredictionResponse> => {
  const response = await apiClient.post<PredictionResponse>(
    `/sessions/${sessionId}/model/predict`,
    inputs
  );
  return response.data;
};
