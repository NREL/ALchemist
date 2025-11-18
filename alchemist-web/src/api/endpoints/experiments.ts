/**
 * Experiments API endpoints
 */
import { apiClient } from '../client';
import type { 
  Experiment,
  ExperimentSummary,
  InitialDesignRequest,
  InitialDesignResponse 
} from '../types';

/**
 * Add a single experiment
 */
export const createExperiment = async (
  sessionId: string,
  experiment: Experiment
): Promise<{ message: string }> => {
  const response = await apiClient.post(
    `/sessions/${sessionId}/experiments`,
    experiment
  );
  return response.data;
};

/**
 * Add multiple experiments
 */
export const createExperimentBatch = async (
  sessionId: string,
  experiments: Experiment[]
): Promise<{ message: string; count: number }> => {
  const response = await apiClient.post(
    `/sessions/${sessionId}/experiments/batch`,
    { experiments }
  );
  return response.data;
};

/**
 * Upload experiments from CSV file
 */
export const uploadExperimentsCSV = async (
  sessionId: string,
  file: File
): Promise<{ message: string; n_experiments: number }> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(
    `/sessions/${sessionId}/experiments/upload`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

/**
 * Get all experiments
 */
export const getExperiments = async (
  sessionId: string
): Promise<{ experiments: any[]; n_experiments: number }> => {
  const response = await apiClient.get(`/sessions/${sessionId}/experiments`);
  return response.data;
};

/**
 * Get experiment summary statistics
 */
export const getExperimentSummary = async (
  sessionId: string
): Promise<ExperimentSummary> => {
  const response = await apiClient.get<ExperimentSummary>(
    `/sessions/${sessionId}/experiments/summary`
  );
  return response.data;
};

/**
 * Generate initial experimental design (DoE)
 */
export const generateInitialDesign = async (
  sessionId: string,
  request: InitialDesignRequest
): Promise<InitialDesignResponse> => {
  const response = await apiClient.post<InitialDesignResponse>(
    `/sessions/${sessionId}/initial-design`,
    request
  );
  return response.data;
};
