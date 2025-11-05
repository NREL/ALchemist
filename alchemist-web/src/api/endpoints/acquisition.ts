/**
 * Acquisition API endpoints
 */
import { apiClient } from '../client';
import type { AcquisitionRequest, AcquisitionResponse } from '../types';

/**
 * Get next experiment suggestions using acquisition function
 */
export const getSuggestions = async (
  sessionId: string,
  request: AcquisitionRequest
): Promise<AcquisitionResponse> => {
  const response = await apiClient.post<AcquisitionResponse>(
    `/sessions/${sessionId}/acquisition/suggest`,
    request
  );
  return response.data;
};
