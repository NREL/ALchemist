/**
 * Acquisition API endpoints
 */
import { apiClient } from '../client';
import type { SuggestRequest, SuggestResponse } from '../types';

/**
 * Get next experiment suggestions using acquisition function
 */
export const getSuggestions = async (
  sessionId: string,
  request: SuggestRequest
): Promise<SuggestResponse> => {
  const response = await apiClient.post<SuggestResponse>(
    `/sessions/${sessionId}/acquisition/suggest`,
    request
  );
  return response.data;
};
