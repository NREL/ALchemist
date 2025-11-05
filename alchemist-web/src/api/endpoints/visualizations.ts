/**
 * API endpoints for visualizations
 */
import { apiClient } from '../client';
import type {
  ContourDataRequest,
  ContourDataResponse,
  ParityDataResponse,
  MetricsDataResponse,
  QQPlotDataResponse,
  CalibrationCurveDataResponse,
  HyperparametersResponse
} from '../types';

/**
 * Get contour plot data for 2D model predictions
 */
export async function getContourData(
  sessionId: string,
  request: ContourDataRequest
): Promise<ContourDataResponse> {
  const response = await apiClient.post<ContourDataResponse>(
    `/sessions/${sessionId}/visualizations/contour`,
    request
  );
  return response.data;
}

/**
 * Get parity plot data (actual vs predicted)
 */
export async function getParityData(
  sessionId: string,
  useCalibrated: boolean = false
): Promise<ParityDataResponse> {
  const response = await apiClient.get<ParityDataResponse>(
    `/sessions/${sessionId}/visualizations/parity`,
    {
      params: { use_calibrated: useCalibrated }
    }
  );
  return response.data;
}

/**
 * Get CV metrics over training size
 */
export async function getMetricsData(
  sessionId: string,
  cvSplits: number = 5
): Promise<MetricsDataResponse> {
  const response = await apiClient.get<MetricsDataResponse>(
    `/sessions/${sessionId}/visualizations/metrics`,
    {
      params: { cv_splits: cvSplits }
    }
  );
  return response.data;
}

/**
 * Get Q-Q plot data for residual analysis
 */
export async function getQQPlotData(
  sessionId: string,
  useCalibrated: boolean = false
): Promise<QQPlotDataResponse> {
  const response = await apiClient.get<QQPlotDataResponse>(
    `/sessions/${sessionId}/visualizations/qq-plot`,
    {
      params: { use_calibrated: useCalibrated }
    }
  );
  return response.data;
}

/**
 * Get calibration curve data (reliability diagram)
 */
export async function getCalibrationCurveData(
  sessionId: string,
  useCalibrated: boolean = false
): Promise<CalibrationCurveDataResponse> {
  const response = await apiClient.get<CalibrationCurveDataResponse>(
    `/sessions/${sessionId}/visualizations/calibration-curve`,
    {
      params: { use_calibrated: useCalibrated }
    }
  );
  return response.data;
}

/**
 * Get model hyperparameters and configuration
 */
export async function getHyperparameters(
  sessionId: string
): Promise<HyperparametersResponse> {
  const response = await apiClient.get<HyperparametersResponse>(
    `/sessions/${sessionId}/visualizations/hyperparameters`
  );
  return response.data;
}
