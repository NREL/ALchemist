/**
 * TypeScript types for ALchemist API
 * These interfaces match the Pydantic models in the FastAPI backend
 */

// ============================================================================
// Session Types
// ============================================================================

export interface Session {
  session_id: string;
  created_at: string;
  ttl_hours: number;
  expires_at: string;
  variable_count: number;
  experiment_count: number;
  model_trained: boolean;
}

export interface CreateSessionRequest {
  ttl_hours?: number;
}

export interface CreateSessionResponse {
  session_id: string;
  created_at: string;
  ttl_hours: number;
  expires_at: string;
}

export interface UpdateTTLRequest {
  ttl_hours: number;
}

// ============================================================================
// Variable Types
// ============================================================================

export type VariableType = 'continuous' | 'discrete' | 'categorical';

// API expects these type values
export type APIVariableType = 'real' | 'integer' | 'categorical';

export interface Variable {
  name: string;
  type: VariableType;
  bounds?: [number, number];  // For continuous/discrete
  categories?: string[];      // For categorical
  unit?: string;
  description?: string;
}

// API request format (what backend expects for POST)
export interface APIVariable {
  name: string;
  type: APIVariableType;
  min?: number;              // For real/integer (API format)
  max?: number;              // For real/integer (API format)
  categories?: string[];     // For categorical
  unit?: string;
  description?: string;
}

// What backend returns in GET (includes bounds array)
export interface VariableDetail {
  name: string;
  type: APIVariableType;  // Backend returns 'real', 'integer', 'categorical'
  bounds?: [number, number] | null;
  categories?: string[] | null;
  unit?: string;
  description?: string;
}

export interface VariablesListResponse {
  variables: VariableDetail[];
  n_variables: number;  // Backend returns n_variables not count
}

// ============================================================================
// Experiment Types
// ============================================================================

export interface Experiment {
  inputs: Record<string, number | string>;
  output?: number;
  noise?: number;
  [key: string]: any;  // Allow indexing by string for dynamic column access
}

export interface ExperimentBatch {
  experiments: Experiment[];
}

export interface ExperimentSummary {
  n_experiments: number;
  has_data: boolean;
  has_noise?: boolean;
  target_stats?: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  feature_names?: string[];
}

// ============================================================================
// Model Types
// ============================================================================

export type ModelBackend = 'sklearn' | 'botorch';
export type KernelType = 'rbf' | 'matern' | 'periodic' | 'rational_quadratic';
export type TransformType = 'normalize' | 'standardize' | null;

export interface TrainModelRequest {
  backend: ModelBackend;
  kernel?: KernelType;
  kernel_params?: Record<string, any>;
  input_transform?: TransformType;
  output_transform?: TransformType;
  calibration_enabled?: boolean;
}

export interface ModelInfo {
  backend: ModelBackend;
  kernel: string;
  hyperparameters: Record<string, any>;
  cv_metrics: {
    rmse: number;
    mae: number;
    mape: number;
    r2: number;
  };
  calibration_factor?: number;
  training_time: number;
}

export interface PredictionInput {
  inputs: Record<string, number | string>[];
}

export interface Prediction {
  inputs: Record<string, number | string>;
  prediction: number;
  uncertainty: number;
}

export interface PredictionResponse {
  predictions: Prediction[];
  n_predictions: number;
}

// ============================================================================
// Acquisition Types
// ============================================================================

export type AcquisitionStrategy = 
  | 'EI' | 'PI' | 'UCB' 
  | 'qEI' | 'qPI' | 'qUCB' | 'qNIPV';

export type OptimizationGoal = 'maximize' | 'minimize';

export interface SuggestRequest {
  strategy: AcquisitionStrategy;
  goal: OptimizationGoal;
  n_suggestions?: number;
  xi?: number;    // For EI/PI
  kappa?: number; // For UCB
}

export interface Suggestion {
  inputs: Record<string, number | string>;
  acquisition_value: number;
}

export interface SuggestResponse {
  suggestions: Suggestion[];
  strategy: AcquisitionStrategy;
  n_suggestions: number;
}

// ============================================================================
// Error Types
// ============================================================================

export interface APIError {
  detail: string;
}
