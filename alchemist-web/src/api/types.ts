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
export type KernelType = 'RBF' | 'Matern' | 'RationalQuadratic';
export type MaternNu = '0.5' | '1.5' | '2.5' | 'inf';

// Sklearn-specific options
export type SklearnInputTransform = 'none' | 'minmax' | 'standard' | 'robust';
export type SklearnOutputTransform = 'none' | 'minmax' | 'standard' | 'robust';
export type SklearnOptimizer = 'CG' | 'BFGS' | 'L-BFGS-B' | 'TNC';

// BoTorch-specific options
export type BoTorchInputTransform = 'none' | 'normalize' | 'standardize';
export type BoTorchOutputTransform = 'none' | 'standardize';

export interface TrainModelRequest {
  backend: ModelBackend;
  kernel: KernelType;
  kernel_params?: {
    nu?: number;  // For Matern kernel
    [key: string]: any;
  };
  input_transform?: string;  // Transform type (backend-specific)
  output_transform?: string;  // Transform type (backend-specific)
  calibration_enabled?: boolean;
}

export interface ModelMetrics {
  rmse: number;
  mae: number;
  r2: number;
  mape?: number;
}

export interface TrainModelResponse {
  success: boolean;
  backend: ModelBackend;
  kernel: KernelType;
  hyperparameters: Record<string, any>;
  metrics: ModelMetrics;
  message: string;
}

export interface ModelInfo {
  backend: ModelBackend | null;
  hyperparameters: Record<string, any> | null;
  metrics: ModelMetrics | null;
  is_trained: boolean;
}

// ============================================================================
// Prediction Types
// ============================================================================

export interface PredictionRequest {
  inputs: Array<Record<string, number | string>>;
}

export interface PredictionResult {
  inputs: Record<string, number | string>;
  prediction: number;
  uncertainty: number;
}

export interface PredictionResponse {
  predictions: PredictionResult[];
  n_predictions: number;
}

// ============================================================================
// Acquisition Types
// ============================================================================

export type AcquisitionStrategy = 'EI' | 'PI' | 'UCB' | 'qEI' | 'qUCB' | 'qNIPV';
export type OptimizationGoal = 'maximize' | 'minimize';

export interface AcquisitionRequest {
  strategy: AcquisitionStrategy;
  goal: OptimizationGoal;
  n_suggestions?: number;
  xi?: number;      // For EI/PI
  kappa?: number;   // For UCB
}

export interface AcquisitionResponse {
  suggestions: Array<Record<string, any>>;
  n_suggestions: number;
}

// ============================================================================
// Find Optimum Types
// ============================================================================

export interface FindOptimumRequest {
  goal: OptimizationGoal;
}

export interface FindOptimumResponse {
  optimum: Record<string, any>;
  predicted_value: number;
  predicted_std: number | null;
  goal: string;
}

// ============================================================================
// Error Types
// ============================================================================

export interface APIError {
  detail: string;
}
