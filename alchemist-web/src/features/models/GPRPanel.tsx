/**
 * GPR Panel - Gaussian Process Model Training
 * Mimics desktop UI gpr_panel.py layout and functionality
 */
import { useState } from 'react';
import { useModelInfo, useTrainModel } from '../../hooks/api/useModels';
import { useExperimentsSummary } from '../../hooks/api/useExperiments';
import { VisualizationsPanel } from '../../components/visualizations';
import type { 
  ModelBackend, 
  KernelType, 
  MaternNu,
  SklearnInputTransform,
  SklearnOutputTransform,
  SklearnOptimizer,
  BoTorchInputTransform,
  BoTorchOutputTransform,
  TrainModelRequest 
} from '../../api/types';
import { CheckCircle2, AlertCircle, Loader2, LineChart } from 'lucide-react';

interface GPRPanelProps {
  sessionId: string;
}

export function GPRPanel({ sessionId }: GPRPanelProps) {
  // State for backend selection
  const [backend, setBackend] = useState<ModelBackend>('sklearn');
  const [advancedEnabled, setAdvancedEnabled] = useState(false);
  
  // Visualizations panel state
  const [showVisualizations, setShowVisualizations] = useState(false);
  
  // Sklearn-specific state
  const [skKernel, setSkKernel] = useState<KernelType>('RBF');
  const [skMaternNu, setSkMaternNu] = useState<MaternNu>('1.5');
  const [skOptimizer, setSkOptimizer] = useState<SklearnOptimizer>('L-BFGS-B');
  const [skInputTransform, setSkInputTransform] = useState<SklearnInputTransform>('none');
  const [skOutputTransform, setSkOutputTransform] = useState<SklearnOutputTransform>('none');
  const [skCalibrateUncertainty, setSkCalibrateUncertainty] = useState(false);
  
  // BoTorch-specific state
  const [btKernel, setBtKernel] = useState<KernelType>('Matern');
  const [btMaternNu, setBtMaternNu] = useState<MaternNu>('2.5');
  const [btInputTransform, setBtInputTransform] = useState<BoTorchInputTransform>('none');
  const [btOutputTransform, setBtOutputTransform] = useState<BoTorchOutputTransform>('none');
  const [btCalibrateUncertainty, setBtCalibrateUncertainty] = useState(false);
  
  // API hooks
  const { data: modelInfo } = useModelInfo(sessionId);
  const { data: experimentsSummary } = useExperimentsSummary(sessionId);
  const trainModel = useTrainModel(sessionId);
  
  // Check if enough data exists
  const hasEnoughData = experimentsSummary?.has_data && 
                        experimentsSummary.n_experiments >= 5;
  
  const handleTrainModel = async () => {
    if (!hasEnoughData) {
      alert('Need at least 5 experiments to train a model');
      return;
    }
    
    // Build request based on backend
    const request: TrainModelRequest = {
      backend,
      kernel: backend === 'sklearn' ? skKernel : btKernel,
    };
    
    // Add kernel params for Matern
    if (backend === 'sklearn' && skKernel === 'Matern') {
      request.kernel_params = { nu: parseFloat(skMaternNu) };
    } else if (backend === 'botorch' && btKernel === 'Matern') {
      request.kernel_params = { nu: parseFloat(btMaternNu) };
    }
    
    // Add transforms if not 'none'
    if (backend === 'sklearn') {
      if (skInputTransform !== 'none') request.input_transform = skInputTransform;
      if (skOutputTransform !== 'none') request.output_transform = skOutputTransform;
      request.calibration_enabled = skCalibrateUncertainty;
    } else {
      if (btInputTransform !== 'none') request.input_transform = btInputTransform;
      if (btOutputTransform !== 'none') request.output_transform = btOutputTransform;
      request.calibration_enabled = btCalibrateUncertainty;
    }
    
    try {
      await trainModel.mutateAsync(request);
    } catch (error) {
      console.error('Training failed:', error);
      // Error is already shown via toast/notification system
    }
  };
  
  return (
    <div className="rounded-lg border bg-card p-6">
      <h2 className="text-2xl font-bold mb-6">Gaussian Process Model</h2>
      
      {/* Backend Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Backend</label>
        <div className="flex gap-2">
          <button
            onClick={() => setBackend('sklearn')}
            className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              backend === 'sklearn'
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            scikit-learn
          </button>
          <button
            onClick={() => setBackend('botorch')}
            className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              backend === 'botorch'
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            botorch
          </button>
        </div>
      </div>
      
      {/* Advanced Options Toggle */}
      <div className="mb-4 p-3 border rounded-lg">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={advancedEnabled}
            onChange={(e) => setAdvancedEnabled(e.target.checked)}
            className="w-4 h-4 rounded border-gray-300"
          />
          <span className="text-sm font-medium">Enable Advanced Options</span>
        </label>
        
        {/* Scikit-learn Options */}
        {backend === 'sklearn' && (
          <div className="mt-4 space-y-4">
            <div>
              <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Kernel Selection
              </label>
              <select
                value={skKernel}
                onChange={(e) => setSkKernel(e.target.value as KernelType)}
                disabled={!advancedEnabled}
                className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
              >
                <option value="RBF">RBF</option>
                <option value="Matern">Matern</option>
                <option value="RationalQuadratic">RationalQuadratic</option>
              </select>
            </div>
            
            {skKernel === 'Matern' && (
              <div>
                <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                  Matern nu
                </label>
                <select
                  value={skMaternNu}
                  onChange={(e) => setSkMaternNu(e.target.value as MaternNu)}
                  disabled={!advancedEnabled}
                  className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
                >
                  <option value="0.5">0.5</option>
                  <option value="1.5">1.5</option>
                  <option value="2.5">2.5</option>
                  <option value="inf">inf</option>
                </select>
              </div>
            )}
            
            <div className="text-xs text-muted-foreground italic">
              Note: Kernel hyperparameters will be automatically optimized.
            </div>
            
            <div>
              <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Optimizer
              </label>
              <select
                value={skOptimizer}
                onChange={(e) => setSkOptimizer(e.target.value as SklearnOptimizer)}
                disabled={!advancedEnabled}
                className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
              >
                <option value="CG">CG</option>
                <option value="BFGS">BFGS</option>
                <option value="L-BFGS-B">L-BFGS-B</option>
                <option value="TNC">TNC</option>
              </select>
            </div>
            
            <div>
              <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Input Scaling
              </label>
              <select
                value={skInputTransform}
                onChange={(e) => setSkInputTransform(e.target.value as SklearnInputTransform)}
                disabled={!advancedEnabled}
                className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
              >
                <option value="none">none</option>
                <option value="minmax">minmax</option>
                <option value="standard">standard</option>
                <option value="robust">robust</option>
              </select>
            </div>
            
            <div>
              <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Output Scaling
              </label>
              <select
                value={skOutputTransform}
                onChange={(e) => setSkOutputTransform(e.target.value as SklearnOutputTransform)}
                disabled={!advancedEnabled}
                className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
              >
                <option value="none">none</option>
                <option value="minmax">minmax</option>
                <option value="standard">standard</option>
                <option value="robust">robust</option>
              </select>
            </div>
            
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={skCalibrateUncertainty}
                onChange={(e) => setSkCalibrateUncertainty(e.target.checked)}
                disabled={!advancedEnabled}
                className="w-4 h-4 rounded border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <span className={`text-sm ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Calibrate Uncertainty
              </span>
            </label>
          </div>
        )}
        
        {/* BoTorch Options */}
        {backend === 'botorch' && (
          <div className="mt-4 space-y-4">
            <div>
              <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Continuous Kernel
              </label>
              <select
                value={btKernel}
                onChange={(e) => setBtKernel(e.target.value as KernelType)}
                disabled={!advancedEnabled}
                className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
              >
                <option value="RBF">RBF</option>
                <option value="Matern">Matern</option>
              </select>
            </div>
            
            {btKernel === 'Matern' && (
              <div>
                <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                  Matern nu
                </label>
                <select
                  value={btMaternNu}
                  onChange={(e) => setBtMaternNu(e.target.value as MaternNu)}
                  disabled={!advancedEnabled}
                  className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
                >
                  <option value="0.5">0.5</option>
                  <option value="1.5">1.5</option>
                  <option value="2.5">2.5</option>
                </select>
              </div>
            )}
            
            <div className="text-xs text-muted-foreground italic">
              BoTorch uses sensible defaults for training parameters.
            </div>
            
            <div>
              <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Input Scaling
              </label>
              <select
                value={btInputTransform}
                onChange={(e) => setBtInputTransform(e.target.value as BoTorchInputTransform)}
                disabled={!advancedEnabled}
                className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
              >
                <option value="none">none</option>
                <option value="normalize">normalize</option>
                <option value="standardize">standardize</option>
              </select>
            </div>
            
            <div>
              <label className={`block text-sm mb-1 ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Output Scaling
              </label>
              <select
                value={btOutputTransform}
                onChange={(e) => setBtOutputTransform(e.target.value as BoTorchOutputTransform)}
                disabled={!advancedEnabled}
                className="w-full px-3 py-2 border rounded-md disabled:opacity-50 disabled:cursor-not-allowed bg-background"
              >
                <option value="none">none</option>
                <option value="standardize">standardize</option>
              </select>
            </div>
            
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={btCalibrateUncertainty}
                onChange={(e) => setBtCalibrateUncertainty(e.target.checked)}
                disabled={!advancedEnabled}
                className="w-4 h-4 rounded border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <span className={`text-sm ${!advancedEnabled ? 'text-muted-foreground' : ''}`}>
                Calibrate Uncertainty
              </span>
            </label>
          </div>
        )}
      </div>
      
      {/* Train Button */}
      <button
        onClick={handleTrainModel}
        disabled={!hasEnoughData || trainModel.isPending}
        className="w-full bg-primary text-primary-foreground px-4 py-3 rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
      >
        {trainModel.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
        {trainModel.isPending ? 'Training Model...' : 'Train Model'}
      </button>
      
      {/* Data validation message */}
      {!hasEnoughData && (
        <div className="mt-2 p-3 bg-amber-500/10 border border-amber-500/20 rounded-md flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-amber-700 dark:text-amber-500">
            {experimentsSummary?.n_experiments
              ? `Need at least 5 experiments (currently ${experimentsSummary.n_experiments})`
              : 'No experimental data loaded. Please load experiments first.'}
          </p>
        </div>
      )}
      
      {/* Training Success Message */}
      {trainModel.isSuccess && trainModel.data && (
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-md flex items-start gap-2">
          <CheckCircle2 className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm">
            <p className="font-medium text-green-700 dark:text-green-500">
              {trainModel.data.message}
            </p>
          </div>
        </div>
      )}
      
      {/* Model Info Display */}
      {modelInfo?.is_trained && (
        <div className="mt-6 p-4 bg-muted/50 rounded-lg space-y-3">
          <h3 className="font-semibold text-sm">Trained Model Info</h3>
          
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <div>
              <span className="text-muted-foreground">Backend:</span>{' '}
              <span className="font-medium">{modelInfo.backend}</span>
            </div>
          </div>
          
          {modelInfo.metrics && (
            <div>
              <h4 className="text-xs font-medium text-muted-foreground mb-2">Cross-Validation Metrics</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">RMSE:</span>{' '}
                  <span className="font-medium">{modelInfo.metrics.rmse.toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">MAE:</span>{' '}
                  <span className="font-medium">{modelInfo.metrics.mae.toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">R²:</span>{' '}
                  <span className="font-medium">{modelInfo.metrics.r2.toFixed(4)}</span>
                </div>
                {modelInfo.metrics.mape !== undefined && (
                  <div>
                    <span className="text-muted-foreground">MAPE:</span>{' '}
                    <span className="font-medium">{modelInfo.metrics.mape.toFixed(2)}%</span>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Hyperparameters - collapsed by default */}
          <details className="text-sm">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground font-medium">
              View Hyperparameters
            </summary>
            <pre className="mt-2 p-2 bg-background rounded text-xs overflow-x-auto">
              {JSON.stringify(modelInfo.hyperparameters, null, 2)}
            </pre>
          </details>
        </div>
      )}
      
      {/* Visualization Button */}
      <button
        onClick={() => setShowVisualizations(true)}
        disabled={!modelInfo?.is_trained}
        className="w-full mt-4 border border-primary text-primary hover:bg-primary hover:text-primary-foreground px-4 py-2 rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-primary flex items-center justify-center gap-2"
      >
        <LineChart className="w-4 h-4" />
        Show Model Visualizations
      </button>
      
      {/* Visualizations Modal */}
      <VisualizationsPanel
        sessionId={sessionId}
        isOpen={showVisualizations}
        onClose={() => setShowVisualizations(false)}
        modelBackend={modelInfo?.backend || undefined}
      />
    </div>
  );
}
