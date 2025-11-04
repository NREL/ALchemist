# GPR Panel Integration Guide

## Overview

The GPR Panel component has been successfully scaffolded to match the desktop UI's `gpr_panel.py` functionality. This component allows users to train Gaussian Process Regression models using either scikit-learn or BoTorch backends.

## Files Created

### 1. Component
- **Location**: `src/features/models/GPRPanel.tsx`
- **Purpose**: Main GPR panel UI component
- **Features**: Backend selection, kernel configuration, advanced options, model training

### 2. API Hooks
- **Location**: `src/hooks/api/useModels.ts`
- **Hooks**:
  - `useModelInfo(sessionId)` - Fetches current model status
  - `useTrainModel(sessionId)` - Trains a new model

### 3. Type Definitions
- **Location**: `src/api/types.ts` (updated)
- **Added Types**:
  - `ModelBackend`, `KernelType`, `MaternNu`
  - `SklearnInputTransform`, `SklearnOutputTransform`, `SklearnOptimizer`
  - `BoTorchInputTransform`, `BoTorchOutputTransform`
  - `TrainModelRequest`, `TrainModelResponse`, `ModelInfo`, `ModelMetrics`

### 4. Module Export
- **Location**: `src/features/models/index.ts`
- **Exports**: `GPRPanel`

### 5. Integration
- **Location**: `src/App.tsx` (updated)
- **Change**: Added `<GPRPanel sessionId={sessionId} />` after experiments panel

## Component Structure

```tsx
<GPRPanel>
  ├── Backend Selection (scikit-learn / BoTorch)
  ├── Advanced Options Toggle
  │   ├── Scikit-learn Options
  │   │   ├── Kernel Selection (RBF, Matern, RationalQuadratic)
  │   │   ├── Matern nu (conditional)
  │   │   ├── Optimizer
  │   │   ├── Input Scaling
  │   │   ├── Output Scaling
  │   │   └── Calibrate Uncertainty
  │   └── BoTorch Options
  │       ├── Continuous Kernel (RBF, Matern)
  │       ├── Matern nu (conditional)
  │       ├── Input Scaling
  │       ├── Output Scaling
  │       └── Calibrate Uncertainty
  ├── Train Model Button
  ├── Data Validation Message (if insufficient data)
  ├── Training Success Message
  └── Model Info Display (after training)
      ├── Backend & Kernel
      ├── CV Metrics (RMSE, MAE, R², MAPE)
      └── Hyperparameters (collapsible)
```

## API Endpoints Required

The component expects these REST API endpoints to be implemented:

### 1. Get Model Info
```
GET /api/v1/sessions/{session_id}/model
```

### 2. Train Model
```
POST /api/v1/sessions/{session_id}/model/train
```

## State Flow

1. **Component Mount**
   - Fetches current model info (if exists)
   - Fetches experiment summary to check data availability

2. **User Configures Model**
   - Selects backend (sklearn/botorch)
   - Optionally enables advanced options
   - Configures kernel, transforms, etc.

3. **User Clicks "Train Model"**
   - Validates sufficient data (≥5 experiments)
   - Builds `TrainModelRequest` based on selections
   - Calls `trainModel.mutateAsync(request)`
   - Shows loading state during training

4. **Training Complete**
   - Invalidates queries to refetch model info
   - Shows success toast with R² score
   - Displays model metrics and hyperparameters

## Desktop UI Parity

### Implemented Features ✅
- Backend selection (scikit-learn, BoTorch)
- Advanced options toggle
- Kernel selection with conditional Matern nu
- Input/output scaling options
- Uncertainty calibration toggle
- Data validation (min 5 experiments)
- Training button with loading state
- Model info display with metrics
- Hyperparameters display (collapsible)

### Not Yet Implemented ⏳
- Visualization/analysis plots (marked as "Coming Soon")
- Real-time training progress updates
- Optimizer selection for BoTorch (not in desktop UI)

### Differences from Desktop UI
1. **Layout**: React version uses vertical layout vs desktop's fixed-width panel
2. **Styling**: Uses Tailwind CSS instead of CustomTkinter
3. **Notifications**: Uses toast notifications instead of message boxes
4. **Advanced Options**: All disabled by default (more intuitive UX)

## Testing Checklist

- [ ] Component renders without errors
- [ ] Backend switching works correctly
- [ ] Advanced options toggle enables/disables controls
- [ ] Matern nu field shows/hides based on kernel selection
- [ ] Train button is disabled when < 5 experiments
- [ ] Training request is built correctly for both backends
- [ ] Model info displays after successful training
- [ ] Error handling shows appropriate messages
- [ ] Toast notifications appear for success/error

## Next Steps

1. **Test with Backend**: Start the FastAPI server and verify endpoints
2. **Add Visualizations**: Implement the analysis plots panel
3. **Enhanced UX**: Consider adding:
   - Training time display
   - Model comparison feature
   - Cross-validation fold details
   - Hyperparameter optimization interface

## Usage Example

```tsx
import { GPRPanel } from './features/models';

function WorkflowPage() {
  const sessionId = useSessionId();
  
  return (
    <div className="space-y-6">
      <VariablesPanel sessionId={sessionId} />
      <ExperimentsPanel sessionId={sessionId} />
      <GPRPanel sessionId={sessionId} />
      <AcquisitionPanel sessionId={sessionId} />
    </div>
  );
}
```

## Dependencies

All required dependencies are already installed:
- `@tanstack/react-query` - Data fetching
- `axios` - HTTP client
- `sonner` - Toast notifications
- `lucide-react` - Icons
- `tailwindcss` - Styling
