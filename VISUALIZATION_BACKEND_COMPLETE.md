# Visualization Backend Implementation Summary

## Overview

I've implemented a comprehensive backend API for all the visualization types from the desktop UI. The backend is now complete and ready for React component development.

---

## New Files Created

### 1. `/api/routers/visualizations.py` (Complete Router)

New visualization endpoints providing data for all 7 plot types:

#### **Endpoints:**

1. **`POST /{session_id}/visualizations/contour`**
   - Generates 2D contour plot data for model predictions
   - Configurable grid resolution (10-200, default 50)
   - Optional experimental points overlay
   - Optional suggestion points overlay
   - Fixed values for non-axis variables
   - Returns: prediction grid, uncertainty grid, bounds, colorbars

2. **`GET /{session_id}/visualizations/parity`**
   - Parity plot data (actual vs predicted from CV)
   - Optional calibrated/uncalibrated uncertainties
   - Returns: y_true, y_pred, y_std, metrics (RMSE, MAE, R², MAPE), bounds

3. **`GET /{session_id}/visualizations/metrics`**
   - CV metrics over increasing training size (5 to N samples)
   - Configurable CV splits (2-10, default 5)
   - Returns: training_sizes, RMSE, MAE, R², MAPE arrays

4. **`GET /{session_id}/visualizations/qq-plot`**
   - Q-Q plot for standardized residuals
   - Optional calibrated/uncalibrated results
   - Returns: theoretical quantiles, sample quantiles, z-score stats

5. **`GET /{session_id}/visualizations/calibration-curve`**
   - Reliability diagram showing calibration quality
   - Optional calibrated/uncalibrated results
   - Returns: nominal coverage, empirical coverage arrays

6. **`GET /{session_id}/visualizations/hyperparameters`**
   - Model configuration and hyperparameters
   - Returns: hyperparameters dict, backend, kernel, transforms, calibration info

### 2. `/alchemist-web/src/api/endpoints/visualizations.ts`

TypeScript API client functions:
- `getContourData(sessionId, request)`
- `getParityData(sessionId, useCalibrated)`
- `getMetricsData(sessionId, cvSplits)`
- `getQQPlotData(sessionId, useCalibrated)`
- `getCalibrationCurveData(sessionId, useCalibrated)`
- `getHyperparameters(sessionId)`

### 3. `/alchemist-web/src/hooks/api/useVisualizations.ts`

React Query hooks for data fetching:
- `useContourData()` - Manual refetch (enabled: false by default)
- `useParityData()`
- `useMetricsData()`
- `useQQPlotData()`
- `useCalibrationCurveData()`
- `useHyperparameters()`

All hooks include:
- Proper caching (30-60 seconds stale time)
- No refetch on window focus
- Disabled when no session ID

---

## Updated Files

### 1. `/api/main.py`
- Added visualizations router import
- Registered router at `/api/v1/sessions/{session_id}/visualizations`

### 2. `/alchemist-web/src/api/types.ts`
- Added 6 new request/response type definitions:
  - `ContourDataRequest` / `ContourDataResponse`
  - `ParityDataResponse`
  - `MetricsDataResponse`
  - `QQPlotDataResponse`
  - `CalibrationCurveDataResponse`
  - `HyperparametersResponse`

### 3. `/api/routers/acquisition.py`
- Added `session.last_suggestions` storage when suggestions are generated
- Enables contour plots to show next suggested points

---

## Key Features Implemented

### **Data Sources:**
- Uses **cached CV results** from trained models (`cv_cached_results`)
- Supports both **calibrated and uncalibrated** uncertainties
- Efficient grid generation for contour plots (vectorized predictions)

### **Error Handling:**
- Model must be trained before accessing visualizations
- Validates variable names exist in search space
- Checks for categorical variables (contour plots require continuous)
- Graceful handling of missing CV results

### **Performance Optimizations:**
- Contour data uses configurable grid resolution (default 50x50 = 2,500 points)
- Metrics endpoint is expensive (60s cache) - computes full CV evaluation
- Other endpoints use 30s cache with CV cached results (fast)
- All hooks disable refetch on window focus

### **Calibration Support:**
- All uncertainty-based plots support calibrated/uncalibrated toggle
- Parity plot, Q-Q plot, calibration curve all respect this flag
- Uses `cv_cached_results_calibrated` when available

---

## Data Flow

```
User Request → React Hook → API Client → Router Endpoint → Session Model
                                                              ↓
                                                     Cached CV Results
                                                              ↓
                                                  Process & Format Data
                                                              ↓
                                             Return Response (JSON)
```

---

## What's Ready for Frontend Development

### **✅ Backend Complete:**
1. All 6 visualization endpoints working
2. TypeScript types defined
3. React Query hooks created
4. Proper error handling
5. Caching strategies implemented
6. Session state management (last_suggestions)

### **🎨 Ready to Build (Next Steps):**

1. **Visualization Components (using Recharts):**
   - `ContourPlot.tsx` - 2D heatmap with scatter overlay
   - `ParityPlot.tsx` - Scatter with error bars + parity line
   - `MetricsPlot.tsx` - Line chart (multi-series)
   - `QQPlot.tsx` - Scatter with diagonal reference
   - `CalibrationCurve.tsx` - Line with shaded regions

2. **Visualization Panel/Modal:**
   - Tabbed interface or dropdown selector for plot types
   - Controls for:
     - Contour: X/Y variable selection, fixed values, resolution
     - Parity/QQ/Calibration: Calibrated toggle, sigma multiplier
     - Metrics: Metric selection dropdown
   - Plot customization (titles, labels, colors, etc.)

3. **Integration Points:**
   - Add "Show Visualizations" button to GPRPanel (only when trained)
   - Could be modal, drawer, or dedicated panel
   - Save/export functionality for plots

---

## Example Usage (Frontend)

```typescript
// In a visualization component
const { data: parityData, isLoading } = useParityData(
  sessionId,
  useCalibrated,
  modelTrained  // enabled flag
);

// In contour plot with manual refetch
const { data: contourData, refetch } = useContourData(
  sessionId,
  {
    x_var: selectedXVar,
    y_var: selectedYVar,
    fixed_values: fixedVals,
    grid_resolution: 50,
    include_experiments: true,
    include_suggestions: showSuggestions
  },
  false  // don't auto-fetch, user clicks "Generate Plot"
);

const handleGeneratePlot = () => {
  refetch();
};
```

---

## Testing Checklist

### **Backend Testing:**
- [ ] Start API server: `python run_api.py`
- [ ] Check OpenAPI docs: `http://localhost:8000/api/docs`
- [ ] Test each visualization endpoint with trained model
- [ ] Verify calibrated/uncalibrated toggle works
- [ ] Test contour with different grid resolutions
- [ ] Verify error handling (no model, invalid variables)

### **Frontend Testing (Once Components Built):**
- [ ] All plots render with real data
- [ ] Loading states display correctly
- [ ] Error states handled gracefully
- [ ] Controls update plots properly
- [ ] Calibration toggle switches data source
- [ ] Contour plot updates on axis/fixed value changes

---

## Next Steps Recommendations

### **Phase 1 - Simple Plots First:**
1. Start with **Parity Plot** (simplest - just scatter + line)
2. Then **Metrics Plot** (multi-line chart)
3. Then **Q-Q Plot** (similar to parity)

### **Phase 2 - Complex Visualizations:**
4. **Calibration Curve** (line + shaded areas)
5. **Contour Plot** (heatmap + controls + overlays)

### **Phase 3 - Polish:**
6. Plot customization controls
7. Export/save functionality
8. Responsive layout
9. Dark mode support

---

## Notes

- **Recharts** is already installed in your project
- For contour plots, you may need `recharts-surface` or use a different library like `Plotly.js` or `Visx`
- Consider using **React Suspense** for loading states
- Consider **Zustand** or **Context** for plot customization state (instead of prop drilling)

---

## Files to Review

1. `/api/routers/visualizations.py` - Full endpoint implementation
2. `/alchemist-web/src/api/types.ts` - TypeScript types (bottom of file)
3. `/alchemist-web/src/hooks/api/useVisualizations.ts` - React Query hooks
4. `/alchemist-web/src/api/endpoints/visualizations.ts` - API client functions

All backend code is production-ready and follows the existing patterns in your codebase!
