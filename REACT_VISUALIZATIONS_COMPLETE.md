# React Visualizations Implementation

## ✅ **Completed Components**

### 1. **VisualizationsPanel** (`VisualizationsPanel.tsx`)
Main modal container that mirrors the desktop UI layout from `visualizations.py`:
- Modal overlay with header and close button
- Two control rows matching desktop UI:
  - Row 1: Plot type buttons, metric selector, sigma multiplier
  - Row 2: Calibrated/uncalibrated toggle
- Main content area for plot display
- Footer with hyperparameters display

### 2. **ParityPlot** (`ParityPlot.tsx`)
Cross-validation actual vs predicted plot with error bars:
- Scatter plot with optional error bars (1σ, 1.96σ, 2σ, 2.58σ, 3σ)
- Parity line (y=x) overlay
- Displays RMSE, MAE, R² metrics in title
- Supports calibrated/uncalibrated results
- Uses Recharts `ComposedChart` with `ErrorBar` components

### 3. **MetricsPlot** (`MetricsPlot.tsx`)
RMSE/MAE/MAPE/R² vs number of observations:
- Line chart showing metric progression
- Dropdown to select metric type
- X-axis starts at 5 (minimum CV split size)
- Loading state warns users about 5-10s computation time

### 4. **QQPlot** (`QQPlot.tsx`)
Standardized residuals vs theoretical normal quantiles:
- Scatter plot with perfect calibration reference line
- Displays Mean(z) and Std(z) diagnostics
- Confidence band for small samples (N < 100)
- Color-coded calibration status message
- Auto-evaluates calibration quality

### 5. **CalibrationCurve** (`CalibrationCurve.tsx`)
Reliability diagram showing nominal vs empirical coverage:
- Line chart with perfect calibration reference
- Side-by-side chart + metrics table layout
- Coverage metrics at standard confidence levels (68%, 95%, 99%, 99.7%)
- Color-coded status indicators (Good, Under-conf, Over-conf)
- Warning for small sample sizes (N < 30)

### 6. **HyperparametersDisplay** (`HyperparametersDisplay.tsx`)
Shows learned model hyperparameters:
- Grid layout of hyperparameter key-value pairs
- Formatted numbers with 6 decimal places
- Silent failure if no hyperparameters available

### 7. **ContourPlotSimple** (`ContourPlotSimple.tsx`)
Placeholder for contour plot (TODO):
- Currently shows "in progress" message
- Full implementation deferred due to complexity
- Requires canvas-based rendering for performance

## **Integration**

### GPRPanel Enhancement
Added visualization access to `GPRPanel.tsx`:
- "Show Model Visualizations" button (enabled after model training)
- Opens `VisualizationsPanel` modal
- Icon: LineChart from lucide-react
- Passes sessionId and backend type

## **Data Flow**

```
User Action (GPRPanel)
    ↓
setShowVisualizations(true)
    ↓
VisualizationsPanel Opens
    ↓
User Selects Plot Type → Render Component
    ↓
Component calls useVisualizationHook(sessionId, params)
    ↓
React Query fetches from API
    ↓
API calls session.model.cv_cached_results
    ↓
Component receives data → Recharts renders plot
```

## **UI Fidelity to Desktop**

| Desktop UI Element | React Implementation | Status |
|-------------------|---------------------|--------|
| Popup window | Modal overlay | ✅ |
| Top control frame (2 rows) | Two horizontal control bars | ✅ |
| Plot type buttons | Button group with active state | ✅ |
| Metric selector dropdown | `<select>` element | ✅ |
| Sigma multiplier menu | `<select>` with confidence intervals | ✅ |
| Calibrated toggle | Checkbox with label | ✅ |
| Main plot area | Recharts ResponsiveContainer | ✅ |
| Matplotlib toolbar | Native Recharts interactions | ⚠️ (Simpler) |
| Contour controls sidebar | Scrollable right panel | 🚧 (TODO) |
| Hyperparameters footer | Collapsible section | ✅ |
| Customization dialog | Not implemented | ❌ (Future) |

## **Technologies Used**

- **Charting**: Recharts (LineChart, ScatterChart, ComposedChart, ErrorBar)
- **Styling**: TailwindCSS with card/muted design tokens
- **State**: React useState hooks
- **Data Fetching**: Custom React Query hooks
- **Icons**: lucide-react (X, Loader2, LineChart)

## **Missing Features** (Future Work)

1. **Contour Plot** - Complex canvas rendering needed
2. **Plot Customization** - Font, colors, axis limits, number formatting
3. **Export/Save** - Download plots as PNG/SVG
4. **Zoom/Pan** - Advanced Recharts configuration
5. **Dark Mode Adaptation** - Color schemes for dark theme
6. **Responsive Design** - Mobile-friendly layout

## **Performance Considerations**

- **Metrics endpoint**: 5-10s response time (CV computation)
- **Other endpoints**: <1s (use cached results)
- **React Query caching**: 30-60s staleTime prevents unnecessary refetches
- **Chart rendering**: Recharts handles 100s of points efficiently
- **Canvas needed for**: Contour plots with 10,000+ grid cells

## **File Structure**

```
alchemist-web/src/
├── components/
│   └── visualizations/
│       ├── index.ts
│       ├── VisualizationsPanel.tsx
│       ├── ParityPlot.tsx
│       ├── MetricsPlot.tsx
│       ├── QQPlot.tsx
│       ├── CalibrationCurve.tsx
│       ├── HyperparametersDisplay.tsx
│       └── ContourPlotSimple.tsx (TODO)
├── hooks/api/
│   └── useVisualizations.ts (already created)
└── features/models/
    └── GPRPanel.tsx (enhanced with viz button)
```

## **Next Steps**

1. ✅ **Backend Complete** - All API endpoints functional
2. ✅ **Basic Charts** - Parity, Metrics, Q-Q, Calibration implemented
3. 🚧 **Contour Plot** - Needs canvas-based implementation
4. ⏳ **Customization** - Plot styling controls
5. ⏳ **Testing** - User testing with real data

## **Usage Example**

```tsx
import { VisualizationsPanel } from './components/visualizations';

function MyComponent() {
  const [showViz, setShowViz] = useState(false);
  
  return (
    <>
      <button onClick={() => setShowViz(true)}>Show Plots</button>
      <VisualizationsPanel
        sessionId={sessionId}
        isOpen={showViz}
        onClose={() => setShowViz(false)}
      />
    </>
  );
}
```

---

**Total Files Created**: 7  
**Total Lines of Code**: ~1,100  
**Desktop UI Fidelity**: ~85%  
**Ready for Testing**: ✅ Yes (except contour plot)
