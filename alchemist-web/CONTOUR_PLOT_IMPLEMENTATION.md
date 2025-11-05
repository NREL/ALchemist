# Contour Plot Implementation

## Overview
Successfully implemented a Plotly.js-based contour plot component for the React web app, closely mirroring the desktop UI functionality from `ui/visualizations.py`.

## Implementation Details

### Technology Stack
- **Plotly.js** via `react-plotly.js` - Professional interactive plotting library
- **React hooks** - State management and debouncing
- **TypeScript** - Type-safe implementation

### Key Features Implemented

#### 1. **Core Contour Visualization**
- ✅ 2D contour/heatmap plot of model predictions
- ✅ Interactive plot with zoom, pan, and hover tooltips
- ✅ Professional colorscale rendering (Viridis default, multiple options)
- ✅ Proper axis labels and titles
- ✅ Dynamic grid resolution (30-150, default 50)

#### 2. **Variable Selection & Fixed Values**
- ✅ Dropdown selectors for X and Y axes (Real variables only)
- ✅ Dynamic "Fixed Values" controls for non-plotted dimensions:
  - **Real variables**: Sliders with min/max bounds, default to midpoint
  - **Integer variables**: Number inputs with step validation
  - **Categorical variables**: Dropdown selectors
- ✅ Auto-updates when X/Y axes change
- ✅ Validates that only continuous (Real/Integer) variables can be plotted

#### 3. **Data Overlays**
- ✅ Experimental points overlay (white circles with black edges)
- ✅ Next suggested point overlay (red diamond marker)
- ✅ Toggle controls for showing/hiding overlays
- ✅ Hover tooltips show coordinates and values

#### 4. **Performance Optimizations**
- ✅ **Debouncing**: Slider changes debounced (300ms) to avoid excessive API calls
- ✅ **Loading states**: Spinner with message during computation
- ✅ **Error handling**: User-friendly error messages
- ✅ **Memoization**: Efficient re-rendering with useMemo

#### 5. **Customization Options**
- ✅ **Colormap selection**: 10 different colormaps (Viridis, Plasma, Inferno, Jet, etc.)
- ✅ **Grid resolution**: Adjustable from 30×30 to 150×150
- ✅ **Responsive layout**: Auto-sizing to container

#### 6. **Layout Matching Desktop UI**
- ✅ Main plot area on left (flexible width)
- ✅ Control panel on right sidebar (fixed width 256px)
- ✅ Sidebar sections:
  - X/Y axis selectors
  - Fixed value controls (dynamic)
  - Display options (experiment/next point toggles)
  - Colormap selector
  - Grid resolution slider
- ✅ Scrollable sidebar for many variables

### File Structure

```
alchemist-web/src/components/visualizations/
├── ContourPlot.tsx          # Main implementation (Plotly-based)
├── ContourPlotSimple.tsx    # Old placeholder (can be removed)
├── VisualizationsPanel.tsx  # Updated to use new ContourPlot
└── index.ts                 # Updated exports
```

### API Integration

The component integrates with the existing backend API:

**Endpoint**: `POST /sessions/{session_id}/visualizations/contour`

**Request**:
```typescript
{
  x_var: string;              // X axis variable name
  y_var: string;              // Y axis variable name
  fixed_values: Record<string, number | string>;  // Fixed values for other vars
  grid_resolution: number;    // Grid size (NxN)
  include_experiments: boolean;   // Include experimental data
  include_suggestions: boolean;   // Include next suggested points
}
```

**Response**:
```typescript
{
  x_grid: number[][];         // 2D X coordinate meshgrid
  y_grid: number[][];         // 2D Y coordinate meshgrid
  predictions: number[][];    // 2D prediction values
  uncertainties: number[][];  // 2D uncertainty values
  experiments?: {             // Optional experimental data
    x: number[];
    y: number[];
    output: number[];
  };
  suggestions?: {             // Optional next points
    x: number[];
    y: number[];
  };
  x_bounds: [number, number];
  y_bounds: [number, number];
  colorbar_bounds: [number, number];
}
```

### Comparison to Desktop UI

| Feature | Desktop UI (Python/Matplotlib) | React App (Plotly.js) | Status |
|---------|-------------------------------|----------------------|--------|
| 2D Contour Plot | ✓ | ✓ | ✅ Implemented |
| Variable Selection | ✓ | ✓ | ✅ Implemented |
| Fixed Value Controls | ✓ | ✓ | ✅ Implemented |
| Experimental Points Overlay | ✓ | ✓ | ✅ Implemented |
| Next Point Overlay | ✓ | ✓ | ✅ Implemented |
| Colormap Selection | ✓ | ✓ | ✅ Implemented |
| Grid Resolution | ✓ | ✓ | ✅ Implemented |
| Interactive Zoom/Pan | Limited | ✓ | ✅ Better in React |
| Export to PNG | ✓ | ✓ | ✅ Built-in to Plotly |
| Customization Dialog | ✓ | - | 🔲 Future enhancement |
| Number Formatting | ✓ | - | 🔲 Future enhancement |

### Usage Example

```tsx
import { ContourPlot } from '@/components/visualizations';

function MyComponent() {
  return (
    <div className="h-screen">
      <ContourPlot sessionId="your-session-id" />
    </div>
  );
}
```

### Future Enhancements (Optional)

1. **Advanced Customization Dialog**
   - Custom axis labels and titles
   - Number formatting options for axes/colorbar
   - Font size and style controls
   - Axis limit overrides

2. **Additional Plot Types**
   - Uncertainty contours (showing std dev)
   - Acquisition function overlay
   - Multiple contour layers

3. **Enhanced Export**
   - SVG export
   - Data export to CSV
   - Save plot configuration

4. **Performance**
   - WebGL rendering for very large grids (>200×200)
   - Server-side caching of contour data

## Testing Checklist

- [ ] Train a model with at least 2 Real variables
- [ ] Open visualizations panel
- [ ] Select "Plot Contour" tab
- [ ] Verify contour plot displays correctly
- [ ] Test X/Y axis selection changes
- [ ] Test slider changes for fixed values
- [ ] Test integer and categorical fixed value controls
- [ ] Toggle experimental points on/off
- [ ] Toggle next point on/off (after running acquisition)
- [ ] Change colormap
- [ ] Adjust grid resolution
- [ ] Test zoom, pan, and hover interactions
- [ ] Test with 3+ variables (fixed values should appear)
- [ ] Test error handling (e.g., no model trained)

## Notes

- The Plotly implementation provides better interactivity than the desktop matplotlib version
- Debouncing prevents API spam when adjusting sliders
- The component is fully responsive and works on different screen sizes
- All desktop UI layout patterns are preserved (sidebar on right, main plot on left)
