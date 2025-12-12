# ALchemist Web UI

React + TypeScript web interface for the ALchemist active learning toolkit.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm
- ALchemist FastAPI backend running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at http://localhost:5173

### Backend Setup

Make sure the ALchemist FastAPI backend is running:

```bash
# From the ALchemist root directory
python -m api.run_api
```

The API should be accessible at http://localhost:8000

## 📦 Project Structure

```
src/
├── api/                    # API client and endpoints
│   ├── client.ts          # Axios configuration
│   ├── types.ts           # TypeScript interfaces
│   └── endpoints/         # API endpoint functions
├── components/            # React components
│   ├── layout/           # Layout components
│   └── ui/               # Reusable UI components
├── features/             # Feature modules
│   ├── sessions/        # Session management
│   ├── variables/       # Search space definition
│   ├── experiments/     # Data management
│   ├── models/          # Model training
│   └── acquisition/     # Next experiment suggestions
├── hooks/               # Custom React hooks
│   └── api/            # API integration hooks
├── lib/                # Utility functions
├── providers/          # React context providers
└── App.tsx            # Main application component
```

## 🛠️ Technology Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TanStack Query** - Server state management
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **React Hook Form** + **Zod** - Form handling and validation
- **Sonner** - Toast notifications
- **Recharts** - Data visualization

## 📝 Development Roadmap

### Phase 1: Foundation ✅
- [x] Project setup with Vite + React + TypeScript
- [x] API client layer with TypeScript types
- [x] React Query provider setup
- [x] Session management hooks
- [x] Basic UI with Tailwind CSS

### Phase 2: Variables Module (Next)
- [ ] Variable definition form
- [ ] Variable list/table
- [ ] Add/edit/delete functionality

### Phase 3: Experiments Module
- [ ] Experiment data table
- [ ] Manual data entry
- [ ] CSV upload

### Phase 4: Model Training
- [ ] Model configuration form
- [ ] Training interface
- [ ] CV metrics display

### Phase 5: Acquisition & Visualizations
- [ ] Strategy selector
- [ ] Contour plots
- [ ] Parity plots

## 🧪 Available Scripts

```bash
npm run dev          # Start dev server with HMR
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

## 📚 Learning Resources

- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
- [TanStack Query Docs](https://tanstack.com/query/latest/docs/react/overview)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)

## 📄 License

BSD 3-Clause (same as ALchemist core)


```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
