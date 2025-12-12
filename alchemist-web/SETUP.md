# ALchemist Web UI - Setup Instructions

## ✅ What's Been Created

The initial project structure has been set up with:

1. **Project Structure** - Organized feature-based directory layout
2. **API Layer** - Complete TypeScript types and endpoint functions
3. **React Query Setup** - Server state management configured
4. **Tailwind CSS** - Styling system configured
5. **Custom Hooks** - Session and variable management hooks
6. **Basic App** - Starter UI with session management

## 🔧 Next Steps

### Step 1: Install Dependencies

**IMPORTANT**: You need to install the npm packages before the app will work.

```bash
# Navigate to the alchemist-web directory
cd alchemist-web

# Install all dependencies (this may take a few minutes)
npm install
```

This will install all the packages listed in `package.json`:
- React Query for API state management
- Axios for HTTP requests
- Tailwind CSS for styling
- Zod for validation
- React Hook Form for forms
- Sonner for toast notifications
- And more...

### Step 2: Verify Backend is Running

Make sure the FastAPI backend is running:

```bash
# In a separate terminal, from the ALchemist root directory
python -m api.run_api
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Start the Development Server

```bash
# In the alchemist-web directory
npm run dev
```

The React app will start on http://localhost:5173

### Step 4: Test the Connection

1. Open http://localhost:5173 in your browser
2. You should see the ALchemist Web welcome screen
3. Try creating a session (button is visible but not yet functional)

## 📁 Project Files Created

### Configuration Files
- ✅ `package.json` - Updated with all dependencies
- ✅ `tailwind.config.js` - Tailwind CSS configuration
- ✅ `postcss.config.js` - PostCSS configuration
- ✅ `.env.development` - Development environment variables
- ✅ `.env.production` - Production environment variables

### API Layer (`src/api/`)
- ✅ `client.ts` - Axios HTTP client with interceptors
- ✅ `types.ts` - TypeScript interfaces matching FastAPI schemas
- ✅ `endpoints/sessions.ts` - Session API calls
- ✅ `endpoints/variables.ts` - Variables API calls
- ✅ `endpoints/experiments.ts` - Experiments API calls
- ✅ `endpoints/models.ts` - Model training API calls
- ✅ `endpoints/acquisition.ts` - Acquisition API calls

### Hooks (`src/hooks/api/`)
- ✅ `useSessions.ts` - Session management with React Query
- ✅ `useVariables.ts` - Variables management with React Query

### Utilities
- ✅ `lib/utils.ts` - Helper functions (class names, formatting)
- ✅ `providers/QueryProvider.tsx` - React Query configuration

### Application
- ✅ `App.tsx` - Main app component with session management
- ✅ `index.css` - Tailwind CSS with theme variables
- ✅ `README.md` - Project documentation

## 🎯 What Works Now

After installing dependencies and starting the dev server:

1. ✅ React app loads at localhost:5173
2. ✅ Tailwind CSS styling applied
3. ✅ Session ID persistence in localStorage
4. ✅ Toast notifications system ready
5. ✅ API client configured to connect to localhost:8000

## 🚧 What's Next (Phase 2)

The next development phase will add the **Variables Module**:

1. Create session management UI
   - Button to create new sessions
   - Session info display
   - TTL extension

2. Create variable definition form
   - Type selector (continuous/discrete/categorical)
   - Bounds input for continuous/discrete
   - Categories list for categorical
   - Unit and description fields

3. Create variable list table
   - Display all defined variables
   - Edit/delete functionality
   - Variable count summary

4. Wire up API calls
   - Connect forms to API endpoints
   - Handle loading/error states
   - Show success/error notifications

## 📚 Key Concepts for Beginners

### TypeScript Types
All API responses have type definitions in `src/api/types.ts`. This gives you autocomplete and type checking.

Example:
```typescript
// TypeScript knows this is a Variable
const variable: Variable = {
  name: "temperature",
  type: "continuous",
  bounds: [100, 500],
  unit: "°C"
};
```

### React Query Hooks
Instead of managing API state manually, we use React Query:

```typescript
// Automatically handles loading, error, and data states
const { data, isLoading, error } = useVariables(sessionId);
```

### Mutations for Updates
For creating/updating data:

```typescript
const createVar = useCreateVariable(sessionId);
createVar.mutate(newVariable); // Automatically refetches data
```

### Tailwind CSS Classes
Instead of writing CSS, we use utility classes:

```tsx
<div className="bg-primary text-white p-4 rounded-lg">
  Styled with Tailwind!
</div>
```

## 🐛 Troubleshooting

### "Cannot find module" errors
Run `npm install` to install all dependencies.

### CSS not working
Make sure Tailwind is properly installed:
```bash
npm install -D tailwindcss postcss autoprefixer
```

### API connection errors
1. Check FastAPI is running: http://localhost:8000/health
2. Check CORS is enabled for localhost:5173
3. Verify `.env.development` has correct API URL

### Port already in use
If port 5173 is busy, Vite will suggest a different port. Just use that one.

## 💡 Tips for Learning

1. **Start with Types** - Look at `src/api/types.ts` to understand the data structures
2. **Follow the Data** - API endpoint → Hook → Component
3. **Use Browser DevTools** - Check the Network tab to see API calls
4. **Console.log Everything** - Add logging to understand data flow
5. **Read the Docs** - Links to learning resources in main README

## ✉️ Questions?

- Check the main README in `alchemist-web/README.md`
- Review FastAPI docs at http://localhost:8000/api/docs
- Look at existing hook patterns in `src/hooks/api/`

Ready to build the Variables module? Let me know when you've completed the installation!
