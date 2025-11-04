# ALchemist Web - Quick Reference

## 🚀 Quick Start Commands

```bash
# First time setup
cd alchemist-web
npm install

# Start development (opens http://localhost:5173)
npm run dev

# In separate terminal - start backend
cd ..
python run_api.py
```

## 📁 Key Files Reference

### API Integration
```typescript
// Import types
import type { Variable, Session } from '@/api/types';

// Import API functions
import { createSession, getVariables } from '@/api/endpoints/sessions';

// Import React Query hooks
import { useSession, useCreateSession } from '@/hooks/api/useSessions';
import { useVariables, useCreateVariable } from '@/hooks/api/useVariables';
```

### Common Patterns

#### Fetch Data
```typescript
const { data, isLoading, error } = useVariables(sessionId);

if (isLoading) return <LoadingSpinner />;
if (error) return <ErrorMessage error={error} />;
return <DataDisplay data={data} />;
```

#### Create/Update Data
```typescript
const createVar = useCreateVariable(sessionId);

const handleSubmit = async (formData: Variable) => {
  try {
    await createVar.mutateAsync(formData);
    // Success! React Query auto-refetches
  } catch (error) {
    // Error handled by mutation
  }
};
```

#### Form with Validation
```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  value: z.number(),
});

function MyForm() {
  const { register, handleSubmit, formState: { errors } } = useForm({
    resolver: zodResolver(schema),
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('name')} />
      {errors.name && <span>{errors.name.message}</span>}
    </form>
  );
}
```

## 🎨 Tailwind CSS Cheat Sheet

### Layout
```tsx
<div className="flex flex-col gap-4">       {/* Flexbox column with gap */}
<div className="grid grid-cols-2 gap-4">   {/* 2-column grid */}
<div className="container mx-auto">        {/* Centered container */}
```

### Spacing
```tsx
p-4     {/* padding: 1rem all sides */}
px-4    {/* padding-left & padding-right */}
py-2    {/* padding-top & padding-bottom */}
m-4     {/* margin */}
gap-4   {/* gap between flex/grid items */}
```

### Colors (from theme)
```tsx
bg-background      {/* Background color */}
bg-primary         {/* Primary color */}
text-foreground    {/* Text color */}
text-muted-foreground  {/* Muted text */}
border-border      {/* Border color */}
```

### Sizing
```tsx
w-full     {/* width: 100% */}
h-full     {/* height: 100% */}
min-h-screen  {/* min-height: 100vh */}
max-w-lg   {/* max-width: 32rem */}
```

### Typography
```tsx
text-sm    {/* font-size: 0.875rem */}
text-lg    {/* font-size: 1.125rem */}
text-2xl   {/* font-size: 1.5rem */}
font-bold  {/* font-weight: 700 */}
```

### Borders & Rounding
```tsx
border          {/* border: 1px solid */}
border-2        {/* border: 2px solid */}
rounded         {/* border-radius: 0.25rem */}
rounded-lg      {/* border-radius: 0.5rem */}
rounded-full    {/* border-radius: 9999px */}
```

### Hover & States
```tsx
hover:bg-primary/90    {/* On hover */}
focus:ring-2           {/* On focus */}
disabled:opacity-50    {/* When disabled */}
```

## 🔧 Debugging Tips

### Check API Connection
```typescript
// In browser console
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(console.log);
```

### View React Query State
```typescript
// Install React Query DevTools
npm install @tanstack/react-query-devtools

// Add to App.tsx
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

<QueryClientProvider client={queryClient}>
  <App />
  <ReactQueryDevtools initialIsOpen={false} />
</QueryClientProvider>
```

### Log Form Data
```typescript
const form = useForm();
console.log(form.watch()); // See all form values in real-time
```

### Check Environment Variables
```typescript
console.log(import.meta.env.VITE_API_BASE_URL);
```

## 🐛 Common Errors & Fixes

### "Cannot find module 'X'"
```bash
npm install
# or specifically
npm install X
```

### CORS Error
Check that FastAPI has CORS enabled for localhost:5173 in `api/main.py`

### Form not submitting
Check browser console for validation errors

### API returns 404
- Check backend is running: `python run_api.py`
- Check API base URL in `.env.development`
- Check session ID is valid

### Styles not applying
- Check Tailwind is installed: `npm install -D tailwindcss`
- Check `tailwind.config.js` exists
- Check `index.css` has `@tailwind` directives

## 📚 File Templates

### New Feature Component
```typescript
// src/features/myfeature/MyComponent.tsx
import { useState } from 'react';
import { useMyData } from '@/hooks/api/useMyData';

interface MyComponentProps {
  sessionId: string;
}

export function MyComponent({ sessionId }: MyComponentProps) {
  const { data, isLoading } = useMyData(sessionId);

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">My Feature</h2>
      {/* Component content */}
    </div>
  );
}
```

### New API Hook
```typescript
// src/hooks/api/useMyData.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as api from '@/api/endpoints/myendpoint';

export function useMyData(sessionId: string) {
  return useQuery({
    queryKey: ['mydata', sessionId],
    queryFn: () => api.getMyData(sessionId),
    enabled: !!sessionId,
  });
}

export function useCreateMyData(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data: MyDataType) => api.createMyData(sessionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mydata', sessionId] });
    },
  });
}
```

## 🎯 Development Workflow

1. **Design** - Plan component structure and data flow
2. **Types** - Define TypeScript interfaces
3. **API** - Create endpoint functions
4. **Hooks** - Create React Query hooks
5. **Components** - Build UI components
6. **Test** - Test with real backend
7. **Polish** - Add loading states, error handling, styling

## 💡 Best Practices

- ✅ Use TypeScript types everywhere
- ✅ Handle loading and error states
- ✅ Validate forms with Zod
- ✅ Show user feedback (toasts)
- ✅ Keep components small and focused
- ✅ Use React Query for server state
- ✅ Use Tailwind utilities over custom CSS
- ✅ Make components reusable

## 🔗 Essential Links

- **Vite Docs**: https://vitejs.dev/
- **React Docs**: https://react.dev/
- **TypeScript Handbook**: https://www.typescriptlang.org/docs/
- **TanStack Query**: https://tanstack.com/query/latest
- **Tailwind CSS**: https://tailwindcss.com/docs
- **Shadcn UI**: https://ui.shadcn.com/
- **React Hook Form**: https://react-hook-form.com/
- **Zod**: https://zod.dev/

Happy coding! 🚀
