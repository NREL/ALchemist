# Phase 2 Development Plan: Variables Module

## 🎯 Goal

Create the variable definition interface that mirrors the desktop UI's `SpaceSetupWindow`.

## 📋 Tasks Breakdown

### Task 1: Session Creation Component

**File**: `src/features/sessions/SessionManager.tsx`

```typescript
// Features to implement:
- Create new session button with loading state
- Display current session info (ID, created date, TTL)
- Extend TTL button
- Delete session button
- Auto-create session on first load if none exists
```

**API Calls Used**:
- `useCreateSession()` - Create new session
- `useSession(sessionId)` - Get session info
- `useUpdateSessionTTL(sessionId)` - Extend TTL
- `useDeleteSession()` - Delete session

### Task 2: Variable Form Component

**File**: `src/features/variables/VariableForm.tsx`

```typescript
// Form fields:
- Variable name (text input, required)
- Variable type (select: continuous/discrete/categorical)
- Conditional fields based on type:
  - Continuous/Discrete: min/max bounds
  - Categorical: list of category values
- Unit (text input, optional)
- Description (textarea, optional)
```

**Libraries**:
- React Hook Form for form state
- Zod for validation schema
- Shadcn Dialog for modal

**Validation Rules**:
```typescript
const variableSchema = z.object({
  name: z.string().min(1, "Name is required"),
  type: z.enum(['continuous', 'discrete', 'categorical']),
  bounds: z.tuple([z.number(), z.number()]).optional(),
  categories: z.array(z.string()).optional(),
  unit: z.string().optional(),
  description: z.string().optional(),
}).refine((data) => {
  // If continuous/discrete, bounds required
  // If categorical, categories required
});
```

### Task 3: Variable List Component

**File**: `src/features/variables/VariableList.tsx`

```typescript
// Display features:
- Table with columns: Name, Type, Bounds/Categories, Unit
- Edit button (opens VariableForm in edit mode)
- Delete button (with confirmation)
- Summary: "3 variables defined"
```

**API Calls Used**:
- `useVariables(sessionId)` - List all variables
- `useCreateVariable(sessionId)` - Add new variable
- React Query invalidation after edits

### Task 4: Main Variables Panel

**File**: `src/features/variables/VariablesPanel.tsx`

```typescript
// Layout:
- Header with "Add Variable" button
- VariableList component
- VariableForm in Dialog (triggered by button or edit)
```

### Task 5: Integrate into App

**File**: `src/App.tsx`

```typescript
// Add tabs or sections:
- Session info (collapsible header)
- Variables section (always visible when session exists)
- Placeholder sections for experiments, models, acquisition
```

## 🎨 UI Components Needed

Install Shadcn UI components:

```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add input
npx shadcn-ui@latest add label
npx shadcn-ui@latest add select
npx shadcn-ui@latest add table
npx shadcn-ui@latest add card
npx shadcn-ui@latest add tabs
npx shadcn-ui@latest add textarea
```

## 📝 Example Code Snippets

### Session Manager

```typescript
function SessionManager() {
  const [sessionId, setSessionId] = useState<string | null>(getStoredSessionId());
  const { data: session, isLoading } = useSession(sessionId);
  const createSession = useCreateSession();

  const handleCreateSession = async () => {
    const newSession = await createSession.mutateAsync();
    setSessionId(newSession.session_id);
  };

  if (!sessionId) {
    return <CreateSessionButton onClick={handleCreateSession} />;
  }

  return <SessionInfo session={session} />;
}
```

### Variable Form

```typescript
function VariableForm({ onSubmit, initialData }: Props) {
  const form = useForm<Variable>({
    resolver: zodResolver(variableSchema),
    defaultValues: initialData,
  });

  const varType = form.watch('type');

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <Input {...form.register('name')} label="Variable Name" />
      
      <Select {...form.register('type')}>
        <option value="continuous">Continuous</option>
        <option value="discrete">Discrete</option>
        <option value="categorical">Categorical</option>
      </Select>

      {(varType === 'continuous' || varType === 'discrete') && (
        <BoundsInput form={form} />
      )}

      {varType === 'categorical' && (
        <CategoriesInput form={form} />
      )}

      <Button type="submit">Add Variable</Button>
    </form>
  );
}
```

### Variable List

```typescript
function VariableList({ sessionId }: Props) {
  const { data, isLoading } = useVariables(sessionId);

  if (isLoading) return <LoadingSpinner />;

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Range/Categories</TableHead>
          <TableHead>Unit</TableHead>
          <TableHead>Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data?.variables.map((variable) => (
          <VariableRow key={variable.name} variable={variable} />
        ))}
      </TableBody>
    </Table>
  );
}
```

## 🧪 Testing Checklist

After implementation, test:

1. ✅ Create new session
2. ✅ Session info displays correctly
3. ✅ Add continuous variable with bounds
4. ✅ Add discrete variable with bounds
5. ✅ Add categorical variable with categories
6. ✅ Variables display in table
7. ✅ Edit existing variable
8. ✅ Delete variable
9. ✅ Form validation works (required fields, min < max)
10. ✅ Loading states display during API calls
11. ✅ Error messages show on API failures
12. ✅ Success toasts appear on success
13. ✅ Session persists across page refresh

## 📊 Desktop UI Reference

From the desktop app (`ui/variables_setup.py`):

- **Window Title**: "Setup Search Space"
- **Fields**: Name, Type dropdown, Min/Max for continuous, Categories for categorical
- **Buttons**: "Add Variable", "Save to File", "Load from File"
- **Table**: Shows all defined variables with edit/delete options

Our web version will match this functionality but with modern web UI patterns.

## 💡 Pro Tips

1. **Start Simple** - Get basic add/list working before edit/delete
2. **Use DevTools** - Check Network tab to see API calls
3. **Add Logging** - Console.log data at each step
4. **Test with Backend** - Make sure FastAPI is running
5. **Incremental Commits** - Commit after each working feature

## 🔗 Resources

- [React Hook Form Docs](https://react-hook-form.com/)
- [Zod Validation](https://zod.dev/)
- [Shadcn UI Components](https://ui.shadcn.com/)
- [TanStack Table](https://tanstack.com/table/latest)

## ⏭️ Next Phase Preview

After Variables module is complete, we'll build:

**Phase 3: Experiments Module**
- Upload CSV files
- Manual data entry
- Display experiment table
- Summary statistics

Ready to start? Begin with Task 1 (Session Manager)!
