import { Box } from '@mui/material';
import AppLayout from '@/components/Layout/AppLayout';
import { useAuth } from '@/hooks/useAuth';

function App() {
  // Non-blocking auth check — fires in background, updates store when done.
  // If auth fails later, apiFetch redirects to /auth/login.
  useAuth();

  return (
    <Box sx={{ height: '100vh', minHeight: '100dvh', display: 'flex', width: '100%' }}>
      <AppLayout />
    </Box>
  );
}

export default App;
