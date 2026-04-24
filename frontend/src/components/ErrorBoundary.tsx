import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Alert, AlertTitle, Box, Button, Stack, Typography } from '@mui/material';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  errorMessage: string;
}

const STORAGE_KEYS_TO_RESET = [
  'hf-agent-sessions',
  'hf-agent-layout',
  'hf-agent-messages',
  'hf-agent-backend-messages',
  'hf-agent-tool-errors',
  'hf-agent-rejected-tools',
];

export default class ErrorBoundary extends Component<Props, State> {
  state: State = {
    hasError: false,
    errorMessage: '',
  };

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      errorMessage: error?.message || 'Unknown UI error',
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Keep this for debugging in browser console.
    // eslint-disable-next-line no-console
    console.error('UI crash caught by ErrorBoundary:', error, errorInfo);
  }

  private handleResetStorage = () => {
    try {
      for (const key of STORAGE_KEYS_TO_RESET) {
        localStorage.removeItem(key);
      }
    } catch {
      // ignore
    }
    window.location.reload();
  };

  private handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 2,
          bgcolor: 'background.default',
        }}
      >
        <Alert severity="error" sx={{ maxWidth: 760, width: '100%' }}>
          <AlertTitle>UI crashed (recovered)</AlertTitle>
          <Typography variant="body2" sx={{ mb: 1 }}>
            The app hit a runtime error and rendered a safe recovery screen instead of a blank page.
          </Typography>
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              fontFamily: 'monospace',
              mb: 1.5,
              whiteSpace: 'pre-wrap',
              opacity: 0.85,
            }}
          >
            {this.state.errorMessage}
          </Typography>
          <Stack direction="row" spacing={1.25}>
            <Button variant="contained" color="error" onClick={this.handleResetStorage}>
              Reset local data
            </Button>
            <Button variant="outlined" color="error" onClick={this.handleReload}>
              Reload
            </Button>
          </Stack>
        </Alert>
      </Box>
    );
  }
}
