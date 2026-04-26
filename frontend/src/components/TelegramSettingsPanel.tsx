import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  MenuItem,
  TextField,
  Typography,
} from '@mui/material';
import { apiFetch } from '@/utils/api';
import type { ExecutionMode } from '@/utils/executionMode';

export default function TelegramSettingsPanel() {
  const [busy, setBusy] = useState(false);
  const [configured, setConfigured] = useState(false);
  const [running, setRunning] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const [maskedToken, setMaskedToken] = useState('');
  const [token, setToken] = useState('');
  const [allowedChatIds, setAllowedChatIds] = useState('');
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('local');
  const [timeoutSeconds, setTimeoutSeconds] = useState('3600');
  const [configPath, setConfigPath] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const res = await apiFetch('/auth/telegram/status');
      if (!res.ok) return;
      const data = await res.json();
      setConfigured(!!data?.configured);
      setRunning(!!data?.running);
      setEnabled(!!data?.enabled);
      setMaskedToken(typeof data?.masked_token === 'string' ? data.masked_token : '');
      setAllowedChatIds(Array.isArray(data?.allowed_chat_ids) ? data.allowed_chat_ids.join(',') : '');
      setExecutionMode(data?.execution_mode === 'sandbox' ? 'sandbox' : 'local');
      setTimeoutSeconds(String(data?.turn_timeout_seconds || 3600));
      setConfigPath(typeof data?.config_path === 'string' ? data.config_path : '');
    } catch {
      // ignore transient errors
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const saveConfig = useCallback(async (clearToken = false, forceEnabled?: boolean) => {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const res = await apiFetch('/auth/telegram/config', {
        method: 'POST',
        body: JSON.stringify({
          token: token.trim(),
          clear_token: clearToken,
          enabled: forceEnabled ?? enabled,
          allowed_chat_ids: allowedChatIds,
          execution_mode: executionMode,
          turn_timeout_seconds: Number(timeoutSeconds) || 3600,
        }),
      });
      if (!res.ok) {
        setError('Failed to save Telegram bot config.');
        return;
      }
      const data = await res.json();
      setToken('');
      setConfigured(!!data?.configured);
      setRunning(!!data?.running);
      setEnabled(!!data?.enabled);
      setMaskedToken(typeof data?.masked_token === 'string' ? data.masked_token : '');
      setAllowedChatIds(Array.isArray(data?.allowed_chat_ids) ? data.allowed_chat_ids.join(',') : '');
      setExecutionMode(data?.execution_mode === 'sandbox' ? 'sandbox' : 'local');
      setTimeoutSeconds(String(data?.turn_timeout_seconds || 3600));
      setConfigPath(typeof data?.config_path === 'string' ? data.config_path : '');
      setMessage(data?.running ? 'Telegram bot is running.' : 'Telegram bot config saved.');
    } catch {
      setError('Failed to save Telegram bot config.');
    } finally {
      setBusy(false);
    }
  }, [allowedChatIds, enabled, executionMode, timeoutSeconds, token]);

  const stopBot = useCallback(async () => {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const res = await apiFetch('/auth/telegram/stop', { method: 'POST' });
      if (!res.ok) {
        setError('Failed to stop Telegram bot.');
        return;
      }
      const data = await res.json();
      setRunning(!!data?.running);
      setEnabled(!!data?.enabled);
      setMessage('Telegram bot stopped.');
    } catch {
      setError('Failed to stop Telegram bot.');
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <Box>
      <Box sx={{ mt: 0.75, display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        <Chip
          size="small"
          label={running ? 'Running' : configured ? 'Configured' : 'Not configured'}
          sx={{
            fontWeight: 600,
            color: running ? '#000' : 'var(--muted-text)',
            bgcolor: running ? 'var(--accent-green)' : 'rgba(255,255,255,0.06)',
            border: '1px solid var(--border)',
          }}
        />
        {maskedToken && (
          <Typography variant="caption" sx={{ color: 'var(--muted-text)', fontFamily: 'monospace' }}>
            {maskedToken}
          </Typography>
        )}
      </Box>
      {!!configPath && (
        <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'var(--muted-text)', fontFamily: 'monospace' }} title={configPath}>
          Config: {configPath}
        </Typography>
      )}
      <Box sx={{ mt: 0.75, display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 1 }}>
        <TextField
          size="small"
          type="password"
          placeholder={configured ? 'Telegram bot token configured' : 'Telegram bot token'}
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
        <TextField
          size="small"
          placeholder="Allowed chat IDs, comma-separated (optional)"
          value={allowedChatIds}
          onChange={(e) => setAllowedChatIds(e.target.value)}
        />
        <TextField
          select
          size="small"
          label="Execution mode"
          value={executionMode}
          onChange={(e) => setExecutionMode(e.target.value === 'sandbox' ? 'sandbox' : 'local')}
        >
          <MenuItem value="local">Local</MenuItem>
          <MenuItem value="sandbox">Sandbox</MenuItem>
        </TextField>
        <TextField
          size="small"
          label="Turn timeout seconds"
          type="number"
          value={timeoutSeconds}
          onChange={(e) => setTimeoutSeconds(e.target.value)}
          inputProps={{ min: 30, step: 30 }}
        />
      </Box>
      <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          size="small"
          onClick={() => saveConfig(false, true)}
          disabled={busy || (!token.trim() && !configured)}
          sx={{ textTransform: 'none', bgcolor: '#FF9D00', color: '#000' }}
        >
          Save & start bot
        </Button>
        <Button
          variant="outlined"
          size="small"
          onClick={() => saveConfig(false, false)}
          disabled={busy}
          sx={{ textTransform: 'none' }}
        >
          Save disabled
        </Button>
        <Button
          variant="outlined"
          size="small"
          onClick={stopBot}
          disabled={busy || !running}
          sx={{ textTransform: 'none' }}
        >
          Stop bot
        </Button>
        <Button
          variant="outlined"
          size="small"
          onClick={() => saveConfig(true, false)}
          disabled={busy || !configured}
          sx={{ textTransform: 'none' }}
        >
          Clear token
        </Button>
      </Box>
      <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'var(--muted-text)', fontFamily: 'monospace' }}>
        Commands: /start, /help, /commands, /new, /status, /gateway on|off, /models, /model &lt;id|number|label&gt;, /sessions, /crons, /cancelcron &lt;id&gt;, /interrupt, /cron [minutes] &lt;prompt&gt;
      </Typography>
      {message && (
        <Alert severity="success" sx={{ mt: 2 }}>
          {message}
        </Alert>
      )}
      {error && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
}
