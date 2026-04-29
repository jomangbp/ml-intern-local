import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Stack,
  Typography,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import SaveOutlinedIcon from '@mui/icons-material/SaveOutlined';
import RestoreIcon from '@mui/icons-material/Restore';
import { apiFetch } from '@/utils/api';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { getPreferredExecutionMode } from '@/utils/executionMode';

interface SavedSessionInfo {
  saved_id: string;
  title: string;
  model?: string | null;
  execution_mode?: string | null;
  message_count: number;
  last_save_time?: string | null;
}

interface SavedSessionsDialogProps {
  open: boolean;
  onClose: () => void;
  currentSessionId?: string | null;
}

function formatSavedTime(value?: string | null): string {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value.slice(0, 19).replace('T', ' ');
  return d.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}

export default function SavedSessionsDialog({ open, onClose, currentSessionId }: SavedSessionsDialogProps) {
  const { createSession, updateSessionTitle, switchSession } = useSessionStore();
  const { setPlan, clearPanel } = useAgentStore();
  const [sessions, setSessions] = useState<SavedSessionInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadSaved = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await apiFetch('/api/saved-sessions?limit=50');
      if (!res.ok) throw new Error(`List failed: ${res.status}`);
      const data = await res.json();
      setSessions(Array.isArray(data.sessions) ? data.sessions : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to list saved sessions.');
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (open) void loadSaved();
  }, [open, loadSaved]);

  const handleSaveCurrent = useCallback(async () => {
    if (!currentSessionId || saving) return;
    setSaving(true);
    setError(null);
    try {
      const res = await apiFetch(`/api/session/${currentSessionId}/save`, { method: 'POST', body: JSON.stringify({}) });
      if (!res.ok) throw new Error(`Save failed: ${res.status}`);
      await loadSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save current session.');
    } finally {
      setSaving(false);
    }
  }, [currentSessionId, saving, loadSaved]);

  const handleResume = useCallback(async (saved: SavedSessionInfo) => {
    setRestoringId(saved.saved_id);
    setError(null);
    try {
      const res = await apiFetch(`/api/saved-sessions/${encodeURIComponent(saved.saved_id)}/resume`, {
        method: 'POST',
        body: JSON.stringify({ mode: 'exact', execution_mode: getPreferredExecutionMode() }),
      });
      if (!res.ok) throw new Error(`Resume failed: ${res.status}`);
      const data = await res.json();
      createSession(data.session_id);
      updateSessionTitle(data.session_id, saved.title || 'Resumed session');
      switchSession(data.session_id);
      setPlan([]);
      clearPanel();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to resume saved session.');
    } finally {
      setRestoringId(null);
    }
  }, [createSession, updateSessionTitle, switchSession, setPlan, clearPanel, onClose]);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>Saved sessions</Typography>
          <Typography variant="caption" sx={{ color: 'var(--muted-text)' }}>
            Resume old research/training context like `resume`.
          </Typography>
        </Box>
        <IconButton onClick={() => void loadSaved()} disabled={busy} size="small">
          <RefreshIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Stack spacing={1.5}>
          {error && <Alert severity="error">{error}</Alert>}
          {currentSessionId && (
            <Button
              variant="outlined"
              startIcon={saving ? <CircularProgress size={16} /> : <SaveOutlinedIcon />}
              onClick={handleSaveCurrent}
              disabled={saving}
              sx={{ textTransform: 'none', alignSelf: 'flex-start' }}
            >
              {saving ? 'Saving...' : 'Save current session'}
            </Button>
          )}
          <Divider />
          {busy ? (
            <Box sx={{ py: 4, display: 'flex', justifyContent: 'center' }}><CircularProgress size={24} /></Box>
          ) : sessions.length === 0 ? (
            <Typography variant="body2" sx={{ color: 'var(--muted-text)', py: 2 }}>
              No saved sessions found yet. Save a session first.
            </Typography>
          ) : (
            <List disablePadding>
              {sessions.map((saved) => (
                <ListItem
                  key={saved.saved_id}
                  divider
                  secondaryAction={
                    <Button
                      size="small"
                      variant="contained"
                      startIcon={restoringId === saved.saved_id ? <CircularProgress size={14} color="inherit" /> : <RestoreIcon />}
                      disabled={!!restoringId}
                      onClick={() => void handleResume(saved)}
                      sx={{ textTransform: 'none' }}
                    >
                      Resume
                    </Button>
                  }
                  sx={{ pl: 0, pr: 12 }}
                >
                  <ListItemText
                    primary={saved.title || saved.saved_id}
                    secondary={`${saved.message_count} messages · ${formatSavedTime(saved.last_save_time)}${saved.model ? ` · ${saved.model}` : ''}`}
                    primaryTypographyProps={{ noWrap: true, sx: { fontWeight: 600 } }}
                    secondaryTypographyProps={{ noWrap: true }}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} sx={{ textTransform: 'none' }}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
