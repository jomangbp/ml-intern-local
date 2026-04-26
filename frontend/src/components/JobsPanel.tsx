import { useEffect, useState, useCallback } from 'react';
import {
  Box, Typography, List, ListItem, ListItemText, ListItemIcon,
  Chip, IconButton, Dialog, DialogTitle, DialogContent, DialogActions,
  Button, Tooltip, LinearProgress, Paper,
} from '@mui/material';
import {
  Stop as StopIcon,
  Delete as KillIcon,
  Refresh as RefreshIcon,
  Description as LogIcon,
  Terminal as JobIcon,
} from '@mui/icons-material';

interface Job {
  job_id: string;
  kind: string;
  command: string;
  status: string;
  exit_code: number | null;
  pid: number | null;
  started_at: number | null;
  ended_at: number | null;
  created_at: number;
  cwd: string;
  created_by: Record<string, string>;
}

interface JobLogsDialogProps {
  jobId: string | null;
  open: boolean;
  onClose: () => void;
}

function JobLogsDialog({ jobId, open, onClose }: JobLogsDialogProps) {
  const [logs, setLogs] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !jobId) return;
    setLoading(true);
    fetch(`/api/jobs/${jobId}/logs?lines=200`)
      .then(r => r.json())
      .then(data => setLogs(data.logs || 'No logs available.'))
      .catch(() => setLogs('Error loading logs.'))
      .finally(() => setLoading(false));
  }, [open, jobId]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        📋 Job Logs — {jobId?.slice(0, 12)}...
      </DialogTitle>
      <DialogContent>
        {loading ? <LinearProgress /> : (
          <Box
            component="pre"
            sx={{
              bgcolor: '#1e1e1e',
              color: '#d4d4d4',
              p: 2,
              borderRadius: 1,
              fontSize: '0.75rem',
              fontFamily: 'monospace',
              maxHeight: '60vh',
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}
          >
            {logs}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

function statusIcon(status: string) {
  switch (status) {
    case 'running': return '🟢';
    case 'completed': return '✅';
    case 'failed': return '❌';
    case 'cancelled': return '⚪';
    case 'killed': return '💀';
    case 'queued': return '⏳';
    case 'unknown': return '❓';
    default: return '❓';
  }
}

function formatElapsed(start: number | null, end: number | null): string {
  if (!start) return '—';
  const e = end || Date.now() / 1000;
  const s = Math.floor(e - start);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}m${rs.toString().padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h${rm.toString().padStart(2, '0')}m`;
}

interface JobsPanelProps {
  collapsed?: boolean;
}

export default function JobsPanel({ collapsed = false }: JobsPanelProps) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(false);
  const [logsJobId, setLogsJobId] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    fetch('/api/jobs?limit=20')
      .then(r => r.json())
      .then(data => setJobs(data))
      .catch(() => setJobs([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleStop = async (jobId: string) => {
    await fetch(`/api/jobs/${jobId}/stop`, { method: 'POST' });
    refresh();
  };

  const handleKill = async (jobId: string) => {
    await fetch(`/api/jobs/${jobId}/kill`, { method: 'POST' });
    refresh();
  };

  const running = jobs.filter(j => j.status === 'running').length;

  if (collapsed) {
    return (
      <Tooltip title={`${running} running jobs`}>
        <Chip
          icon={<JobIcon />}
          label={running}
          size="small"
          color={running > 0 ? 'success' : 'default'}
          variant="outlined"
          sx={{ cursor: 'pointer' }}
        />
      </Tooltip>
    );
  }

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="h6" sx={{ fontSize: '0.95rem' }}>
          🔧 Jobs {running > 0 && <Chip label={`${running} running`} size="small" color="success" sx={{ ml: 1 }} />}
        </Typography>
        <IconButton size="small" onClick={refresh} disabled={loading}>
          <RefreshIcon fontSize="small" />
        </IconButton>
      </Box>

      {jobs.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          No jobs yet. Jobs appear when training or scripts are run.
        </Typography>
      ) : (
        <List dense sx={{ flex: 1, overflow: 'auto' }}>
          {jobs.map(job => (
            <ListItem
              key={job.job_id}
              sx={{
                borderBottom: '1px solid',
                borderColor: 'divider',
                '&:hover': { bgcolor: 'action.hover' },
              }}
              secondaryAction={
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  <Tooltip title="View logs">
                    <IconButton size="small" onClick={() => setLogsJobId(job.job_id)}>
                      <LogIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  {job.status === 'running' && (
                    <>
                      <Tooltip title="Stop (SIGTERM)">
                        <IconButton size="small" onClick={() => handleStop(job.job_id)}>
                          <StopIcon fontSize="small" color="warning" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Kill (SIGKILL)">
                        <IconButton size="small" onClick={() => handleKill(job.job_id)}>
                          <KillIcon fontSize="small" color="error" />
                        </IconButton>
                      </Tooltip>
                    </>
                  )}
                </Box>
              }
            >
              <ListItemIcon sx={{ minWidth: 32 }}>
                <Typography variant="body2">{statusIcon(job.status)}</Typography>
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                    {job.command.slice(0, 80)}{job.command.length > 80 ? '...' : ''}
                  </Typography>
                }
                secondary={
                  <Typography variant="caption" color="text.secondary">
                    {job.kind} · {job.job_id.slice(0, 12)} · {formatElapsed(job.started_at, job.ended_at)}
                    {job.exit_code != null && job.exit_code !== 0 && ` · exit ${job.exit_code}`}
                  </Typography>
                }
              />
            </ListItem>
          ))}
        </List>
      )}

      <JobLogsDialog
        jobId={logsJobId}
        open={logsJobId !== null}
        onClose={() => setLogsJobId(null)}
      />
    </Paper>
  );
}
