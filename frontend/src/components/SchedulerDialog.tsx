import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  IconButton,
  MenuItem,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import CancelIcon from '@mui/icons-material/Cancel';
import RefreshIcon from '@mui/icons-material/Refresh';
import ScheduleIcon from '@mui/icons-material/Schedule';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { apiFetch } from '@/utils/api';
import type { ExecutionMode } from '@/utils/executionMode';

type SchedulerStatus = 'scheduled' | 'running' | 'checking' | 'completed' | 'cancelled' | 'failed' | string;

interface SchedulerTaskConfig {
  task_name?: string;
  interval_minutes?: number;
  interval_seconds?: number;
  repeat?: boolean;
  max_runs?: number;
  command?: string;
  work_dir?: string;
  training_match?: string;
  match_mode?: 'substring' | 'regex' | string;
  stop_if_running?: boolean;
  grace_seconds?: number;
  log_path?: string;
  kind?: string;
  prompt?: string;
  session_id?: string;
  run_immediately?: boolean;
}

interface SchedulerProcessMatch {
  pid?: number;
  ppid?: number;
  command?: string;
}

interface SchedulerStopResult {
  attempted_pids?: number[];
  sigkill_pids?: number[];
  errors?: string[];
}

interface SchedulerCheck {
  finished_at?: string;
  command_exit_code?: number | string | null;
  matching_processes?: SchedulerProcessMatch[];
  stop_result?: SchedulerStopResult | null;
  output_tail?: string;
  prompt_submitted?: boolean;
  prompt?: string;
}

interface SchedulerTask {
  task_id?: string;
  status?: SchedulerStatus;
  runner_pid?: number;
  runner_alive?: boolean;
  runs_completed?: number;
  config?: SchedulerTaskConfig;
  last_check?: SchedulerCheck;
  log_path?: string;
  created_at?: string;
  finished_at?: string;
  error?: string;
}

interface SchedulerDialogProps {
  open: boolean;
  onClose: () => void;
  activeExecutionMode: ExecutionMode | null;
  activeSessionId: string | null;
}

function parseError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return 'Unknown scheduler error';
}

async function parseApiError(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: string };
    if (typeof data.detail === 'string') return data.detail;
  } catch {
    // ignore and use status text
  }
  return `${res.status} ${res.statusText}`.trim();
}

function statusColor(status?: SchedulerStatus): 'default' | 'primary' | 'success' | 'warning' | 'error' {
  switch ((status || '').toLowerCase()) {
    case 'completed':
      return 'success';
    case 'running':
    case 'checking':
      return 'primary';
    case 'scheduled':
      return 'warning';
    case 'failed':
      return 'error';
    default:
      return 'default';
  }
}

function compactDate(value?: string): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export default function SchedulerDialog({ open, onClose, activeExecutionMode, activeSessionId }: SchedulerDialogProps) {
  const [mode, setMode] = useState<'prompt' | 'watchdog'>('prompt');
  const [taskName, setTaskName] = useState('Training cron');
  const [intervalMinutes, setIntervalMinutes] = useState('30');
  const [repeat, setRepeat] = useState(true);
  const [maxRuns, setMaxRuns] = useState('');
  const [trainingMatch, setTrainingMatch] = useState('scripts/train.py');
  const [matchMode, setMatchMode] = useState<'substring' | 'regex'>('substring');
  const [stopIfRunning, setStopIfRunning] = useState(true);
  const [graceSeconds, setGraceSeconds] = useState('30');
  const [command, setCommand] = useState('');
  const [workDir, setWorkDir] = useState('.');
  const [commandTimeoutSeconds, setCommandTimeoutSeconds] = useState('120');
  const [cronPrompt, setCronPrompt] = useState('Check training progress and stop the training if it is still running.');

  const [tasks, setTasks] = useState<SchedulerTask[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const isLocal = activeExecutionMode === 'local';

  const payload = useMemo(() => {
    const base = {
      task_name: taskName.trim() || (mode === 'prompt' ? 'Training cron' : 'Training watchdog'),
      interval_minutes: Number(intervalMinutes),
      repeat,
      max_runs: maxRuns.trim() ? Number(maxRuns) : 0,
    };
    if (mode === 'prompt') {
      return {
        ...base,
        prompt: cronPrompt.trim(),
        session_id: activeSessionId || '',
      };
    }
    return {
      ...base,
      training_match: trainingMatch.trim(),
      match_mode: matchMode,
      stop_if_running: stopIfRunning,
      grace_seconds: Number(graceSeconds),
      command: command.trim(),
      work_dir: workDir.trim() || '.',
      command_timeout_seconds: Number(commandTimeoutSeconds),
    };
  }, [
    activeSessionId,
    command,
    commandTimeoutSeconds,
    cronPrompt,
    graceSeconds,
    intervalMinutes,
    matchMode,
    maxRuns,
    mode,
    repeat,
    stopIfRunning,
    taskName,
    trainingMatch,
    workDir,
  ]);

  const refreshTasks = useCallback(async () => {
    const res = await apiFetch('/api/scheduler/tasks');
    if (!res.ok) throw new Error(await parseApiError(res));
    const data = (await res.json()) as { tasks?: SchedulerTask[] };
    setTasks(Array.isArray(data.tasks) ? data.tasks : []);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    let cancelled = false;
    const refresh = async () => {
      try {
        const res = await apiFetch('/api/scheduler/tasks');
        if (!res.ok) return;
        const data = (await res.json()) as { tasks?: SchedulerTask[] };
        if (!cancelled) setTasks(Array.isArray(data.tasks) ? data.tasks : []);
      } catch {
        // ignore transient refresh errors
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [open]);

  const validatePayload = useCallback((): string | null => {
    if (!Number.isFinite(payload.interval_minutes) || payload.interval_minutes <= 0) {
      return 'Interval must be greater than 0 minutes.';
    }
    if (payload.max_runs < 0 || !Number.isFinite(payload.max_runs)) {
      return 'Max runs must be empty, 0, or a positive number.';
    }
    if (mode === 'prompt') {
      if (!activeSessionId) return 'Create or select a session before scheduling a prompt cron.';
      if (!('prompt' in payload) || !payload.prompt) return 'Prompt is required.';
      return null;
    }
    if (!('command' in payload) || !('training_match' in payload) || (!payload.command && !payload.training_match)) {
      return 'Provide a training process match, a command, or both.';
    }
    if (payload.stop_if_running && payload.training_match.length < 6) {
      return 'Training match is too broad. Use a more specific command substring.';
    }
    if (!Number.isFinite(payload.grace_seconds) || payload.grace_seconds < 0) {
      return 'Grace seconds must be 0 or greater.';
    }
    if (!Number.isFinite(payload.command_timeout_seconds) || payload.command_timeout_seconds <= 0) {
      return 'Command timeout must be greater than 0 seconds.';
    }
    return null;
  }, [activeSessionId, mode, payload]);

  const createTask = useCallback(async () => {
    const validationError = validatePayload();
    if (validationError) {
      setError(validationError);
      return;
    }
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const res = await apiFetch('/api/scheduler/tasks', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await parseApiError(res));
      const data = (await res.json()) as { task_id?: string; output?: string; tasks?: SchedulerTask[] };
      setMessage(data.task_id ? `Scheduled ${mode === 'prompt' ? 'cron' : 'watchdog'} ${data.task_id}.` : (data.output || 'Scheduled task.'));
      setTasks(Array.isArray(data.tasks) ? data.tasks : tasks);
      await refreshTasks();
    } catch (e) {
      setError(parseError(e));
    } finally {
      setBusy(false);
    }
  }, [mode, payload, refreshTasks, tasks, validatePayload]);

  const runOnce = useCallback(async () => {
    const validationError = validatePayload();
    if (validationError) {
      setError(validationError);
      return;
    }
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const res = await apiFetch('/api/scheduler/run-once', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await parseApiError(res));
      const data = (await res.json()) as { output?: string };
      setMessage(data.output || 'Ran check once.');
      await refreshTasks();
    } catch (e) {
      setError(parseError(e));
    } finally {
      setBusy(false);
    }
  }, [payload, refreshTasks, validatePayload]);

  const cancelTask = useCallback(async (taskId: string) => {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const res = await apiFetch(`/api/scheduler/tasks/${encodeURIComponent(taskId)}/cancel`, { method: 'POST' });
      if (!res.ok) throw new Error(await parseApiError(res));
      setMessage(`Cancelled ${taskId}.`);
      await refreshTasks();
    } catch (e) {
      setError(parseError(e));
    } finally {
      setBusy(false);
    }
  }, [refreshTasks]);

  const manualRefresh = useCallback(async () => {
    setBusy(true);
    setError('');
    try {
      await refreshTasks();
    } catch (e) {
      setError(parseError(e));
    } finally {
      setBusy(false);
    }
  }, [refreshTasks]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ScheduleIcon fontSize="small" />
        Training Scheduler
      </DialogTitle>
      <DialogContent dividers>
        {!isLocal && mode === 'watchdog' && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Watchdog controls operate on the machine running the backend. Use Local mode for training watchdogs.
          </Alert>
        )}
        <Alert severity="info" sx={{ mb: 2 }}>
          Chat shortcut: <Box component="span" sx={{ fontFamily: 'monospace' }}>/cron [minutes] &lt;prompt to send to agent&gt;</Box>
        </Alert>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1.5 }}>
          <TextField
            select
            size="small"
            label="Scheduler type"
            value={mode}
            onChange={(e) => {
              const next = e.target.value as 'prompt' | 'watchdog';
              setMode(next);
              setTaskName(next === 'prompt' ? 'Training cron' : 'Training watchdog');
              setRepeat(next === 'prompt');
            }}
          >
            <MenuItem value="prompt">Prompt cron: send prompt to agent</MenuItem>
            <MenuItem value="watchdog">Process watchdog: check/stop training</MenuItem>
          </TextField>
          <TextField
            size="small"
            label="Task name"
            value={taskName}
            onChange={(e) => setTaskName(e.target.value)}
          />
          <TextField
            size="small"
            label="Every (minutes)"
            type="number"
            value={intervalMinutes}
            onChange={(e) => setIntervalMinutes(e.target.value)}
            inputProps={{ min: 0.01, step: 1 }}
          />
          {mode === 'prompt' ? (
            <TextField
              size="small"
              label="Prompt to send to agent"
              value={cronPrompt}
              onChange={(e) => setCronPrompt(e.target.value)}
              multiline
              minRows={4}
              helperText="Equivalent to: /cron [minutes] <this prompt>"
              sx={{ gridColumn: { xs: 'auto', md: '1 / span 2' } }}
            />
          ) : (
            <>
              <TextField
                size="small"
                label="Training process match"
                value={trainingMatch}
                onChange={(e) => setTrainingMatch(e.target.value)}
                helperText="Specific substring or regex from the process command line"
                sx={{ gridColumn: { xs: 'auto', md: '1 / span 2' } }}
              />
              <TextField
                select
                size="small"
                label="Match mode"
                value={matchMode}
                onChange={(e) => setMatchMode(e.target.value as 'substring' | 'regex')}
              >
                <MenuItem value="substring">Substring</MenuItem>
                <MenuItem value="regex">Regex</MenuItem>
              </TextField>
              <TextField
                size="small"
                label="Grace seconds before SIGKILL"
                type="number"
                value={graceSeconds}
                onChange={(e) => setGraceSeconds(e.target.value)}
                inputProps={{ min: 0, step: 1 }}
              />
              <TextField
                size="small"
                label="Optional check command"
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                helperText="Runs before matching/stopping, e.g. tail -n 20 training_log.txt"
                sx={{ gridColumn: { xs: 'auto', md: '1 / span 2' } }}
              />
              <TextField
                size="small"
                label="Command working directory"
                value={workDir}
                onChange={(e) => setWorkDir(e.target.value)}
              />
              <TextField
                size="small"
                label="Command timeout seconds"
                type="number"
                value={commandTimeoutSeconds}
                onChange={(e) => setCommandTimeoutSeconds(e.target.value)}
                inputProps={{ min: 1, step: 1 }}
              />
            </>
          )}
        </Box>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mt: 1.5 }}>
          {mode === 'watchdog' && (
            <FormControlLabel
              control={<Switch checked={stopIfRunning} onChange={(e) => setStopIfRunning(e.target.checked)} />}
              label="Stop matching training if still running"
            />
          )}
          <FormControlLabel
            control={<Switch checked={repeat} onChange={(e) => setRepeat(e.target.checked)} />}
            label={mode === 'prompt' ? 'Repeat prompt every interval' : 'Repeat every interval'}
          />
          {repeat && (
            <TextField
              size="small"
              label="Max runs (0 = unlimited)"
              type="number"
              value={maxRuns}
              onChange={(e) => setMaxRuns(e.target.value)}
              inputProps={{ min: 0, step: 1 }}
              sx={{ maxWidth: 190 }}
            />
          )}
        </Stack>

        <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            startIcon={<ScheduleIcon />}
            onClick={createTask}
            disabled={busy}
            sx={{ textTransform: 'none', bgcolor: '#FF9D00', color: '#000' }}
          >
            {mode === 'prompt' ? 'Schedule /cron' : 'Schedule watchdog'}
          </Button>
          <Tooltip title={mode === 'prompt' ? 'Sends the configured prompt to the active agent now.' : 'Runs the configured check immediately. If stop is enabled, matching processes will be stopped now.'}>
            <span>
              <Button
                variant="outlined"
                startIcon={<PlayArrowIcon />}
                onClick={runOnce}
                disabled={busy}
                sx={{ textTransform: 'none' }}
              >
                {mode === 'prompt' ? 'Send prompt now' : 'Run check now'}
              </Button>
            </span>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={manualRefresh}
            disabled={busy}
            sx={{ textTransform: 'none' }}
          >
            Refresh
          </Button>
        </Stack>

        {message && (
          <Alert severity="success" sx={{ mt: 2, whiteSpace: 'pre-wrap' }}>
            {message}
          </Alert>
        )}
        {error && (
          <Alert severity="error" sx={{ mt: 2, whiteSpace: 'pre-wrap' }}>
            {error}
          </Alert>
        )}

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700 }}>
          Scheduled tasks
        </Typography>
        {tasks.length === 0 ? (
          <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontFamily: 'monospace' }}>
            No scheduler tasks yet.
          </Typography>
        ) : (
          <Stack spacing={1}>
            {tasks.map((task) => {
              const taskId = task.task_id || 'unknown';
              const config = task.config || {};
              const matches = task.last_check?.matching_processes || [];
              const stopped = task.last_check?.stop_result?.attempted_pids || [];
              const isPromptCron = config.kind === 'prompt_cron' || !!config.prompt;
              const canCancel = ['scheduled', 'running', 'checking'].includes((task.status || '').toLowerCase());
              return (
                <Box
                  key={taskId}
                  sx={{
                    p: 1.25,
                    border: '1px solid var(--border)',
                    borderRadius: 1.5,
                    bgcolor: 'rgba(255,255,255,0.03)',
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, alignItems: 'flex-start' }}>
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="body2" sx={{ fontWeight: 700 }}>
                        {config.task_name || taskId}
                      </Typography>
                      <Typography variant="caption" sx={{ display: 'block', color: 'var(--muted-text)', fontFamily: 'monospace' }}>
                        {taskId}{isPromptCron ? ` · session ${config.session_id || '—'}` : ` · PID ${task.runner_pid || '—'}${task.runner_pid ? ` (${task.runner_alive ? 'alive' : 'done'})` : ''}`}
                      </Typography>
                    </Box>
                    <Stack direction="row" spacing={0.75} alignItems="center">
                      <Chip size="small" color={statusColor(task.status)} label={task.status || 'unknown'} />
                      {canCancel && (
                        <Tooltip title="Cancel watchdog">
                          <IconButton size="small" onClick={() => cancelTask(taskId)} disabled={busy}>
                            <CancelIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Stack>
                  </Box>
                  <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'var(--muted-text)' }}>
                    Interval: {config.interval_minutes ?? '—'} min · Repeat: {config.repeat ? 'yes' : 'no'} · Runs: {task.runs_completed ?? 0}
                  </Typography>
                  {config.prompt && (
                    <Typography variant="caption" sx={{ display: 'block', color: 'var(--muted-text)', fontFamily: 'monospace', wordBreak: 'break-word' }}>
                      prompt: {config.prompt}
                    </Typography>
                  )}
                  {config.training_match && (
                    <Typography variant="caption" sx={{ display: 'block', color: 'var(--muted-text)', fontFamily: 'monospace', wordBreak: 'break-word' }}>
                      match: {config.training_match}
                    </Typography>
                  )}
                  {config.command && (
                    <Typography variant="caption" sx={{ display: 'block', color: 'var(--muted-text)', fontFamily: 'monospace', wordBreak: 'break-word' }}>
                      command: {config.command}
                    </Typography>
                  )}
                  <Typography variant="caption" sx={{ display: 'block', color: 'var(--muted-text)' }}>
                    {isPromptCron
                      ? `Last send: ${compactDate(task.last_check?.finished_at)} · submitted: ${task.last_check?.prompt_submitted === undefined ? '—' : String(task.last_check.prompt_submitted)}`
                      : `Last check: ${compactDate(task.last_check?.finished_at)} · matches: ${matches.length} · stopped pids: ${stopped.length ? stopped.join(', ') : 'none'}`}
                  </Typography>
                  {!isPromptCron && (
                    <Typography variant="caption" sx={{ display: 'block', color: 'var(--muted-text)', fontFamily: 'monospace', wordBreak: 'break-word' }}>
                      log: {task.log_path || config.log_path || '—'}
                    </Typography>
                  )}
                  {task.error && (
                    <Alert severity="error" sx={{ mt: 1 }}>{task.error}</Alert>
                  )}
                </Box>
              );
            })}
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} sx={{ textTransform: 'none' }}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
