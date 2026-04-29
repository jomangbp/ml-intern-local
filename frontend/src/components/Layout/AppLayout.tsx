import { useCallback, useRef, useEffect, useState } from 'react';
import {
  Avatar,
  Box,
  Drawer,
  Typography,
  IconButton,
  Alert,
  AlertTitle,
  Snackbar,
  Chip,
  Tooltip,
  Link,
  useMediaQuery,
  useTheme,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  MenuItem,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import DarkModeOutlinedIcon from '@mui/icons-material/DarkModeOutlined';
import LightModeOutlinedIcon from '@mui/icons-material/LightModeOutlined';
import SettingsOutlinedIcon from '@mui/icons-material/SettingsOutlined';
import LoginIcon from '@mui/icons-material/Login';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ScheduleIcon from '@mui/icons-material/Schedule';
import RestoreIcon from '@mui/icons-material/Restore';

import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { useLayoutStore } from '@/store/layoutStore';
import SessionSidebar from '@/components/SessionSidebar/SessionSidebar';
import SessionChat from '@/components/SessionChat';
import CodePanel from '@/components/CodePanel/CodePanel';
import WelcomeScreen from '@/components/WelcomeScreen/WelcomeScreen';
import SchedulerDialog from '@/components/SchedulerDialog';
import SavedSessionsDialog from '@/components/SavedSessionsDialog';
import JobsPanel from '@/components/JobsPanel';
import ApprovalInbox from '@/components/ApprovalInbox';
import { apiFetch } from '@/utils/api';
import { getPreferredExecutionMode, setPreferredExecutionMode, type ExecutionMode } from '@/utils/executionMode';

const DRAWER_WIDTH = 260;
const RUNNING_JOB_STATES = new Set(['RUNNING', 'PENDING', 'QUEUED', 'STARTING']);
const CODEX_DEVICE_AUTH_URL = 'https://auth.openai.com/codex/device';

function isLikelyTrainingCommand(text?: string): boolean {
  if (!text) return false;
  return /(train|trainer|fine[- ]?tune|torchrun|accelerate\s+launch|deepspeed|\bepoch\b|\bsft\b|\bdpo\b|\bgrpo\b)/i.test(text);
}

function findLatestRunningJobToolCallId(jobStatuses: Record<string, string>): string | null {
  const entries = Object.entries(jobStatuses);
  for (let i = entries.length - 1; i >= 0; i--) {
    const [toolCallId, status] = entries[i];
    if (RUNNING_JOB_STATES.has((status || '').toUpperCase())) {
      return toolCallId;
    }
  }
  return null;
}

export default function AppLayout() {
  const { sessions, activeSessionId, markExpired } = useSessionStore();
  const {
    isConnected,
    llmHealthError,
    setLlmHealthError,
    user,
    activityStatus,
    jobStatuses,
    jobUrls,
    trackioUrls,
  } = useAgentStore();
  const {
    isLeftSidebarOpen,
    isRightPanelOpen,
    rightPanelWidth,
    themeMode,
    setRightPanelWidth,
    setLeftSidebarOpen,
    toggleLeftSidebar,
    toggleTheme,
    setRightPanelOpen,
  } = useLayoutStore();

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const [showExpiredToast, setShowExpiredToast] = useState(false);
  const [preferredExecutionMode, setPreferredExecutionModeState] = useState<ExecutionMode>(() => getPreferredExecutionMode());
  const [activeExecutionMode, setActiveExecutionMode] = useState<ExecutionMode | null>(null);
  const disconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [schedulerOpen, setSchedulerOpen] = useState(false);
  const [savedSessionsOpen, setSavedSessionsOpen] = useState(false);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [codexConnected, setCodexConnected] = useState(false);
  const [codexMessage, setCodexMessage] = useState('');
  const [minimaxConfigured, setMinimaxConfigured] = useState(false);
  const [zaiConfigured, setZaiConfigured] = useState(false);
  const [workspaceEnvFile, setWorkspaceEnvFile] = useState('');
  const [minimaxToken, setMinimaxToken] = useState('');
  const [zaiToken, setZaiToken] = useState('');
  const [telegramConfigured, setTelegramConfigured] = useState(false);
  const [telegramRunning, setTelegramRunning] = useState(false);
  const [telegramEnabled, setTelegramEnabled] = useState(false);
  const [telegramMaskedToken, setTelegramMaskedToken] = useState('');
  const [telegramToken, setTelegramToken] = useState('');
  const [telegramAllowedChatIds, setTelegramAllowedChatIds] = useState('');
  const [telegramExecutionMode, setTelegramExecutionMode] = useState<ExecutionMode>('local');
  const [telegramTimeoutSeconds, setTelegramTimeoutSeconds] = useState('3600');
  const [telegramConfigPath, setTelegramConfigPath] = useState('');
  const [telegramMessage, setTelegramMessage] = useState('');
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [providerTestMessage, setProviderTestMessage] = useState('');
  const codexPopupRef = useRef<Window | null>(null);

  const isResizing = useRef(false);

  const activeRunningToolCallId =
    activityStatus.type === 'tool' && activityStatus.toolName === 'hf_jobs'
      ? (activityStatus.toolCallId || findLatestRunningJobToolCallId(jobStatuses))
      : findLatestRunningJobToolCallId(jobStatuses);

  const isLocalTrainingRunning =
    activeExecutionMode === 'local' &&
    activityStatus.type === 'tool' &&
    activityStatus.toolName === 'bash' &&
    isLikelyTrainingCommand(activityStatus.description);

  const hasRunningTraining = !!activeRunningToolCallId || isLocalTrainingRunning;
  const runningJobUrl = activeRunningToolCallId ? jobUrls[activeRunningToolCallId] : undefined;
  const runningTrackioUrl = activeRunningToolCallId ? trackioUrls[activeRunningToolCallId] : undefined;

  const handleRunningBadgeClick = useCallback(() => {
    if (runningTrackioUrl) {
      window.open(runningTrackioUrl, '_blank', 'noopener,noreferrer');
      return;
    }
    if (runningJobUrl) {
      window.open(runningJobUrl, '_blank', 'noopener,noreferrer');
      return;
    }
    if (isLocalTrainingRunning) {
      setRightPanelOpen(true);
    }
  }, [runningTrackioUrl, runningJobUrl, isLocalTrainingRunning, setRightPanelOpen]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return;
    const newWidth = window.innerWidth - e.clientX;
    const maxWidth = window.innerWidth * 0.6;
    const minWidth = 300;
    if (newWidth > minWidth && newWidth < maxWidth) {
      setRightPanelWidth(newWidth);
    }
  }, [setRightPanelWidth]);

  const stopResizing = useCallback(() => {
    isResizing.current = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', stopResizing);
    document.body.style.cursor = 'default';
  }, [handleMouseMove]);

  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopResizing);
    document.body.style.cursor = 'col-resize';
  }, [handleMouseMove, stopResizing]);

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', stopResizing);
    };
  }, [handleMouseMove, stopResizing]);

  // -- LLM health check on mount -----------------------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch('/api/health/llm');
        const data = await res.json();
        if (!cancelled && data.status === 'error') {
          setLlmHealthError({
            error: data.error || 'Unknown LLM error',
            errorType: data.error_type || 'unknown',
            model: data.model,
          });
        } else if (!cancelled) {
          setLlmHealthError(null);
        }
      } catch {
        // Backend unreachable -- not an LLM issue, ignore
      }
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!activeSessionId) {
      setActiveExecutionMode(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch(`/api/session/${activeSessionId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        const mode = data?.execution_mode === 'local' ? 'local' : 'sandbox';
        setActiveExecutionMode(mode);
      } catch {
        // ignore transient fetch errors
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  const toggleLocalOption = useCallback(() => {
    const next: ExecutionMode = preferredExecutionMode === 'local' ? 'sandbox' : 'local';
    setPreferredExecutionModeState(next);
    setPreferredExecutionMode(next);
  }, [preferredExecutionMode]);

  const refreshSettings = useCallback(async () => {
    try {
      const [codexRes, providerRes, telegramRes] = await Promise.all([
        apiFetch('/auth/codex/status'),
        apiFetch('/auth/providers/status'),
        apiFetch('/auth/telegram/status'),
      ]);

      if (codexRes.ok) {
        const codexData = await codexRes.json();
        setCodexConnected(!!codexData.logged_in);
        if (codexData.logged_in && codexData.username) {
          setCodexMessage(`Codex connected as ${codexData.username}`);
        }
      }

      if (providerRes.ok) {
        const providerData = await providerRes.json();
        setMinimaxConfigured(!!providerData?.minimax?.configured);
        setZaiConfigured(!!providerData?.zai?.configured);
        setWorkspaceEnvFile(typeof providerData?.workspace_env_file === 'string' ? providerData.workspace_env_file : '');
      }

      if (telegramRes.ok) {
        const telegramData = await telegramRes.json();
        setTelegramConfigured(!!telegramData?.configured);
        setTelegramRunning(!!telegramData?.running);
        setTelegramEnabled(!!telegramData?.enabled);
        setTelegramMaskedToken(typeof telegramData?.masked_token === 'string' ? telegramData.masked_token : '');
        setTelegramAllowedChatIds(Array.isArray(telegramData?.allowed_chat_ids) ? telegramData.allowed_chat_ids.join(',') : '');
        setTelegramExecutionMode(telegramData?.execution_mode === 'sandbox' ? 'sandbox' : 'local');
        setTelegramTimeoutSeconds(String(telegramData?.turn_timeout_seconds || 3600));
        setTelegramConfigPath(typeof telegramData?.config_path === 'string' ? telegramData.config_path : '');
      }
    } catch {
      // ignore transient settings fetch errors
    }
  }, []);

  const openSettings = useCallback(() => {
    setSettingsError(null);
    setSettingsOpen(true);
    refreshSettings();
  }, [refreshSettings]);

  const handleSettingsCodexLogin = useCallback(async () => {
    setSettingsBusy(true);
    setSettingsError(null);

    if (!codexConnected) {
      codexPopupRef.current = window.open(
        CODEX_DEVICE_AUTH_URL,
        'codex-device-auth',
        'noopener,noreferrer,width=520,height=760'
      );
      if (!codexPopupRef.current) {
        setSettingsError('Popup blocked. Use "Open device page" or allow popups for this site.');
      }
    }

    try {
      const res = await apiFetch('/auth/codex/login', {
        method: 'POST',
        body: JSON.stringify({ force: false }),
      });
      const data = await res.json();
      setCodexConnected(!!data.logged_in);
      const message = typeof data.message === 'string' ? data.message : 'Codex login flow initiated.';
      setCodexMessage(message);

      const urlMatch = message.match(/https?:\/\/\S+/);
      if (urlMatch?.[0] && codexPopupRef.current && !codexPopupRef.current.closed) {
        codexPopupRef.current.location.href = urlMatch[0];
        codexPopupRef.current.focus();
      }
    } catch {
      setSettingsError('Failed to start Codex OAuth flow.');
    } finally {
      setSettingsBusy(false);
    }
  }, [codexConnected]);

  const handleSaveProviderTokens = useCallback(async () => {
    setSettingsBusy(true);
    setSettingsError(null);
    setProviderTestMessage('');
    try {
      const res = await apiFetch('/auth/providers/tokens', {
        method: 'POST',
        body: JSON.stringify({
          minimax_api_key: minimaxToken.trim(),
          zai_api_key: zaiToken.trim(),
        }),
      });
      if (!res.ok) {
        setSettingsError('Failed to save MiniMax/ZAI tokens.');
        return;
      }
      setMinimaxToken('');
      setZaiToken('');
      await refreshSettings();
    } catch {
      setSettingsError('Failed to save MiniMax/ZAI tokens.');
    } finally {
      setSettingsBusy(false);
    }
  }, [minimaxToken, zaiToken, refreshSettings]);

  const handleClearProviderTokens = useCallback(async () => {
    setSettingsBusy(true);
    setSettingsError(null);
    setProviderTestMessage('');
    try {
      const res = await apiFetch('/auth/providers/tokens', {
        method: 'POST',
        body: JSON.stringify({ clear_minimax: true, clear_zai: true }),
      });
      if (!res.ok) {
        setSettingsError('Failed to clear MiniMax/ZAI tokens.');
        return;
      }
      await refreshSettings();
    } catch {
      setSettingsError('Failed to clear MiniMax/ZAI tokens.');
    } finally {
      setSettingsBusy(false);
    }
  }, [refreshSettings]);

  const handleSaveTelegramConfig = useCallback(async (clearToken = false, forceEnabled?: boolean) => {
    setSettingsBusy(true);
    setSettingsError(null);
    setTelegramMessage('');
    try {
      const res = await apiFetch('/auth/telegram/config', {
        method: 'POST',
        body: JSON.stringify({
          token: telegramToken.trim(),
          clear_token: clearToken,
          enabled: forceEnabled ?? telegramEnabled,
          allowed_chat_ids: telegramAllowedChatIds,
          execution_mode: telegramExecutionMode,
          turn_timeout_seconds: Number(telegramTimeoutSeconds) || 3600,
        }),
      });
      if (!res.ok) {
        setSettingsError('Failed to save Telegram bot config.');
        return;
      }
      const data = await res.json();
      setTelegramToken('');
      setTelegramConfigured(!!data?.configured);
      setTelegramRunning(!!data?.running);
      setTelegramEnabled(!!data?.enabled);
      setTelegramMaskedToken(typeof data?.masked_token === 'string' ? data.masked_token : '');
      setTelegramAllowedChatIds(Array.isArray(data?.allowed_chat_ids) ? data.allowed_chat_ids.join(',') : '');
      setTelegramExecutionMode(data?.execution_mode === 'sandbox' ? 'sandbox' : 'local');
      setTelegramTimeoutSeconds(String(data?.turn_timeout_seconds || 3600));
      setTelegramConfigPath(typeof data?.config_path === 'string' ? data.config_path : '');
      setTelegramMessage(data?.running ? 'Telegram bot is running.' : 'Telegram bot config saved.');
    } catch {
      setSettingsError('Failed to save Telegram bot config.');
    } finally {
      setSettingsBusy(false);
    }
  }, [telegramAllowedChatIds, telegramEnabled, telegramExecutionMode, telegramTimeoutSeconds, telegramToken]);

  const handleStopTelegram = useCallback(async () => {
    setSettingsBusy(true);
    setSettingsError(null);
    setTelegramMessage('');
    try {
      const res = await apiFetch('/auth/telegram/stop', { method: 'POST' });
      if (!res.ok) {
        setSettingsError('Failed to stop Telegram bot.');
        return;
      }
      const data = await res.json();
      setTelegramRunning(!!data?.running);
      setTelegramEnabled(!!data?.enabled);
      setTelegramMessage('Telegram bot stopped.');
    } catch {
      setSettingsError('Failed to stop Telegram bot.');
    } finally {
      setSettingsBusy(false);
    }
  }, []);

  const handleTestProvider = useCallback(async (provider: 'minimax' | 'zai') => {
    setSettingsBusy(true);
    setSettingsError(null);
    setProviderTestMessage('');
    try {
      const res = await apiFetch('/auth/providers/test', {
        method: 'POST',
        body: JSON.stringify({ provider }),
      });
      if (!res.ok) {
        setSettingsError(`Failed to test ${provider}.`);
        return;
      }
      const data = await res.json();
      if (data.ok) {
        setProviderTestMessage(`${provider.toUpperCase()} OK${data.preview ? ` — ${data.preview}` : ''}`);
      } else {
        setSettingsError(`${provider.toUpperCase()} test failed: ${data.error || 'unknown error'}`);
      }
    } catch {
      setSettingsError(`Failed to test ${provider}.`);
    } finally {
      setSettingsBusy(false);
    }
  }, []);

  const hasAnySessions = sessions.length > 0;

  // Debounced "session expired" toast
  useEffect(() => {
    if (!isConnected && activeSessionId) {
      disconnectTimer.current = setTimeout(() => setShowExpiredToast(true), 2000);
    } else {
      if (disconnectTimer.current) clearTimeout(disconnectTimer.current);
      disconnectTimer.current = null;
      setShowExpiredToast(false);
    }
    return () => {
      if (disconnectTimer.current) clearTimeout(disconnectTimer.current);
    };
  }, [isConnected, activeSessionId]);

  const handleSessionDead = useCallback(
    (deadSessionId: string) => {
      // Backend lost this session — mark it expired so the chat shows a
      // recovery banner instead of either silently failing or eagerly
      // creating a new backend session (which would pay a summary-call
      // cost for sessions the user may never revisit).
      markExpired(deadSessionId);
    },
    [markExpired],
  );

  // Close sidebar on mobile after selecting a session
  const handleSidebarClose = useCallback(() => {
    if (isMobile) setLeftSidebarOpen(false);
  }, [isMobile, setLeftSidebarOpen]);

  // -- LLM error toast helper --------------------------------------------
  const llmErrorTitle = llmHealthError
    ? llmHealthError.errorType === 'credits'
      ? 'API Credits Exhausted'
      : llmHealthError.errorType === 'auth'
      ? 'Invalid API Key'
      : llmHealthError.errorType === 'rate_limit'
      ? 'Rate Limited'
      : llmHealthError.errorType === 'network'
      ? 'LLM Provider Unreachable'
      : 'LLM Error'
    : '';

  // -- Welcome screen: no sessions at all ---------------------------------
  if (!hasAnySessions) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'auto',
          overflowX: 'hidden',
          WebkitOverflowScrolling: 'touch',
        }}
      >
        <WelcomeScreen />
      </Box>
    );
  }

  // -- Sidebar drawer -----------------------------------------------------
  const sidebarDrawer = (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      anchor="left"
      open={isLeftSidebarOpen}
      onClose={() => setLeftSidebarOpen(false)}
      ModalProps={{ keepMounted: true }}
      sx={{
        '& .MuiDrawer-paper': {
          boxSizing: 'border-box',
          width: DRAWER_WIDTH,
          borderRight: '1px solid',
          borderColor: 'divider',
          top: 0,
          height: '100%',
          bgcolor: 'var(--panel)',
        },
      }}
    >
      <SessionSidebar onClose={handleSidebarClose} />
    </Drawer>
  );

  // -- Main chat interface ------------------------------------------------
  return (
    <Box sx={{ display: 'flex', width: '100%', height: '100%' }}>
      {/* -- Left Sidebar ------------------------------------------------- */}
      {isMobile ? (
        sidebarDrawer
      ) : (
        <Box
          component="nav"
          sx={{
            width: isLeftSidebarOpen ? DRAWER_WIDTH : 0,
            flexShrink: 0,
            transition: isResizing.current ? 'none' : 'width 0.2s',
            overflow: 'hidden',
          }}
        >
          {sidebarDrawer}
        </Box>
      )}

      {/* -- Main Content (header + chat + code panel) -------------------- */}
      <Box
        sx={{
          flexGrow: 1,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          transition: isResizing.current ? 'none' : 'width 0.2s',
          overflow: 'hidden',
          minWidth: 0,
        }}
      >
        {/* -- Top Header Bar --------------------------------------------- */}
        <Box sx={{
          height: { xs: 52, md: 60 },
          px: { xs: 1, md: 2 },
          display: 'flex',
          alignItems: 'center',
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.default',
          zIndex: 1200,
          flexShrink: 0,
        }}>
          <IconButton onClick={toggleLeftSidebar} size="small">
            {isLeftSidebarOpen && !isMobile ? <ChevronLeftIcon /> : <MenuIcon />}
          </IconButton>

          <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 0.75 }}>
            <Box
              component="img"
              src="/smolagents.webp"
              alt="smolagents"
              sx={{ width: { xs: 20, md: 22 }, height: { xs: 20, md: 22 } }}
            />
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 700,
                color: 'var(--text)',
                letterSpacing: '-0.01em',
                fontSize: { xs: '0.88rem', md: '0.95rem' },
              }}
            >
              ML Intern
            </Typography>
            {activeExecutionMode && (
              <Chip
                size="small"
                label={activeExecutionMode === 'local' ? 'Local PC' : 'HF Sandbox'}
                sx={{
                  height: 20,
                  fontSize: '0.65rem',
                  fontWeight: 600,
                  color: 'var(--text)',
                  bgcolor: activeExecutionMode === 'local' ? 'rgba(255, 157, 0, 0.18)' : 'rgba(255,255,255,0.08)',
                  border: '1px solid',
                  borderColor: activeExecutionMode === 'local' ? 'rgba(255, 157, 0, 0.55)' : 'var(--border)',
                }}
              />
            )}
            {hasRunningTraining && (
              <Chip
                size="small"
                clickable
                onClick={handleRunningBadgeClick}
                label={isLocalTrainingRunning ? 'LOCAL TRAINING RUNNING' : 'TRAINING RUNNING'}
                sx={{
                  height: 20,
                  fontSize: '0.64rem',
                  fontWeight: 700,
                  color: '#000',
                  bgcolor: isLocalTrainingRunning ? '#FF9D00' : 'var(--accent-green)',
                  animation: 'pulseHeaderRun 1.8s ease-in-out infinite',
                  cursor: 'pointer',
                  '@keyframes pulseHeaderRun': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                  },
                }}
              />
            )}
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {hasRunningTraining && runningJobUrl && (
              <Link
                href={runningJobUrl}
                target="_blank"
                rel="noopener noreferrer"
                underline="hover"
                sx={{
                  display: { xs: 'none', sm: 'inline-flex' },
                  fontFamily: 'monospace',
                  fontSize: '0.66rem',
                  color: 'var(--accent-yellow)',
                  mr: 0.25,
                }}
              >
                HF Job
              </Link>
            )}
            {hasRunningTraining && runningTrackioUrl && (
              <Link
                href={runningTrackioUrl}
                target="_blank"
                rel="noopener noreferrer"
                underline="hover"
                sx={{
                  display: { xs: 'none', sm: 'inline-flex' },
                  fontFamily: 'monospace',
                  fontSize: '0.66rem',
                  color: 'var(--accent-green)',
                  mr: 0.25,
                }}
              >
                Trackio
              </Link>
            )}
            {isLocalTrainingRunning && (
              <Link
                component="button"
                onClick={() => setRightPanelOpen(true)}
                underline="hover"
                sx={{
                  display: { xs: 'none', sm: 'inline-flex' },
                  fontFamily: 'monospace',
                  fontSize: '0.66rem',
                  color: '#FF9D00',
                  mr: 0.25,
                }}
              >
                Local Logs
              </Link>
            )}
            <Tooltip title="Toggle Local PC option for new sessions">
              <Chip
                size="small"
                clickable
                onClick={toggleLocalOption}
                label={preferredExecutionMode === 'local' ? 'Local: ON' : 'Local: OFF'}
                sx={{
                  height: 22,
                  fontSize: '0.66rem',
                  fontWeight: 600,
                  color: preferredExecutionMode === 'local' ? '#000' : 'var(--muted-text)',
                  bgcolor: preferredExecutionMode === 'local' ? '#FF9D00' : 'rgba(255,255,255,0.06)',
                  border: '1px solid',
                  borderColor: preferredExecutionMode === 'local' ? '#FF9D00' : 'var(--border)',
                  display: { xs: 'none', sm: 'inline-flex' },
                }}
              />
            </Tooltip>
            <Tooltip title="Resume saved session">
              <IconButton
                onClick={() => setSavedSessionsOpen(true)}
                size="small"
                sx={{
                  color: 'text.secondary',
                  '&:hover': { color: '#FF9D00' },
                }}
              >
                <RestoreIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Configure training scheduler / watchdog">
              <IconButton
                onClick={() => setSchedulerOpen(true)}
                size="small"
                sx={{
                  color: activeExecutionMode === 'local' || preferredExecutionMode === 'local' ? '#FF9D00' : 'text.secondary',
                  '&:hover': { color: '#FF9D00' },
                }}
              >
                <ScheduleIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Jobs & Approvals">
              <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', mx: 0.5 }}>
                <JobsPanel collapsed />
                <ApprovalInbox collapsed />
              </Box>
            </Tooltip>
            <Tooltip title="Settings">
              <IconButton
                onClick={openSettings}
                size="small"
                sx={{
                  color: 'text.secondary',
                  '&:hover': { color: 'primary.main' },
                }}
              >
                <SettingsOutlinedIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <IconButton
              onClick={toggleTheme}
              size="small"
              sx={{
                color: 'text.secondary',
                '&:hover': { color: 'primary.main' },
              }}
            >
              {themeMode === 'dark' ? <LightModeOutlinedIcon fontSize="small" /> : <DarkModeOutlinedIcon fontSize="small" />}
            </IconButton>

            {user?.picture ? (
              <Avatar
                src={user.picture}
                alt={user.username || 'User'}
                sx={{ width: 28, height: 28, ml: 0.5 }}
              />
            ) : user?.username ? (
              <Avatar
                sx={{
                  width: 28,
                  height: 28,
                  ml: 0.5,
                  bgcolor: 'primary.main',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                }}
              >
                {user.username[0].toUpperCase()}
              </Avatar>
            ) : null}
          </Box>
        </Box>

        {/* -- Chat + Code Panel ------------------------------------------ */}
        <Box
          sx={{
            flexGrow: 1,
            display: 'flex',
            overflow: 'hidden',
          }}
        >
          {/* Chat area */}
          <Box
            component="main"
            className="chat-pane"
            sx={{
              flexGrow: 1,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              background: 'var(--body-gradient)',
              p: { xs: 1.5, sm: 2, md: 3 },
              minWidth: 0,
            }}
          >
            {activeSessionId ? (
              // Render ALL sessions — each owns its own useAgentChat.
              // Only the active one renders visible UI (others return null).
              sessions.map((s) => (
                <SessionChat
                  key={s.id}
                  sessionId={s.id}
                  isActive={s.id === activeSessionId}
                  onSessionDead={handleSessionDead}
                />
              ))
            ) : (
              <Box
                sx={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                  gap: 2,
                  px: 2,
                }}
              >
                <Typography variant="h5" color="text.secondary" sx={{ fontFamily: 'monospace', fontSize: { xs: '1rem', md: '1.5rem' } }}>
                  NO SESSION SELECTED
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace', fontSize: { xs: '0.75rem', md: '0.875rem' } }}>
                  Initialize a session via the sidebar
                </Typography>
              </Box>
            )}
          </Box>

          {/* Code panel -- inline on desktop, overlay drawer on mobile */}
          {isRightPanelOpen && !isMobile && (
            <>
              <Box
                onMouseDown={startResizing}
                sx={{
                  width: '4px',
                  cursor: 'col-resize',
                  bgcolor: 'divider',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background-color 0.2s',
                  flexShrink: 0,
                  '&:hover': { bgcolor: 'primary.main' },
                }}
              >
                <DragIndicatorIcon
                  sx={{ fontSize: '0.8rem', color: 'text.secondary', pointerEvents: 'none' }}
                />
              </Box>
              <Box
                sx={{
                  width: rightPanelWidth,
                  flexShrink: 0,
                  height: '100%',
                  overflow: 'hidden',
                  borderLeft: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'var(--panel)',
                }}
              >
                <CodePanel />
              </Box>
            </>
          )}
        </Box>
      </Box>

      {/* Code panel -- drawer overlay on mobile */}
      {isMobile && (
        <Drawer
          anchor="bottom"
          open={isRightPanelOpen}
          onClose={() => useLayoutStore.getState().setRightPanelOpen(false)}
          sx={{
            '& .MuiDrawer-paper': {
              height: '75vh',
              borderTopLeftRadius: 16,
              borderTopRightRadius: 16,
              bgcolor: 'var(--panel)',
            },
          }}
        >
          <CodePanel />
        </Drawer>
      )}

      <SchedulerDialog
        open={schedulerOpen}
        onClose={() => setSchedulerOpen(false)}
        activeExecutionMode={activeExecutionMode}
        activeSessionId={activeSessionId}
      />

      <SavedSessionsDialog
        open={savedSessionsOpen}
        onClose={() => setSavedSessionsOpen(false)}
        currentSessionId={activeSessionId}
      />

      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Chat Settings</DialogTitle>
        <DialogContent dividers>
          <Typography variant="caption" sx={{ color: 'var(--muted-text)' }}>
            Local mode default for new sessions
          </Typography>
          <Box sx={{ mt: 0.75, mb: 2, display: 'flex', gap: 1 }}>
            <Chip
              size="small"
              clickable
              onClick={toggleLocalOption}
              label={preferredExecutionMode === 'local' ? 'Local: ON' : 'Local: OFF'}
              sx={{
                fontWeight: 600,
                color: preferredExecutionMode === 'local' ? '#000' : 'var(--muted-text)',
                bgcolor: preferredExecutionMode === 'local' ? '#FF9D00' : 'rgba(255,255,255,0.06)',
                border: '1px solid',
                borderColor: preferredExecutionMode === 'local' ? '#FF9D00' : 'var(--border)',
              }}
            />
          </Box>

          <Typography variant="caption" sx={{ color: 'var(--muted-text)' }}>
            Codex OAuth (for GPT-5.3 / GPT-5.4 / GPT-5.5)
          </Typography>
          <Box sx={{ mt: 0.75, mb: 2, display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="small"
              startIcon={<LoginIcon sx={{ fontSize: 14 }} />}
              onClick={handleSettingsCodexLogin}
              disabled={settingsBusy}
              sx={{ textTransform: 'none', bgcolor: codexConnected ? 'var(--accent-green)' : '#FF9D00', color: '#000' }}
            >
              {codexConnected ? 'Codex Connected' : 'Connect Codex'}
            </Button>
            <Button
              variant="outlined"
              size="small"
              component="a"
              href={CODEX_DEVICE_AUTH_URL}
              target="_blank"
              rel="noopener noreferrer"
              startIcon={<OpenInNewIcon sx={{ fontSize: 13 }} />}
              sx={{ textTransform: 'none' }}
            >
              Open device page
            </Button>
          </Box>
          {!!codexMessage && (
            <Typography variant="caption" sx={{ display: 'block', mb: 2, color: 'var(--muted-text)' }}>
              {codexMessage}
            </Typography>
          )}

          <Typography variant="caption" sx={{ color: 'var(--muted-text)' }}>
            Provider tokens (MiniMax / ZAI)
          </Typography>
          {!!workspaceEnvFile && (
            <Typography
              variant="caption"
              sx={{ display: 'block', mt: 0.5, color: 'var(--muted-text)', fontFamily: 'monospace' }}
              title={workspaceEnvFile}
            >
              Stored in: {workspaceEnvFile}
            </Typography>
          )}
          <Box sx={{ mt: 0.75, display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 1 }}>
            <TextField
              size="small"
              type="password"
              placeholder={minimaxConfigured ? 'MiniMax token configured' : 'MiniMax API key'}
              value={minimaxToken}
              onChange={(e) => setMinimaxToken(e.target.value)}
            />
            <TextField
              size="small"
              type="password"
              placeholder={zaiConfigured ? 'ZAI token configured' : 'ZAI API key'}
              value={zaiToken}
              onChange={(e) => setZaiToken(e.target.value)}
            />
          </Box>
          <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="small"
              onClick={handleSaveProviderTokens}
              disabled={settingsBusy}
              sx={{ textTransform: 'none', bgcolor: '#FF9D00', color: '#000' }}
            >
              Save tokens
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={handleClearProviderTokens}
              disabled={settingsBusy || (!minimaxConfigured && !zaiConfigured)}
              sx={{ textTransform: 'none' }}
            >
              Clear tokens
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={() => handleTestProvider('minimax')}
              disabled={settingsBusy || !minimaxConfigured}
              sx={{ textTransform: 'none' }}
            >
              Test MiniMax
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={() => handleTestProvider('zai')}
              disabled={settingsBusy || !zaiConfigured}
              sx={{ textTransform: 'none' }}
            >
              Test ZAI
            </Button>
          </Box>

          {providerTestMessage && (
            <Alert severity="success" sx={{ mt: 2 }}>
              {providerTestMessage}
            </Alert>
          )}

          <Typography variant="caption" sx={{ display: 'block', mt: 2.5, color: 'var(--muted-text)' }}>
            Telegram bot
          </Typography>
          <Box sx={{ mt: 0.75, display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Chip
              size="small"
              label={telegramRunning ? 'Running' : telegramConfigured ? 'Configured' : 'Not configured'}
              sx={{
                fontWeight: 600,
                color: telegramRunning ? '#000' : 'var(--muted-text)',
                bgcolor: telegramRunning ? 'var(--accent-green)' : 'rgba(255,255,255,0.06)',
                border: '1px solid var(--border)',
              }}
            />
            {telegramMaskedToken && (
              <Typography variant="caption" sx={{ color: 'var(--muted-text)', fontFamily: 'monospace' }}>
                {telegramMaskedToken}
              </Typography>
            )}
          </Box>
          {!!telegramConfigPath && (
            <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'var(--muted-text)', fontFamily: 'monospace' }} title={telegramConfigPath}>
              Config: {telegramConfigPath}
            </Typography>
          )}
          <Box sx={{ mt: 0.75, display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 1 }}>
            <TextField
              size="small"
              type="password"
              placeholder={telegramConfigured ? 'Telegram bot token configured' : 'Telegram bot token'}
              value={telegramToken}
              onChange={(e) => setTelegramToken(e.target.value)}
            />
            <TextField
              size="small"
              placeholder="Allowed chat IDs, comma-separated (optional)"
              value={telegramAllowedChatIds}
              onChange={(e) => setTelegramAllowedChatIds(e.target.value)}
            />
            <TextField
              select
              size="small"
              label="Execution mode"
              value={telegramExecutionMode}
              onChange={(e) => setTelegramExecutionMode(e.target.value === 'sandbox' ? 'sandbox' : 'local')}
            >
              <MenuItem value="local">Local</MenuItem>
              <MenuItem value="sandbox">Sandbox</MenuItem>
            </TextField>
            <TextField
              size="small"
              label="Turn timeout seconds"
              type="number"
              value={telegramTimeoutSeconds}
              onChange={(e) => setTelegramTimeoutSeconds(e.target.value)}
              inputProps={{ min: 30, step: 30 }}
            />
          </Box>
          <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="small"
              onClick={() => handleSaveTelegramConfig(false, true)}
              disabled={settingsBusy || (!telegramToken.trim() && !telegramConfigured)}
              sx={{ textTransform: 'none', bgcolor: '#FF9D00', color: '#000' }}
            >
              Save & start bot
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={() => handleSaveTelegramConfig(false, false)}
              disabled={settingsBusy}
              sx={{ textTransform: 'none' }}
            >
              Save disabled
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={handleStopTelegram}
              disabled={settingsBusy || !telegramRunning}
              sx={{ textTransform: 'none' }}
            >
              Stop bot
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={() => handleSaveTelegramConfig(true, false)}
              disabled={settingsBusy || !telegramConfigured}
              sx={{ textTransform: 'none' }}
            >
              Clear token
            </Button>
          </Box>
          <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'var(--muted-text)', fontFamily: 'monospace' }}>
            Commands: /start, /help, /commands, /new, /status, /sessions, /crons, /cancelcron &lt;id&gt;, /interrupt, /cron [minutes] &lt;prompt&gt;
          </Typography>
          {telegramMessage && (
            <Alert severity="success" sx={{ mt: 2 }}>
              {telegramMessage}
            </Alert>
          )}

          {settingsError && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              {settingsError}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)} sx={{ textTransform: 'none' }}>Close</Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={showExpiredToast}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        onClose={() => setShowExpiredToast(false)}
      >
        <Alert
          severity="warning"
          variant="filled"
          onClose={() => setShowExpiredToast(false)}
          sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}
        >
          Task expired — create a new task to continue.
        </Alert>
      </Snackbar>
      <Snackbar
        open={!!llmHealthError}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        onClose={() => setLlmHealthError(null)}
      >
        <Alert
          severity="error"
          variant="filled"
          onClose={() => setLlmHealthError(null)}
          sx={{ fontSize: '0.8rem', maxWidth: 480 }}
        >
          <AlertTitle sx={{ fontWeight: 700, fontSize: '0.85rem' }}>
            {llmErrorTitle}
          </AlertTitle>
          {llmHealthError && (
            <Typography variant="body2" sx={{ fontSize: '0.78rem', opacity: 0.9 }}>
              {llmHealthError.model} — {llmHealthError.error.slice(0, 150)}
            </Typography>
          )}
        </Alert>
      </Snackbar>
    </Box>
  );
}
