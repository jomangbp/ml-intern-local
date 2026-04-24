import { useState, useCallback, useEffect, useRef, type ReactNode } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Alert,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import GroupAddIcon from '@mui/icons-material/GroupAdd';
import LoginIcon from '@mui/icons-material/Login';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';
import { getPreferredExecutionMode, setPreferredExecutionMode, type ExecutionMode } from '@/utils/executionMode';
import { isInIframe, triggerLogin } from '@/hooks/useAuth';
import { useOrgMembership } from '@/hooks/useOrgMembership';

const HF_ORANGE = '#FF9D00';
const ORG_JOIN_URL =
  'https://huggingface.co/organizations/ml-agent-explorers/share/GzPMJUivoFPlfkvFtIqEouZKSytatKQSZT';
const CODEX_DEVICE_AUTH_URL = 'https://auth.openai.com/codex/device';

// ---------------------------------------------------------------------------
// ChecklistStep sub-component
// ---------------------------------------------------------------------------

type StepStatus = 'completed' | 'active' | 'locked';

interface ChecklistStepProps {
  stepNumber: number;
  title: string;
  description: string;
  status: StepStatus;
  lockedReason?: string;
  actionLabel?: string;
  onAction?: () => void;
  actionIcon?: ReactNode;
  actionHref?: string;
  loading?: boolean;
  isLast?: boolean;
}

function StepIndicator({ status, stepNumber }: { status: StepStatus; stepNumber: number }) {
  if (status === 'completed') {
    return <CheckCircleIcon sx={{ fontSize: 28, color: 'var(--accent-green)' }} />;
  }
  return (
    <Box
      sx={{
        width: 28,
        height: 28,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.8rem',
        fontWeight: 700,
        ...(status === 'active'
          ? { bgcolor: HF_ORANGE, color: '#000' }
          : { bgcolor: 'transparent', border: '2px solid var(--border)', color: 'var(--muted-text)' }),
      }}
    >
      {stepNumber}
    </Box>
  );
}

function ChecklistStep({
  stepNumber,
  title,
  description,
  status,
  lockedReason,
  actionLabel,
  onAction,
  actionIcon,
  actionHref,
  loading = false,
  isLast = false,
}: ChecklistStepProps) {
  const btnSx = {
    px: 3,
    py: 0.75,
    fontSize: '0.85rem',
    fontWeight: 700,
    textTransform: 'none' as const,
    borderRadius: '10px',
    whiteSpace: 'nowrap' as const,
    textDecoration: 'none',
    ...(status === 'active'
      ? {
          bgcolor: HF_ORANGE,
          color: '#000',
          boxShadow: '0 2px 12px rgba(255, 157, 0, 0.25)',
          '&:hover': { bgcolor: '#FFB340', boxShadow: '0 4px 20px rgba(255, 157, 0, 0.4)' },
        }
      : {
          bgcolor: 'rgba(255,255,255,0.04)',
          color: 'var(--muted-text)',
          '&.Mui-disabled': { bgcolor: 'rgba(255,255,255,0.04)', color: 'var(--muted-text)' },
        }),
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 3,
        py: 2.5,
        borderLeft: '3px solid',
        borderLeftColor:
          status === 'completed'
            ? 'var(--accent-green)'
            : status === 'active'
              ? HF_ORANGE
              : 'transparent',
        ...(!isLast && { borderBottom: '1px solid var(--border)' }),
        opacity: status === 'locked' ? 0.55 : 1,
        transition: 'opacity 0.2s, border-color 0.2s',
      }}
    >
      <StepIndicator status={status} stepNumber={stepNumber} />

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="subtitle2"
          sx={{
            fontWeight: 600,
            fontSize: '0.92rem',
            color: status === 'completed' ? 'var(--muted-text)' : 'var(--text)',
            ...(status === 'completed' && { textDecoration: 'line-through', textDecorationColor: 'var(--muted-text)' }),
          }}
        >
          {title}
        </Typography>
        <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.8rem', mt: 0.25, lineHeight: 1.5 }}>
          {status === 'locked' && lockedReason ? lockedReason : description}
        </Typography>
      </Box>

      {status === 'completed' ? (
        <Typography variant="caption" sx={{ color: 'var(--accent-green)', fontWeight: 600, fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
          Done
        </Typography>
      ) : actionLabel ? (
        actionHref ? (
          <Button
            variant="contained"
            size="small"
            component="a"
            href={actionHref}
            target="_blank"
            rel="noopener noreferrer"
            disabled={status === 'locked'}
            startIcon={actionIcon}
            sx={btnSx}
            onClick={onAction}
          >
            {actionLabel}
          </Button>
        ) : (
          <Button
            variant="contained"
            size="small"
            disabled={status === 'locked' || loading}
            startIcon={loading ? <CircularProgress size={16} color="inherit" /> : actionIcon}
            onClick={onAction}
            sx={btnSx}
          >
            {loading ? 'Loading...' : actionLabel}
          </Button>
        )
      ) : null}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// WelcomeScreen
// ---------------------------------------------------------------------------

export default function WelcomeScreen() {
  const { createSession } = useSessionStore();
  const { setPlan, clearPanel, user } = useAgentStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>(() => getPreferredExecutionMode());
  const [codexBusy, setCodexBusy] = useState(false);
  const [codexConnected, setCodexConnected] = useState(false);
  const [codexMessage, setCodexMessage] = useState<string>('');
  const codexPopupRef = useRef<Window | null>(null);

  const [providerBusy, setProviderBusy] = useState(false);
  const [providerMessage, setProviderMessage] = useState('');
  const [minimaxToken, setMinimaxToken] = useState('');
  const [zaiToken, setZaiToken] = useState('');
  const [minimaxConfigured, setMinimaxConfigured] = useState(false);
  const [zaiConfigured, setZaiConfigured] = useState(false);

  const inIframe = isInIframe();
  const isAuthenticated = !!user?.authenticated;
  const isDevUser = user?.username === 'dev';

  // Iframe: localStorage-based org tracking (no auth token available)
  const [iframeOrgJoined, setIframeOrgJoined] = useState(() => {
    try { return localStorage.getItem('hf-agent-org-joined') === '1'; } catch { return false; }
  });
  const joinLinkOpened = useRef(false);

  // Auto-advance when user returns from org join link (iframe only)
  useEffect(() => {
    if (!inIframe) return;
    const handleVisibility = () => {
      if (document.visibilityState !== 'visible' || !joinLinkOpened.current) return;
      joinLinkOpened.current = false;
      try { localStorage.setItem('hf-agent-org-joined', '1'); } catch { /* ignore */ }
      setIframeOrgJoined(true);
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, [inIframe]);

  const isOrgMember = inIframe ? iframeOrgJoined : !!user?.orgMember;

  // Poll for org membership once authenticated (skipped in dev mode and iframe)
  const popupRef = useOrgMembership(isAuthenticated && !isDevUser && !inIframe && !isOrgMember);

  // ---- Actions ----

  const handleJoinOrg = useCallback(() => {
    if (inIframe) {
      // Iframe: open link, track via visibilitychange + localStorage
      joinLinkOpened.current = true;
      window.open(ORG_JOIN_URL, '_blank', 'noopener,noreferrer');
      return;
    }
    // Direct: open as popup, auto-close via polling
    const popup = window.open(ORG_JOIN_URL, 'hf-org-join', 'noopener');
    if (popup) {
      popupRef.current = popup;
    } else {
      window.open(ORG_JOIN_URL, '_blank', 'noopener,noreferrer');
    }
  }, [popupRef, inIframe]);

  const refreshCodexStatus = useCallback(async () => {
    if (inIframe) return;
    try {
      const res = await apiFetch('/auth/codex/status');
      if (!res.ok) return;
      const data = await res.json();
      setCodexConnected(!!data.logged_in);
      if (data.logged_in && data.username) {
        setCodexMessage(`Codex connected as ${data.username}`);
      }
    } catch {
      // ignore
    }
  }, [inIframe]);

  const refreshProviderStatus = useCallback(async () => {
    if (inIframe) return;
    try {
      const res = await apiFetch('/auth/providers/status');
      if (!res.ok) return;
      const data = await res.json();
      setMinimaxConfigured(!!data?.minimax?.configured);
      setZaiConfigured(!!data?.zai?.configured);
    } catch {
      // ignore
    }
  }, [inIframe]);

  useEffect(() => {
    refreshCodexStatus();
    refreshProviderStatus();
  }, [refreshCodexStatus, refreshProviderStatus]);

  const handleCodexLogin = useCallback(async () => {
    setCodexBusy(true);
    setError(null);

    // Open popup synchronously from click event so browser doesn't block it.
    if (!codexConnected) {
      codexPopupRef.current = window.open(
        CODEX_DEVICE_AUTH_URL,
        'codex-device-auth',
        'noopener,noreferrer,width=520,height=760'
      );
      if (!codexPopupRef.current) {
        setError('Popup blocked. Please allow popups for this site and click Connect Codex again.');
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

      // If backend returned a specific URL in text, reuse popup and navigate there.
      const urlMatch = message.match(/https?:\/\/\S+/);
      if (urlMatch?.[0] && codexPopupRef.current && !codexPopupRef.current.closed) {
        codexPopupRef.current.location.href = urlMatch[0];
        codexPopupRef.current.focus();
      }
    } catch {
      setError('Failed to start Codex OAuth flow.');
    } finally {
      setCodexBusy(false);
    }
  }, [codexConnected, setError]);

  const handleSaveProviderTokens = useCallback(async () => {
    setProviderBusy(true);
    setError(null);
    setProviderMessage('');
    try {
      const res = await apiFetch('/auth/providers/tokens', {
        method: 'POST',
        body: JSON.stringify({
          minimax_api_key: minimaxToken.trim(),
          zai_api_key: zaiToken.trim(),
        }),
      });
      if (!res.ok) {
        setError('Failed to save MiniMax/ZAI tokens.');
        return;
      }
      const data = await res.json();
      setMinimaxConfigured(!!data?.minimax?.configured);
      setZaiConfigured(!!data?.zai?.configured);
      setProviderMessage('Provider tokens saved for this login session.');
      setMinimaxToken('');
      setZaiToken('');
    } catch {
      setError('Failed to save MiniMax/ZAI tokens.');
    } finally {
      setProviderBusy(false);
    }
  }, [minimaxToken, zaiToken]);

  const handleClearProviderTokens = useCallback(async () => {
    setProviderBusy(true);
    setError(null);
    setProviderMessage('');
    try {
      const res = await apiFetch('/auth/providers/tokens', {
        method: 'POST',
        body: JSON.stringify({ clear_minimax: true, clear_zai: true }),
      });
      if (!res.ok) {
        setError('Failed to clear MiniMax/ZAI tokens.');
        return;
      }
      setMinimaxConfigured(false);
      setZaiConfigured(false);
      setProviderMessage('Provider tokens cleared.');
      setMinimaxToken('');
      setZaiToken('');
    } catch {
      setError('Failed to clear MiniMax/ZAI tokens.');
    } finally {
      setProviderBusy(false);
    }
  }, []);

  const handleStartSession = useCallback(async () => {
    if (isCreating) return;
    setIsCreating(true);
    setError(null);

    try {
      const response = await apiFetch('/api/session', {
        method: 'POST',
        body: JSON.stringify({ execution_mode: executionMode }),
      });
      if (response.status === 503) {
        const data = await response.json();
        setError(data.detail || 'Server is at capacity. Please try again later.');
        return;
      }
      if (response.status === 401) {
        triggerLogin();
        return;
      }
      if (!response.ok) {
        setError('Failed to create session. Please try again.');
        return;
      }
      const data = await response.json();
      createSession(data.session_id);
      setPlan([]);
      clearPanel();
    } catch {
      // Redirect may throw — ignore
    } finally {
      setIsCreating(false);
    }
  }, [isCreating, executionMode, createSession, setPlan, clearPanel]);

  // ---- Step status helpers ----

  const signInStatus: StepStatus = isAuthenticated ? 'completed' : 'active';
  const joinOrgStatus: StepStatus = isOrgMember ? 'completed' : isAuthenticated ? 'active' : 'locked';
  // Do not block session start on org membership.
  const startStatus: StepStatus = isAuthenticated ? 'active' : 'locked';

  const canStartNow = isDevUser || isAuthenticated;
  const floatingStartLabel = isCreating ? 'Starting...' : canStartNow ? 'Start Session' : 'Sign in to Start';

  const handleFloatingStart = useCallback(() => {
    if (isCreating) return;
    if (canStartNow) {
      void handleStartSession();
      return;
    }
    triggerLogin();
  }, [isCreating, canStartNow, handleStartSession]);

  // Space URL for iframe "Open ML Intern" step
  const spaceHost =
    typeof window !== 'undefined'
      ? window.location.hostname.includes('.hf.space')
        ? window.location.origin
        : 'https://smolagents-ml-intern.hf.space'
      : '';

  return (
    <Box
      sx={{
        width: '100%',
        minHeight: '100%',
        height: 'auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        background: 'var(--body-gradient)',
        py: { xs: 3, md: 4 },
        pb: { xs: 6, md: 10 },
        overflowY: 'auto',
        overflowX: 'hidden',
        WebkitOverflowScrolling: 'touch',
      }}
    >
      {/* Logo */}
      <Box
        component="img"
        src="/smolagents.webp"
        alt="smolagents"
        sx={{ width: 80, height: 80, mb: 2.5, display: 'block' }}
      />

      {/* Title */}
      <Typography
        variant="h2"
        sx={{
          fontWeight: 800,
          color: 'var(--text)',
          mb: 1,
          letterSpacing: '-0.02em',
          fontSize: { xs: '1.8rem', md: '2.4rem' },
        }}
      >
        ML Intern
      </Typography>

      {/* Description */}
      <Typography
        variant="body1"
        sx={{
          color: 'var(--muted-text)',
          maxWidth: 480,
          mb: 4,
          lineHeight: 1.7,
          fontSize: '0.9rem',
          textAlign: 'center',
          px: 2,
          '& strong': { color: 'var(--text)', fontWeight: 600 },
        }}
      >
        Your personal <strong>ML agent</strong>. It reads <strong>papers</strong>, finds <strong>datasets</strong>, trains <strong>models</strong>, and iterates until the numbers go up. Instructions in. Trained model out.
      </Typography>

      <Box
        sx={{
          width: '100%',
          maxWidth: 520,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 1.5,
          px: 0.5,
        }}
      >
        <Typography variant="caption" sx={{ color: 'var(--muted-text)', fontSize: '0.72rem' }}>
          Execution mode
        </Typography>
        <ToggleButtonGroup
          value={executionMode}
          exclusive
          size="small"
          onChange={(_, next: ExecutionMode | null) => {
            if (!next) return;
            setExecutionMode(next);
            setPreferredExecutionMode(next);
          }}
          sx={{
            '& .MuiToggleButton-root': {
              color: 'var(--muted-text)',
              borderColor: 'var(--border)',
              fontSize: '0.72rem',
              textTransform: 'none',
              px: 1.2,
              py: 0.35,
            },
            '& .Mui-selected': {
              color: 'var(--text)',
              bgcolor: 'rgba(255,255,255,0.08) !important',
            },
          }}
        >
          <ToggleButton value="sandbox">HF Sandbox</ToggleButton>
          <ToggleButton value="local">Local PC</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Typography
        variant="caption"
        sx={{ mb: 2.5, color: 'var(--muted-text)', fontSize: '0.72rem', textAlign: 'center', px: 2 }}
      >
        Local PC mode runs tools on this machine (no HF sandbox deployment).
      </Typography>

      {!inIframe && (
        <Box
          sx={{
            width: '100%',
            maxWidth: 520,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 1.5,
            border: '1px solid var(--border)',
            bgcolor: 'var(--surface)',
            borderRadius: '10px',
            px: 1.5,
            py: 1,
            mb: 1.5,
            mx: 2,
          }}
        >
          <Typography variant="caption" sx={{ color: 'var(--muted-text)', fontSize: '0.75rem' }}>
            Optional: connect Codex OAuth for GPT-5.3 / GPT-5.4 models.
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexShrink: 0 }}>
            <Button
              variant="contained"
              size="small"
              startIcon={codexBusy ? <CircularProgress size={14} color="inherit" /> : <LoginIcon sx={{ fontSize: 14 }} />}
              onClick={handleCodexLogin}
              disabled={codexBusy}
              sx={{
                px: 1.5,
                py: 0.45,
                fontSize: '0.74rem',
                textTransform: 'none',
                borderRadius: '8px',
                bgcolor: codexConnected ? 'var(--accent-green)' : HF_ORANGE,
                color: '#000',
                '&:hover': {
                  bgcolor: codexConnected ? 'var(--accent-green)' : '#FFB340',
                },
              }}
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
              sx={{
                px: 1.1,
                py: 0.45,
                fontSize: '0.72rem',
                textTransform: 'none',
                borderRadius: '8px',
                borderColor: 'var(--border)',
                color: 'var(--muted-text)',
                '&:hover': {
                  borderColor: HF_ORANGE,
                  color: 'var(--text)',
                },
              }}
            >
              Fallback
            </Button>
          </Box>
        </Box>
      )}

      {codexMessage && !inIframe && (
        <Typography
          variant="caption"
          sx={{
            mb: 1.5,
            color: 'var(--muted-text)',
            maxWidth: 520,
            px: 2,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
          title={codexMessage}
        >
          {codexMessage}
        </Typography>
      )}

      {!inIframe && (
        <Box
          sx={{
            width: '100%',
            maxWidth: 520,
            border: '1px solid var(--border)',
            bgcolor: 'var(--surface)',
            borderRadius: '10px',
            px: 1.5,
            py: 1.2,
            mb: 1.8,
            mx: 2,
          }}
        >
          <Typography variant="caption" sx={{ color: 'var(--muted-text)', fontSize: '0.75rem', display: 'block', mb: 1 }}>
            Optional: add MiniMax / ZAI API tokens (for GPT alternatives in model picker).
          </Typography>

          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr auto auto' }, gap: 1, alignItems: 'center' }}>
            <TextField
              size="small"
              type="password"
              placeholder={minimaxConfigured ? 'MiniMax token configured' : 'MiniMax API key'}
              value={minimaxToken}
              onChange={(e) => setMinimaxToken(e.target.value)}
              sx={{
                '& .MuiInputBase-root': { bgcolor: 'rgba(255,255,255,0.03)', color: 'var(--text)' },
                '& .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border)' },
              }}
            />
            <TextField
              size="small"
              type="password"
              placeholder={zaiConfigured ? 'ZAI token configured' : 'ZAI API key'}
              value={zaiToken}
              onChange={(e) => setZaiToken(e.target.value)}
              sx={{
                '& .MuiInputBase-root': { bgcolor: 'rgba(255,255,255,0.03)', color: 'var(--text)' },
                '& .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border)' },
              }}
            />
            <Button
              variant="contained"
              size="small"
              disabled={providerBusy}
              onClick={handleSaveProviderTokens}
              sx={{
                px: 1.4,
                py: 0.55,
                fontSize: '0.72rem',
                textTransform: 'none',
                borderRadius: '8px',
                bgcolor: HF_ORANGE,
                color: '#000',
                '&:hover': { bgcolor: '#FFB340' },
              }}
            >
              {providerBusy ? 'Saving...' : 'Save'}
            </Button>
            <Button
              variant="outlined"
              size="small"
              disabled={providerBusy || (!minimaxConfigured && !zaiConfigured)}
              onClick={handleClearProviderTokens}
              sx={{
                px: 1.1,
                py: 0.55,
                fontSize: '0.72rem',
                textTransform: 'none',
                borderRadius: '8px',
                borderColor: 'var(--border)',
                color: 'var(--muted-text)',
                '&:hover': {
                  borderColor: HF_ORANGE,
                  color: 'var(--text)',
                },
              }}
            >
              Clear
            </Button>
          </Box>

          {(providerMessage || minimaxConfigured || zaiConfigured) && (
            <Typography variant="caption" sx={{ display: 'block', mt: 0.9, color: 'var(--muted-text)', fontSize: '0.72rem' }}>
              {providerMessage || `Configured: ${minimaxConfigured ? 'MiniMax ' : ''}${zaiConfigured ? 'ZAI' : ''}`}
            </Typography>
          )}
        </Box>
      )}

      {/* ── Checklist ──────────────────────────────────────────── */}
      <Box
        sx={{
          width: '100%',
          maxWidth: 520,
          bgcolor: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '12px',
          overflow: 'hidden',
          mx: 2,
        }}
      >
        {isDevUser ? (
          /* Dev mode: single step */
          <ChecklistStep
            stepNumber={1}
            title="Start Session"
            description="Launch an AI agent session for ML engineering."
            status="active"
            actionLabel="Start Session"
            actionIcon={<RocketLaunchIcon sx={{ fontSize: 16 }} />}
            onAction={handleStartSession}
            loading={isCreating}
            isLast
          />
        ) : inIframe ? (
          /* Iframe: 2 steps */
          <>
            <ChecklistStep
              stepNumber={1}
              title="Join ML Agent Explorers"
              description="Get free access to GPUs, inference APIs, and Hub resources."
              status={isOrgMember ? 'completed' : 'active'}
              actionLabel="Join Organization"
              actionIcon={<GroupAddIcon sx={{ fontSize: 16 }} />}
              onAction={handleJoinOrg}
            />
            <ChecklistStep
              stepNumber={2}
              title="Open ML Intern"
              description="Open the agent in a full browser tab to get started."
              status={'active'}
              actionLabel="Open ML Intern"
              actionIcon={<OpenInNewIcon sx={{ fontSize: 16 }} />}
              actionHref={spaceHost}
              isLast
            />
          </>
        ) : (
          /* Direct access: 3 steps */
          <>
            <ChecklistStep
              stepNumber={1}
              title="Sign in with Hugging Face"
              description="Authenticate to access GPU resources and model APIs."
              status={signInStatus}
              actionLabel="Sign in"
              actionIcon={<LoginIcon sx={{ fontSize: 16 }} />}
              onAction={() => triggerLogin()}
            />
            <ChecklistStep
              stepNumber={2}
              title="Join ML Agent Explorers (optional)"
              description="Optional: get free access to shared GPU resources and Hub perks."
              status={joinOrgStatus}
              lockedReason="Sign in first to continue."
              actionLabel="Join Organization"
              actionIcon={<GroupAddIcon sx={{ fontSize: 16 }} />}
              onAction={handleJoinOrg}
            />
            <ChecklistStep
              stepNumber={3}
              title="Start Session"
              description="Launch an AI agent session for ML engineering."
              status={startStatus}
              lockedReason="Sign in first to continue."
              actionLabel="Start Session"
              actionIcon={<RocketLaunchIcon sx={{ fontSize: 16 }} />}
              onAction={handleStartSession}
              loading={isCreating}
              isLast
            />
          </>
        )}
      </Box>

      {/* Polling hint when waiting for org join */}
      {isAuthenticated && !isOrgMember && !isDevUser && !inIframe && (
        <Typography
          variant="caption"
          sx={{ mt: 2, color: 'var(--muted-text)', fontSize: '0.75rem', textAlign: 'center' }}
        >
          This page updates automatically when you join the organization.
        </Typography>
      )}

      {/* Error */}
      {error && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setError(null)}
          sx={{
            mt: 3,
            maxWidth: 400,
            fontSize: '0.8rem',
            borderColor: HF_ORANGE,
            color: 'var(--text)',
          }}
        >
          {error}
        </Alert>
      )}

      {/* Footnote */}
      <Typography
        variant="caption"
        sx={{ mt: 4, color: 'var(--muted-text)', opacity: 0.5, fontSize: '0.7rem' }}
      >
        Conversations are stored locally in your browser.
      </Typography>

      {/* Desktop sticky CTA fallback: always clickable even if scroll/layout breaks */}
      <Box
        sx={{
          position: 'fixed',
          left: '50%',
          bottom: 14,
          transform: 'translateX(-50%)',
          zIndex: 2000,
          px: 1,
          width: { xs: 'calc(100% - 16px)', sm: 'auto' },
          pointerEvents: 'none',
        }}
      >
        <Button
          variant="contained"
          onClick={handleFloatingStart}
          disabled={isCreating}
          startIcon={isCreating ? <CircularProgress size={16} color="inherit" /> : <RocketLaunchIcon sx={{ fontSize: 16 }} />}
          sx={{
            pointerEvents: 'auto',
            width: { xs: '100%', sm: 'auto' },
            minWidth: 220,
            py: 0.9,
            px: 2.5,
            borderRadius: '999px',
            fontWeight: 700,
            textTransform: 'none',
            bgcolor: HF_ORANGE,
            color: '#000',
            boxShadow: '0 8px 24px rgba(0,0,0,0.35)',
            '&:hover': { bgcolor: '#FFB340' },
          }}
        >
          {floatingStartLabel}
        </Button>
      </Box>
    </Box>
  );
}
