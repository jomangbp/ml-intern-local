import { useEffect, useState, useCallback } from 'react';
import {
  Box, Typography, List, ListItem, ListItemText, ListItemIcon,
  Chip, IconButton, Dialog, DialogTitle, DialogContent, DialogActions,
  Button, Tooltip, Paper,
} from '@mui/material';
import {
  CheckCircle as ApproveIcon,
  Cancel as RejectIcon,
  Refresh as RefreshIcon,
  Description as DetailsIcon,
  Shield as ApprovalIcon,
} from '@mui/icons-material';

interface Approval {
  approval_id: string;
  session_id: string;
  tools: Array<{
    tool: string;
    arguments: Record<string, unknown>;
    tool_call_id: string;
  }>;
  status: string;
  created_at: number;
  expires_at: number;
  resolved_at: number | null;
  platform: string;
  chat_id: string | number | null;
}

function toolSummary(tool: { tool: string; arguments: Record<string, unknown> }): string {
  const { tool: name, arguments: args } = tool;
  switch (name) {
    case 'bash':
      return `💻 ${(args.command as string || '').slice(0, 80)}`;
    case 'write_file':
    case 'edit_file':
      return `✏️ ${name}: ${(args.path as string || '').slice(0, 60)}`;
    case 'local_training':
      return `🏋️ ${(args.script as string || '').slice(0, 60)}`;
    default:
      return `🔧 ${name}`;
  }
}

function toolDetails(tool: { tool: string; arguments: Record<string, unknown> }): string {
  const { tool: name, arguments: args } = tool;
  const parts = Object.entries(args).slice(0, 5).map(([k, v]) => {
    const s = String(v);
    return `  ${k}: ${s.length > 150 ? s.slice(0, 150) + '...' : s}`;
  });
  return `${name}\n${parts.join('\n')}`;
}

function timeAgo(ts: number): string {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

interface ApprovalDetailsDialogProps {
  approval: Approval | null;
  open: boolean;
  onClose: () => void;
}

function ApprovalDetailsDialog({ approval, open, onClose }: ApprovalDetailsDialogProps) {
  if (!approval) return null;
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>📋 Approval Details</DialogTitle>
      <DialogContent>
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>ID:</strong> {approval.approval_id.slice(0, 20)}...
        </Typography>
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>Session:</strong> {approval.session_id.slice(0, 12)}...
        </Typography>
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>Created:</strong> {timeAgo(approval.created_at)}
        </Typography>
        <Typography variant="body2" sx={{ mb: 2 }}>
          <strong>Expires:</strong> {approval.expires_at ? timeAgo(approval.expires_at) : '—'}
        </Typography>
        <Divider />
        {approval.tools.map((t, i) => (
          <Box key={i} sx={{ mb: 1 }}>
            <Box
              component="pre"
              sx={{
                bgcolor: '#1e1e1e',
                color: '#d4d4d4',
                p: 1,
                borderRadius: 1,
                fontSize: '0.75rem',
                fontFamily: 'monospace',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
              }}
            >
              {toolDetails(t)}
            </Box>
          </Box>
        ))}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

function Divider(_props: { sx?: React.CSSProperties }) {
  return <Box sx={{ height: 1, bgcolor: 'divider' }} />;
}

interface ApprovalInboxProps {
  collapsed?: boolean;
}

export default function ApprovalInbox({ collapsed = false }: ApprovalInboxProps) {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [loading, setLoading] = useState(false);
  const [detailsApproval, setDetailsApproval] = useState<Approval | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    fetch('/api/approvals?status=pending')
      .then(r => r.json())
      .then(data => setApprovals(data))
      .catch(() => setApprovals([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleApprove = async (id: string) => {
    await fetch(`/api/approvals/${id}/approve`, { method: 'POST' });
    refresh();
  };

  const handleReject = async (id: string) => {
    await fetch(`/api/approvals/${id}/reject`, { method: 'POST' });
    refresh();
  };

  const pending = approvals.filter(a => a.status === 'pending').length;

  if (collapsed) {
    return (
      <Tooltip title={`${pending} pending approvals`}>
        <Chip
          icon={<ApprovalIcon />}
          label={pending}
          size="small"
          color={pending > 0 ? 'warning' : 'default'}
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
          🔐 Approvals {pending > 0 && <Chip label={`${pending} pending`} size="small" color="warning" sx={{ ml: 1 }} />}
        </Typography>
        <IconButton size="small" onClick={refresh} disabled={loading}>
          <RefreshIcon fontSize="small" />
        </IconButton>
      </Box>

      {approvals.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          No pending approvals.
        </Typography>
      ) : (
        <List dense sx={{ flex: 1, overflow: 'auto' }}>
          {approvals.map(approval => (
            <ListItem
              key={approval.approval_id}
              sx={{
                borderBottom: '1px solid',
                borderColor: 'divider',
                bgcolor: approval.status === 'pending' ? 'warning.50' : 'transparent',
                '&:hover': { bgcolor: 'action.hover' },
              }}
              secondaryAction={
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  <Tooltip title="Details">
                    <IconButton size="small" onClick={() => setDetailsApproval(approval)}>
                      <DetailsIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  {approval.status === 'pending' && (
                    <>
                      <Tooltip title="Approve">
                        <IconButton size="small" onClick={() => handleApprove(approval.approval_id)} color="success">
                          <ApproveIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Reject">
                        <IconButton size="small" onClick={() => handleReject(approval.approval_id)} color="error">
                          <RejectIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </>
                  )}
                </Box>
              }
            >
              <ListItemIcon sx={{ minWidth: 32 }}>
                <Typography variant="body2">
                  {approval.status === 'pending' ? '⏳' : approval.status === 'approved' ? '✅' : approval.status === 'rejected' ? '❌' : '⏰'}
                </Typography>
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                    {approval.tools.map(toolSummary).join(' · ')}
                  </Typography>
                }
                secondary={
                  <Typography variant="caption" color="text.secondary">
                    {approval.approval_id.slice(0, 12)} · {timeAgo(approval.created_at)}
                    {approval.platform && ` · from ${approval.platform}`}
                  </Typography>
                }
              />
            </ListItem>
          ))}
        </List>
      )}

      <ApprovalDetailsDialog
        approval={detailsApproval}
        open={detailsApproval !== null}
        onClose={() => setDetailsApproval(null)}
      />
    </Paper>
  );
}
