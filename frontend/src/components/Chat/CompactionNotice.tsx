import { Box, Stack, Typography } from '@mui/material';
import type { CompactionNotice } from '@/store/agentStore';
import StorageIcon from '@mui/icons-material/Storage';

interface CompactionNoticeProps {
  notice: CompactionNotice;
}

export default function CompactionNoticeComponent({ notice }: CompactionNoticeProps) {
  const saved = notice.tokensSaved;
  const pct = notice.oldTokens > 0
    ? Math.round((saved / notice.oldTokens) * 100)
    : 0;

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', my: 1.5 }}>
      <Stack
        direction="row"
        alignItems="center"
        spacing={1}
        sx={{
          px: 2,
          py: 0.75,
          borderRadius: 1,
          bgcolor: 'var(--surface)',
          border: '1px solid var(--border)',
          opacity: 0.85,
        }}
      >
        <StorageIcon sx={{ fontSize: 14, color: 'var(--muted-text)' }} />
        <Typography
          sx={{
            fontFamily: 'monospace',
            fontSize: '0.72rem',
            color: 'var(--muted-text)',
            fontWeight: 600,
            letterSpacing: '0.02em',
          }}
        >
          Compacted from{' '}
          <strong>{notice.oldTokens.toLocaleString()}</strong>
          {' '}tokens{' '}
          <span style={{ color: 'var(--accent-green)' }}>
            (-{saved.toLocaleString()}, {pct}%)
          </span>
        </Typography>
      </Stack>
    </Box>
  );
}
