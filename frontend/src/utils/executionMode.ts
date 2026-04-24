export type ExecutionMode = 'sandbox' | 'local';

const KEY = 'ml-intern-execution-mode';

export function getPreferredExecutionMode(): ExecutionMode {
  if (typeof window === 'undefined') return 'sandbox';
  const raw = window.localStorage.getItem(KEY);
  return raw === 'local' ? 'local' : 'sandbox';
}

export function setPreferredExecutionMode(mode: ExecutionMode): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(KEY, mode);
}
