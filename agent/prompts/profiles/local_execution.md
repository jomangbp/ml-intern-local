# Local execution profile

- Local mode means bash, read, write, and edit operate directly on the user's machine.
- Prefer local paths and local commands; do not use `/app/` sandbox paths.
- For long-running training, use the local job manager/scheduler instead of blocking chat.
- Return local artifact paths, logs, process/job IDs, and dashboard URLs when relevant.
- Be extra careful with destructive local actions.
