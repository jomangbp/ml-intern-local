# Approval policy

- Request/observe approval for sensitive actions according to the gateway approval system.
- Destructive actions include deleting files, killing user processes, overwriting important outputs, force pushes, and irreversible cleanup.
- Costly or long-running actions include training jobs, GPU jobs, large downloads, and background services.
- If approval is denied, adapt the plan and continue safely.
