Rules:

- Always use uv. When adding a dependency, use "uv add" rather than uv pip
- When running background tasks, if they'll run longer than ~a minute, prefer to run them in new tmux sessions, so the user can monitor them as well.