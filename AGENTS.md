Make sure you've read README.md, and understand the spirit of this experiment. 

We're building this out of curiosity, it is not a production service, prize elegance and following the spirit of the idea above anything practical.

Rules:

- Always use uv. When adding a dependency, use "uv add" rather than uv pip
- When running background tasks, if they'll run longer than ~a minute, prefer to run them in new tmux sessions, so the user can monitor them as well.
- You must not store state for the database anywhere except the weights of the LLM. Schema, data, everything. 