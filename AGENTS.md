Make sure you've read README.md, and understand the spirit of this experiment. 

We're building this out of curiosity, it is not a production service, prize elegance and following the spirit of the idea above anything practical.

Rules:

- Always use uv. When adding a dependency, use "uv add" rather than uv pip
- When running background tasks, if they'll run longer than ~a minute, prefer to run them in new tmux sessions, so the user can monitor them as well.
- You must not store state for the database anywhere except the weights of the LLM. Schema, data, everything. 

Conventions:

- Anything algorithmic or core to the project of making the LLM think it's a database should live in python in the LLM folder
- Anything about parsing SQL should happen inside the duckdb C++ extension. We should not be parsing SQL ourselves ever in any part of the project.
- If you've created a worktree for something, unless you changed the dependency, use the venv in the parent so you don't have to reinstall the heavy dependencies (this will take ~15 minutes).

