## Brief Project Structure

- **open_agent/**: Project-specific code built on top of the public `smolagents` framework.

  - **run_hle.py**: Entry point to run HLE evaluation workflows and orchestrate agents.
  - **tools/**: Custom tools (search, inspection, web crawling, scoring, reformulation, etc.) used by agents during problem solving.
  - **data_utils/**: Dataset loaders and lightweight preprocessing (e.g., HLE subset loader).
  - **agent_kb/**: Agent knowledge-base utilities (prompting, retrieval helpers, interfaces).
  - **rag/**: Retrieval-Augmented Generation building blocks (embeddings, storages, loaders, retrievers, message types, utilities).
  - **reflectors/**: Components for reflection/refinement loops during agent reasoning.

- **src/**: External or vendored libraries.

  - **smolagents/**: Public agent framework used as the base runtime and abstractions.
  - **pyproject.toml**: Build metadata for the `src`-based packages.

- **configs/**: Configuration files for experiments (e.g., HLE run settings).

- **data/**: Data directory (datasets, caches, or intermediate artifacts; structure may vary by task).

- **output/**: Experiment outputs (answers, logs, run-time configs), organized by run name and timestamp.

- **downloads/**: Downloaded assets or artifacts used during runs.

- **assets/**: Static assets for docs/demos.

- **visualization/**: Simple visualization scripts/utilities for inspecting results.

- **scripts/**: Shell scripts to reproduce common runs (e.g., `run_hle.sh`).

- **workflow/**: Reserved for pipeline/automation definitions (empty or optional).

- **requirements.txt**: Python dependencies for the project environment.

- **.env_template**: Example environment variables (e.g., tokens) to copy into `.env`.