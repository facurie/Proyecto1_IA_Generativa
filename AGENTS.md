# Repository Guidelines

## Project Structure & Module Organization

This Python project builds a staged TinyStories language model. Root files (`00_dataset.py`, `01_tokenizer.py`, `02_pretraining.py`) are the editable sources for matching notebooks. Later stages are described in `assignment.md`; their files may not exist yet. `data.py` loads datasets, `environment.py` configures paths and devices, and `tools/` contains notebook generation and Windows certificate helpers. Read `README.md` for setup and `assignment.md` for stage requirements. Runtime datasets, caches, and model artifacts belong in ignored directories such as `.hf_cache/` and `checkpoints/`.

## Build, Test, and Development Commands

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python verify_setup.py
python tools/py_to_notebook.py --check
```

Activate `.venv` before installing dependencies. `verify_setup.py` checks packages, device setup, and Hugging Face Hub access; it needs a network connection. The notebook check reports `.ipynb` files whose cells differ from their `.py` sources (outputs are ignored). After changing a stage, run `python tools/py_to_notebook.py --only 01_tokenizer` (substitute the stage name) to regenerate its notebook; regeneration keeps the outputs of code cells whose source did not change, and warns about the ones that need re-running (`--limpiar` drops all outputs). Run stages from the root so relative `checkpoints/` paths resolve. Colab with a T4 GPU is the target for full training.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase configuration constants such as `SMOKE_TEST`. Preserve the numbered stage filenames and Spanish instructional prose. In stage sources, use `# %%` for code cells and `# %% [markdown]` with a triple-quoted string for Markdown cells. Edit the `.py` source, then regenerate the `.ipynb`; do not hand-edit generated notebook JSON. No formatter or linter configuration is committed.

## Testing Guidelines

There is no committed automated test suite or coverage threshold. Run `python tools/py_to_notebook.py --check` after notebook edits. For behavioral changes, run the affected stage with `SMOKE_TEST` enabled where available and inspect its output; Stage 2 enables this automatically on CPU. Stage 2 requires the tokenizer artifact from Stage 1. Keep downloaded data and checkpoints out of commits.

## Commit & Pull Request Guidelines

Recent commits use short descriptive subjects in English or Spanish; no formal prefix scheme is evident. Write a concise subject naming the changed stage or purpose, for example `Update stage 1 tokenizer checks`. In a pull request, summarize the stage or helper changed, include validation commands and results, link a relevant issue if one exists, and attach plots or notebook output when results change. Commit regenerated notebooks alongside their source files.
