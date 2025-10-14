# AutoDot Local LLM Manager

The `misc/scripts/autodot_llm_manager.py` helper provides a lightweight GUI for
teams that want AutoDot to run exclusively on top of local large language models
that can be fetched from Hugging Face.  The script downloads model snapshots,
prepares a LangChain pipeline, exposes a small prompt sandbox for quick smoke
tests, and can now wire LangChain agents into a headless Godot session so local
models can apply project changes automatically.

## Quick start

```bash
pip install huggingface_hub langchain langchain-community transformers torch
python3 misc/scripts/autodot_llm_manager.py
```

The GUI presents curated defaults spanning OpenAI's GPT-OSS 20B release,
Mistral's 7B Instruct v0.2 and Nemo Instruct 24.07 checkpoints, Meta's Llama 4
Maverick and Llama 4 Scout models, plus Microsoft Phi-3 Mini 4k Instruct.  Each
entry stores the associated Hugging Face repository id, so launching the
download simply requires pressing **Download / update model**.  You can freely
replace the repository id to target any other public or private model.

Snapshot downloads run in the background and default to a dedicated cache under
`~/.cache/autodot/models`.  Hugging Face tokens are optional but can be supplied
for gated models or rate-limit exemptions.  Once a download completes you can
press **Prepare LangChain pipeline** to build a local text-generation pipeline
with the configured sampling parameters.

The prompt sandbox on the left becomes active after the pipeline is ready.  Use
it to sanity check responses or verify that weights and tokenizer files were
resolved correctly.  The log panel at the bottom records every background
operation, highlights missing dependencies and reports errors with actionable
messages so that the script “just works” even on fresh machines.

## Controlling Godot from LangChain

When LangChain is available the **Godot automation** panel lets you register a
tool-enabled agent that can execute arbitrary GDScript snippets inside a
headless editor session.  Provide the path to a Godot executable (or leave it
blank to use `godot` from `PATH`), optionally point it at a project directory and
press **Register Godot LangChain agent**.  After the agent is ready, enable the
checkbox so prompts are routed through it.

The registered LangChain tool expects the model to supply valid GDScript body
statements.  They are wrapped into a temporary script that runs in headless
mode, which makes it possible to automate scene edits, import assets or call
editor APIs such as `EditorInterface`.  The tool captures stdout and stderr from
the Godot process, so each prompt reports exactly what happened during the
automation run.

Because the helper is completely standalone, it can be shipped with project
templates, CI provisioning steps or onboarding documentation while keeping the
core AutoDot editor free of additional runtime dependencies.
