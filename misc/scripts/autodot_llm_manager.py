#!/usr/bin/env python3
"""Interactive GUI helper for managing AutoDot local LLM backends.

This tool focuses on keeping AutoDot's large language model (LLM) workflow entirely
local.  It allows creators to download models directly from Hugging Face, prepare
them for use with LangChain and quickly sanity check the resulting pipeline.  The
GUI exposes sensible defaults for popular model families (OpenAI GPT-OSS,
Mistral, Llama 4 and Phi) while still enabling advanced users to target any
other repository.

The script intentionally avoids depending on the Godot editor runtime so that it
can be executed as a standalone helper during project setup.  When a Godot
binary is available it can optionally expose a LangChain agent that executes
GDScript automation against a project:

```
python3 misc/scripts/autodot_llm_manager.py
```

Requirements:

* huggingface_hub
* langchain
* langchain-community (LangChain split package)
* transformers
* torch (or any backend supported by transformers for inference)

The GUI remains functional even when some of these packages are missing – it will
surface clear instructions in the log panel instead of crashing.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError as exc:  # pragma: no cover - tkinter is part of stdlib but may be missing in headless envs
    raise SystemExit(
        "tkinter is required to run the AutoDot LLM manager GUI. "
        "Install the python3-tk package or use a Python distribution with tkinter support."
    ) from exc


# -- Optional runtime dependencies -------------------------------------------------------------

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None  # type: ignore[assignment]

try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:  # pragma: no cover - optional dependency
    HuggingFacePipeline = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from langchain.agents import AgentType, initialize_agent  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    AgentType = None  # type: ignore[assignment]
    initialize_agent = None  # type: ignore[assignment]

LangChainToolType = None
try:  # pragma: no cover - optional dependency
    from langchain.tools import Tool as LangChainToolType  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    try:
        from langchain.agents import Tool as LangChainToolType  # type: ignore[assignment]
    except ImportError:  # pragma: no cover - optional dependency
        LangChainToolType = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = AutoTokenizer = pipeline = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import langflow  # type: ignore[unused-import]
    import langflow.server  # type: ignore[unused-import]
except ImportError:  # pragma: no cover - optional dependency
    langflow = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import langflow_community  # type: ignore[unused-import]
except ImportError:  # pragma: no cover - optional dependency
    langflow_community = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import langflow_embedded_chat  # type: ignore[unused-import]
except ImportError:  # pragma: no cover - optional dependency
    langflow_embedded_chat = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from langflow.load import run_flow_from_json
except ImportError:  # pragma: no cover - optional dependency
    run_flow_from_json = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class ModelOption:
    """Represents a default model choice exposed in the GUI."""

    label: str
    repo_id: str
    revision: Optional[str] = None
    description: Optional[str] = None


DEFAULT_MODELS: Dict[str, ModelOption] = {
    option.label: option
    for option in (
        ModelOption(
            label="OpenAI GPT-OSS 20B",
            repo_id="openai/gpt-oss-20b",
            description=(
                "OpenAI's GPT-OSS 20B checkpoint built for local inference. Requires a "
                "machine with ample RAM/VRAM or quantisation."
            ),
        ),
        ModelOption(
            label="Mistral 7B Instruct v0.2",
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            description=(
                "Instruction tuned Mistral model. Requires a system with sufficient RAM/VRAM "
                "or quantisation before interactive use."
            ),
        ),
        ModelOption(
            label="Mistral Nemo Instruct 24.07",
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            description=(
                "Latest Mistral Nemo instruct release, tuned for agentic workflows. Accept "
                "the Mistral license on Hugging Face before downloading."
            ),
        ),
        ModelOption(
            label="Llama 4 Maverick",
            repo_id="meta-llama/Llama-4-Maverick",
            description=(
                "Meta's Llama 4 Maverick release geared towards creative generation. Access "
                "requires accepting the Meta Llama 4 license on Hugging Face."
            ),
        ),
        ModelOption(
            label="Llama 4 Scout",
            repo_id="meta-llama/Llama-4-Scout",
            description=(
                "Meta's Llama 4 Scout model focused on reasoning and tool usage. Accept the "
                "Meta Llama 4 license on Hugging Face before use."
            ),
        ),
        ModelOption(
            label="Phi-3 Mini 4k Instruct",
            repo_id="microsoft/phi-3-mini-4k-instruct",
            description="Compact Microsoft Phi model geared towards tool enabled workflows.",
        ),
    )
}


def _format_exception(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}" if exc else "Unknown error"


class GodotCommandTool:
    """Wraps headless Godot execution so LangChain agents can apply edits."""

    def __init__(self, executable: Path, project_dir: Optional[Path]) -> None:
        self.executable = executable
        self.project_dir = project_dir

    def run(self, gdscript_body: str) -> str:
        script_content = self._wrap_script(gdscript_body)
        with tempfile.NamedTemporaryFile("w", suffix=".gd", delete=False) as temp_script:
            temp_script.write(script_content)
            temp_path = Path(temp_script.name)

        cmd = [str(self.executable), "--headless", "--quit"]
        if self.project_dir is not None:
            cmd.extend(["--path", str(self.project_dir)])
        cmd.extend(["--script", str(temp_path)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        finally:
            temp_path.unlink(missing_ok=True)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        outcome = "Result: success" if result.returncode == 0 else f"Result: failure (exit {result.returncode})"
        chunks = [outcome]
        if stdout:
            chunks.append("STDOUT:\n" + stdout)
        if stderr:
            chunks.append("STDERR:\n" + stderr)
        return "\n\n".join(chunks)

    def as_langchain_tool(self):
        if LangChainToolType is None:
            raise RuntimeError("LangChain Tool class unavailable.")
        description = (
            "Execute GDScript statements inside a temporary headless Godot session. "
            "Input must contain valid lines for a `_init()` body. Use it to modify scenes, "
            "resources or run editor automation tasks."
        )
        return LangChainToolType(name="godot_command", func=self.run, description=description)

    @staticmethod
    def _wrap_script(body: str) -> str:
        indented_body = "\n".join("    " + line if line.strip() else "" for line in body.splitlines())
        if not indented_body:
            indented_body = "    pass"
        return "\n".join(
            [
                "extends SceneTree",
                "",
                "func _init():",
                indented_body,
                "    get_tree().quit()",
            ]
        )


class LLMManagerGUI:
    """Encapsulates the Tkinter GUI wiring and background tasks."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AutoDot Local LLM Manager")
        self.root.minsize(900, 650)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.langchain_llm = None
        self.langchain_agent = None
        self.loaded_model_path: Optional[Path] = None

        self.model_choice = tk.StringVar(value=next(iter(DEFAULT_MODELS)))
        self.custom_repo = tk.StringVar(value=DEFAULT_MODELS[self.model_choice.get()].repo_id)
        self.revision_var = tk.StringVar()
        self.hf_token_var = tk.StringVar()
        self.base_dir_var = tk.StringVar(value=str(Path.home() / ".cache" / "autodot" / "models"))
        self.trust_remote_code = tk.BooleanVar(value=True)
        self.max_new_tokens = tk.IntVar(value=256)
        self.temperature = tk.DoubleVar(value=0.7)
        self.godot_exec_var = tk.StringVar()
        self.godot_project_var = tk.StringVar()
        self.use_agent_var = tk.BooleanVar(value=False)
        self.backend_choice = tk.StringVar(value="pipeline")

        self.langflow_host_var = tk.StringVar(value="127.0.0.1")
        self.langflow_port_var = tk.IntVar(value=7860)
        self.langflow_workspace_var = tk.StringVar(
            value=str(Path.home() / ".cache" / "autodot" / "langflow")
        )
        self.langflow_flow_file_var = tk.StringVar()
        self.langflow_input_component_var = tk.StringVar(value="ChatInput")
        self.langflow_tool_component_var = tk.StringVar(value="GodotToolToggle")
        self.langflow_allow_tools_var = tk.BooleanVar(value=False)
        self.langflow_status_var = tk.StringVar(value="LangFlow idle.")

        self.langflow_process: Optional[subprocess.Popen[str]] = None
        self.langflow_monitor_thread: Optional[threading.Thread] = None
        self.langflow_server_url: Optional[str] = None
        self.langflow_flow_path: Optional[Path] = None

        self.prompt_input: Optional[ScrolledText] = None
        self.prompt_output: Optional[ScrolledText] = None
        self.log_view: Optional[ScrolledText] = None
        self.status_label: Optional[ttk.Label] = None
        self.backend_tool_checkbox: Optional[ttk.Checkbutton] = None

        self._build_gui()
        self.backend_choice.trace_add("write", lambda *_: self._update_backend_controls())
        self._update_backend_controls()
        self._process_log_queue()
        self._update_dependency_warnings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- GUI construction ------------------------------------------------------------------

    def _build_gui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        for i in range(3):
            main.rowconfigure(i, weight=0)
        main.rowconfigure(3, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)

        # Model selection --------------------------------------------------------------
        selection_frame = ttk.LabelFrame(main, text="Model selection", padding=10)
        selection_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=(0, 0), pady=(0, 12))

        ttk.Label(selection_frame, text="Default models").grid(row=0, column=0, sticky="w")
        model_combobox = ttk.Combobox(
            selection_frame,
            textvariable=self.model_choice,
            state="readonly",
            values=list(DEFAULT_MODELS.keys()),
            width=40,
        )
        model_combobox.grid(row=1, column=0, sticky="ew", pady=4)
        model_combobox.bind("<<ComboboxSelected>>", self._on_model_selected)

        ttk.Label(selection_frame, text="Custom Hugging Face repository").grid(row=0, column=1, sticky="w")
        repo_entry = ttk.Entry(selection_frame, textvariable=self.custom_repo, width=40)
        repo_entry.grid(row=1, column=1, sticky="ew", pady=4, padx=(12, 0))

        ttk.Label(selection_frame, text="Revision / branch (optional)").grid(row=0, column=2, sticky="w")
        ttk.Entry(selection_frame, textvariable=self.revision_var, width=24).grid(row=1, column=2, sticky="ew", padx=(12, 0), pady=4)

        ttk.Label(selection_frame, text="Destination directory").grid(row=2, column=0, sticky="w", pady=(12, 0))
        dest_entry = ttk.Entry(selection_frame, textvariable=self.base_dir_var)
        dest_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)

        browse_button = ttk.Button(selection_frame, text="Browse…", command=self._on_browse_dir)
        browse_button.grid(row=3, column=2, sticky="ew", padx=(12, 0), pady=4)

        ttk.Label(selection_frame, text="Hugging Face token (if required)").grid(row=4, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(selection_frame, textvariable=self.hf_token_var, show="*", width=30).grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Checkbutton(selection_frame, text="Trust remote code", variable=self.trust_remote_code).grid(
            row=4, column=2, rowspan=2, sticky="w", padx=(12, 0), pady=(12, 0)
        )

        download_button = ttk.Button(selection_frame, text="Download / update model", command=self._start_download)
        download_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(12, 0))

        # LangChain preparation --------------------------------------------------------
        pipeline_frame = ttk.LabelFrame(main, text="LangChain pipeline", padding=10)
        pipeline_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 12))

        ttk.Label(pipeline_frame, text="Max new tokens").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(pipeline_frame, from_=16, to=2048, increment=16, textvariable=self.max_new_tokens, width=8).grid(
            row=1, column=0, sticky="w"
        )

        ttk.Label(pipeline_frame, text="Temperature").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Spinbox(pipeline_frame, from_=0.0, to=2.0, increment=0.05, textvariable=self.temperature, width=8).grid(
            row=1, column=1, sticky="w", padx=(12, 0)
        )

        ttk.Button(pipeline_frame, text="Prepare LangChain pipeline", command=self._start_prepare_pipeline).grid(
            row=1, column=2, sticky="ew", padx=(12, 0)
        )

        automation_frame = ttk.LabelFrame(main, text="Godot automation", padding=10)
        automation_frame.grid(row=1, column=2, sticky="nsew", padx=(12, 0), pady=(0, 12))
        automation_frame.columnconfigure(0, weight=1)

        ttk.Label(automation_frame, text="Godot executable (blank to auto-detect)").grid(row=0, column=0, sticky="w")
        ttk.Entry(automation_frame, textvariable=self.godot_exec_var).grid(row=1, column=0, sticky="ew", pady=4)

        ttk.Label(automation_frame, text="Project directory (optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(automation_frame, textvariable=self.godot_project_var).grid(row=3, column=0, sticky="ew", pady=4)

        ttk.Button(
            automation_frame,
            text="Register Godot LangChain agent",
            command=self._start_register_godot_agent,
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

        ttk.Checkbutton(
            automation_frame,
            text="Use Godot automation during prompts",
            variable=self.use_agent_var,
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))

        # Prompt interface ------------------------------------------------------------
        prompt_frame = ttk.LabelFrame(main, text="Prompt sandbox", padding=10)
        prompt_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        prompt_frame.rowconfigure(1, weight=1)
        prompt_frame.columnconfigure(0, weight=1)

        backend_frame = ttk.Frame(prompt_frame)
        backend_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        backend_frame.columnconfigure(3, weight=1)

        ttk.Label(backend_frame, text="Execution backend:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            backend_frame,
            text="LangChain pipeline",
            variable=self.backend_choice,
            value="pipeline",
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(
            backend_frame,
            text="LangFlow flow",
            variable=self.backend_choice,
            value="langflow",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.backend_tool_checkbox = ttk.Checkbutton(
            backend_frame,
            text="Allow Godot tools",
            variable=self.use_agent_var,
        )
        self.backend_tool_checkbox.grid(row=0, column=3, sticky="w")

        self.prompt_input = ScrolledText(prompt_frame, height=12, wrap="word")
        self.prompt_input.grid(row=1, column=0, sticky="nsew")
        self.prompt_input.insert("1.0", "You can run a quick prompt once a model has been loaded.")

        run_prompt_button = ttk.Button(prompt_frame, text="Run prompt", command=self._start_prompt)
        run_prompt_button.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        output_frame = ttk.LabelFrame(main, text="Model output", padding=10)
        output_frame.grid(row=2, column=1, sticky="nsew", padx=(12, 0), pady=(0, 12))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.prompt_output = ScrolledText(output_frame, height=12, wrap="word", state="disabled")
        self.prompt_output.grid(row=0, column=0, sticky="nsew")

        # LangFlow -------------------------------------------------------------------
        langflow_frame = ttk.LabelFrame(main, text="LangFlow workspace", padding=10)
        langflow_frame.grid(row=2, column=2, sticky="nsew", padx=(12, 0), pady=(0, 12))
        langflow_frame.columnconfigure(1, weight=1)

        ttk.Label(langflow_frame, text="Host").grid(row=0, column=0, sticky="w")
        ttk.Entry(langflow_frame, textvariable=self.langflow_host_var, width=16).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(langflow_frame, text="Port").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(langflow_frame, from_=1024, to=65535, textvariable=self.langflow_port_var, width=6).grid(
            row=0,
            column=3,
            sticky="w",
            pady=4,
        )

        ttk.Label(langflow_frame, text="Workspace directory").grid(row=1, column=0, sticky="w")
        ttk.Entry(langflow_frame, textvariable=self.langflow_workspace_var).grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(langflow_frame, text="Browse…", command=self._browse_langflow_workspace).grid(
            row=2, column=3, sticky="ew", padx=(12, 0), pady=4
        )

        ttk.Label(langflow_frame, text="Flow JSON").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(langflow_frame, textvariable=self.langflow_flow_file_var).grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(langflow_frame, text="Browse…", command=self._browse_langflow_flow).grid(
            row=4, column=3, sticky="ew", padx=(12, 0), pady=4
        )
        ttk.Button(langflow_frame, text="Use selected flow", command=self._set_langflow_flow).grid(
            row=5, column=0, columnspan=4, sticky="ew"
        )

        component_frame = ttk.Frame(langflow_frame)
        component_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        component_frame.columnconfigure(1, weight=1)
        component_frame.columnconfigure(3, weight=1)
        ttk.Label(component_frame, text="Input component").grid(row=0, column=0, sticky="w")
        ttk.Entry(component_frame, textvariable=self.langflow_input_component_var, width=18).grid(
            row=0, column=1, sticky="ew", padx=(4, 12)
        )
        ttk.Label(component_frame, text="Tool toggle component").grid(row=0, column=2, sticky="w")
        ttk.Entry(component_frame, textvariable=self.langflow_tool_component_var, width=18).grid(
            row=0, column=3, sticky="ew"
        )

        ttk.Checkbutton(
            langflow_frame,
            text="Expose Godot tools to flow",
            variable=self.langflow_allow_tools_var,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

        button_row = ttk.Frame(langflow_frame)
        button_row.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        button_row.columnconfigure(1, weight=1)
        ttk.Button(button_row, text="Launch workspace", command=self._start_langflow_workspace).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(button_row, text="Stop workspace", command=self._stop_langflow_workspace).grid(
            row=0, column=1, sticky="ew", padx=(12, 0)
        )
        ttk.Button(button_row, text="Open LangFlow UI", command=self._open_langflow_ui).grid(
            row=0, column=2, sticky="ew", padx=(12, 0)
        )

        ttk.Label(langflow_frame, textvariable=self.langflow_status_var, wraplength=240, justify="left").grid(
            row=9, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        # Log ------------------------------------------------------------------------
        log_frame = ttk.LabelFrame(main, text="Log", padding=10)
        log_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_view = ScrolledText(log_frame, height=10, wrap="word", state="disabled")
        self.log_view.grid(row=0, column=0, sticky="nsew")

        self.status_label = ttk.Label(main, text="Ready.")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    # -- Utility methods --------------------------------------------------------------

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def _set_status(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.config(text=message)

    def _process_log_queue(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if self.log_view is not None:
                self.log_view.configure(state="normal")
                self.log_view.insert("end", msg + "\n")
                self.log_view.configure(state="disabled")
                self.log_view.see("end")
        self.root.after(150, self._process_log_queue)

    def _update_dependency_warnings(self) -> None:
        missing: list[str] = []
        if snapshot_download is None:
            missing.append("huggingface_hub")
        if HuggingFacePipeline is None:
            missing.append("langchain-community")
        if initialize_agent is None or LangChainToolType is None or AgentType is None:
            missing.append("langchain")
        if AutoModelForCausalLM is None or AutoTokenizer is None or pipeline is None:
            missing.append("transformers")
        if torch is None:
            missing.append("torch")
        if langflow is None:
            missing.append("langflow")
        if langflow_community is None:
            missing.append("langflow-community")
        if langflow_embedded_chat is None:
            missing.append("langflow-embedded-chat")

        if missing:
            self._log(
                "Missing optional dependencies: "
                + ", ".join(missing)
                + ". Install them with 'pip install huggingface_hub langchain langchain-community transformers torch langflow langflow-community langflow-embedded-chat'."
            )

    def _update_backend_controls(self) -> None:
        backend = self.backend_choice.get()
        if self.backend_tool_checkbox is not None:
            if backend == "pipeline":
                self.backend_tool_checkbox.state(["!disabled"])
            else:
                self.backend_tool_checkbox.state(["disabled"])
        if backend != "pipeline" and self.use_agent_var.get():
            self.use_agent_var.set(False)

    # -- Event handlers ----------------------------------------------------------------

    def _on_model_selected(self, _event: object) -> None:
        option = DEFAULT_MODELS[self.model_choice.get()]
        self.custom_repo.set(option.repo_id)
        if option.revision:
            self.revision_var.set(option.revision)
        else:
            self.revision_var.set("")
        if option.description:
            self._log(option.description)

    def _on_browse_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.base_dir_var.get())
        if chosen:
            self.base_dir_var.set(chosen)

    def _browse_langflow_workspace(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.langflow_workspace_var.get())
        if chosen:
            self.langflow_workspace_var.set(chosen)

    def _browse_langflow_flow(self) -> None:
        chosen = filedialog.askopenfilename(
            initialdir=self.langflow_flow_file_var.get() or str(Path.home()),
            filetypes=(("LangFlow export", "*.json"), ("All files", "*")),
        )
        if chosen:
            self.langflow_flow_file_var.set(chosen)

    def _set_langflow_flow(self) -> None:
        if run_flow_from_json is None:
            messagebox.showerror(
                "LangFlow runtime missing",
                "Install langflow and langflow-community to execute flows.",
            )
            return

        path_value = self.langflow_flow_file_var.get().strip()
        if not path_value:
            messagebox.showwarning("Flow not selected", "Choose a LangFlow flow JSON export first.")
            return

        flow_path = Path(path_value).expanduser()
        if not flow_path.exists():
            messagebox.showerror("Flow missing", f"Flow file {flow_path} does not exist.")
            return

        self.langflow_flow_path = flow_path
        self.langflow_status_var.set(f"LangFlow flow ready: {flow_path.name}")
        self._log(f"LangFlow flow set to {flow_path}")

    def _start_download(self) -> None:
        if snapshot_download is None:
            messagebox.showerror("Missing dependency", "Install huggingface_hub to enable downloads.")
            return

        repo_id = self.custom_repo.get().strip()
        if not repo_id:
            messagebox.showwarning("Repository missing", "Please specify a Hugging Face repository id (e.g. org/model).")
            return

        base_dir = Path(self.base_dir_var.get()).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        target_dir = base_dir / repo_id.replace("/", "__")

        revision = self.revision_var.get().strip() or None
        token = self.hf_token_var.get().strip() or None

        def task() -> None:
            self._set_status(f"Downloading {repo_id}…")
            self._log(f"Starting snapshot download for {repo_id} -> {target_dir}")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=token,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            except Exception as exc:  # pragma: no cover - network interaction
                self._log(f"Download failed: {_format_exception(exc)}")
                self._set_status("Download failed.")
                return

            self._log(f"Model files available at {target_dir}")
            self._set_status("Model downloaded.")
            self.loaded_model_path = target_dir

        self._run_background(task)

    def _start_prepare_pipeline(self) -> None:
        if HuggingFacePipeline is None or AutoModelForCausalLM is None or AutoTokenizer is None or pipeline is None:
            messagebox.showerror(
                "Missing dependency",
                "Install langchain, langchain-community and transformers to build the pipeline.",
            )
            return

        model_dir = self.loaded_model_path or self._infer_model_path()
        if model_dir is None or not model_dir.exists():
            messagebox.showwarning(
                "Model not available",
                "Download a model first or set the destination to an existing snapshot.",
            )
            return

        max_tokens = self.max_new_tokens.get()
        temperature = self.temperature.get()
        trust_remote = self.trust_remote_code.get()

        def task() -> None:
            self._set_status("Preparing LangChain pipeline…")
            self._log(f"Loading model from {model_dir}")

            dtype = None
            device = "cpu"
            if torch is not None:
                if torch.cuda.is_available():  # pragma: no cover - GPU unavailable in CI
                    device = "cuda"
                    dtype = torch.float16
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover - macOS only
                    device = "mps"
                    dtype = torch.float16
                else:
                    dtype = torch.float32

            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=trust_remote)
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                    trust_remote_code=trust_remote,
                    torch_dtype=dtype,
                )
                generation_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto" if device != "cpu" else None,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                self.langchain_llm = HuggingFacePipeline(pipeline=generation_pipeline)
            except Exception as exc:  # pragma: no cover - heavy runtime logic
                self._log(f"LangChain pipeline failure: {_format_exception(exc)}")
                self._set_status("Pipeline initialisation failed.")
                return

            self._log("LangChain pipeline ready.")
            self._set_status("Pipeline ready.")
            self.langchain_agent = None

        self._run_background(task)

    def _start_register_godot_agent(self) -> None:
        if initialize_agent is None or LangChainToolType is None or AgentType is None:
            messagebox.showerror(
                "LangChain missing",
                "Install langchain>=0.1 to enable Godot automation.",
            )
            return

        if self.langchain_llm is None:
            messagebox.showinfo("Pipeline missing", "Prepare a LangChain pipeline before enabling automation.")
            return

        exec_value = self.godot_exec_var.get().strip()
        if exec_value:
            exec_path = Path(exec_value).expanduser()
        else:
            detected = shutil.which("godot")
            exec_path = Path(detected) if detected else None

        if exec_path is None:
            messagebox.showerror(
                "Executable missing",
                "Specify a Godot executable or add it to PATH for auto-detection.",
            )
            return

        if not exec_path.exists():
            messagebox.showerror("Invalid path", f"Godot executable not found at {exec_path}.")
            return

        project_value = self.godot_project_var.get().strip()
        project_path: Optional[Path] = None
        if project_value:
            project_path = Path(project_value).expanduser()
            if not project_path.exists():
                messagebox.showerror("Invalid project", f"Project directory {project_path} does not exist.")
                return

        def task() -> None:
            self._set_status("Registering Godot automation…")
            self._log(
                f"Setting up Godot tool using executable {exec_path}" + (f" with project {project_path}" if project_path else "")
            )

            try:
                tool = GodotCommandTool(exec_path, project_path)
                langchain_tool = tool.as_langchain_tool()
                agent = initialize_agent(
                    tools=[langchain_tool],
                    llm=self.langchain_llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True,
                )
            except Exception as exc:  # pragma: no cover - depends on external binaries
                self._log(f"Failed to register automation: {_format_exception(exc)}")
                self._set_status("Automation failed.")
                return

            self.langchain_agent = agent
            self._log("Godot automation ready. Enable the checkbox to allow prompts to use it.")
            self._set_status("Automation ready.")

        self._run_background(task)

    def _start_langflow_workspace(self) -> None:
        if langflow is None:
            messagebox.showerror(
                "LangFlow not available",
                "Install langflow, langflow-community and langflow-embedded-chat to launch the workspace.",
            )
            return

        if self.langflow_process is not None and self.langflow_process.poll() is None:
            messagebox.showinfo("Already running", "LangFlow workspace is already active.")
            return

        workspace_dir = Path(self.langflow_workspace_var.get()).expanduser()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        host = self.langflow_host_var.get().strip() or "127.0.0.1"
        port = self.langflow_port_var.get()
        cmd = [
            sys.executable,
            "-m",
            "langflow",
            "run",
            "--host",
            host,
            "--port",
            str(port),
        ]
        env = os.environ.copy()
        env.setdefault("LANGFLOW_HOME", str(workspace_dir))

        self._log(f"Launching LangFlow workspace with command: {' '.join(cmd)}")
        self.langflow_status_var.set("Starting LangFlow workspace…")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self._log(f"Failed to launch LangFlow: {_format_exception(exc)}")
            self.langflow_status_var.set("LangFlow launch failed.")
            return

        self.langflow_process = process
        self.langflow_server_url = f"http://{host}:{port}"
        self.langflow_status_var.set(f"LangFlow workspace running at {self.langflow_server_url}")
        self._log(f"LangFlow workspace started at {self.langflow_server_url} (storage: {workspace_dir})")

        monitor = threading.Thread(target=self._monitor_langflow_process, args=(process,), daemon=True)
        monitor.start()
        self.langflow_monitor_thread = monitor

    def _stop_langflow_workspace(self) -> None:
        process = self.langflow_process
        if process is None or process.poll() is not None:
            self._log("LangFlow workspace is not running.")
            self.langflow_status_var.set("LangFlow idle.")
            self.langflow_process = None
            return

        self._log("Stopping LangFlow workspace…")
        self.langflow_status_var.set("Stopping LangFlow workspace…")
        try:
            process.terminate()
        except Exception as exc:
            self._log(f"Failed to terminate LangFlow workspace: {_format_exception(exc)}")
            return

        threading.Thread(target=self._finalize_langflow_stop, args=(process,), daemon=True).start()

    def _finalize_langflow_stop(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._log("LangFlow workspace did not exit in time; killing…")
            process.kill()
            process.wait()
        self._log("LangFlow workspace stopped.")
        if self.langflow_process is process:
            self.langflow_process = None
            self.langflow_server_url = None
        self.root.after(0, lambda: self.langflow_status_var.set("LangFlow idle."))

    def _monitor_langflow_process(self, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    clean = line.rstrip()
                    if clean:
                        self._log(f"[LangFlow] {clean}")
        finally:
            return_code = process.wait()
            self._log(f"LangFlow workspace exited with code {return_code}")
            if self.langflow_process is process:
                self.langflow_process = None
                self.langflow_server_url = None
                self.root.after(0, lambda: self.langflow_status_var.set("LangFlow idle."))

    def _open_langflow_ui(self) -> None:
        if langflow is None:
            messagebox.showerror("LangFlow not available", "Install langflow to open the workspace UI.")
            return

        url = self.langflow_server_url
        if url is None:
            host = self.langflow_host_var.get().strip() or "127.0.0.1"
            port = self.langflow_port_var.get()
            url = f"http://{host}:{port}"

        self._log(f"Opening LangFlow UI at {url}")
        webbrowser.open(url)

    def _start_prompt(self) -> None:
        if self.prompt_input is None or self.prompt_output is None:
            return

        prompt = self.prompt_input.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Prompt missing", "Type a prompt into the sandbox first.")
            return

        backend = self.backend_choice.get()

        if backend == "pipeline" and self.langchain_llm is None:
            messagebox.showinfo("Pipeline not ready", "Prepare a LangChain pipeline before running prompts.")
            return

        if backend == "langflow":
            if run_flow_from_json is None:
                messagebox.showerror(
                    "LangFlow runtime missing",
                    "Install langflow and langflow-community to execute flows.",
                )
                return
            if self.langflow_flow_path is None:
                messagebox.showwarning("Flow not ready", "Select a LangFlow flow JSON export before running prompts.")
                return

        use_agent = self.use_agent_var.get()
        agent = self.langchain_agent

        def task() -> None:
            if backend == "langflow":
                executor_name = "LangFlow flow"
            else:
                executor = agent if (use_agent and agent is not None) else self.langchain_llm
                executor_name = "LangChain agent" if executor is agent else "LangChain pipeline"
            self._set_status("Running prompt…")
            self._log(f"Invoking {executor_name}…")
            try:
                if backend == "langflow":
                    response_text = self._execute_langflow_prompt(prompt)
                else:
                    if executor is agent:
                        response_payload = agent.invoke({"input": prompt})
                        if isinstance(response_payload, dict) and "output" in response_payload:
                            response_text = str(response_payload["output"])
                        else:
                            response_text = str(response_payload)
                    else:
                        response_text = str(self.langchain_llm.invoke(prompt))
            except Exception as exc:  # pragma: no cover - runtime dependent
                self._log(f"Prompt execution failed: {_format_exception(exc)}")
                self._set_status("Prompt failed.")
                return

            def update_output() -> None:
                if self.prompt_output is not None:
                    self.prompt_output.configure(state="normal")
                    self.prompt_output.delete("1.0", "end")
                    self.prompt_output.insert("end", response_text)
                    self.prompt_output.configure(state="disabled")
            self.root.after(0, update_output)
            self._log("Prompt completed.")
            self._set_status("Prompt completed.")

        self._run_background(task)

    def _execute_langflow_prompt(self, prompt: str) -> str:
        if run_flow_from_json is None or self.langflow_flow_path is None:
            raise RuntimeError("LangFlow runtime unavailable.")

        tweaks: Dict[str, Dict[str, Any]] = {}
        input_component = self.langflow_input_component_var.get().strip()
        if input_component:
            tweaks.setdefault(input_component, {})["input_value"] = prompt

        tool_component = self.langflow_tool_component_var.get().strip()
        if tool_component:
            tweaks.setdefault(tool_component, {})["enabled"] = self.langflow_allow_tools_var.get()

        try:
            result = run_flow_from_json(
                str(self.langflow_flow_path),
                input_value=prompt,
                tweaks=tweaks or None,
            )
        except TypeError:
            if tweaks:
                result = run_flow_from_json(str(self.langflow_flow_path), tweaks=tweaks)
            else:
                result = run_flow_from_json(str(self.langflow_flow_path), input_value=prompt)

        return self._stringify_langflow_result(result)

    def _stringify_langflow_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "outputs" in result:
                return self._stringify_langflow_result(result["outputs"])
            if "output" in result:
                return self._stringify_langflow_result(result["output"])
            return "\n".join(f"{key}: {self._stringify_langflow_result(value)}" for key, value in result.items())
        if isinstance(result, (list, tuple)):
            return "\n\n".join(self._stringify_langflow_result(item) for item in result)
        return str(result)

    def _on_close(self) -> None:
        self._stop_langflow_workspace()
        self.root.destroy()

    # -- Background worker helpers ----------------------------------------------------

    def _run_background(self, task: Callable[[], None]) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Task running", "Please wait for the current task to finish.")
            return

        def wrapper() -> None:
            try:
                task()
            finally:
                self.worker_thread = None

        self.worker_thread = threading.Thread(target=wrapper, daemon=True)
        self.worker_thread.start()

    def _infer_model_path(self) -> Optional[Path]:
        repo_id = self.custom_repo.get().strip()
        if not repo_id:
            return None
        base_dir = Path(self.base_dir_var.get()).expanduser()
        return (base_dir / repo_id.replace("/", "__")).resolve()

    # -- Lifecycle --------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    LLMManagerGUI().run()


if __name__ == "__main__":
    main()
