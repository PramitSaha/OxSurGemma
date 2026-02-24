"""
Surgical co-pilot entrypoint. Uses MedRAX agent with surgical tools.

Run from repository root:
  python -m surgical_copilot.main [--gradio] [--port 8585]
  # or after: pip install -e .
  python -m surgical_copilot.main --gradio
"""
import os
import sys
from pathlib import Path

# If run as script from inside surgical_copilot/, add repo root to path
if __name__ == "__main__" or "surgical_copilot" not in sys.modules:
    _root = Path(__file__).resolve().parent.parent
    if _root not in sys.path:
        sys.path.insert(0, str(_root))

from surgical_copilot import config

_medrax = config.MEDRAX_DIR
if not _medrax.exists():
    print(
        f"MedRAX not found at {_medrax}. From the repository root, run:\n"
        "  git clone https://github.com/bowang-lab/MedRAX.git\n"
        "  pip install -e MedRAX\n"
        "Or set MEDRAX_DIR to your MedRAX clone.",
        file=sys.stderr,
    )
    sys.exit(1)
if str(_medrax) not in sys.path:
    sys.path.insert(0, str(_medrax))

print("Loading dependencies (this may take 2-5 minutes on first run)...", flush=True)
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# Import tools so they register
import surgical_copilot.tools  # noqa: F401
from surgical_copilot.registry import get_tools, get_registered_names
from surgical_copilot.tools.frame_attributes import is_frame_attributes_available
from surgical_copilot.tools.object_detection_merged import is_object_detection_merged_available
from surgical_copilot.tools.rag_retrieval import is_rag_retrieval_available


def _load_system_prompt():
    """Load surgical co-pilot or medical assistant prompt."""
    prompt_file = getattr(
        config, "SYSTEM_PROMPT_FILE", config.MEDRAX_DIR / "medrax" / "docs" / "system_prompts.txt"
    )
    # Prefer prompts in our package
    local_prompts = Path(__file__).resolve().parent / "docs" / "system_prompts.txt"
    if local_prompts.exists():
        prompt_file = local_prompts
    if not prompt_file.exists():
        return "You are a surgical co-pilot. Use the provided tools to analyze images and answer questions."
    from medrax.utils import load_prompts_from_file
    prompts = load_prompts_from_file(str(prompt_file))
    key = getattr(config, "SURGICAL_PROMPT_KEY", "SURGICAL_COPILOT")
    return prompts.get(key, prompts.get("MEDICAL_ASSISTANT", list(prompts.values())[0] if prompts else prompt_file.read_text(encoding="utf-8").strip()))


# MedGemma model IDs for --medgemma CLI choice
MEDGEMMA_4B = "google/medgemma-4b-it"
MEDGEMMA_27B = "google/medgemma-27b-it"


def create_agent(
    tools_to_use=None,
    model=None,
    temperature=None,
    log_dir=None,
    openai_kwargs=None,
    llm_backend=None,
    local_model=None,
):
    """
    Create the surgical co-pilot agent (MedRAX Agent + surgical tools).

    Args:
        tools_to_use: List of tool names, or None to use config.TOOLS_TO_USE / all.
        model: OpenAI model name (when backend=openai) or ignored for local.
        temperature: Model temperature.
        log_dir: Directory for tool-call logs.
        openai_kwargs: Extra kwargs for ChatOpenAI (e.g. api_key, base_url).
        llm_backend: "openai" (closed-source API) or "local" (MedGemma etc.). Default from config.
        local_model: When llm_backend="local", Hugging Face model ID (e.g. google/medgemma-4b-it).
                    Overrides config.LOCAL_MODEL if set.

    Returns:
        MedRAX Agent instance.
    """
    print("  Importing MedRAX Agent...", flush=True)
    from medrax.agent import Agent
    print("  Getting tools...", flush=True)

    tools_to_use = tools_to_use if tools_to_use is not None else list(config.TOOLS_TO_USE)
    # Exclude frame_attributes if checkpoint missing (avoids endless retry loop)
    if "frame_attributes" in tools_to_use and not is_frame_attributes_available():
        tools_to_use = [t for t in tools_to_use if t != "frame_attributes"]
        print(
            "frame_attributes excluded: checkpoint not found. Train with: "
            "cd frame_attributes_tasks && python train_cholec20_multilabel.py --epochs 30",
            file=sys.stderr,
        )
    # Exclude object_detection_merged if weights missing
    if "object_detection_merged" in tools_to_use and not is_object_detection_merged_available():
        tools_to_use = [t for t in tools_to_use if t != "object_detection_merged"]
        print(
            "object_detection_merged excluded: best_detector_balanced.pt not found. Train with: "
            "python object_detection/train_detector_balanced.py --data_dir ... --save_weights object_detection/best_detector_balanced.pt",
            file=sys.stderr,
        )
    # Exclude rag_retrieval if RAG index not built
    if "rag_retrieval" in tools_to_use and not is_rag_retrieval_available():
        tools_to_use = [t for t in tools_to_use if t != "rag_retrieval"]
        print(
            "rag_retrieval excluded: RAG index not found. Build with: "
            "cd surgical_rag && python run_pipeline.py && python build_index.py",
            file=sys.stderr,
        )
    model = model or config.MODEL
    temperature = temperature if temperature is not None else config.TEMPERATURE
    log_dir = log_dir or config.LOG_DIR
    llm_backend = llm_backend or getattr(config, "LLM_BACKEND", "openai")
    openai_kwargs = dict(openai_kwargs or {})
    print("  Creating LLM...", flush=True)

    if llm_backend == "openai":
        if not openai_kwargs.get("api_key"):
            openai_kwargs["api_key"] = getattr(config, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
        if not openai_kwargs.get("api_key"):
            print(
                "OpenAI API key is required when LLM_BACKEND=openai. Set OPENAI_API_KEY or:\n"
                "  export OPENAI_API_KEY='your-key-here'",
                file=sys.stderr,
            )
            sys.exit(1)
        llm = ChatOpenAI(model=model, temperature=temperature, **openai_kwargs)
    elif llm_backend == "local":
        from surgical_copilot.llm_local import MedGemmaChatModel

        # Default: MedGemma 4B. CLI (--medgemma 4b|27b) overrides.
        local_model_id = local_model or getattr(config, "LOCAL_MODEL", MEDGEMMA_4B)
        use_27b = local_model_id == MEDGEMMA_27B
        lora_root = getattr(config, "TOOL_USE_LORA_ROOT", Path(__file__).resolve().parent.parent / "tool_use_lora_checkpoints")
        preferred = lora_root / ("medgemma-27b-tool-use-lora" if use_27b else "medgemma-4b-tool-use-lora")
        if preferred.is_dir() and (preferred / "adapter_config.json").exists():
            lora_path = str(preferred)
        else:
            fallback = getattr(config, "TOOL_USE_LORA_ADAPTER", None)
            lora_path = str(fallback) if fallback else None
        hf_cache = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or getattr(config, "HF_CACHE_DIR", None)
        hf_token = getattr(config, "HF_TOKEN", None) or os.environ.get("HF_TOKEN")
        llm = MedGemmaChatModel(
            model_id=local_model_id,
            temperature=temperature,
            cache_dir=hf_cache,
            hf_token=hf_token,
            lora_adapter_path=lora_path,
        )
    else:
        print(
            f"Unknown LLM_BACKEND={llm_backend}. Use 'openai' or 'local'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Loading tools: {tools_to_use}...", flush=True)
    tool_list = get_tools(tools_to_use=tools_to_use)
    checkpointer = MemorySaver()
    system_prompt = _load_system_prompt()
    print("  Initializing Agent...", flush=True)

    agent = Agent(
        model=llm,
        tools=tool_list,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        log_tools=True,
        log_dir=log_dir,
    )
    agent._raw_model = llm  # unbound model for parallel-tools-then-answer (single answer, hide tool output)
    print("  Agent initialization complete.", flush=True)
    return agent


def run_chat(agent, thread_id="default"):
    """Simple interactive chat with the agent."""
    from langchain_core.messages import HumanMessage

    config_key = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    print("Surgical co-pilot. Tools:", list(agent.tools.keys()))
    print("Say something or type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "quit":
            break
        for chunk in agent.workflow.stream(
            {"messages": [HumanMessage(content=text)]},
            config=config_key,
        ):
            for node, state in chunk.items():
                msgs = state.get("messages", [])
                if msgs and hasattr(msgs[-1], "content") and msgs[-1].content:
                    print("Copilot:", msgs[-1].content)
        print()
    print("Bye.")


"""
def run_gradio(agent, server_name="0.0.0.0", server_port=8585, share=True):
    print("  Importing Gradio...", flush=True)
    import gradio as gr
    print("  Importing gradio_demo...", flush=True)
    from surgical_copilot.gradio_demo import create_demo

    # tools_dict: name -> tool instance (for optional use in UI, e.g. display names)
    tools_dict = dict(agent.tools)
    print("  Creating Gradio demo interface...", flush=True)
    demo, custom_css = create_demo(agent, tools_dict)
    print("  Setting up queue...", flush=True)
    demo.queue()  # Required for gr.Progress() to display during upload processing
    # Gradio: TTS, extracted frames, and Gradio's cache — add to allowed_paths so tools can read files.
    tts_dir = Path(os.environ.get("TMPDIR", os.getcwd())) / "surgical_copilot_tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(os.getcwd()) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    project_temp = project_root / "temp"
    project_temp.mkdir(parents=True, exist_ok=True)
    gradio_temp = Path(os.environ.get("GRADIO_TEMP_DIR", "/tmp"))
    launch_kw = dict(
        server_name=server_name,
        server_port=server_port,
        share=share,
        allowed_paths=[str(tts_dir), str(temp_dir), str(project_temp), str(gradio_temp)],
        css=custom_css,
        theme=gr.themes.Base(),
    )
    print(f"  Launching Gradio on {server_name}:{server_port} (share={share})...", flush=True)
    demo.launch(**launch_kw)


if __name__ == "__main__":
    print("Starting surgical co-pilot...", flush=True)
    import argparse
    parser = argparse.ArgumentParser(description="Surgical co-pilot: CLI chat or Gradio demo")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio web demo (default: terminal chat)",
    )
    parser.add_argument("--port", type=int, default=8585, help="Gradio server port (default: 8585)")
    parser.add_argument("--no-share", action="store_true", help="Disable Gradio share link")
    args = parser.parse_args()
    print("Arguments parsed. Creating agent...", flush=True)

    agent = create_agent()
    print("Agent created successfully.", flush=True)
    if args.gradio:
        print("Starting Gradio server...", flush=True)
        run_gradio(agent, server_port=args.port, share=not args.no_share)
    else:
        run_chat(agent)
"""

def run_gradio(agent, server_port=8585, share=False):
    """
    Launches the Gradio web interface with the new futuristic theme.
    """
    try:
        from surgical_copilot.gradio_demo import create_demo
    except ImportError as e:
        print(f"❌ Failed to import Gradio demo: {e}", flush=True)
        return

    print("Building futuristic UI...", flush=True)
    
    # 1. Capture both the demo object AND the custom CSS string
    # We pass an empty dict {} for tools if you aren't using dynamic tool loading yet
    demo, custom_css = create_demo(agent, {}) 

    print(f"🚀 Launching Gradio server on port {server_port}...", flush=True)
    
    # 2. Pass the CSS directly to the launch() method here
    demo.launch(
        server_name="0.0.0.0",       # Allows external connections (required for most servers)
        server_port=server_port,
        share=share,
        css=custom_css,              # <--- APPLIES THE THEME
        allowed_paths=["temp", ".", "/"], # Ensures video/images can be served
        show_error=True
    )

if __name__ == "__main__":
    import argparse
    
    print("Initializing Surgical Co-Pilot System...", flush=True)
    
    parser = argparse.ArgumentParser(description="Surgical co-pilot: CLI chat or Gradio demo")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio web demo (default: terminal chat)",
    )
    parser.add_argument("--port", type=int, default=8585, help="Gradio server port (default: 8585)")
    parser.add_argument("--no-share", action="store_true", help="Disable Gradio share link")
    parser.add_argument(
        "--medgemma",
        choices=["4b", "27b"],
        default=None,
        help="Local LLM: use MedGemma 4B or 27B (only when LLM_BACKEND=local). Default: 4b.",
    )
    parser.add_argument(
        "--hf-cache",
        default=None,
        metavar="PATH",
        help="Hugging Face cache directory for model downloads (default: project dir hf_cache). E.g. ./hf_cache or /path/to/my/cache.",
    )
    parser.add_argument(
        "--cvs",
        action="store_true",
        help="Use new CVS model (single ResNet18, new_cvs_model_1). If not set, use old ensemble (vgg, resnet, resnet18, densenet).",
    )
    args = parser.parse_args()

    # CVS tool mode: --cvs → new single model; no flag → old four-backbone ensemble
    os.environ["SURGICAL_CVS_USE_NEW_MODEL"] = "1" if args.cvs else "0"
    if args.cvs:
        print("  CVS: using new model (ResNet18, new_cvs_model_1)", flush=True)
    else:
        print("  CVS: using old ensemble (vgg, resnet, resnet18, densenet)", flush=True)

    print("Arguments parsed. Loading AI Agent...", flush=True)

    # Use project-local HF cache by default (config sets it); CLI overrides
    if args.hf_cache is not None:
        os.environ["TRANSFORMERS_CACHE"] = os.path.abspath(args.hf_cache)
        os.environ["HF_HOME"] = os.path.abspath(args.hf_cache)
        print(f"  HF cache: {os.path.abspath(args.hf_cache)}", flush=True)

    local_model_arg = None
    if args.medgemma:
        local_model_arg = MEDGEMMA_4B if args.medgemma == "4b" else MEDGEMMA_27B
        print(f"  Using MedGemma: {args.medgemma} ({local_model_arg})", flush=True)

    agent = create_agent(local_model=local_model_arg)
    print("✅ Agent created successfully.", flush=True)

    if args.gradio:
        run_gradio(agent, server_port=args.port, share=not args.no_share)
    else:
        run_chat(agent)