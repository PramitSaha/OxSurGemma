"""
Configuration for the surgical co-pilot.
Edit TOOLS_TO_USE to enable/disable tools; add new tool names after registering them.
"""
import os
from pathlib import Path

# Default set of tools. Use None to load all registered tools.
TOOLS_TO_USE = [
    "surgical_scene_segmentation",
    "phase_detection",
    "instrument_tracking",
    "object_detection_merged",
    "frame_attributes",
    "ssg_vqa",
    "triplet_recognition",
    "critical_view_of_safety",
    "rag_retrieval",
]

# Path to MedRAX repo (for agent and prompts). Clone: git clone https://github.com/bowang-lab/MedRAX.git
# Set MEDRAX_DIR env to override. Default: MedRAX as sibling of this repo.
_here = Path(__file__).resolve().parent
REPO_ROOT = _here.parent
MEDRAX_DIR = Path(
    os.environ.get(
        "MEDRAX_DIR",
        REPO_ROOT / "MedRAX",
    )
)

# System prompt: use surgical-specific prompt file or MedRAX default
SYSTEM_PROMPT_FILE = MEDRAX_DIR / "medrax" / "docs" / "system_prompts.txt"
SURGICAL_PROMPT_KEY = "SURGICAL_COPILOT"  # add [SURGICAL_COPILOT] to prompts file, or use MEDICAL_ASSISTANT

# Model / API
MODEL = os.environ.get("SURGICAL_COPILOT_MODEL", "gpt-4o")
TEMPERATURE = float(os.environ.get("SURGICAL_COPILOT_TEMPERATURE", "0.3"))  # Lower from 0.7 = faster
LOG_DIR = os.environ.get("SURGICAL_COPILOT_LOG_DIR", "logs")

# LLM backend: "local" uses MedGemma + tool-use LoRA from tool_use_lora_checkpoints; "openai" uses GPT-4o.
# Set SURGICAL_COPILOT_LLM_BACKEND=openai to use OpenAI instead.
LLM_BACKEND = os.environ.get("SURGICAL_COPILOT_LLM_BACKEND", "local")
# When LLM_BACKEND="local": base model (must match the LoRA adapter: 4b or 27b)
LOCAL_MODEL = os.environ.get("SURGICAL_COPILOT_LOCAL_MODEL", "google/medgemma-4b-it")
# Tool-use LoRA adapter path (under tool_use_lora_checkpoints). First existing 4b/27b is used.
TOOL_USE_LORA_ROOT = REPO_ROOT / "tool_use_lora_checkpoints"
TOOL_USE_LORA_PATH = os.environ.get("SURGICAL_COPILOT_TOOL_USE_LORA") or None  # None = auto-detect below
def _default_tool_use_lora_path():
    if TOOL_USE_LORA_PATH:
        return Path(TOOL_USE_LORA_PATH)
    for name in ("medgemma-4b-tool-use-lora", "medgemma-27b-tool-use-lora"):
        p = TOOL_USE_LORA_ROOT / name
        if p.is_dir() and (p / "adapter_config.json").exists():
            return p
    return TOOL_USE_LORA_ROOT / "medgemma-4b-tool-use-lora"  # default path even if missing
TOOL_USE_LORA_ADAPTER = _default_tool_use_lora_path()
# Hugging Face cache: use env if set, otherwise repo-local path
_default_hf_cache = REPO_ROOT / "hf_cache"
HF_CACHE_DIR = (
    os.environ.get("TRANSFORMERS_CACHE")
    or os.environ.get("HF_HOME")
    or os.environ.get("SURGICAL_COPILOT_HF_CACHE")
    or str(_default_hf_cache)
)
# Ensure env is set so transformers/huggingface_hub use project cache when no env was set
if "TRANSFORMERS_CACHE" not in os.environ and "HF_HOME" not in os.environ:
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    os.environ["HF_HOME"] = HF_CACHE_DIR
try:
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
except OSError:
    pass
# Hugging Face token for gated models (e.g. MedGemma). Accept terms at huggingface.co, then: huggingface-cli login or export HF_TOKEN=hf_xxx
HF_TOKEN = os.environ.get("HF_TOKEN")

# OpenAI API key. Required only when LLM_BACKEND="openai".
# Option A: export OPENAI_API_KEY=... (preferred). Option B: set below (do not commit).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or None  # or set to "sk-..." to hardcode
