# OxSurGemma - A Surgical AI-Copilot for Laparoscopic Cholecystectomy powered by MedGemma

A MedGemma-powered surgical co-pilot for laparoscopic cholecystectomy that combines a multi-tool agent framework with specialised vision models, LLM-based tool routing, speech I/O, and RAG retrieval.

## 🎥 Demo Videos

Demonstration of the Surgical AI Co-Pilot assisting during laparoscopic cholecystectomy. Click on the videos below to explore.
<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=YsTkc-kAEr0">
        <img src="https://img.youtube.com/vi/YsTkc-kAEr0/0.jpg" width="400">
      </a>
      <br>
      <b> Part 1 </b>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=f9_w6XSPPZI">
        <img src="https://img.youtube.com/vi/f9_w6XSPPZI/0.jpg" width="400">
      </a>
      <br>
      <b> Part 2 </b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=P4oygElgkSk">
        <img src="https://img.youtube.com/vi/P4oygElgkSk/0.jpg" width="400">
      </a>
      <br>
      <b> Part 3 </b>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=5Cfh3c_hNr0">
        <img src="https://img.youtube.com/vi/5Cfh3c_hNr0/0.jpg" width="400">
      </a>
      <br>
      <b> Part 4 </b>
    </td>
  </tr>
</table>

## 🤗 Model Zoo (Hugging Face)

| Component | Weights |
|---|---|
| MedGemma-27B-LoRA (Fine-tuned for Tool use and Surgeon-facing Response)| [pramit-saha/oxsurgemma-tool-use-lora-27b](https://huggingface.co/pramit-saha/oxsurgemma-tool-use-lora-27b) |
| MedGemma-4B-LoRA (Fine-tuned for Tool use and Surgeon-facing Response)| [pramit-saha/oxsurgemma-tool-use-lora-4b](https://huggingface.co/pramit-saha/oxsurgemma-tool-use-lora-4b) |
| MedGemma-4B-LoRA (Fine-tuned for Surgical Scene Graph VQA) | [pramit-saha/oxsurgemma-ssg-vqa](https://huggingface.co/pramit-saha/oxsurgemma-ssg-vqa) |
| Critical View of Safety (ResNet18, ResNet50, DenseNet, EfficientNet) | [pramit-saha/oxsurgemma-cvs](https://huggingface.co/pramit-saha/oxsurgemma-cvs) |
| Surgical Instrument and Critical Anatomy Detection (YOLOv8) | [pramit-saha/oxsurgemma-object-detection](https://huggingface.co/pramit-saha/oxsurgemma-object-detection) |
| Surgical Scene Segmentation (YOLOv8-seg) | [pramit-saha/oxsurgemma-scene-segmentation](https://huggingface.co/pramit-saha/oxsurgemma-scene-segmentation) |
| Surgical Phase Detection (ResNet50) | [pramit-saha/oxsurgemma-phase-detection](https://huggingface.co/pramit-saha/oxsurgemma-phase-detection) |
| Surgical Triplet Recognition (ResNet50) | [pramit-saha/oxsurgemma-cholect50-heads](https://huggingface.co/pramit-saha/oxsurgemma-cholect50-heads) |
| Surgical Monitoring Module (ResNet50) | [pramit-saha/oxsurgemma-frame-attributes](https://huggingface.co/pramit-saha/oxsurgemma-frame-attributes) |


## Quick start

```bash

# 1. Clone OxSurGemma and MedRAX (agent framework)
git clone https://github.com/PramitSaha/OxSurGemma.git
cd OxSurGemma

# 2. Install OxSurGemma dependencies (from repo root)
pip install -e .

# 3. Environment (for local LLM: Hugging Face token)
cp .env.example .env
set HF_TOKEN=... (accept MedGemma terms at huggingface.co first)

# 4. Run (uses MedGemma 4B + tool-use LoRA by default)
./run.sh                      # terminal chat
./run.sh --gradio --port 8585 # web UI
# Or: python -m surgical_copilot.main [--gradio] [--port 8585]
```

See [Setup](#setup) and [Running](#running) below for details.

## System overview

The system uses a **LLM-based agent** (LangGraph + LangChain) to orchestrate surgical analysis tools. A user query (text or voice) is routed to the appropriate tool(s) via an LLM. **By default the LLM is local:** MedGemma (4B or 27B) with a **tool-use LoRA adapter** from `tool_use_lora_checkpoints/`. The user can download the Huggingface model weights and place them in corresponding folders as mentioned below:

### Tools and models

| Tool | Model | Architecture | Expected file location |
|------|-------|---------------|---------|
| **Phase Detection** | ResNet50 | 8 phases | `phase_detection_workflow/best_phase.pt` |
| **Scene Segmentation** | YOLOv8-seg | 13 classes | `scene_segmentation_utils/runs/best.pt` |
| **Critical View of Safety** | ColeNet (ResNet18 or ensemble) | 3 CVS criteria | `cvs_models/log/best_model.pth` |
| **Frame Attributes** | ResNet50 + dual heads | 4 operators + 10 conditions | `frame_attributes_tasks/cholec20_multilabel_checkpoints/best_cholec20_multilabel.pt` |
| **Triplet Recognition** | ResNet50 (3 heads) + YOLOv8 | Tool/verb/target + 100-class triplet | `instrument_triplet_tasks/cholect50_checkpoints/best_*.pt` |
| **Object Detection** | YOLOv8 | 13 classes (anatomy + instruments) | `object_detection/best_detector_balanced.pt` |
| **Instrument Tracking** | YOLOv8 | 6 or 7 tools | `instrument_triplet_tasks/runs/tool/best.pt` |
| **SSG VQA** | MedGemma-4B + LoRA | Visual QA | `ssg_vqa_finetuning/checkpoint-7400/`|
| **RAG Retrieval** | ChromaDB + embeddings | Textbook retrieval | `surgical_rag/data/rag_index/` |
| **Speech I/O** | Whisper / TTS | ASR and synthesis | Downloaded at runtime |

## Project layout

The repo is organised so that the **core app** (`surgical_copilot/`), **model weights**, and **pipelines** (e.g. `surgical_rag/`) are clearly separated. Optional tools are excluded at startup if their checkpoints are missing.

### Directory structure

```
OxSurGemma/
├── run.sh                      # Launcher: ./run.sh [--gradio] [--port 8585]
├── .env.example                # Copy to .env and set HF_TOKEN or OPENAI_API_KEY
├── pyproject.toml              # pip install -e . from repo root
├── requirements.txt            # Python dependencies (root)
│
├── surgical_copilot/           # Main package (entrypoint: main.py)
│   ├── main.py                 # Entrypoint: CLI chat or Gradio UI
│   ├── config.py               # Config: LLM backend, paths, tool-use LoRA
│   ├── registry.py             # Tool registry
│   ├── llm_local.py            # Local MedGemma + LoRA adapter loading
│   ├── parallel_tools.py       # Parallel tool execution (Gradio)
│   ├── gradio_demo.py          # Gradio web UI
│   ├── tools/                  # Tool implementations
│   │   ├── phase_detection.py, scene_segmentation.py, critical_view_of_safety.py
│   │   ├── frame_attributes.py, triplet_recognition.py, object_detection_merged.py
│   │   ├── instrument_tracking.py, ssg_vqa.py, rag_retrieval.py, base.py
│   ├── audio/                  # STT / TTS (Gradio)
│   └── docs/system_prompts.txt # Agent system prompts
│
│
├── tool_use_lora_checkpoints/  # MedGemma LoRA for tool-use routing (default LLM)
│   ├── medgemma-4b-tool-use-lora/
│   └── medgemma-27b-tool-use-lora/
│
├── phase_detection_workflow/   # Phase detection
│   ├── best_phase.pt           # Standalone ResNet50 (8 phases)
│   └── workflow_codes/         # M2CAI16 alternative (optional)
│
├── scene_segmentation_utils/   # Scene segmentation
│   ├── cholecseg8k_yolov8.py   # CLASS_NAMES (required at runtime)
│   └── runs/best.pt
│
├── frame_attributes_tasks/     # Frame attributes (inference only)
│   ├── cholec20_model.py
│   └── cholec20_multilabel_checkpoints/best_cholec20_multilabel.pt
│
├── instrument_triplet_tasks/   # Instrument/triplet (inference only)
│   ├── cholect50_model.py
│   ├── cholect50_checkpoints/  # best_tool.pt, best_verb.pt, best_target.pt
│   └── runs/                   # YOLO tool + triplet weights
│
├── object_detection/           # Merged object detector
│   └── best_detector_balanced.pt
│
├── cvs_models/                 # Critical View of Safety
│   ├── colenet/                # Model definition
│   ├── run_cvs_inference.py
│   └── log/                    # Weights (new_cvs_model_1/, colenet_*/)
│
├── ssg_vqa_finetuning/         # SSG VQA LoRA
│   └── surgical_vqa_sft_ssg/checkpoints/checkpoint-7400/
│
└── surgical_rag/               # RAG retrieval (runtime: code + index only)
    ├── __init__.py
    ├── rag_retrieval.py
    └── data/rag_index/         # ChromaDB index (build once)
```

## Setup

### 1. Clone MedRAX (required)

```bash
cd surgical_copilot_release
git clone https://github.com/bowang-lab/MedRAX.git
pip install -e MedRAX
```

### 2. Install Python dependencies

From the repository root:

```bash
pip install -e .
```

Key packages: `torch`, `transformers`, `peft`, `langchain`, `langgraph`, `gradio`, `chromadb`, `sentence-transformers`, `ultralytics`, etc. (see `requirements.txt` and `pyproject.toml`).

### 3. Environment variables

**Default (local LLM — MedGemma + tool-use LoRA):**

- `HF_TOKEN` — Hugging Face token (required for gated MedGemma). Accept the model terms at [huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it), then `huggingface-cli login` or `export HF_TOKEN=hf_...`.
- Optional: `TRANSFORMERS_CACHE` or `HF_HOME` — cache directory for downloads (default: repo `hf_cache/`).

## Running

All commands from the **repository root**.

### Default: terminal chat (MedGemma 4B + tool-use LoRA)

```bash
./run.sh
# or: python -m surgical_copilot.main
```

### Gradio web UI

```bash
./run.sh --gradio --port 8585
# or: python -m surgical_copilot.main --gradio --port 8585
```

### Choosing 4B vs 27B (local LLM)

By default the app uses **MedGemma 4B** and the **4B tool-use LoRA** from `tool_use_lora_checkpoints/`. From the command line you can switch to 27B:

```bash
# 4B (default)
./run.sh --gradio
python -m surgical_copilot.main --medgemma 4b

# 27B (needs more VRAM)
python -m surgical_copilot.main --gradio --medgemma 27b
```

### Using OpenAI instead of local MedGemma

```bash
export SURGICAL_COPILOT_LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
./run.sh --gradio
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--gradio` | Launch Gradio web UI (default: terminal chat) |
| `--medgemma 4b` | Use MedGemma 4B + 4B tool-use LoRA (default) |
| `--medgemma 27b` | Use MedGemma 27B + 27B tool-use LoRA |
| `--port PORT` | Gradio server port (default: 8585) |
| `--no-share` | Disable Gradio share link |
| `--hf-cache PATH` | Hugging Face cache directory |
| `--cvs` | Use single ResNet18 CVS model (default: 4-model ensemble) |

## RAG index

The RAG tool uses a pre-built ChromaDB index under `surgical_rag/data/rag_index/`. If that folder exists and contains `chroma.sqlite3`, the tool is enabled. To rebuild the index you need the full surgical_rag pipeline (not included in this minimal runtime layout); the release assumes the index is already built.
