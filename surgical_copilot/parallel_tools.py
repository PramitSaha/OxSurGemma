"""
Run all tools in parallel on the current frame, then get a single LLM answer.
Use this mode to hide individual tool outputs and show only the synthesized answer.
On Pause: run all tools except VQA and cache results; on question use cache for answer.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage


def _parse_overlay_path_from_text(text: str) -> Optional[str]:
    """Parse overlay path from tool output (e.g. 'Overlay path: /path/to/seg_overlay_123.png'). Returns path if file exists."""
    if not text:
        return None
    for prefix in ("Overlay path: ", "Overlay path:", "Overlay saved to: ", "Overlay saved to:"):
        idx = text.find(prefix)
        if idx == -1:
            continue
        start = idx + len(prefix)
        rest = text[start:].strip()
        path = rest.split("\n")[0].strip().rstrip(".,;")
        if path and len(path) > 3:
            if Path(path).exists():
                return path
            if Path(path.replace("\\", "/")).exists():
                return path.replace("\\", "/")
    return None

# Exclude these from preload on Pause (question-dependent; answer from cache instead)
TOOLS_TO_SKIP_ON_PAUSE = ("ssg_vqa", "rag_retrieval")

# Tool name -> (args builder from image_path, question) -> dict of args
def _tool_args(image_path: Optional[str], question: str) -> Dict[str, dict]:
    """Build minimal args for each tool. question used for VQA and RAG."""
    base = {"image_path": image_path, "path": image_path} if image_path else {}
    vqa = {**base, "question": question or "What is shown in this surgical image?"}
    return {
        "surgical_scene_segmentation": {**base},
        "phase_detection": {"video_or_frames_path": image_path or "", "image_path": image_path, **base},
        "instrument_tracking": {**base},
        "frame_attributes": {**base},
        "ssg_vqa": vqa,
        "triplet_recognition": {**base},
        "critical_view_of_safety": {**base},
        "rag_retrieval": {"query": question or "What are key steps and safety in cholecystectomy?"},
    }


def _run_one_tool(tools: Dict[str, Any], name: str, args: dict) -> tuple[str, str]:
    """Run one tool; return (name, result_string). Catches errors."""
    if name not in tools:
        return name, "(tool not loaded)"
    tool = tools[name]
    # Skip if args are missing required (e.g. no image for image-based tools)
    if not args:
        return name, "(no args)"
    if tool.name == "rag_retrieval":
        if not args.get("query"):
            return name, "(no query)"
    elif tool.name in ("phase_detection", "surgical_scene_segmentation", "instrument_tracking",
                       "frame_attributes", "ssg_vqa", "triplet_recognition", "critical_view_of_safety"):
        if not args.get("image_path") and not args.get("video_or_frames_path") and not args.get("path"):
            return name, "(no image)"
    try:
        out = str(tool.invoke(args))
        # Keep enough of overlay tools' output so "Overlay path: ..." is not truncated
        cap = 3500 if tool.name in ("surgical_scene_segmentation", "object_detection_merged") else 2000
        return name, out[:cap]
    except Exception as e:
        return name, f"(error: {e})"


def run_all_tools_parallel(
    tools: Dict[str, Any],
    image_path: Optional[str],
    question: str,
) -> Dict[str, str]:
    """Run all tools in parallel. question used for VQA and RAG. Returns tool_name -> result text."""
    arg_map = _tool_args(image_path, question)
    return _run_tools_from_arg_map(tools, arg_map)


def _run_tools_from_arg_map(tools: Dict[str, Any], arg_map: Dict[str, dict]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    lock = threading.Lock()

    def run(name: str, args: dict) -> None:
        n, text = _run_one_tool(tools, name, args)
        with lock:
            results[n] = text

    threads = []
    for name, args in arg_map.items():
        if name not in tools:
            continue
        t = threading.Thread(target=run, args=(name, args))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return results


def _tool_args_no_vqa_no_rag(image_path: Optional[str]) -> Dict[str, dict]:
    """Args for tools run on Pause only (no VQA, no RAG). Used to preload/cache."""
    base = {"image_path": image_path, "path": image_path} if image_path else {}
    return {
        "surgical_scene_segmentation": {**base},
        "phase_detection": {"video_or_frames_path": image_path or "", "image_path": image_path, **base} if image_path else {},
        "instrument_tracking": {**base},
        "frame_attributes": {**base},
        "triplet_recognition": {**base},
        "critical_view_of_safety": {**base},
    }


def _agent_tools_dict(agent: Any) -> Dict[str, Any]:
    """Get agent tools as a name -> tool dict (handles list or dict)."""
    raw = getattr(agent, "tools", None) or {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return {getattr(t, "name", None): t for t in raw if getattr(t, "name", None)}
    if hasattr(raw, "items"):
        return dict(raw)
    return {}


def run_tools_parallel_no_vqa(agent: Any, image_path: Optional[str]) -> Dict[str, str]:
    """
    Run all tools in parallel except VQA and RAG (ssg_vqa, rag_retrieval).
    Call on Pause to load models and cache results; then answer from cache when user asks.
    """
    if not image_path or not Path(image_path).exists():
        return {}
    arg_map = _tool_args_no_vqa_no_rag(image_path)
    all_tools = _agent_tools_dict(agent)
    tools = {k: v for k, v in all_tools.items() if k not in TOOLS_TO_SKIP_ON_PAUSE}
    return _run_tools_from_arg_map(tools, {k: v for k, v in arg_map.items() if k in tools})


def format_tool_results(results: Dict[str, str]) -> str:
    """Format tool results for the LLM context."""
    lines = []
    for name, text in sorted(results.items()):
        snippet = (text.strip() or "(no output)")[:1500]
        lines.append(f"- **{name}**: {snippet}")
    return "\n".join(lines)


def get_single_answer(
    raw_model: Any,
    system_prompt: str,
    tool_results: Dict[str, str],
    user_question: str,
    image_path: Optional[str] = None,
) -> str:
    """
    Invoke the raw LLM (no tools) with tool results + user question; return the answer text.
    """
    formatted = format_tool_results(tool_results)
    q_lower = (user_question or "").lower()
    is_cvs_question = "critical_view_of_safety" in tool_results and any(w in q_lower for w in ["safe", "cvs", "cut"])
    if is_cvs_question:
        answer_instruction = (
            "You MUST include in your answer: (1) All three CVS criteria from the critical_view_of_safety result—"
            "two structures visible, cystic plate dissected, hepatocystic triangle cleared—"
            "with achieved/not achieved and their scores. (2) A clear explanation of why it is or is not safe to cut "
            "based on these details. Do not give only a brief summary; state the explicit criteria and your reasoning."
        )
    else:
        answer_instruction = (
            "Provide a single concise answer based only on the tool results above. Do not repeat raw tool output."
        )
    user_content = (
        "The following are results from all surgical tools run in parallel on the current frame.\n\n"
        "Tool results:\n" + formatted + "\n\n"
        "User question: " + (user_question or "Summarize what you see and any relevant findings.") + "\n\n"
        + answer_instruction
    )
    messages = [HumanMessage(content=user_content)]
    if system_prompt:
        messages = [SystemMessage(content=system_prompt)] + messages
    response = raw_model.invoke(messages)
    return (getattr(response, "content", None) or str(response) or "").strip()


def run_parallel_tools_and_answer(
    agent: Any,
    image_path: Optional[str],
    question: str,
) -> Tuple[str, Optional[str]]:
    """
    Run all tools in parallel, then get one synthesized answer. Use when you want
    to hide individual tool outputs and show only the final answer.
    Returns (answer_text, overlay_path). overlay_path is set if segmentation or object_detection produced an overlay (for UI to display).
    """
    raw = getattr(agent, "_raw_model", None)
    if not raw:
        return "[Error: parallel answer mode requires agent._raw_model]", None
    results = run_all_tools_parallel(agent.tools, image_path, question)
    overlay_path = None
    for tool_name in ("surgical_scene_segmentation", "object_detection_merged"):
        if tool_name in results:
            overlay_path = _parse_overlay_path_from_text(results[tool_name])
            if overlay_path:
                break
    answer = get_single_answer(
        raw,
        getattr(agent, "system_prompt", "") or "",
        results,
        question,
        image_path,
    )
    return answer, overlay_path


def answer_from_cached_results(agent: Any, cached_results: Dict[str, str], user_question: str) -> str:
    """
    Produce a single answer from precomputed tool results (e.g. from Pause).
    Does not call any tools; uses the raw LLM with cached results.
    """
    raw = getattr(agent, "_raw_model", None)
    if not raw:
        return "[Error: answer_from_cached requires agent._raw_model]"
    if not cached_results:
        return "[No cached tool results. Pause on a frame first.]"
    return get_single_answer(
        raw,
        getattr(agent, "system_prompt", "") or "",
        cached_results,
        user_question or "Summarize the findings.",
        None,
    )
