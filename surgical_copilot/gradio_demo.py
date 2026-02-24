import gradio as gr
from gradio import ChatMessage
import cv2
import time
import re
import os
import tempfile
import threading
import asyncio
import queue
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
from langchain_core.messages import HumanMessage

# --- 1. HELPER FUNCTIONS ---

def _convert_to_mp4(input_path: str) -> str:
    """Convert to H.264 MP4 only when needed. Skip for .mp4; for other formats use original if OpenCV can read it."""
    path_obj = Path(input_path)
    # Already MP4: use as-is (no re-encode; OpenCV and frame extraction work fine)
    if path_obj.suffix.lower() == ".mp4":
        return str(path_obj)
    # If OpenCV can read it, use original and avoid slow FFmpeg re-encode
    if _video_duration_sec(str(path_obj)) is not None:
        return str(path_obj)
    output_path = path_obj.parent / f"{path_obj.stem}_web.mp4"
    if output_path.exists():
        return str(output_path)
    print(f"[FFmpeg] Converting {input_path} to H.264 (format not readable by OpenCV)...", flush=True)
    command = [
        "ffmpeg", "-y", "-i", str(input_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(output_path)
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return str(output_path)
    except Exception:
        return input_path

def _video_duration_sec(path: str) -> Optional[float]:
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return float(frame_count) / float(fps) if frame_count > 0 else None
    except Exception: return None

def _extract_frame(video_path: str, time_sec: float, out_dir: Path) -> Optional[str]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None: return None
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"frame_{Path(video_path).stem}_{int(time_sec*1000)}.png"
        cv2.imwrite(str(out_path), frame)
        return str(out_path) if out_path.exists() else None
    except Exception: return None

def _parse_segment_object_name(message: str) -> Optional[str]:
    """Extract requested structure from segment-like message, e.g. 'segment the liver' or 'draw the boundaries of the gallbladder' -> object name."""
    if not message or not isinstance(message, str):
        return None
    q = message.strip().lower()
    prefixes = (
        "segment the ", "segment ", "segmentation the ", "segmentation ", "segment only the ", "segment only ",
        "draw the boundaries of ", "draw boundaries of ", "draw the boundary of ", "draw boundary of ",
        "draw the edges of ", "draw edges of ", "draw the edge of ", "outline the ", "outline ",
        "show boundaries of ", "mark boundaries of ", "trace boundaries of ", "trace the ",
        "where is the ", "where is ", "where's the ", "where's ", "where are the ", "where are ",
    )
    for prefix in prefixes:
        if q.startswith(prefix):
            obj = q[len(prefix):].strip()
            # Take first phrase (stop at comma, period, question mark, or end)
            obj = obj.split(",")[0].split(".")[0].rstrip("?").strip()
            if obj and len(obj) < 50:
                return obj if obj else None
    return None


def _message_content_to_str(content: Any) -> str:
    """Get plain text from message content (string or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts) if parts else str(content)
    return str(content)


def _parse_overlay_path(tool_content: str) -> Optional[str]:
    """Parse overlay path from tool output or LLM reply (e.g. 'Overlay path: ...' or 'at the provided path: ...')."""
    text = _message_content_to_str(tool_content) if not isinstance(tool_content, str) else tool_content
    for prefix in (
        "Overlay path: ", "Overlay saved to: ", "Overlay path:", "Overlay saved to:",
        "at the provided path: ", "provided path: ", "at the provided path:", "provided path:",
    ):
        p = prefix if prefix.endswith(" ") else (prefix + " ")
        if p in text or (prefix in text and not prefix.endswith(" ")):
            start = text.index(prefix) + len(prefix)
            rest = text[start:].strip()
            path = rest.split("\n")[0].strip().rstrip(".,;)\"']")
            if path and len(path) > 3:
                if Path(path).exists():
                    return path
                if Path(path.replace("\\", "/")).exists():
                    return path.replace("\\", "/")
    return None

# Play/pause state for frame auto-advance (scrub)
_video_playing = False
_PLAY_STEP_SEC = 0.5   # real seconds between frame updates
_PLAY_SPEED = 2.0      # 2x playback speed (video time advance per step)

# --- 2. CHAT INTERFACE CLASS ---

class ChatInterface:
    def __init__(self, agent: Any, tools_dict: dict):
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.current_thread_id: Optional[str] = None
        self.current_image_path: Optional[str] = None

    def handle_upload(self, file_path: Optional[str]) -> Optional[str]:
        if not file_path: return None
        self.current_image_path = file_path
        return file_path

    def add_message(self, message: str, selected_frame_path: Optional[str], history: List[ChatMessage]) -> Tuple[List[ChatMessage], gr.Textbox, str]:
        history = history or []
        if selected_frame_path and Path(selected_frame_path).exists():
            history.append(ChatMessage(role="user", content={"path": selected_frame_path}))
        if message and message.strip():
            history.append(ChatMessage(role="user", content=message.strip()))
        return history, gr.Textbox(value="", interactive=False), (message or "").strip()

    def _route_tool(self, question: str) -> Optional[str]:
        q = (question or "").lower().strip()
        if not q: return None
        # "Where is X?" / "Where's the X?" → run both segmentation and object detection
        if any(phrase in q for phrase in ["where is the ", "where is ", "where's the ", "where's ", "where are the ", "where are "]):
            return "where_location"
        # "Right/correct tools for this phase?" → need both phase and instruments (no single tool)
        if any(w in q for w in ["right tool", "right tools", "correct tool", "correct tools", "right instrument", "right instruments", "correct instrument", "correct instruments"]) and any(w in q for w in ["phase", "this phase", "current phase"]):
            return None
        # Describe the scene / this image / what you see → VQA (not RAG)
        if any(phrase in q for phrase in ["describe the scene", "describe this", "describe what you see", "describe the image", "describe the frame", "describe what's", "describe what is in"]):
            return None  # fall through to ssg_vqa or agent choice
        # Textbook/guideline/procedure + describe/explain (procedure, not scene) → RAG
        if any(w in q for w in ["guideline", "textbook", "procedure", "how do i", "what are the steps", "what are all the steps", "steps of", "critical view", "cholecystectomy", "describe", "explain"]): return "rag_retrieval"
        # Object detection (bounding boxes): must come BEFORE segmentation so "detect structures" etc. use detector not segmenter
        if any(phrase in q for phrase in [
            "detect object", "detect objects", "detect the object", "detect the objects",
            "detect structure", "detect structures", "detect the structure", "detect the structures",
            "object detection", "run object detection", "do object detection",
            "what objects", "which objects", "show objects", "find objects",
            "detect anatomy", "detect anatomy and", "detect anatomy and instruments",
        ]): return "object_detection_merged"
        if "detect" in q and any(w in q for w in ["structure", "object", "anatomy", "instrument", "element"]): return "object_detection_merged"
        if any(w in q for w in ["instrument", "tool"]): return "instrument_tracking"
        # Step/stage/process → agent uses RAG (procedure) + triplet_recognition + optionally phase_detection (no single tool forced)
        if any(w in q for w in ["step", "stage", "process"]):
            return None
        # "what are all the phases?" / "list of phases" → answer from knowledge (no tool); current-phase questions → phase_detection
        if any(w in q for w in ["phase"]):
            if any(phrase in q for phrase in ["all the phase", "all phase", "list of phase", "how many phase", "what phase exist", "name the phase"]):
                return None  # agent answers from system prompt (full list), do not force phase_detection
            return "phase_detection"
        # Segmentation: segment, draw boundaries/edges, outline, etc.
        if any(w in q for w in ["segment", "anatomy", "segmentation", "structures", "what structures", "what anatomy", "draw boundary", "draw boundaries", "draw the boundary", "draw the boundaries", "draw edge", "draw edges", "outline", "outline the", "show boundaries", "mark boundaries", "trace boundaries", "trace the"]):
            return "surgical_scene_segmentation"
        if any(w in q for w in ["safe", "cvs"]): return "critical_view_of_safety"
        if any(w in q for w in ["occlusion", "blood"]): return "frame_attributes"
        if any(w in q for w in ["who", "action"]): return "triplet_recognition"
        return "ssg_vqa"

    def _extract_text_from_chat_message(self, m: ChatMessage) -> Optional[str]:
        """Get plain text from a Gradio ChatMessage (skip image-only entries)."""
        content = getattr(m, "content", None)
        if content is None:
            return None
        if isinstance(content, dict) and content.get("path"):
            return None  # image-only user message
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text", "").strip()
                    if t:
                        return t
            return None
        return str(content).strip() or None

    def _build_conversation_context(self, chat_history: List[ChatMessage], exclude_last_user: bool = True) -> str:
        """Build a short prior-conversation summary so the agent can relate follow-ups (e.g. 'and the phase?') to the last few Q&A."""
        if not chat_history or len(chat_history) < 2:
            return ""
        msgs = chat_history[:-1] if (exclude_last_user and getattr(chat_history[-1], "role", None) == "user") else chat_history
        if len(msgs) < 2:
            return ""
        recent = []
        for m in msgs[-8:]:  # last 4 exchanges
            role = getattr(m, "role", None)
            text = self._extract_text_from_chat_message(m)
            if not text:
                continue
            text = text[:400] + ("..." if len(text) > 400 else "")
            if role == "user":
                recent.append(f"User: {text}")
            elif role == "assistant":
                recent.append(f"Assistant: {text}")
        if not recent:
            return ""
        return "Recent conversation (use for follow-ups; same image/frame applies):\n" + "\n".join(recent) + "\n\n"

    def _build_user_content(self, message: str, image_path: Optional[str], chat_history: List[ChatMessage]) -> str:
        parts = []
        # Include last few Q&A so the agent can answer follow-ups like "and the phase?", "what about instruments?"
        prior = self._build_conversation_context(chat_history or [])
        if prior:
            parts.append(prior)
        if message:
            q = message.strip()
            tool = self._route_tool(q)
            if tool:
                parts.append(f"You MUST call this tool only: {tool}.")
                if tool == "critical_view_of_safety":
                    parts.append(
                        "In your response you MUST: (1) State all three CVS criteria from the tool output—"
                        "two structures visible, cystic plate dissected, hepatocystic triangle cleared—"
                        "with achieved/not achieved and their scores. (2) Explain clearly why it is or is not safe to cut "
                        "based on these details. Do not give a brief summary; give the explicit criteria and your reasoning."
                    )
            parts.append(f"User question: {q}")
        if image_path and Path(image_path).exists():
            parts.append(f"image_path: {image_path}")
        return "\n\n".join(parts) if parts else ""

    async def process_message(
        self,
        message: str,
        selected_frame_path: Optional[str],
        chat_history: List[ChatMessage],
        current_overlay_path: Optional[str] = None,
        cached_tool_results: Optional[dict] = None,
        cached_tool_results_image_path: Optional[str] = None,
    ):
        chat_history = chat_history or []
        if not self.current_thread_id: self.current_thread_id = str(time.time())

        # Don't touch the video in this chain — yield display_update=None so the frame stays intact (no loading, no clear)
        yield chat_history, selected_frame_path, "", None, None

        # "Remove overlay" — clear overlay and show original frame without calling the agent
        q = (message or "").strip().lower()
        if q in ("remove overlay", "remove the overlay", "clear overlay", "remove overlay please", "clear the overlay"):
            chat_history.append(ChatMessage(role="assistant", content="Overlay removed."))
            base_img = selected_frame_path if selected_frame_path and Path(selected_frame_path).exists() else None
            yield chat_history, selected_frame_path, "", None, (base_img, None)
            return

        image_path = self.current_image_path or selected_frame_path
        has_image = image_path and Path(image_path).exists()

        route = self._route_tool((message or "").strip())

        # "Where is X?" — run both segmentation and object detection; show segmentation overlay and synthesize both in chat
        if route == "where_location" and has_image and image_path:
            object_name = _parse_segment_object_name((message or "").strip())
            seg_tool = self.tools_dict.get("surgical_scene_segmentation")
            if not seg_tool and hasattr(self.agent, "tools"):
                agent_tools = self.agent.tools if isinstance(self.agent.tools, dict) else {getattr(t, "name", None): t for t in (self.agent.tools or []) if getattr(t, "name", None)}
                seg_tool = agent_tools.get("surgical_scene_segmentation")
            if not seg_tool:
                try:
                    from surgical_copilot.registry import get_tools
                    seg_list = get_tools(tools_to_use=["surgical_scene_segmentation"])
                    seg_tool = seg_list[0] if seg_list else None
                except Exception:
                    seg_tool = None
            det_tool = self.tools_dict.get("object_detection_merged")
            if not det_tool and hasattr(self.agent, "tools"):
                agent_tools = self.agent.tools if isinstance(self.agent.tools, dict) else {getattr(t, "name", None): t for t in (self.agent.tools or []) if getattr(t, "name", None)}
                det_tool = agent_tools.get("object_detection_merged")
            if not det_tool:
                try:
                    from surgical_copilot.registry import get_tools
                    det_list = get_tools(tools_to_use=["object_detection_merged"])
                    det_tool = det_list[0] if det_list else None
                except Exception:
                    det_tool = None
            if not seg_tool and not det_tool:
                chat_history.append(ChatMessage(role="assistant", content="Segmentation and object detection are not available. Please ensure the required model weights are in place."))
                yield chat_history, selected_frame_path, "", None, None
                return
            seg_out, det_out = None, None
            seg_err, det_err = None, None

            def _run_where_seg():
                nonlocal seg_out, seg_err
                if not seg_tool:
                    return
                try:
                    seg_args = {"image_path": image_path}
                    if object_name:
                        seg_args["object_name"] = object_name
                    seg_out = seg_tool.invoke(seg_args)
                except Exception as e:
                    seg_err = e

            def _run_where_det():
                nonlocal det_out, det_err
                if not det_tool:
                    return
                try:
                    det_out = det_tool.invoke({"image_path": image_path})
                except Exception as e:
                    det_err = e

            t1 = threading.Thread(target=_run_where_seg, daemon=True)
            t2 = threading.Thread(target=_run_where_det, daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            overlay_path = None
            if seg_out:
                overlay_path = _parse_overlay_path(str(seg_out))
                if overlay_path and not Path(overlay_path).exists():
                    overlay_path = None
            parts = []
            if seg_err:
                parts.append(f"Segmentation error: {seg_err}")
            elif seg_out:
                raw = str(seg_out).split(" Overlay path:")[0] if " Overlay path:" in str(seg_out) else str(seg_out)
                if "Structures:" in raw:
                    structures_part = raw[raw.index("Structures:") + len("Structures:"):].strip().rstrip(".")
                    names = [n.strip() for n in structures_part.split(",") if n.strip()][:8]
                    parts.append("Segmentation: " + (", ".join(names) if names else "structures detected in the frame"))
                else:
                    parts.append("Segmentation completed for the frame.")
            if det_err:
                parts.append(f"Object detection error: {det_err}")
            elif det_out:
                raw = str(det_out).split("Overlay path:")[0] if "Overlay path:" in str(det_out) else str(det_out)
                parts.append("Object detection: " + raw.strip() if raw.strip() else "Object detection completed.")
            summary = " ".join(parts) if parts else "Location: I ran segmentation and object detection on this frame."
            if object_name:
                summary = f"For \"{object_name}\": " + summary
            chat_history.append(ChatMessage(role="assistant", content=summary))
            yield chat_history, selected_frame_path, "", None, (overlay_path, overlay_path) if overlay_path else None
            return

        # "Segment" / anatomy: always run segmentation and show overlay (don't use cache so user gets the overlay)
        if route == "surgical_scene_segmentation" and has_image and image_path:
            seg_tool = self.tools_dict.get("surgical_scene_segmentation")
            if not seg_tool and hasattr(self.agent, "tools"):
                agent_tools = self.agent.tools if isinstance(self.agent.tools, dict) else {getattr(t, "name", None): t for t in (self.agent.tools or []) if getattr(t, "name", None)}
                seg_tool = agent_tools.get("surgical_scene_segmentation")
            if not seg_tool:
                try:
                    from surgical_copilot.registry import get_tools
                    seg_list = get_tools(tools_to_use=["surgical_scene_segmentation"])
                    seg_tool = seg_list[0] if seg_list else None
                except Exception:
                    seg_tool = None
            if seg_tool:
                # Optional object filter, e.g. "segment the liver" -> object_name="liver"
                object_name = _parse_segment_object_name((message or "").strip())
                seg_args = {"image_path": image_path}
                if object_name:
                    seg_args["object_name"] = object_name
                frame_for_display = selected_frame_path if (selected_frame_path and Path(selected_frame_path).exists()) else image_path
                result_holder = []
                def _run_seg():
                    try:
                        out = seg_tool.invoke(seg_args)
                        result_holder.append(("ok", out))
                    except Exception as e:
                        result_holder.append(("err", e))
                worker = threading.Thread(target=_run_seg, daemon=True)
                worker.start()
                worker.join()
                if not result_holder:
                    yield chat_history, selected_frame_path, "", None, None
                    return
                status, out = result_holder[0]
                if status == "err":
                    chat_history.append(ChatMessage(role="assistant", content=f"Segmentation error: {out}"))
                    yield chat_history, selected_frame_path, "", None, None
                    return
                overlay_path = _parse_overlay_path(str(out))
                if overlay_path and Path(overlay_path).exists():
                    if object_name:
                        summary = f"I've segmented the {object_name}. The overlay highlights it in the frame."
                    else:
                        raw = str(out).split(" Overlay path:")[0] if " Overlay path:" in str(out) else str(out)
                        if "Structures:" in raw:
                            structures_part = raw[raw.index("Structures:") + len("Structures:"):].strip().rstrip(".")
                            names = [n.strip() for n in structures_part.split(",") if n.strip()][:8]
                            summary = "I've segmented the scene. The overlay highlights: " + ", ".join(names) + "." if names else "Segmentation complete. The overlay shows the detected anatomical structures and instruments in the frame."
                        else:
                            summary = "Segmentation complete. The overlay shows the detected anatomical structures and instruments in the frame."
                    chat_history.append(ChatMessage(role="assistant", content=summary))
                    yield chat_history, selected_frame_path, "", None, (overlay_path, overlay_path)
                else:
                    chat_history.append(ChatMessage(role="assistant", content="Segmentation ran but the overlay could not be displayed. Try again or ask to segment a specific structure."))
                    yield chat_history, selected_frame_path, "", None, None
                return

        # Object detection: run object_detection_merged directly and show overlay (same pattern as segmentation)
        if route == "object_detection_merged" and has_image and image_path:
            det_tool = self.tools_dict.get("object_detection_merged")
            if not det_tool and hasattr(self.agent, "tools"):
                agent_tools = self.agent.tools if isinstance(self.agent.tools, dict) else {getattr(t, "name", None): t for t in (self.agent.tools or []) if getattr(t, "name", None)}
                det_tool = agent_tools.get("object_detection_merged")
            if not det_tool:
                try:
                    from surgical_copilot.registry import get_tools
                    det_list = get_tools(tools_to_use=["object_detection_merged"])
                    det_tool = det_list[0] if det_list else None
                except Exception:
                    det_tool = None
            if det_tool:
                result_holder = []
                def _run_det():
                    try:
                        out = det_tool.invoke({"image_path": image_path})
                        result_holder.append(("ok", out))
                    except Exception as e:
                        result_holder.append(("err", e))
                worker = threading.Thread(target=_run_det, daemon=True)
                worker.start()
                worker.join()
                if not result_holder:
                    yield chat_history, selected_frame_path, "", None, None
                    return
                status, out = result_holder[0]
                if status == "err":
                    chat_history.append(ChatMessage(role="assistant", content=f"Object detection error: {out}"))
                    yield chat_history, selected_frame_path, "", None, None
                    return
                overlay_path = _parse_overlay_path(str(out))
                if overlay_path and Path(overlay_path).exists():
                    raw = str(out).split("Overlay path:")[0] if "Overlay path:" in str(out) else str(out)
                    if "Anatomy:" in raw or "Tools/instruments:" in raw:
                        summary = "I've run object detection on this frame. The overlay shows the detected anatomy and instruments (bounding boxes)."
                    else:
                        summary = "Object detection complete. The overlay shows the detected anatomy and instruments in the frame."
                    chat_history.append(ChatMessage(role="assistant", content=summary))
                    yield chat_history, selected_frame_path, "", None, (overlay_path, overlay_path)
                else:
                    chat_history.append(ChatMessage(role="assistant", content=str(out).split("Overlay path:")[0].strip() if "Overlay path:" in str(out) else str(out)))
                    yield chat_history, selected_frame_path, "", None, None
                return
            # Route was object_detection_merged but tool not available — don't fall through to agent (which might call segmentation)
            chat_history.append(ChatMessage(role="assistant", content="Object detection is not available: merged detector weights (best_detector_balanced.pt) not found. Place the weights in the project root or object_detection/ and restart."))
            yield chat_history, selected_frame_path, "", None, None
            return

        # Prefer cached results (from Pause) when they match the current frame — no tool calls
        use_cache = (
            has_image
            and (message or "").strip()
            and cached_tool_results
            and cached_tool_results_image_path == image_path
        )
        if use_cache:
            yield chat_history, selected_frame_path, "", None, None
            try:
                from surgical_copilot.parallel_tools import answer_from_cached_results
                prior = self._build_conversation_context(chat_history)
                question_for_llm = (prior + "\nCurrent question: " + (message or "").strip()) if prior else (message or "").strip()
                answer = answer_from_cached_results(self.agent, cached_tool_results, question_for_llm)
                chat_history.append(ChatMessage(role="assistant", content=answer or "(No answer generated.)"))
                tts_path = None
                if answer:
                    try:
                        from surgical_copilot.audio.tts import text_to_speech
                        Path("temp/tts").mkdir(parents=True, exist_ok=True)
                        tts_path = text_to_speech(answer[:500], f"temp/tts/response_{int(time.time())}.mp3")
                    except Exception:
                        pass
                yield chat_history, selected_frame_path, "", tts_path, None
                return
            except Exception as e:
                chat_history.append(ChatMessage(role="assistant", content=f"Error: {e}"))
                yield chat_history, selected_frame_path, "", None, None
                return

        # Parallel mode: run all tools in parallel, show only the single synthesized answer (hide tool output)
        # Skip for segment and object-detection so we use agent stream and get overlay from execute chunk
        if has_image and (message or "").strip() and route not in ("surgical_scene_segmentation", "object_detection_merged", "where_location"):
            yield chat_history, selected_frame_path, "", None, None
            answer_holder = []
            error_holder = []
            # Include prior conversation so LLM can answer follow-ups (e.g. "and the instruments?", "what about the phase?")
            prior = self._build_conversation_context(chat_history)
            question_for_llm = (prior + "\nCurrent question: " + (message or "").strip()) if prior else (message or "").strip()

            def _run_parallel():
                try:
                    from surgical_copilot.parallel_tools import run_parallel_tools_and_answer
                    ans, overlay = run_parallel_tools_and_answer(self.agent, image_path, question_for_llm)
                    answer_holder.append((ans, overlay))
                except Exception as e:
                    error_holder.append(e)

            worker = threading.Thread(target=_run_parallel, daemon=True)
            worker.start()
            worker.join()  # wait without yielding to avoid blinking

            if error_holder:
                chat_history.append(ChatMessage(role="assistant", content=f"Error: {error_holder[0]}"))
                yield chat_history, selected_frame_path, "", None, None
                return
            answer, overlay_path = answer_holder[0] if answer_holder else (None, None)
            chat_history.append(ChatMessage(role="assistant", content=answer or "(No answer generated.)"))
            tts_path = None
            if answer:
                try:
                    from surgical_copilot.audio.tts import text_to_speech
                    Path("temp/tts").mkdir(parents=True, exist_ok=True)
                    tts_path = text_to_speech(answer[:500], f"temp/tts/response_{int(time.time())}.mp3")
                except Exception:
                    pass
            display_update = (overlay_path, overlay_path) if overlay_path and Path(overlay_path).exists() else None
            yield chat_history, selected_frame_path, "", tts_path, display_update
            return

        user_content = self._build_user_content(message, image_path, chat_history)
        if not user_content:
            yield chat_history, selected_frame_path, "", None, None
            return

        config_key = {"configurable": {"thread_id": self.current_thread_id}, "recursion_limit": 10}
        messages = [HumanMessage(content=user_content)]
        chunk_queue = queue.Queue()

        yield chat_history, selected_frame_path, "", None, None

        def _stream_producer():
            try:
                for chunk in self.agent.workflow.stream({"messages": messages}, config=config_key):
                    chunk_queue.put(chunk)
            except Exception as e:
                chunk_queue.put(("_error", str(e)))
            finally:
                chunk_queue.put(None)

        threading.Thread(target=_stream_producer, daemon=True).start()
        loop = asyncio.get_running_loop()

        last_assistant_text = ""
        stream_overlay_path = None
        try:
            while True:
                try:
                    chunk = await loop.run_in_executor(None, lambda: chunk_queue.get(timeout=0.4))
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                if isinstance(chunk, tuple) and chunk[0] == "_error":
                    chat_history.append(ChatMessage(role="assistant", content=f"Error: {chunk[1]}"))
                    yield chat_history, selected_frame_path, "", None, None
                    return
                if not isinstance(chunk, dict):
                    continue

                for node_name, state in chunk.items():
                    msgs = state.get("messages", [])
                    if not msgs:
                        continue
                    last = msgs[-1]
                    content = getattr(last, "content", "")

                    if node_name == "process" and isinstance(content, str):
                        content = re.sub(r"temp/[^\s]*", "", content).strip()
                        if content:
                            last_assistant_text = content
                            # Don't yield here — accumulate and yield once at end to avoid blinking
                    elif node_name == "execute":
                        overlay_path = None
                        for m in msgs:
                            c = getattr(m, "content", "")
                            c_str = _message_content_to_str(c)
                            overlay_path = _parse_overlay_path(c_str)
                            if overlay_path and Path(overlay_path).exists():
                                break
                        if overlay_path and Path(overlay_path).exists():
                            stream_overlay_path = overlay_path
                            yield chat_history, selected_frame_path, "", None, (overlay_path, overlay_path)
        except Exception as e:
            chat_history.append(ChatMessage(role="assistant", content=f"Error: {str(e)}"))
            yield chat_history, selected_frame_path, "", None, None
            return

        # Single chat update at end of stream — no per-chunk yields, so no blinking
        if last_assistant_text:
            chat_history.append(ChatMessage(role="assistant", content=last_assistant_text))
            # Fallback: if we never got overlay from execute chunk, parse path from assistant reply (e.g. "at the provided path: /path/to/seg_overlay_123.png")
            if stream_overlay_path is None:
                parsed = _parse_overlay_path(last_assistant_text)
                if parsed and Path(parsed).exists():
                    stream_overlay_path = parsed
                if stream_overlay_path is None:
                    # Match full path to seg_overlay or detection overlay file (path may follow "path:", "provided path:", or stand alone)
                    for pattern in (r"/\S*seg_overlay_\d+\.png", r"[^\s]*/seg_overlay_\d+\.png", r"/\S*detection_\d+_\S+\.png"):
                        match = re.search(pattern, last_assistant_text)
                        if match:
                            p = match.group(0).strip().rstrip(".,;)\"]")
                            if Path(p).exists():
                                stream_overlay_path = p
                                break

        tts_path = None
        if last_assistant_text:
            try:
                from surgical_copilot.audio.tts import text_to_speech
                tts_dir = Path("temp/tts")
                tts_dir.mkdir(parents=True, exist_ok=True)
                tts_path = text_to_speech(last_assistant_text[:500], str(tts_dir / f"response_{int(time.time())}.mp3"))
            except ImportError: pass
            except Exception: pass

        # Final yield: only send display_update when we have a new overlay; else None so frame stays intact
        display_update = (stream_overlay_path, stream_overlay_path) if stream_overlay_path and Path(stream_overlay_path).exists() else None
        yield chat_history, selected_frame_path, "", tts_path, display_update

# --- 3. UI CREATION FUNCTION ---

def create_demo(agent: Any, tools_dict: dict) -> Tuple[gr.Blocks, str]:
    interface = ChatInterface(agent, tools_dict)

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Roboto+Mono:wght@400;700&display=swap');
    html, body { height: 100%; margin: 0; overflow: auto; }
    body, .gradio-container { background-color: #050505 !important; color: #e0f0ff !important; font-family: 'Rajdhani', sans-serif !important; min-height: 100vh !important; }
    .gradio-container .container { min-height: calc(100vh - 16px) !important; padding-bottom: 8px !important; }
    .main-content-row { height: calc(100vh - 130px) !important; min-height: 600px !important; display: flex !important; align-items: stretch !important; }
    .video-column, .chat-column { min-height: 0 !important; }
    .chat-column { display: flex !important; flex-direction: column !important; height: 100% !important; }
    /* First child = frame-alert-group wrapper: do not grow (avoids white space below real-time monitoring) */
    .chat-column > div:first-child { flex: 0 0 auto !important; min-height: 0 !important; overflow: visible !important; display: flex !important; flex-direction: column !important; }
    /* Second child = chatbot wrapper: take remaining space */
    .chat-column > div:nth-child(2) { flex: 1 1 auto !important; min-height: 0 !important; overflow: auto !important; display: flex !important; flex-direction: column !important; }
    .chat-column .chat-window, .chat-column [class*="chatbot"] { flex: 1 1 auto !important; min-height: 0 !important; overflow: auto !important; }
    .chat-column .input-spacer { flex: 0 0 auto !important; margin-top: auto !important; margin-bottom: 8px !important; }
    .chat-column .gr-form { flex: 0 0 auto !important; }
    #frame-alert-box, #frame-alert-box *, .frame-alert-box, .frame-alert-box * { color: #ffffff !important; }
    .frame-alert-group { display: flex !important; flex-direction: column !important; gap: 0 !important; margin-bottom: 2px !important; }
    .frame-alert-group > * { margin: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; }
    .frame-alert-group + div { margin-top: 0 !important; padding-top: 0 !important; }
    .chat-column .frame-alert-group ~ * { margin-top: 0 !important; }
    .frame-alert-title { margin: 0 !important; padding: 0 0 2px 0 !important; font-size: 0.85em !important; color: rgba(0, 243, 255, 0.9) !important; font-weight: 600 !important; letter-spacing: 0.05em !important; line-height: 1.2 !important; text-transform: uppercase !important; }
    .frame-alert-group .frame-alert-title + * { margin-top: 0 !important; }
    .frame-alert-box { margin: 0 !important; padding: 10px 12px !important; font-size: 0.9em !important; min-height: 4em !important; max-height: 6em !important; overflow: auto !important; background: #0d1f35 !important; border: 1px solid rgba(0, 243, 255, 0.4) !important; border-radius: 4px !important; text-shadow: 0 0 1px #000, 0 1px 2px #000 !important; text-transform: uppercase !important; }
    /* Voice bar: compact row, half width each, fit side by side */
    #voice-bar-row { display: flex !important; width: 100% !important; gap: 6px !important; min-height: 0 !important; max-height: 112px !important; align-items: stretch !important; }
    #voice-bar-row > div { flex: 1 1 0 !important; min-width: 0 !important; max-width: 50% !important; overflow: hidden !important; }
    #voice-uplink, #tts-response { min-height: 0 !important; }
    #tts-response audio { max-height: 80px !important; }
    .frame-alert-box span:not([style*="color"]) { color: #ffffff !important; }
    .frame-alert-box span[style*="color"] { text-shadow: 0 0 1px #000, 0 1px 2px #000 !important; }
    .frame-alert-box span[style*="ff4444"], .frame-alert-box .frame-alarm { color: #ff4444 !important; }
    /* No slider/range inside chat column (avoids slider-in-slider from TTS audio) */
    .chat-column input[type=range], .chat-column [type=range] { display: none !important; }
    h1, h2, h3 { color: #00f3ff; text-shadow: 0 0 10px rgba(0, 243, 255, 0.5); font-family: 'Rajdhani', sans-serif; }
    #main-title { white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; font-size: min(2.75rem, 2.2vw) !important; line-height: 1.2 !important; margin: 0 !important; }
    #main-title span { color: #ffeb3b !important; text-shadow: 0 0 8px rgba(255, 235, 59, 0.6); }
    .glass-panel { background: rgba(10, 15, 30, 0.6) !important; backdrop-filter: blur(10px); border: 1px solid rgba(0, 243, 255, 0.2); box-shadow: 0 0 15px rgba(0, 243, 255, 0.05); border-radius: 4px; }
    .video-column { width: 100% !important; max-width: 100% !important; }
    .video-container { border: 1px solid #00f3ff; position: relative; width: 100% !important; max-width: 100% !important; min-height: 1150px; aspect-ratio: 16/9; display: flex !important; flex-direction: column !important; }
    .video-container > div { flex: 1 1 auto !important; min-height: 0 !important; width: 100% !important; overflow: hidden !important; }
    .video-container::before { content: "LIVE FEED // SIGNAL: STABLE"; position: absolute; top: -25px; left: 0; font-family: 'Roboto Mono', monospace; font-size: 0.7em; color: #00f3ff; }
    #video-main-display, #video-main-display > div, #video-main-display .contain, #video-main-display .image-container, .video-main-display, .video-main-display > div, .video-container .image-container { width: 100% !important; height: 100% !important; max-width: 100% !important; min-height: 950px !important; }
    #video-main-display img, .video-container img, .video-container .image-container img { width: 100% !important; height: 100% !important; object-fit: cover !important; display: block !important; }
    .video-container .upload-box, .video-container [data-testid="file"] { min-height: 950px !important; width: 100% !important; display: flex !important; align-items: center !important; justify-content: center !important; }
    .chat-window { background: rgba(0, 0, 0, 0.8) !important; border-left: 2px solid #00f3ff; font-family: 'Roboto Mono', monospace; font-size: 2em !important; }
    .message-user { color: #00f3ff !important; font-weight: bold; }
    textarea, input { background: #0a0f14 !important; border: 1px solid #333 !important; color: #00f3ff !important; font-family: 'Roboto Mono', monospace !important; }
    input[type=range] { filter: hue-rotate(180deg); }
    .input-spacer { margin-bottom: 8px !important; }
    /* Hide Gradio loading spinner so the video frame stays visible and no rotating logo in chat/input when submitting */
    .gradio-container .loading, .gradio-container [class*='loading'], .gradio-container .progress-bar, .gradio-container [class*='spinner'] { display: none !important; }
    .chat-column .loading, .chat-column [class*='loading'], .chat-column [class*='spinner'], .chat-column .progress-bar { display: none !important; }
    .chat-column .animate-spin, .chat-column [class*='animate'] { display: none !important; }
    .gr-box .loading, .gr-form .loading, [data-testid="chatbot"] .loading, [data-testid="chatbot"] [class*='spinner'] { display: none !important; visibility: hidden !important; }
    """

    with gr.Blocks(title="OxSurGemma - Laparoscopic Cholecystectomy AI-Copilot", css=custom_css) as demo:
        with gr.Row(elem_classes="glass-panel"):
            with gr.Column(scale=3): gr.HTML("<h1 id=\"main-title\"><span style=\"color: #ffeb3b;\">OxSurGemma</span> - A Surgical AI-Copilot for Laparoscopic Cholecystectomy</h1>")
            with gr.Column(scale=1): gr.HTML("<div style='text-align: right; color: #00f3ff; font-family: Roboto Mono;'>SYS: ONLINE <br> GPU: ACTIVE</div>")

        with gr.Row(elem_classes="main-content-row"):
            # Left: Media — one window: when empty shows upload; when loaded shows video/frame (no separate upload row)
            with gr.Column(scale=6, elem_classes="video-column"):
                with gr.Group(elem_classes="glass-panel video-container", elem_id="video-container"):
                    upload_file = gr.File(
                        label="Drop video or image here",
                        file_types=["video", "image"],
                        file_count="single",
                        elem_classes="upload-box",
                        height=1230,
                    )
                    main_display = gr.Image(
                        label="Surgical Feed — use slider below to change frame / overlay",
                        type="filepath",
                        height=1230,
                        show_label=False,
                        elem_id="video-main-display",
                        elem_classes=["video-main-display"],
                        visible=False,
                    )
                    video_player = gr.Video(label="Video", interactive=False, height=0, visible=False)
                
                with gr.Row():
                    time_slider = gr.Slider(minimum=0, maximum=100, value=0, step=0.1, label="FRAME SELECTOR // move to change the image above", interactive=True)
                with gr.Row():
                    play_btn = gr.Button("▶ Play", variant="primary")
                    pause_btn = gr.Button("⏸ Pause", variant="secondary")
                upload_status = gr.Markdown("WAITING FOR INPUT...", elem_classes="status-text")

            # Right: Chat
            with gr.Column(scale=4, elem_classes="glass-panel chat-column"):
                with gr.Group(elem_classes=["frame-alert-group"]):
                    frame_alert_title = gr.Markdown("**Real-time Surgery monitoring**", elem_id="frame-alert-title", elem_classes=["frame-alert-title"], visible=True)
                    frame_alert_box = gr.Markdown("", elem_id="frame-alert-box", elem_classes=["frame-alert-box"], visible=True)
                chatbot = gr.Chatbot([], elem_classes="chat-window", height=400, show_label=False, avatar_images=(None, None), render_markdown=True)
                
                # --- INPUT AREA (STACKED) ---
                
                # 1. Text Input (Top)
                txt = gr.Textbox(
                    show_label=False, 
                    placeholder="I am a surgical co-pilot, how can I assist?", 
                    max_lines=1, 
                    container=False,
                    elem_classes="input-spacer"
                )

                # 2. Voice bar: click mic to record, click again to stop → auto-sends
                with gr.Row(elem_id="voice-bar-row", equal_height=True):
                    with gr.Column(scale=1):
                        mic = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Voice — click mic to start, click again to stop (sends automatically)",
                            elem_id="voice-uplink",
                        )
                    with gr.Column(scale=1):
                        tts_audio = gr.Audio(label="Response (voice)", autoplay=True, visible=True, elem_id="tts-response")

        video_path_state = gr.State()
        selected_frame_path = gr.State()
        overlay_state = gr.State(None)  # path to current overlay image, or None
        display_update_state = gr.State(None)  # None or (main_display_path, overlay_path) — only set when segmentation/remove overlay
        msg_state = gr.State("")
        cached_tool_results = gr.State(None)  # dict of tool_name -> result (set on Pause, no VQA)
        cached_tool_results_image_path = gr.State(None)

        # Conditions that trigger a red ALARM in the frame alert box
        _FRAME_ALARM_ATTRS = ("occluded", "smoke", "bleeding", "stainedlens", "blurred", "reflection", "undercoverage")
        # User-friendly labels for all conditions (so occlusion, smoke, blood etc. are always visible)
        _CONDITION_LABELS = {
            "visibility": "visibility", "visible": "visibility", "crowded": "crowded",
            "occluded": "occlusion", "bleeding": "blood", "smoke": "smoke", "blurred": "blur",
            "undercoverage": "undercoverage", "reflection": "reflection", "stainedlens": "stained lens",
        }

        def _run_frame_attributes_summary(frame_path: Optional[str]) -> str:
            """Run Cholec20 multilabel on the frame and return 'operators | cond1, cond2'."""
            if not frame_path or not Path(frame_path).exists():
                return ""
            try:
                tool = interface.tools_dict.get("frame_attributes")
                if not tool and hasattr(interface, "agent") and getattr(interface.agent, "tools", None):
                    agent_tools = interface.agent.tools
                    if isinstance(agent_tools, dict):
                        tool = agent_tools.get("frame_attributes")
                    else:
                        for t in (agent_tools or []):
                            if getattr(t, "name", None) == "frame_attributes":
                                tool = t
                                break
                if not tool:
                    return ""
                out = tool.invoke({"image_path": frame_path})
                if not out or not isinstance(out, str):
                    return ""
                lines = out.strip().split("\n")
                ops, conds = "", ""
                for line in lines:
                    if "Operators present:" in line:
                        ops = line.split("Operators present:")[-1].strip() or "none"
                    elif "Active conditions:" in line:
                        conds = line.split("Active conditions:")[-1].strip() or "none"
                return f"{ops} | {conds}" if (ops or conds) else ""
            except Exception:
                return ""

        # Full names for operator categories (MSLH, MSRH, ASRH, NULL)
        _OPERATOR_FULL_NAMES = {
            "mslh": "main surgeon left hand (MSLH)",
            "msrh": "main surgeon right hand (MSRH)",
            "asrh": "assistant surgeon right hand (ASRH)",
            "null": "null operator (NULL)",
        }

        def _format_frame_alert(summary: str) -> str:
            """Operators: full form only, never NULL. Conditions: only visibility gets ✓/✗; others show label only."""
            if not summary or "|" not in summary:
                return summary or ""
            ops_part, conds_part = summary.split("|", 1)
            ops_part = ops_part.strip()
            conds_part = conds_part.strip()
            # Parse conditions: "occluded (conf 0.99), crowded (conf 0.5)" -> keys; include all (visibility gets ✓/✗)
            raw_tokens = [t.strip() for t in conds_part.split(",") if t.strip()] if conds_part and conds_part.lower() != "none" else []
            cond_list = []
            for t in raw_tokens:
                key = t.split(" (")[0].strip().lower() if " (" in t else t.lower()
                cond_list.append(key)
            # Operators and conditions on same line (side by side)
            ops_str = ""
            if ops_part and ops_part.lower() != "none":
                op_tokens = [o.strip() for o in ops_part.split(",") if o.strip()]
                op_tokens = [o for o in op_tokens if "null" not in o.lower()]
                op_full = ", ".join(_OPERATOR_FULL_NAMES.get(o.lower(), o) if o.upper() in ("MSLH", "MSRH", "ASRH", "NULL") else o for o in op_tokens)
                if op_full:
                    ops_str = f"**Operators:** {op_full}"
            cond_str = ""
            if cond_list:
                parts = []
                _VISIBILITY_KEYS = {"visibility", "visible"}
                for c in cond_list:
                    label = _CONDITION_LABELS.get(c, c)
                    if c in _VISIBILITY_KEYS:
                        sym = "✗" if c in _FRAME_ALARM_ATTRS else "✓"
                        parts.append(f"{sym} {label}")
                    else:
                        parts.append(label)
                cond_str = f"**Conditions:** {', '.join(parts)}"
            else:
                cond_str = "**Conditions:** ✓ View clear"
            line = "  ·  ".join(filter(None, [ops_str, cond_str]))
            if any(c in _FRAME_ALARM_ATTRS for c in cond_list):
                alarming = [c for c in cond_list if c in _FRAME_ALARM_ATTRS]
                alarm_labels = [_CONDITION_LABELS.get(a, a) for a in alarming]
                alarm_span = f"<span class='frame-alarm' style='color:#ff4444;font-weight:bold'>⚠ ALARM: {', '.join(alarm_labels)}</span>"
                line = (line + "  ·  " + alarm_span) if line else alarm_span
            return line or ""

        def _play_loop(vid_path, current_t):
            """Generator that yields frame updates for playback. Runs Cholec20 multilabel on every frame; alert shown in small box above chat."""
            global _video_playing
            try:
                path = Path(vid_path) if vid_path is not None else None
            except (TypeError, ValueError):
                path = None
            if not path or not path.exists():
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                return
            duration = _video_duration_sec(str(path)) or 10.0
            _video_playing = True
            t = float(current_t) if current_t is not None else 0.0
            if t >= duration:
                t = 0.0
            while _video_playing and t <= duration:
                frame = _extract_frame(str(path), t, interface.upload_dir)
                if frame:
                    interface.handle_upload(frame)
                    summary = _run_frame_attributes_summary(frame)
                    alert_text = _format_frame_alert(summary) if summary else ""
                    yield (
                        gr.update(value=min(t, duration)),
                        frame,
                        None,
                        f"▶ Playing {t:.1f}s / {duration:.1f}s",
                        frame,
                        alert_text,
                    )
                t += _PLAY_STEP_SEC * _PLAY_SPEED
                time.sleep(_PLAY_STEP_SEC)
            _video_playing = False
            yield gr.update(), gr.update(), gr.update(), "⏸ Paused", gr.update(), gr.update()

        def play_frames_generator(vid_path, current_t):
            yield from _play_loop(vid_path, current_t)

        def toggle_play_on_click(evt, arg2, arg3):
            """Click on video: if playing then pause; if paused then play. Gradio .select() passes (evt, input1, input2); in practice (evt, time_slider, video_path_state)."""
            global _video_playing
            if _video_playing:
                _video_playing = False
                yield gr.update(), gr.update(), gr.update(), "⏸ Paused", gr.update(), gr.update()
                return
            # inputs=[video_path_state, time_slider] → arg2 can be path or slider, arg3 the other; if arg2 is float it's slider
            if isinstance(arg2, (int, float)):
                current_t, vid_path = arg2, arg3
            else:
                vid_path, current_t = arg2, arg3
            yield from _play_loop(vid_path, current_t)

        def pause_playback():
            global _video_playing
            _video_playing = False
            return "⏸ Paused — running tools…"

        def on_pause_cache_tools(frame_path):
            """On Pause: run all tools in parallel except VQA and cache results for answer-from-cache."""
            # Use state frame path or fallback to interface's current image (set by upload/play/slider)
            path = frame_path
            if not path or not Path(path).exists():
                path = getattr(interface, "current_image_path", None)
            if not path or not Path(path).exists():
                return None, None, "⏸ Paused — load a video/image first, then pause."
            try:
                from surgical_copilot.parallel_tools import run_tools_parallel_no_vqa
                results = run_tools_parallel_no_vqa(interface.agent, path)
                if not results:
                    return None, None, "⏸ Paused — no tools ran (agent may have no tools loaded)"
                return results, path, "⏸ Paused — tool results cached (ask a question)"
            except Exception as e:
                return None, None, f"⏸ Paused — cache error: {e}"

        def handle_upload_unified(file):
            if file is None:
                return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), None, None, gr.update(visible=False), gr.update(), "WAITING FOR INPUT...", None, None]
            path = str(file)
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                web_path = _convert_to_mp4(path)
                dur = _video_duration_sec(web_path) or 10.0
                frame = _extract_frame(web_path, 0.0, interface.upload_dir)
                if frame: interface.handle_upload(frame)
                return [gr.update(value=web_path, visible=False), gr.update(visible=False), gr.update(maximum=dur, visible=True), web_path, frame, gr.update(value=frame, visible=True), None, f"VIDEO LOADED // {dur:.2f}s — use slider to scrub", None, None]
            interface.handle_upload(path)
            return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, path, gr.update(value=path, visible=True), None, "IMAGE READY", None, None]

        upload_file.upload(handle_upload_unified, inputs=[upload_file], outputs=[video_player, upload_file, time_slider, video_path_state, selected_frame_path, main_display, overlay_state, upload_status, cached_tool_results, cached_tool_results_image_path])

        def slider_change(vid, val):
            # Run Cholec20 multilabel on this frame; show alert in box above chat. Do not output to time_slider (avoids slider-in-slider).
            if not vid:
                return gr.update(), gr.update(), gr.update(), "NO SIGNAL", None, None, ""
            frame = _extract_frame(vid, val, interface.upload_dir)
            if frame:
                interface.handle_upload(frame)
                summary = _run_frame_attributes_summary(frame)
                alert_text = _format_frame_alert(summary) if summary else ""
                return frame, frame, None, f"SEEK: {val:.2f}s", None, None, alert_text
            return gr.update(), gr.update(), gr.update(), "SEEK ERROR", None, None, ""

        # Only update frame on release; do not pass time_slider as output to avoid nested slider
        time_slider.release(slider_change, inputs=[video_path_state, time_slider], outputs=[selected_frame_path, main_display, overlay_state, upload_status, cached_tool_results, cached_tool_results_image_path, frame_alert_box])

        play_btn.click(
            play_frames_generator,
            inputs=[video_path_state, time_slider],
            outputs=[time_slider, main_display, overlay_state, upload_status, selected_frame_path, frame_alert_box],
        )
        pause_btn.click(pause_playback, inputs=[], outputs=[upload_status]).then(
            on_pause_cache_tools,
            inputs=[selected_frame_path],
            outputs=[cached_tool_results, cached_tool_results_image_path, upload_status],
        )

        # Click on video area to toggle play / pause (Image uses .select() for click; it passes event then inputs)
        main_display.select(
            toggle_play_on_click,
            inputs=[video_path_state, time_slider],
            outputs=[time_slider, main_display, overlay_state, upload_status, selected_frame_path, frame_alert_box],
        )

        def enable_input():
            return gr.update(interactive=True, placeholder="I am a surgical co-pilot, how can I assist?")

        def apply_display_update(display_update):
            """Update main_display and overlay_state only when process_message set a new image (segmentation/remove overlay). Leaves frame intact otherwise."""
            if display_update is None:
                return gr.update(), gr.update()
            main_path, overlay_path = display_update
            # Use gr.update(value=...) so the Image component actually refreshes with the overlay image
            path_str = str(main_path) if main_path else None
            if path_str and Path(path_str).exists():
                return (gr.update(value=path_str, visible=True), overlay_path if overlay_path is not None else None)
            return (gr.update(), gr.update())

        # 1. TEXT SUBMIT — chat chain does NOT output to main_display/overlay_state so the frame stays visible and no loading on video. show_progress=hidden avoids rotating Gradio loader in chat/input.
        txt.submit(interface.add_message, [txt, selected_frame_path, chatbot], [chatbot, txt, msg_state], show_progress="hidden") \
           .then(interface.process_message, [msg_state, selected_frame_path, chatbot, overlay_state, cached_tool_results, cached_tool_results_image_path], [chatbot, selected_frame_path, txt, tts_audio, display_update_state], show_progress="hidden") \
           .then(apply_display_update, [display_update_state], [main_display, overlay_state], show_progress="hidden") \
           .then(enable_input, None, [txt])

        # 2. VOICE SUBMIT: mic -> transcribe -> show in chat -> process -> TTS playback
        def _audio_in_to_path(audio_in):
            """Extract file path from Gradio Audio value (str, dict, list, or object with .path)."""
            if audio_in is None:
                return None
            if isinstance(audio_in, str) and audio_in.strip():
                return audio_in.strip()
            if isinstance(audio_in, dict):
                p = audio_in.get("path") or audio_in.get("name") or audio_in.get("url")
                if p and isinstance(p, str):
                    return p.replace("file://", "", 1) if p.startswith("file://") else p
                return None
            if isinstance(audio_in, (list, tuple)) and audio_in:
                return _audio_in_to_path(audio_in[0])
            if hasattr(audio_in, "path"):
                return getattr(audio_in, "path", None)
            return None

        def transcribe_logic(audio_in):
            """Take mic recording (path str, dict, or list), return transcribed text for chat and msg_state."""
            path = _audio_in_to_path(audio_in)
            if not path or not str(path).strip():
                return "**(No recording.)** Click the **microphone** to start, speak, then click the **microphone again** to stop — your message is sent automatically. Or upload an audio file."
            path = str(path).strip()
            # Gradio can write the file slightly after stop_recording; retry so server sees it
            for _ in range(12):
                if Path(path).exists():
                    break
                time.sleep(0.5)
            if not Path(path).exists():
                return "(Recording not ready — wait and try again. On remote connections the file may take longer.)"
            try:
                from surgical_copilot.audio.stt import transcribe_audio
                out = transcribe_audio(path)
                text = (out or "").strip()
                if text.startswith("Error:") or text.startswith("[Voice error:"):
                    return text
                return text if text else "(No speech detected — try again.)"
            except ImportError:
                return "[Voice: Install pip install faster-whisper (or openai-whisper) for speech-to-text.]"
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"[Voice error: {e}]"

        def clear_mic():
            return None

        mic.stop_recording(transcribe_logic, inputs=[mic], outputs=[msg_state], show_progress="hidden") \
           .then(interface.add_message, [msg_state, selected_frame_path, chatbot], [chatbot, txt, msg_state], show_progress="hidden") \
           .then(interface.process_message, [msg_state, selected_frame_path, chatbot, overlay_state, cached_tool_results, cached_tool_results_image_path], [chatbot, selected_frame_path, txt, tts_audio, display_update_state], show_progress="hidden") \
           .then(apply_display_update, [display_update_state], [main_display, overlay_state], show_progress="hidden") \
           .then(enable_input, None, [txt]) \
           .then(clear_mic, None, [mic])

    return demo, custom_css