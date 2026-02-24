"""
Shared input schemas and base for surgical co-pilot tools.
"""
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, Field, model_validator


class SurgicalImageInput(BaseModel):
    """Input for tools that take a single surgical/endoscopy image."""

    image_path: str = Field(
        ...,
        description="Path to the surgical or endoscopy image file (e.g. JPG, PNG)",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_path_alias(cls, data: Any) -> Any:
        """Accept 'path' as alias for 'image_path' (agent may output either)."""
        if isinstance(data, dict) and "path" in data and "image_path" not in data:
            data = {**data, "image_path": data["path"]}
        return data


class SurgicalVideoInput(BaseModel):
    """Input for tools that take a video or frame sequence."""

    video_or_frames_path: str = Field(
        ...,
        description="Path to a video file, directory of frames, or single image",
    )
    frame_index: Optional[int] = Field(
        None,
        description="Optional frame index if using a video or ordered frames",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_path_aliases(cls, data: Any) -> Any:
        """Accept 'image_path' or 'path' as alias for 'video_or_frames_path' (agent may output either)."""
        if isinstance(data, dict):
            p = data.get("video_or_frames_path") or data.get("image_path") or data.get("path")
            if p and "video_or_frames_path" not in data:
                data = {**data, "video_or_frames_path": p}
        return data


class SurgicalQuestionInput(BaseModel):
    """Input for VQA-style tools: image + question."""

    image_path: str = Field(..., description="Path to the surgical/endoscopy image")
    question: str = Field(
        default="What is shown in this surgical image?",
        description="Natural language question about the image",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_path_alias_and_default_question(cls, data: Any) -> Any:
        """Accept 'path' as alias for 'image_path'; default question when missing."""
        if isinstance(data, dict):
            if "path" in data and "image_path" not in data:
                data = {**data, "image_path": data["path"]}
            if not data.get("question"):
                data = {**data, "question": "What is shown in this surgical image?"}
        return data
