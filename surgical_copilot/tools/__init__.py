"""
Surgical co-pilot tools. Import this module to register all tools with the registry.
Add new tools by:
  1. Creating a new file under surgical_copilot/tools/ (e.g. my_tool.py)
  2. Defining a BaseTool subclass and a factory function
  3. Registering with @register("tool_name") and calling the factory in ALL_TOOLS below
  4. Importing the module here so it runs on load
"""
from surgical_copilot.tools.base import (
    SurgicalImageInput,
    SurgicalQuestionInput,
    SurgicalVideoInput,
)

# Import each tool module so decorators run and tools are registered.
from surgical_copilot.tools import scene_segmentation  # noqa: F401
from surgical_copilot.tools import phase_detection  # noqa: F401
from surgical_copilot.tools import instrument_tracking  # noqa: F401
from surgical_copilot.tools import ssg_vqa  # noqa: F401
from surgical_copilot.tools import triplet_recognition  # noqa: F401
from surgical_copilot.tools import critical_view_of_safety  # noqa: F401
from surgical_copilot.tools import frame_attributes  # noqa: F401
from surgical_copilot.tools import object_detection_merged  # noqa: F401
from surgical_copilot.tools import rag_retrieval  # noqa: F401

__all__ = [
    "SurgicalImageInput",
    "SurgicalQuestionInput",
    "SurgicalVideoInput",
]
